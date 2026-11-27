//! `train plan` and `train critic`: the same encoder, different heads.
//!
//! Plan (good examples only): cross-entropy on the recipe, L1 on
//! `start_bars` and `swap_bars` (in units of `BARS_SCALE` bars), binary
//! cross-entropy with soft targets on the 80 × 8 lane gains.
//! Critic (all examples): binary cross-entropy on "good" and on every issue.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, ensure};
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use sone_infer::config::{BARS_SCALE, BeatConfig, BeatKind};
use sone_models::{BeatNet, BeatOut};
use serde::{Deserialize, Serialize};

use crate::config::{Optim, Run};
use crate::data_examples::{self, BeatBatch, ExamplesData, Packed};
use crate::losses::{bce_with_logits, cross_entropy};
use crate::run::{Loader, Log, Meter, RunDir};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BeatModel {
    #[serde(default)]
    pub dim: Option<usize>,
    #[serde(default)]
    pub layers: Option<usize>,
    #[serde(default)]
    pub heads: Option<usize>,
    #[serde(default)]
    pub ff: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatLoss {
    /// Plan: recipe, bars, lanes. Critic: good, issues.
    #[serde(default = "one")]
    pub recipe: f64,
    #[serde(default = "one")]
    pub bars: f64,
    #[serde(default = "one")]
    pub lanes: f64,
    #[serde(default = "one")]
    pub good: f64,
    #[serde(default = "one")]
    pub issues: f64,
    /// Positive weight of the issue labels (each issue is rare).
    #[serde(default = "one")]
    pub issue_pos_weight: f64,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatTrainConfig {
    pub run: Run,
    pub optim: Optim,
    #[serde(default)]
    pub model: BeatModel,
    pub data: ExamplesData,
    pub loss: BeatLoss,
}

fn model_config(kind: BeatKind, m: &BeatModel) -> BeatConfig {
    let mut c = match kind {
        BeatKind::Cortex => BeatConfig::cortex(),
        BeatKind::Cingulate => BeatConfig::cingulate(),
    };
    c.dim = m.dim.unwrap_or(c.dim);
    c.layers = m.layers.unwrap_or(c.layers);
    c.heads = m.heads.unwrap_or(c.heads);
    c.ff = m.ff.unwrap_or(c.ff);
    c
}

pub fn terms(kind: BeatKind) -> &'static [&'static str] {
    match kind {
        BeatKind::Cortex => &["recipe", "bars", "lanes"],
        BeatKind::Cingulate => &["good", "issues"],
    }
}

struct In<B: Backend> {
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    g: Tensor<B, 2>,
}

fn inputs<B: Backend>(c: &BeatConfig, x: &BeatBatch, device: &B::Device) -> In<B> {
    let n = x.n;
    In {
        a: Tensor::from_data(TensorData::new(x.a.clone(), [n, c.a_len, c.a_features()]), device),
        b: Tensor::from_data(TensorData::new(x.b.clone(), [n, c.b_len, c.features]), device),
        g: Tensor::from_data(TensorData::new(x.g.clone(), [n, c.globals()]), device),
    }
}

fn losses<B: Backend>(c: &BeatConfig, out: BeatOut<B>, x: &BeatBatch, l: &BeatLoss, device: &B::Device) -> Vec<Tensor<B, 1>> {
    let n = x.n;
    match out {
        BeatOut::Cortex { recipe, bars, lanes } => {
            let one_hot = Tensor::<B, 2>::from_data(TensorData::new(x.recipe.clone(), [n, 4]), device);
            let tb = Tensor::<B, 2>::from_data(TensorData::new(x.bars.clone(), [n, 2]), device);
            let tl = Tensor::<B, 3>::from_data(TensorData::new(x.lanes.clone(), [n, c.a_len, c.lanes]), device);
            vec![
                cross_entropy(recipe, one_hot).mean() * l.recipe,
                ((bars - tb).abs() / BARS_SCALE as f64).mean() * l.bars,
                bce_with_logits(lanes, tl, 1.0).mean() * l.lanes,
            ]
        }
        BeatOut::Cingulate(logits) => {
            let k = 1 + c.issues();
            let y = Tensor::<B, 2>::from_data(TensorData::new(x.labels.clone(), [n, k]), device);
            vec![
                bce_with_logits(logits.clone().narrow(1, 0, 1), y.clone().narrow(1, 0, 1), 1.0).mean() * l.good,
                bce_with_logits(logits.narrow(1, 1, k - 1), y.narrow(1, 1, k - 1), l.issue_pos_weight).mean() * l.issues,
            ]
        }
    }
}

pub fn run<B: AutodiffBackend>(kind: BeatKind, cfg: BeatTrainConfig, config_path: &Path, device: B::Device) -> Result<()> {
    let c = model_config(kind, &cfg.model);
    let name = if kind == BeatKind::Cortex { "plan" } else { "critic" };
    B::seed(&device, cfg.run.seed);
    let dir = RunDir::create(&crate::config::expand(&cfg.run.dir), config_path)?;
    let (mut train, mut valid) = (Packed::default(), Packed::default());
    let keep = |ex: &sone_core::Example| kind == BeatKind::Cingulate || ex.label.good == 1;
    let separate_valid = !cfg.data.valid.is_empty();
    data_examples::read(&cfg.data.train, cfg.data.max_examples, keep, |ex| {
        if !separate_valid && data_examples::held_out(&ex.id, cfg.data.valid_fraction) {
            valid.push(&c, &ex);
        } else {
            train.push(&c, &ex);
        }
    })?;
    if separate_valid {
        data_examples::read(&cfg.data.valid, None, keep, |ex| valid.push(&c, &ex))?;
    }
    ensure!(train.n > 0, "no {name} training examples in {:?}", cfg.data.train);
    println!("{name}: {:.3} M parameters, {} training examples, {} validation examples", c.num_params() as f64 / 1e6, train.n, valid.n);

    let mut model = BeatNet::<B>::new(&c, cfg.run.seed, &device);
    let [a, b, g] = train.stats(&c);
    model.input_a.set(&a.0, &a.1, &device);
    model.input_b.set(&b.0, &b.1, &device);
    model.input_g.set(&g.0, &g.1, &device);
    let mut ocfg = AdamWConfig::new().with_weight_decay(cfg.optim.weight_decay);
    if let Some(clip) = cfg.optim.grad_clip {
        ocfg = ocfg.with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(clip)));
    }
    let mut optim = ocfg.init::<B, BeatNet<B>>();
    let mut start = 0;
    if let Some((step, m, o)) = dir.load(model.clone(), optim.clone(), &device)? {
        println!("resuming at step {step}");
        (start, model, optim) = (step, m, o);
    }
    let train = Arc::new(train);
    let (tc, cc, bs) = (train.clone(), c.clone(), cfg.run.batch_size);
    let loader = Loader::spawn(cfg.run.loader_threads, cfg.run.seed ^ start as u64, move |rng| Ok(tc.random_batch(&cc, bs, rng)));
    let names = terms(kind);
    let mut columns = vec!["step", "lr", "loss"];
    columns.extend(names);
    columns.extend(["examples_per_s", "elapsed_s"]);
    let mut log = Log::open(&dir.root.join("train.csv"), &columns)?;
    let vcols: &[&str] = match kind {
        BeatKind::Cortex => &["step", "loss", "recipe_acc", "bars_mae", "lanes_mae"],
        BeatKind::Cingulate => &["step", "loss", "good_acc", "issue_f1_macro", "issue_exact"],
    };
    let mut vlog = Log::open(&dir.root.join("valid.csv"), vcols)?;
    let mut meter = Meter::<B>::new(1 + names.len(), &device);
    let t0 = Instant::now();
    let checkpoint_every = cfg.run.checkpoint_every.unwrap_or(cfg.run.valid_every);
    for step in start..cfg.run.steps {
        let x = loader.next()?;
        let i = inputs::<B>(&c, &x, &device);
        let parts = losses(&c, model.forward(i.a, i.b, i.g), &x, &cfg.loss, &device);
        let loss = parts.iter().skip(1).fold(parts[0].clone(), |s, p| s + p.clone());
        let mut shown = vec![loss.clone().detach()];
        shown.extend(parts.iter().map(|p| p.clone().detach()));
        meter.add(&shown, x.n);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let lr = cfg.optim.lr_at(step, cfg.run.steps);
        model = optim.step(lr, model, grads);
        let done = step + 1;
        if done % cfg.run.log_every == 0 || done == cfg.run.steps {
            let (means, rate) = meter.take();
            let mut row = vec![done as f64, lr];
            row.extend(means);
            row.extend([rate, t0.elapsed().as_secs_f64()]);
            log.row(&row)?;
        }
        if done % cfg.run.valid_every == 0 || done == cfg.run.steps {
            if valid.n > 0 {
                let v = validate(&model.valid(), &c, &valid, &cfg.loss, &device);
                let mut row = vec![done as f64];
                row.extend(v);
                vlog.row(&row)?;
            }
            let path = dir.export(name, done, &c, &model.valid().export())?;
            println!("exported {}", path.display());
        }
        if done % checkpoint_every == 0 || done == cfg.run.steps {
            dir.save(done, &model, &optim)?;
        }
    }
    Ok(())
}

fn validate<B: Backend>(model: &BeatNet<B>, c: &BeatConfig, data: &Packed, l: &BeatLoss, device: &B::Device) -> Vec<f64> {
    let mut loss = 0.0;
    let (mut a1, mut a2, mut a3, mut count) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    let issues = c.issues();
    let mut conf = vec![[0usize; 3]; issues]; // hits, predicted, true
    let idx: Vec<usize> = (0..data.n).collect();
    for chunk in idx.chunks(256) {
        let x = data.batch(c, chunk);
        let i = inputs::<B>(c, &x, device);
        let out = model.forward(i.a, i.b, i.g);
        match &out {
            BeatOut::Cortex { recipe, bars, lanes } => {
                let r = sone_models::export::values(recipe.clone());
                let b = sone_models::export::values(bars.clone());
                let g = sone_models::export::values(sigmoid(lanes.clone()));
                for k in 0..x.n {
                    let best = (0..4).max_by(|&p, &q| r[k * 4 + p].total_cmp(&r[k * 4 + q])).unwrap_or(0);
                    a1 += x.recipe[k * 4 + best] as f64;
                    a2 += ((b[k * 2] - x.bars[k * 2]).abs() + (b[k * 2 + 1] - x.bars[k * 2 + 1]).abs()) as f64 / 2.0;
                }
                a3 += g.iter().zip(&x.lanes).map(|(p, t)| (p - t).abs() as f64).sum::<f64>() / (c.a_len * c.lanes) as f64;
            }
            BeatOut::Cingulate(logits) => {
                let p = sone_models::export::values(sigmoid(logits.clone()));
                let k1 = 1 + issues;
                for k in 0..x.n {
                    a1 += f64::from((p[k * k1] > 0.5) == (x.labels[k * k1] > 0.5));
                    let mut exact = true;
                    for j in 0..issues {
                        let (pr, tr) = (p[k * k1 + 1 + j] > 0.5, x.labels[k * k1 + 1 + j] > 0.5);
                        conf[j][0] += usize::from(pr && tr);
                        conf[j][1] += usize::from(pr);
                        conf[j][2] += usize::from(tr);
                        exact &= pr == tr;
                    }
                    a3 += f64::from(exact);
                }
            }
        }
        let parts = losses(c, out, &x, l, device);
        loss += parts.into_iter().map(|p| p.into_scalar().elem::<f64>()).sum::<f64>() * x.n as f64;
        count += x.n;
    }
    let n = count.max(1) as f64;
    match c.kind {
        BeatKind::Cortex => vec![loss / n, a1 / n, a2 / n, a3 / n],
        BeatKind::Cingulate => {
            // macro F1 over the issues that occur in the validation set
            let f1: Vec<f64> = conf.iter().filter(|x| x[2] > 0).map(|x| 2.0 * x[0] as f64 / (x[1] + x[2]) as f64).collect();
            vec![loss / n, a1 / n, if f1.is_empty() { f64::NAN } else { f1.iter().sum::<f64>() / f1.len() as f64 }, a3 / n]
        }
    }
}

pub fn run_cortex<B: AutodiffBackend>(cfg: BeatTrainConfig, config_path: &Path, device: B::Device) -> Result<()> {
    run::<B>(BeatKind::Cortex, cfg, config_path, device)
}

pub fn run_cingulate<B: AutodiffBackend>(cfg: BeatTrainConfig, config_path: &Path, device: B::Device) -> Result<()> {
    run::<B>(BeatKind::Cingulate, cfg, config_path, device)
}
