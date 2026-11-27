//! `train ear`: frame heads (beat, downbeat, voice, line end, phrase start)
//! with binary cross-entropy, masked per song where a label is unknown;
//! key with cross-entropy; tuning with L1.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use sone_infer::TympanumConfig;
use sone_infer::tympanum::TUNING_RANGE;
use sone_models::{TympanumNet, TympanumOut};
use serde::{Deserialize, Serialize};

use crate::config::{Optim, Run};
use crate::data_tympanum::{self, TympanumBatch, TympanumData, TympanumSong};
use crate::losses::{bce_with_logits, cross_entropy};
use crate::run::{Loader, Log, Meter, RunDir};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TympanumModel {
    #[serde(default)]
    pub channels: Option<usize>,
    #[serde(default)]
    pub gru_hidden: Option<usize>,
    #[serde(default)]
    pub gru_layers: Option<usize>,
}

impl TympanumModel {
    pub fn config(&self) -> TympanumConfig {
        let mut c = TympanumConfig::default();
        if let Some(v) = self.channels {
            c.channels = v;
        }
        if let Some(v) = self.gru_hidden {
            c.gru_hidden = v;
        }
        if let Some(v) = self.gru_layers {
            c.gru_layers = v;
        }
        c
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TympanumLoss {
    /// Positive-class weight per frame head (events are rare frames).
    pub pos_weight: [f64; 5],
    /// Weight of each frame head in the loss.
    pub head_weight: [f64; 5],
    pub key: f64,
    pub tuning: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TympanumTrainConfig {
    pub run: Run,
    pub optim: Optim,
    #[serde(default)]
    pub model: TympanumModel,
    pub data: TympanumData,
    pub loss: TympanumLoss,
}

pub const TERMS: [&str; 7] = ["beat", "downbeat", "voice", "line_end", "phrase_start", "key", "tuning"];

struct Tensors<B: Backend> {
    mel: Tensor<B, 3>,
    targets: Tensor<B, 3>,
    head_mask: Tensor<B, 2>,
    key: Tensor<B, 2>,
    tuning: Tensor<B, 1>,
    tuning_mask: Tensor<B, 1>,
}

fn tensors<B: Backend>(b: TympanumBatch, mels: usize, device: &B::Device) -> Tensors<B> {
    let (n, t) = (b.batch, b.frames);
    Tensors {
        mel: Tensor::from_data(TensorData::new(b.mel, [n, t, mels]), device),
        targets: Tensor::from_data(TensorData::new(b.targets, [n, t, 5]), device),
        head_mask: Tensor::from_data(TensorData::new(b.head_mask, [n, 5]), device),
        key: Tensor::from_data(TensorData::new(b.key, [n, 24]), device),
        tuning: Tensor::from_data(TensorData::new(b.tuning, [n]), device),
        tuning_mask: Tensor::from_data(TensorData::new(b.tuning_mask, [n]), device),
    }
}

/// The seven loss terms, each a scalar: five frame heads, key, tuning.
fn losses<B: Backend>(out: &TympanumOut<B>, t: &Tensors<B>, c: &TympanumLoss) -> Vec<Tensor<B, 1>> {
    let [n, frames, _] = out.frames.dims();
    let mut terms = Vec::new();
    for h in 0..5 {
        let x = out.frames.clone().narrow(2, h, 1).reshape([n, frames]);
        let y = t.targets.clone().narrow(2, h, 1).reshape([n, frames]);
        let per_song = bce_with_logits(x, y, c.pos_weight[h]).mean_dim(1).reshape([n]);
        let mask = t.head_mask.clone().narrow(1, h, 1).reshape([n]);
        terms.push((per_song * mask.clone()).sum() / (mask.sum() + 1e-6));
    }
    let known = t.key.clone().sum_dim(1).reshape([n]);
    terms.push(cross_entropy(out.key.clone(), t.key.clone()).sum() / (known.sum() + 1e-6));
    let cents = out.tuning.clone().reshape([n]).tanh() * TUNING_RANGE as f64;
    let err = (cents - t.tuning.clone()).abs() / TUNING_RANGE as f64 * t.tuning_mask.clone();
    terms.push(err.sum() / (t.tuning_mask.clone().sum() + 1e-6));
    terms
}

fn total<B: Backend>(terms: &[Tensor<B, 1>], c: &TympanumLoss) -> Tensor<B, 1> {
    let mut sum = terms[5].clone() * c.key + terms[6].clone() * c.tuning;
    for h in 0..5 {
        sum = sum + terms[h].clone() * c.head_weight[h];
    }
    sum
}

pub fn run<B: AutodiffBackend>(cfg: TympanumTrainConfig, config_path: &Path, device: B::Device) -> Result<()> {
    let c = cfg.model.config();
    B::seed(&device, cfg.run.seed);
    let dir = RunDir::create(&crate::config::expand(&cfg.run.dir), config_path)?;
    let train = data_tympanum::index(&cfg.data.train)?;
    let valid = data_tympanum::index(&cfg.data.valid)?;
    ensure!(!train.is_empty(), "no Ear label files in {:?}", cfg.data.train);
    let len = (cfg.data.chunk_seconds * c.sample_rate as f64) as usize;
    println!("ear: {:.3} M parameters, {} training songs, {} validation songs, {} fps", c.num_params() as f64 / 1e6, train.len(), valid.len(), c.fps());

    let mut model = TympanumNet::<B>::new(&c, cfg.run.seed, &device);
    let mut ocfg = AdamWConfig::new().with_weight_decay(cfg.optim.weight_decay);
    if let Some(clip) = cfg.optim.grad_clip {
        ocfg = ocfg.with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(clip)));
    }
    let mut optim = ocfg.init::<B, TympanumNet<B>>();
    let mut start = 0;
    if let Some((step, m, o)) = dir.load(model.clone(), optim.clone(), &device)? {
        println!("resuming at step {step}");
        (start, model, optim) = (step, m, o);
    }
    let (bs, cc) = (cfg.run.batch_size, c.clone());
    let loader = Loader::spawn(cfg.run.loader_threads, cfg.run.seed ^ start as u64, move |rng| data_tympanum::sample(&train, bs, len, &cc, rng));
    let mut columns = vec!["step", "lr", "loss"];
    columns.extend(TERMS);
    columns.extend(["examples_per_s", "elapsed_s"]);
    let mut log = Log::open(&dir.root.join("train.csv"), &columns)?;
    let mut vlog = Log::open(&dir.root.join("valid.csv"), &["step", "loss", "beat_f", "downbeat_f", "voice_acc", "key_acc", "tuning_mae_cents"])?;
    let mut meter = Meter::<B>::new(1 + TERMS.len(), &device);
    let t0 = Instant::now();
    let checkpoint_every = cfg.run.checkpoint_every.unwrap_or(cfg.run.valid_every);
    for step in start..cfg.run.steps {
        let batch = loader.next()?;
        let n = batch.batch;
        let t = tensors::<B>(batch, c.n_mels, &device);
        let out = model.forward(t.mel.clone());
        let terms = losses(&out, &t, &cfg.loss);
        let loss = total(&terms, &cfg.loss);
        let mut shown = vec![loss.clone().detach()];
        shown.extend(terms.iter().map(|x| x.clone().detach()));
        meter.add(&shown, n);
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
            if let Some(v) = validate(&model.valid(), &valid, len, &cfg, &c, &device)? {
                vlog.row(&[done as f64, v[0], v[1], v[2], v[3], v[4], v[5]])?;
            }
            let path = dir.export("tympanum", done, &c, &model.valid().export())?;
            println!("exported {}", path.display());
        }
        if done % checkpoint_every == 0 || done == cfg.run.steps {
            dir.save(done, &model, &optim)?;
        }
    }
    Ok(())
}

/// Loss, beat and downbeat F-measure (±70 ms), voice frame accuracy, key
/// accuracy and tuning error over the validation songs.
fn validate<B: Backend>(model: &TympanumNet<B>, songs: &[TympanumSong], len: usize, cfg: &TympanumTrainConfig, c: &TympanumConfig, device: &B::Device) -> Result<Option<[f64; 6]>> {
    if songs.is_empty() {
        return Ok(None);
    }
    let (mut loss, mut chunks) = (0.0, 0usize);
    let mut f = [[0usize; 3]; 2]; // per head: hits, predicted, true
    let (mut voice_ok, mut voice_n) = (0usize, 0usize);
    let (mut key_ok, mut key_n, mut tun_err, mut tun_n) = (0usize, 0usize, 0.0f64, 0usize);
    let tolerance = (0.07 * c.fps()).round() as i64;
    for song in songs {
        for batch in data_tympanum::song_chunks(song, len, cfg.data.valid_seconds, c)? {
            let head_mask = batch.head_mask.clone();
            let targets = batch.targets.clone();
            let frames = batch.frames;
            let t = tensors::<B>(batch, c.n_mels, device);
            let out = model.forward(t.mel.clone());
            loss += total(&losses(&out, &t, &cfg.loss), &cfg.loss).into_scalar().elem::<f64>();
            chunks += 1;
            let probs = sone_models::export::values(sigmoid(out.frames.clone()));
            for (slot, h) in [(0usize, 0usize), (1, 1)] {
                if head_mask[h] == 0.0 {
                    continue;
                }
                let p: Vec<f32> = (0..frames).map(|i| probs[i * 5 + h]).collect();
                let pred = peaks(&p, 0.5);
                let truth: Vec<i64> = (0..frames).filter(|&i| targets[i * 5 + h] == 1.0).map(|i| i as i64).collect();
                let hits = truth.iter().filter(|&&g| pred.iter().any(|&q| (q - g).abs() <= tolerance)).count();
                f[slot][0] += hits;
                f[slot][1] += pred.len();
                f[slot][2] += truth.len();
            }
            if head_mask[2] == 1.0 {
                for i in 0..frames {
                    voice_ok += usize::from((probs[i * 5 + 2] > 0.5) == (targets[i * 5 + 2] > 0.5));
                    voice_n += 1;
                }
            }
            if let Some(k) = song.key {
                let logits = sone_models::export::values(out.key.clone());
                let best = (0..24).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap_or(0);
                key_ok += usize::from(best == k);
                key_n += 1;
            }
            if let Some(cents) = song.label.tuning_cents {
                let raw = sone_models::export::values(out.tuning.clone())[0];
                tun_err += (TUNING_RANGE * raw.tanh() - cents).abs() as f64;
                tun_n += 1;
            }
        }
    }
    let fm = |x: [usize; 3]| if x[1] + x[2] == 0 { f64::NAN } else { 2.0 * x[0] as f64 / (x[1] + x[2]) as f64 };
    let ratio = |a: usize, b: usize| if b == 0 { f64::NAN } else { a as f64 / b as f64 };
    Ok(Some([loss / chunks.max(1) as f64, fm(f[0]), fm(f[1]), ratio(voice_ok, voice_n), ratio(key_ok, key_n), if tun_n == 0 { f64::NAN } else { tun_err / tun_n as f64 }]))
}

/// Frames that are local maxima (±2 frames) above `threshold`.
pub fn peaks(p: &[f32], threshold: f32) -> Vec<i64> {
    (0..p.len())
        .filter(|&i| p[i] >= threshold && (i.saturating_sub(2)..(i + 3).min(p.len())).all(|j| p[j] <= p[i] && (j >= i || p[j] < p[i])))
        .map(|i| i as i64)
        .collect()
}
