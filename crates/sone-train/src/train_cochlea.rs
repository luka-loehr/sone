//! `train stems`: supervised (real stems) or as the student of teacher
//! models (teacher stems as targets): the program is the same, only the
//! song directories differ.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use sone_infer::CochleaConfig;
use sone_infer::chunk::Plan;
use sone_models::{Stft, CochleaNet};
use serde::{Deserialize, Serialize};

use crate::config::{Optim, Run};
use crate::data_cochlea::{self, Sampler, Song, CochleaData};
use crate::losses::{COCHLEA_TERMS, CochleaLossConfig, cochlea_loss};
use crate::run::{Loader, Log, Meter, RunDir, samples, sdr};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CochleaModel {
    /// `small`, `base` or `large`.
    pub size: String,
    #[serde(default)]
    pub dim: Option<usize>,
    #[serde(default)]
    pub blocks: Option<usize>,
    #[serde(default)]
    pub mask_hidden: Option<usize>,
}

impl CochleaModel {
    pub fn config(&self) -> Result<CochleaConfig> {
        let mut c = CochleaConfig::size(&self.size).ok_or_else(|| anyhow::anyhow!("unknown size {}", self.size))?;
        if let Some(d) = self.dim {
            c.dim = d;
        }
        if let Some(b) = self.blocks {
            let cycle = [1, 2, 4, 8, 16, 32];
            c.blocks = b;
            c.dilations = (0..b).map(|i| cycle[i % cycle.len()]).collect();
        }
        if let Some(m) = self.mask_hidden {
            c.mask_hidden = m;
        }
        c.validate()?;
        Ok(c)
    }
}

/// How validation (and `sone-infer`) cut a song: chunk and overlap in seconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunking {
    pub chunk_seconds: f64,
    pub overlap_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CochleaTrainConfig {
    pub run: Run,
    pub optim: Optim,
    pub model: CochleaModel,
    pub data: CochleaData,
    pub loss: CochleaLossConfig,
    pub infer: Chunking,
}

pub fn run<B: AutodiffBackend>(cfg: CochleaTrainConfig, config_path: &Path, device: B::Device) -> Result<()> {
    let model_cfg = cfg.model.config()?;
    B::seed(&device, cfg.run.seed);
    let dir = RunDir::create(&crate::config::expand(&cfg.run.dir), config_path)?;
    let len = samples(cfg.data.chunk_seconds, model_cfg.hop);
    let mixture_dir = cfg.data.mixture_dir.as_ref().map(|p| crate::config::expand(p));
    let train_songs = data_cochlea::index(&cfg.data.train, mixture_dir.as_deref())?;
    let valid_songs: Vec<Song> = data_cochlea::index(&cfg.data.valid, mixture_dir.as_deref())?.into_iter().take(cfg.data.valid_songs).collect();
    ensure!(!train_songs.is_empty(), "no training songs found in {:?}", cfg.data.train);
    println!(
        "stems {}: {:.2} M parameters, {} training songs, {} validation songs, chunks of {:.2} s",
        cfg.model.size,
        model_cfg.num_params() as f64 / 1e6,
        train_songs.len(),
        valid_songs.len(),
        len as f64 / 44_100.0
    );

    let mut model = CochleaNet::<B>::new(&model_cfg, cfg.run.seed, &device);
    let mut optim_cfg = AdamWConfig::new().with_weight_decay(cfg.optim.weight_decay);
    if let Some(clip) = cfg.optim.grad_clip {
        optim_cfg = optim_cfg.with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(clip)));
    }
    let mut optim = optim_cfg.init::<B, CochleaNet<B>>();
    let mut start = 0;
    if let Some((step, m, o)) = dir.load(model.clone(), optim.clone(), &device)? {
        println!("resuming at step {step}");
        (start, model, optim) = (step, m, o);
    }

    let sampler = Sampler::new(train_songs, cfg.data.clone(), len)?;
    let batch_size = cfg.run.batch_size;
    let loader = Loader::spawn(cfg.run.loader_threads, cfg.run.seed ^ start as u64, move |rng| sampler.batch(batch_size, rng));
    let stft = Stft::<B>::new(model_cfg.n_fft, model_cfg.hop, &device);
    let multires: Vec<Stft<B>> = cfg.loss.multires_ffts.iter().map(|&n| Stft::new(n, n / 4, &device)).collect();
    let mut columns = vec!["step", "lr", "loss"];
    columns.extend(COCHLEA_TERMS);
    columns.extend(["examples_per_s", "elapsed_s"]);
    let mut log = Log::open(&dir.root.join("train.csv"), &columns)?;
    let mut vlog = Log::open(
        &dir.root.join("valid.csv"),
        &["step", "sdr_vocals", "sdr_drums", "sdr_bass", "sdr_other", "sdr_mean", "leak", "mix_sdr_mean"],
    )?;
    let mut meter = Meter::<B>::new(1 + COCHLEA_TERMS.len(), &device);
    let t0 = Instant::now();
    let checkpoint_every = cfg.run.checkpoint_every.unwrap_or(cfg.run.valid_every);
    if start == 0 {
        // the untrained network, as a reference point for the log
        if let Some(v) = validate(&model.valid(), &valid_songs, &cfg, &model_cfg, &device)? {
            vlog.row(&[0.0, v.sdr[0], v.sdr[1], v.sdr[2], v.sdr[3], v.sdr.iter().sum::<f64>() / 4.0, v.leak, v.mix_sdr.iter().sum::<f64>() / 4.0])?;
        }
    }

    for step in start..cfg.run.steps {
        let batch = loader.next()?;
        let mix = Tensor::<B, 3>::from_data(TensorData::new(batch.mix, [batch.batch, 2, batch.len]), &device);
        let target = Tensor::<B, 4>::from_data(TensorData::new(batch.stems, [batch.batch, 4, 2, batch.len]), &device);
        let out = model.forward(&stft, mix);
        let loss = cochlea_loss(&out, target, &stft, &multires, &cfg.loss);
        let total = loss.total(&cfg.loss);
        let mut terms = vec![total.clone().detach()];
        terms.extend(loss.terms().into_iter().map(|t| t.detach()));
        meter.add(&terms, batch.batch);
        let grads = GradientsParams::from_grads(total.backward(), &model);
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
            let v = validate(&model.valid(), &valid_songs, &cfg, &model_cfg, &device)?;
            if let Some(v) = v {
                vlog.row(&[done as f64, v.sdr[0], v.sdr[1], v.sdr[2], v.sdr[3], v.sdr.iter().sum::<f64>() / 4.0, v.leak, v.mix_sdr.iter().sum::<f64>() / 4.0])?;
            }
            let path = dir.export("cochlea", done, &model_cfg, &model.valid().export())?;
            println!("exported {}", path.display());
        }
        if done % checkpoint_every == 0 || done == cfg.run.steps {
            dir.save(done, &model, &optim)?;
        }
    }
    Ok(())
}

pub struct Validation {
    /// SDR per stem, mean over songs.
    pub sdr: [f64; 4],
    /// SDR of the mixture used as every stem's estimate ("doing nothing").
    pub mix_sdr: [f64; 4],
    pub leak: f64,
}

/// Separate every validation song in overlapping chunks, as `sone-infer`
/// does, and measure SDR per stem against the targets.
pub fn validate<B: Backend>(model: &CochleaNet<B>, songs: &[Song], cfg: &CochleaTrainConfig, c: &CochleaConfig, device: &B::Device) -> Result<Option<Validation>> {
    if songs.is_empty() {
        return Ok(None);
    }
    let stft = Stft::<B>::new(c.n_fft, c.hop, device);
    let chunk = samples(cfg.infer.chunk_seconds, c.hop);
    let overlap = samples(cfg.infer.overlap_seconds, c.hop).min(chunk - c.hop);
    let (mut sdr_sum, mut mix_sum, mut leak_sum) = ([0.0f64; 4], [0.0f64; 4], 0.0);
    for song in songs {
        let start = (cfg.data.valid_offset_seconds * 44_100.0) as u64;
        let start = if start as usize + chunk < song.len as usize { start } else { 0 };
        let n = ((cfg.data.valid_seconds * 44_100.0) as usize).min(song.len as usize - start as usize);
        let targets: Vec<[Vec<f32>; 2]> = (0..4).map(|s| song.read_stem(s, start, n)).collect::<Result<_>>()?;
        let mix: [Vec<f32>; 2] = std::array::from_fn(|ch| (0..n).map(|i| targets.iter().map(|t| t[ch][i]).sum()).collect());
        let est = separate(model, &stft, &mix, chunk, overlap, device);
        for s in 0..4 {
            let flat_t: Vec<f32> = targets[s].concat();
            sdr_sum[s] += sdr(&flat_t, &est[s].concat());
            mix_sum[s] += sdr(&flat_t, &mix.concat());
        }
        leak_sum += leak_db(&targets, &est);
    }
    let k = songs.len() as f64;
    Ok(Some(Validation { sdr: sdr_sum.map(|v| v / k), mix_sdr: mix_sum.map(|v| v / k), leak: leak_sum / k }))
}

/// Chunked separation with the Burn model (the same plan and cross-fade as `sone-infer`).
pub fn separate<B: Backend>(model: &CochleaNet<B>, stft: &Stft<B>, mix: &[Vec<f32>; 2], chunk: usize, overlap: usize, device: &B::Device) -> Vec<[Vec<f32>; 2]> {
    let n = mix[0].len();
    let plan = Plan::new(n, chunk, overlap);
    let mut out: Vec<[Vec<f32>; 2]> = (0..4).map(|_| [vec![0.0; n], vec![0.0; n]]).collect();
    for (i, &start) in plan.starts.iter().enumerate() {
        let mut data = Vec::with_capacity(2 * chunk);
        for ch in mix {
            data.extend((start..start + chunk).map(|t| ch.get(t).copied().unwrap_or(0.0)));
        }
        let wave = model.forward(stft, Tensor::from_data(TensorData::new(data, [1, 2, chunk]), device)).wave;
        let v = sone_models::export::values(wave);
        for s in 0..3 {
            for ch in 0..2 {
                let src = &v[(s * 2 + ch) * chunk..][..chunk];
                for j in 0..chunk.min(n - start) {
                    out[s][ch][start + j] += plan.weight(i, j) * src[j];
                }
            }
        }
    }
    for ch in 0..2 {
        for j in 0..n {
            out[3][ch][j] = mix[ch][j] - out[0][ch][j] - out[1][ch][j] - out[2][ch][j];
        }
    }
    out
}

/// Voice left in the instrumental: energy of (drums + bass + other estimate
/// − their target) in 20 ms windows where the vocal target is louder than
/// the instrumental target, relative to the vocal energy there, in dB.
/// Lower is better; it is a coarse proxy for what the leak loss penalises.
pub fn leak_db(targets: &[[Vec<f32>; 2]], est: &[[Vec<f32>; 2]]) -> f64 {
    let n = targets[0][0].len();
    let win = 882;
    let (mut err, mut voc) = (1e-12f64, 1e-12f64);
    for w0 in (0..n).step_by(win) {
        let w1 = (w0 + win).min(n);
        let (mut ev, mut ei, mut ee) = (0.0f64, 0.0f64, 0.0f64);
        for ch in 0..2 {
            for i in w0..w1 {
                let v = targets[0][ch][i] as f64;
                let inst_t: f64 = (1..4).map(|s| targets[s][ch][i] as f64).sum();
                let inst_e: f64 = (1..4).map(|s| est[s][ch][i] as f64).sum();
                ev += v * v;
                ei += inst_t * inst_t;
                ee += (inst_e - inst_t).powi(2);
            }
        }
        if ev > ei {
            voc += ev;
            err += ee;
        }
    }
    10.0 * (err / voc).log10()
}

/// Separate songs with an exported file through `sone-infer` (no Burn) and
/// print SDR per stem next to the mixture baseline: `train eval-cochlea`.
pub fn eval_file(weights: &Path, roots: &[std::path::PathBuf], mixture_dir: Option<&Path>, seconds: f64, chunking: &Chunking) -> Result<()> {
    let net = sone_infer::Cochlea::load(sone_infer::WeightFile::from_vec(std::fs::read(weights)?)?)?;
    let c = net.config.clone();
    let songs = data_cochlea::index(roots, mixture_dir)?;
    ensure!(!songs.is_empty(), "no songs found");
    let chunk = samples(chunking.chunk_seconds, c.hop);
    let overlap = samples(chunking.overlap_seconds, c.hop).min(chunk - c.hop);
    let mut ws = sone_infer::CochleaWorkspace::default();
    println!("| song | vocals | drums | bass | other | mixture baseline (mean) | CPU s per s audio |");
    println!("|---|---|---|---|---|---|---|");
    let mut all = [0.0f64; 4];
    for song in &songs {
        let n = ((seconds * 44_100.0) as usize).min(song.len as usize);
        let targets: Vec<[Vec<f32>; 2]> = (0..4).map(|s| song.read_stem(s, 0, n)).collect::<Result<_>>()?;
        let mix: [Vec<f32>; 2] = std::array::from_fn(|ch| (0..n).map(|i| targets.iter().map(|t| t[ch][i]).sum()).collect());
        let mut out = vec![[Vec::new(), Vec::new()]; 4];
        let t0 = Instant::now();
        net.separate(&mut ws, &mix[0], &mix[1], chunk, overlap, &mut out);
        let cpu = t0.elapsed().as_secs_f64() / (n as f64 / 44_100.0);
        let s: Vec<f64> = (0..4).map(|i| sdr(&targets[i].concat(), &out[i].concat())).collect();
        let base: f64 = (0..4).map(|i| sdr(&targets[i].concat(), &mix.concat())).sum::<f64>() / 4.0;
        for i in 0..4 {
            all[i] += s[i];
        }
        println!("| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.3} |", song.name, s[0], s[1], s[2], s[3], base, cpu);
    }
    let k = songs.len() as f64;
    println!("mean SDR (dB): vocals {:.2}, drums {:.2}, bass {:.2}, other {:.2}", all[0] / k, all[1] / k, all[2] / k, all[3] / k);
    Ok(())
}
