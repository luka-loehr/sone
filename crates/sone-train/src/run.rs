//! What every training program shares: the run directory (checkpoints,
//! logs, exports), background data loading, the CSV log, SDR.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, sync_channel};
use std::time::Instant;

use anyhow::{Context, Result};
use burn::module::AutodiffModule;
use burn::optim::Optimizer;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use sone_infer::random::Rng;
use sone_infer::st::NamedTensor;
use serde::{Deserialize, Serialize};

type Rec = BinFileRecorder<FullPrecisionSettings>;

/// `<dir>/checkpoint/{model,optim}.bin` + `state.json`, `<dir>/export/`,
/// `<dir>/train.csv`, `<dir>/valid.csv`, `<dir>/config.toml` (a copy).
pub struct RunDir {
    pub root: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct State {
    step: usize,
}

impl RunDir {
    pub fn create(root: &Path, config_path: &Path) -> Result<Self> {
        std::fs::create_dir_all(root.join("checkpoint"))?;
        std::fs::create_dir_all(root.join("export"))?;
        std::fs::copy(config_path, root.join("config.toml")).ok();
        Ok(Self { root: root.to_path_buf() })
    }

    /// Save model and optimizer state; files are written under temporary
    /// names and renamed, so a crash leaves the previous checkpoint intact.
    pub fn save<B: AutodiffBackend, M: AutodiffModule<B>, O: Optimizer<M, B>>(&self, step: usize, model: &M, optim: &O) -> Result<()> {
        let dir = self.root.join("checkpoint");
        let rec = Rec::new();
        // the recorder sets the extension itself ("model_new" -> "model_new.bin")
        model.clone().save_file(dir.join("model_new"), &rec).context("saving model")?;
        rec.record(optim.to_record(), dir.join("optim_new")).context("saving optimizer")?;
        std::fs::rename(dir.join("model_new.bin"), dir.join("model.bin"))?;
        std::fs::rename(dir.join("optim_new.bin"), dir.join("optim.bin"))?;
        std::fs::write(dir.join("state.json"), serde_json::to_string(&State { step })?)?;
        Ok(())
    }

    /// The step, model and optimizer of the last checkpoint, if there is one.
    pub fn load<B: AutodiffBackend, M: AutodiffModule<B>, O: Optimizer<M, B>>(&self, model: M, optim: O, device: &B::Device) -> Result<Option<(usize, M, O)>> {
        let dir = self.root.join("checkpoint");
        let Ok(state) = std::fs::read_to_string(dir.join("state.json")) else { return Ok(None) };
        let state: State = serde_json::from_str(&state)?;
        let rec = Rec::new();
        let model = model.load_file(dir.join("model"), &rec, device).context("loading model checkpoint")?;
        let record = rec.load(dir.join("optim"), device).context("loading optimizer checkpoint")?;
        Ok(Some((state.step, model, optim.load_record(record))))
    }

    /// `export/<name>-<step>-{f32,f16}.safetensors` and `<name>-latest-*`.
    pub fn export(&self, name: &str, step: usize, config: &impl Serialize, tensors: &[NamedTensor]) -> Result<PathBuf> {
        let dir = self.root.join("export");
        let mut latest16 = PathBuf::new();
        for (half, tag) in [(false, "f32"), (true, "f16")] {
            let bytes = sone_models::export::to_safetensors(name, config, tensors, half)?;
            let path = dir.join(format!("{name}-{step:07}-{tag}.safetensors"));
            std::fs::write(&path, &bytes)?;
            let latest = dir.join(format!("{name}-latest-{tag}.safetensors"));
            std::fs::write(&latest, &bytes)?;
            if half {
                latest16 = latest;
            }
        }
        Ok(latest16)
    }
}

/// Training data made on background threads, `threads · 2` items ahead.
pub struct Loader<T> {
    rx: Receiver<Result<T>>,
}

impl<T: Send + 'static> Loader<T> {
    pub fn spawn<F>(threads: usize, seed: u64, make: F) -> Self
    where
        F: Fn(&mut Rng) -> Result<T> + Send + Sync + 'static,
    {
        let (tx, rx) = sync_channel(threads.max(1) * 2);
        let make = Arc::new(make);
        for i in 0..threads.max(1) {
            let (tx, make) = (tx.clone(), make.clone());
            std::thread::spawn(move || {
                let mut rng = Rng::new(seed.wrapping_mul(1000).wrapping_add(i as u64 + 1));
                while tx.send(make(&mut rng)).is_ok() {}
            });
        }
        Self { rx }
    }

    pub fn next(&self) -> Result<T> {
        self.rx.recv().context("data loader stopped")?
    }
}

/// Appends rows to a CSV file and prints them.
pub struct Log {
    file: File,
    columns: Vec<String>,
}

impl Log {
    pub fn open(path: &Path, columns: &[&str]) -> Result<Self> {
        let new = !path.exists();
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        if new {
            writeln!(file, "{}", columns.join(","))?;
        }
        Ok(Self { file, columns: columns.iter().map(|s| s.to_string()).collect() })
    }

    pub fn row(&mut self, values: &[f64]) -> Result<()> {
        let text: Vec<String> = values.iter().map(|v| format!("{v:.6}")).collect();
        writeln!(self.file, "{}", text.join(","))?;
        self.file.flush()?;
        let shown: Vec<String> = self
            .columns
            .iter()
            .zip(values)
            .map(|(c, v)| match c.as_str() {
                "step" => format!("step {v:.0}"),
                "lr" => format!("lr {v:.2e}"),
                _ => format!("{c} {v:.4}"),
            })
            .collect();
        println!("{}", shown.join(" | "));
        Ok(())
    }
}

/// Sums of loss terms over the steps between two log lines, read back
/// from the device only when printed.
pub struct Meter<B: Backend> {
    sums: Vec<Tensor<B, 1>>,
    steps: usize,
    examples: usize,
    since: Instant,
}

impl<B: Backend> Meter<B> {
    pub fn new(terms: usize, device: &B::Device) -> Self {
        Self { sums: (0..terms).map(|_| Tensor::zeros([1], device)).collect(), steps: 0, examples: 0, since: Instant::now() }
    }

    pub fn add(&mut self, terms: &[Tensor<B, 1>], examples: usize) {
        for (s, t) in self.sums.iter_mut().zip(terms) {
            *s = s.clone() + t.clone();
        }
        self.steps += 1;
        self.examples += examples;
    }

    /// Mean of every term and examples per second; resets.
    pub fn take(&mut self) -> (Vec<f64>, f64) {
        let device = self.sums[0].device();
        let n = self.steps.max(1) as f64;
        let means = self.sums.iter().map(|s| s.clone().into_scalar().elem::<f64>() / n).collect();
        let rate = self.examples as f64 / self.since.elapsed().as_secs_f64();
        *self = Self::new(self.sums.len(), &device);
        (means, rate)
    }
}

/// Signal-to-distortion ratio in dB (the MDX / "new SDR" form: one ratio
/// over the whole signal, not BSS Eval's frame median).
pub fn sdr(reference: &[f32], estimate: &[f32]) -> f64 {
    let (mut num, mut den) = (1e-8f64, 1e-8f64);
    for (r, e) in reference.iter().zip(estimate) {
        num += (*r as f64) * (*r as f64);
        den += ((r - e) as f64).powi(2);
    }
    10.0 * (num / den).log10()
}

/// Seconds to a whole number of hops.
pub fn samples(seconds: f64, hop: usize) -> usize {
    (((seconds * sone_core::SAMPLE_RATE as f64) / hop as f64).round() as usize).max(1) * hop
}
