//! The TOML configs of the training programs (`cookbooks/<model>.toml`).
//! Every file has `[run]` and `[optim]`; `[model]`, `[data]` and `[loss]`
//! depend on the network. `--set section.key=value` overrides any entry.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendName {
    /// GPU: Metal on the Mac, Vulkan on Linux (through wgpu).
    Wgpu,
    /// CPU.
    Ndarray,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub backend: BackendName,
    pub seed: u64,
    pub steps: usize,
    pub batch_size: usize,
    /// Checkpoints, logs and exports go here.
    pub dir: PathBuf,
    #[serde(default = "default_log_every")]
    pub log_every: usize,
    pub valid_every: usize,
    /// Defaults to `valid_every`.
    #[serde(default)]
    pub checkpoint_every: Option<usize>,
    /// Threads that read and augment training data.
    #[serde(default = "default_threads")]
    pub loader_threads: usize,
}

fn default_log_every() -> usize {
    20
}
fn default_threads() -> usize {
    4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Optim {
    pub lr: f64,
    pub warmup_steps: usize,
    /// Cosine decay from `lr` to `lr · min_lr_ratio` over the steps after warm-up.
    #[serde(default = "default_min_lr")]
    pub min_lr_ratio: f64,
    #[serde(default = "default_wd")]
    pub weight_decay: f32,
    /// Per-tensor gradient norm clip.
    #[serde(default)]
    pub grad_clip: Option<f32>,
}

fn default_min_lr() -> f64 {
    0.05
}
fn default_wd() -> f32 {
    1e-4
}

impl Optim {
    pub fn lr_at(&self, step: usize, total: usize) -> f64 {
        if step < self.warmup_steps {
            return self.lr * (step + 1) as f64 / self.warmup_steps as f64;
        }
        let span = total.saturating_sub(self.warmup_steps).max(1);
        let p = ((step - self.warmup_steps) as f64 / span as f64).min(1.0);
        let min = self.lr * self.min_lr_ratio;
        min + (self.lr - min) * 0.5 * (1.0 + (std::f64::consts::PI * p).cos())
    }
}

/// Read a config, applying `section.key=value` overrides (values parsed as
/// TOML, falling back to a string).
pub fn load<T: DeserializeOwned>(path: &Path, overrides: &[String]) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut value: toml::Table = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    for o in overrides {
        let (key, raw) = o.split_once('=').with_context(|| format!("override {o}: expected key=value"))?;
        let parsed: toml::Value = toml::from_str::<toml::Table>(&format!("v = {raw}"))
            .ok()
            .and_then(|mut t| t.remove("v"))
            .unwrap_or_else(|| toml::Value::String(raw.to_string()));
        let parts: Vec<&str> = key.split('.').collect();
        let mut table = &mut value;
        for p in &parts[..parts.len() - 1] {
            table = match table.entry(p.to_string()).or_insert_with(|| toml::Value::Table(Default::default())) {
                toml::Value::Table(t) => t,
                _ => bail!("override {key}: {p} is not a table"),
            };
        }
        table.insert(parts[parts.len() - 1].to_string(), parsed);
    }
    toml::Value::Table(value).try_into().with_context(|| format!("config {}", path.display()))
}

/// Expand a leading `~/` (relative paths stay relative to the working directory).
pub fn expand(p: &Path) -> PathBuf {
    if let Some(rest) = p.to_str().and_then(|s| s.strip_prefix("~/"))
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    p.to_path_buf()
}
