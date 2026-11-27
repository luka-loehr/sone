//! `sone-train <cochlea|tympanum|cortex|cingulate> --config <file.toml> [--set section.key=value]...`
//! `sone-train synth <cochlea|tympanum|examples> --out <path> [--n N] [--seconds S] [--seed S]`
//! `sone-train eval-cochlea --weights <file.safetensors> --songs <dir> [--seconds S]`
//!
//! A run resumes from `<run.dir>/checkpoint` when one is there.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, NdArray, Wgpu};
use sone_train::config::{self, BackendName};

struct Args {
    rest: Vec<String>,
}

impl Args {
    fn value(&self, flag: &str) -> Option<String> {
        self.rest.iter().position(|a| a == flag).and_then(|i| self.rest.get(i + 1).cloned())
    }
    fn values(&self, flag: &str) -> Vec<String> {
        self.rest.windows(2).filter(|w| w[0] == flag).map(|w| w[1].clone()).collect()
    }
    fn num<T: std::str::FromStr>(&self, flag: &str, default: T) -> Result<T> {
        match self.value(flag) {
            Some(v) => v.parse().map_err(|_| anyhow::anyhow!("{flag}: cannot parse {v}")),
            None => Ok(default),
        }
    }
}

fn backend_of(path: &Path, overrides: &[String]) -> Result<BackendName> {
    #[derive(serde::Deserialize)]
    struct Only {
        run: RunOnly,
    }
    #[derive(serde::Deserialize)]
    struct RunOnly {
        backend: BackendName,
    }
    Ok(config::load::<Only>(path, overrides)?.run.backend)
}

macro_rules! dispatch {
    ($backend:expr, $($f:ident)::+, $cfg:expr, $path:expr) => {
        match $backend {
            BackendName::Wgpu => $($f)::+::<Autodiff<Wgpu>>($cfg, $path, WgpuDevice::default()),
            BackendName::Ndarray => $($f)::+::<Autodiff<NdArray>>($cfg, $path, Default::default()),
        }
    };
}

fn main() -> Result<()> {
    let mut argv = std::env::args().skip(1);
    let command = argv.next().unwrap_or_default();
    let args = Args { rest: argv.collect() };
    let overrides = args.values("--set");
    match command.as_str() {
        "cochlea" | "tympanum" | "cortex" | "cingulate" | "stems" | "ear" | "plan" | "critic" => {
            let path = PathBuf::from(args.value("--config").context("--config <file.toml> is required")?);
            let backend = backend_of(&path, &overrides)?;
            match command.as_str() {
                "cochlea" | "stems" => {
                    let cfg = config::load(&path, &overrides)?;
                    dispatch!(backend, sone_train::train_cochlea::run, cfg, &path)
                }
                "tympanum" | "ear" => {
                    let cfg = config::load(&path, &overrides)?;
                    dispatch!(backend, sone_train::train_tympanum::run, cfg, &path)
                }
                "cortex" | "plan" => {
                    let cfg = config::load(&path, &overrides)?;
                    dispatch!(backend, sone_train::train_beat::run_cortex, cfg, &path)
                }
                _ => {
                    let cfg = config::load(&path, &overrides)?;
                    dispatch!(backend, sone_train::train_beat::run_cingulate, cfg, &path)
                }
            }
        }
        "synth" => {
            let what = args.rest.first().cloned().unwrap_or_default();
            let out = PathBuf::from(args.value("--out").context("--out is required")?);
            let seed = args.num("--seed", 1u64)?;
            match what.as_str() {
                "cochlea" | "stems" => sone_train::synth::cochlea_dataset(&out, args.num("--n", 8)?, args.num("--seconds", 12.0)?, seed),
                "tympanum" | "ear" => sone_train::synth::tympanum_dataset(&out, args.num("--n", 8)?, args.num("--seconds", 20.0)?, seed),
                "examples" => sone_train::synth::examples_file(&out, args.num("--n", 2000)?, seed),
                _ => bail!("synth: cochlea, tympanum or examples"),
            }
        }
        "eval-cochlea" | "eval-stems" => {
            let weights = PathBuf::from(args.value("--weights").context("--weights is required")?);
            let songs: Vec<PathBuf> = args.values("--songs").into_iter().map(PathBuf::from).collect();
            let mixture_dir = args.value("--mixture-dir").map(PathBuf::from);
            let chunking = sone_train::train_cochlea::Chunking { chunk_seconds: args.num("--chunk", 8.0)?, overlap_seconds: args.num("--overlap", 1.0)? };
            sone_train::train_cochlea::eval_file(&weights, &songs, mixture_dir.as_deref(), args.num("--seconds", 60.0)?, &chunking)
        }
        _ => {
            eprintln!("usage: sone-train <cochlea|tympanum|cortex|cingulate> --config <file.toml> [--set section.key=value]...");
            eprintln!("       sone-train synth <cochlea|tympanum|examples> --out <path> [--n N] [--seconds S] [--seed S]");
            eprintln!("       sone-train eval-cochlea --weights <file.safetensors> --songs <dir> [--seconds S] [--chunk S] [--overlap S]");
            std::process::exit(2);
        }
    }
}
