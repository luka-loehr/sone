//! Losslessly pack the 4 anatomical organ `.safetensors` files (`cochlea`, `tympanum`,
//! `cortex`, `cingulate`) into one unified Hugging Face `sone-6.5M` `model.safetensors`
//! bundle (`cochlea.*`, `tympanum.*`, `cortex.*`, `cingulate.*`) and verify zero-copy
//! `Sone::load` parity.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sone_infer::{Sone, WeightFile, merge_bundle};

fn find_organ(exports: &Path, organ: &str, legacy: &[&str], suffix: &str) -> Result<PathBuf> {
    let primary = exports.join(format!("sone-{organ}-{suffix}.safetensors"));
    if primary.exists() {
        return Ok(primary);
    }
    for cand in legacy {
        let p = exports.join(format!("{cand}-{suffix}.safetensors"));
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!("missing {organ} ({suffix}) checkpoint in {}", exports.display())
}

fn pack_precision(exports: &Path, prefix: &str, out_path: &Path, suffix: &str) -> Result<()> {
    let cochlea_p = find_organ(exports, "cochlea", &["stems-best"], suffix)?;
    let tympanum_p = find_organ(exports, "tympanum", &["ear-best"], suffix)?;
    let cortex_p = find_organ(exports, "cortex", &["plan-v2-best", "plan-best"], suffix)?;
    let cingulate_p = find_organ(exports, "cingulate", &["critic-v2-best", "critic-best"], suffix)?;

    let cochlea_b = std::fs::read(&cochlea_p).with_context(|| format!("reading {}", cochlea_p.display()))?;
    let tympanum_b = std::fs::read(&tympanum_p).with_context(|| format!("reading {}", tympanum_p.display()))?;
    let cortex_b = std::fs::read(&cortex_p).with_context(|| format!("reading {}", cortex_p.display()))?;
    let cingulate_b = std::fs::read(&cingulate_p).with_context(|| format!("reading {}", cingulate_p.display()))?;

    let mut extra = BTreeMap::new();
    extra.insert("name".to_string(), prefix.to_string());
    extra.insert("precision".to_string(), suffix.to_uppercase());
    extra.insert("organs".to_string(), "cochlea,tympanum,cortex,cingulate".to_string());

    let bundle_bytes = merge_bundle(
        &[
            ("cochlea", &cochlea_b),
            ("tympanum", &tympanum_b),
            ("cortex", &cortex_b),
            ("cingulate", &cingulate_b),
        ],
        &extra,
    )?;

    let loaded = Sone::load(WeightFile::from_vec(bundle_bytes.clone())?)?;
    std::fs::write(out_path, &bundle_bytes)?;
    println!(
        "Packed unified {} ({:.2} MB, {} bytes) -> verified Sone::load OK!",
        out_path.display(),
        loaded.weights_bytes() as f64 / 1e6,
        loaded.weights_bytes()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exports = PathBuf::from(args.first().map(String::as_str).unwrap_or("/home/user/bot/exports"));
    let prefix = args.get(1).map(String::as_str).unwrap_or("sone-6.5M");

    pack_precision(&exports, prefix, &exports.join(format!("{prefix}-f16.safetensors")), "f16")?;
    pack_precision(&exports, prefix, &exports.join(format!("{prefix}-f32.safetensors")), "f32")?;
    std::fs::copy(
        exports.join(format!("{prefix}-f16.safetensors")),
        exports.join("model.safetensors"),
    )?;
    println!("Copied compact FP16 unified model -> {}", exports.join("model.safetensors").display());
    Ok(())
}
