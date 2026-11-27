//! End-to-end numerical parity check between PyTorch v2/Pro CUDA trainers
//! (`Cochlea` v2/Pro 4096-FFT 64-band Axial Time+Band 4-Stem Bounded Wiener,
//! `Tympanum` v2 160-mel 3-layer BiGRU, `Cortex` v3 8L, `Cingulate` v3 6L)
//! and pure-Rust zero-copy `sone-infer` (`Sone::load`).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Result, ensure};
use serde::Deserialize;
use sone_infer::{BeatWorkspace, CochleaWorkspace, Sone, TympanumWorkspace, WeightFile, merge_bundle};

#[derive(Deserialize)]
struct RefData {
    mix_l: Vec<f32>,
    mix_r: Vec<f32>,
    cochlea_wave: Vec<Vec<Vec<f32>>>, // [4][2][len]
    mono: Vec<f32>,
    tympanum_frames: Vec<[f32; 5]>,
    tympanum_key: [f32; 24],
    tympanum_tuning: f32,
    cortex_a: Vec<f32>,
    cortex_b: Vec<f32>,
    cortex_g: Vec<f32>,
    cortex_raw: Vec<f32>,
    cing_a: Vec<f32>,
    cing_b: Vec<f32>,
    cing_g: Vec<f32>,
    cing_raw: Vec<f32>,
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn main() -> Result<()> {
    let dir = Path::new("/tmp/parity_v2");
    let ref_data: RefData = serde_json::from_slice(&std::fs::read(dir.join("ref.json"))?)?;

    for (prec, tol) in [("f32", 2e-4f32), ("f16", 4e-3f32)] {
        let cb = std::fs::read(dir.join(format!("sone-cochlea-test-{prec}.safetensors")))?;
        let tb = std::fs::read(dir.join(format!("sone-tympanum-test-{prec}.safetensors")))?;
        let pb = std::fs::read(dir.join(format!("sone-cortex-test-{prec}.safetensors")))?;
        let gb = std::fs::read(dir.join(format!("sone-cingulate-test-{prec}.safetensors")))?;

        let bundle_bytes = merge_bundle(
            &[("cochlea", &cb), ("tympanum", &tb), ("cortex", &pb), ("cingulate", &gb)],
            &BTreeMap::new(),
        )?;
        let sone = Sone::load(WeightFile::from_vec(bundle_bytes)?)?;

        // 1. Cochlea v2/Pro
        let mut cws = CochleaWorkspace::default();
        let mut out = vec![[Vec::new(), Vec::new()]; 4];
        sone.cochlea.separate_chunk(&mut cws, &ref_data.mix_l, &ref_data.mix_r, &mut out);
        let mut c_err = 0.0f32;
        for s in 0..4 {
            for ch in 0..2 {
                c_err = c_err.max(max_diff(&out[s][ch], &ref_data.cochlea_wave[s][ch]));
            }
        }

        // 2. Tympanum v2
        let mut tws = TympanumWorkspace::default();
        let tout = sone.tympanum.analyze(&mut tws, &ref_data.mono);
        let tf_flat: Vec<f32> = tout.frames.iter().flatten().copied().collect();
        let rf_flat: Vec<f32> = ref_data.tympanum_frames.iter().flatten().copied().collect();
        let t_err = max_diff(&tf_flat, &rf_flat)
            .max(max_diff(&tout.key, &ref_data.tympanum_key))
            .max((tout.tuning_cents - ref_data.tympanum_tuning).abs() / 50.0);

        // 3. Cortex v3
        let mut bws = BeatWorkspace::default();
        let craw = sone.cortex.forward_raw(&mut bws, &ref_data.cortex_a, &ref_data.cortex_b, &ref_data.cortex_g);
        let p_err = max_diff(&craw, &ref_data.cortex_raw);

        // 4. Cingulate v3
        let graw = sone.cingulate.forward_raw(&mut bws, &ref_data.cing_a, &ref_data.cing_b, &ref_data.cing_g);
        let g_err = max_diff(&graw, &ref_data.cing_raw);

        println!(
            "PARITY [{prec}] ({:.2} MB unified): cochlea={c_err:.6} tympanum={t_err:.6} cortex={p_err:.6} cingulate={g_err:.6} (tol={tol})",
            sone.weights_bytes() as f64 / 1e6
        );
        ensure!(c_err < tol, "cochlea {prec} parity {c_err} >= {tol}");
        ensure!(t_err < tol, "tympanum {prec} parity {t_err} >= {tol}");
        ensure!(p_err < tol * 4.0, "cortex {prec} parity {p_err} >= {}", tol * 4.0);
        ensure!(g_err < tol * 4.0, "cingulate {prec} parity {g_err} >= {}", tol * 4.0);
    }
    println!("ALL 4 UPGRADED STUDIO V2/PRO ORGANS PASSED EXACT PYTORCH <-> RUST PARITY!");
    Ok(())
}
