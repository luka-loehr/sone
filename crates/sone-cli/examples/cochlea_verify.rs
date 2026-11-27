use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use sone_infer::{Cochlea, CochleaWorkspace, WeightFile};
use sone_render::files;

const HELD_OUT: [&str; 6] = [
    "7yoqSltRmGVlJF34OdLyZG",
    "6N185ACZ5M6TYngUuOPpoW",
    "6wPupZx0WFmkj9wnCrCUJU",
    "0hkR93a0LriV2CQUhSXw3h",
    "0FlcmZ13DGRmscCroro9GO",
    "1YwWHm6Ad1fizVPaQ8nrco",
];

fn sdr_db(target: &[[f32; 2]], est: &[[f32; 2]]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (t, e) in target.iter().zip(est) {
        for c in 0..2 {
            let tv = f64::from(t[c]);
            let dv = tv - f64::from(e[c]);
            num += tv * tv;
            den += dv * dv;
        }
    }
    10.0 * ((num + 1e-12) / (den + 1e-12)).log10()
}

/// MDX23-style 1s chunk-median SDR over active chunks (target RMS > -42 dBFS).
fn active_chunk_median_sdr_db(target: &[[f32; 2]], est: &[[f32; 2]]) -> Option<f64> {
    let win = 44_100usize;
    let min_e = 10.0f64.powf(-42.0 / 10.0);
    let mut vals = Vec::new();
    let n = target.len().min(est.len());
    for s0 in (0..n).step_by(win) {
        let s1 = (s0 + win).min(n);
        if s1 - s0 < win / 2 {
            break;
        }
        let mut e = 0.0f64;
        for f in &target[s0..s1] {
            e += f64::from(f[0]).powi(2) + f64::from(f[1]).powi(2);
        }
        let rms2 = e / ((s1 - s0) * 2) as f64;
        if rms2 >= min_e {
            vals.push(sdr_db(&target[s0..s1], &est[s0..s1]));
        }
    }
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let m = vals.len() / 2;
    Some(if vals.len() % 2 == 1 { vals[m] } else { 0.5 * (vals[m - 1] + vals[m]) })
}

fn rms_db(sig: &[[f32; 2]]) -> f64 {
    if sig.is_empty() {
        return -120.0;
    }
    let mut e = 0.0f64;
    for f in sig {
        e += f64::from(f[0]).powi(2) + f64::from(f[1]).powi(2);
    }
    10.0 * (e / (sig.len() * 2) as f64 + 1e-12).log10()
}

fn leak_db(voc_t: &[[f32; 2]], inst_t: &[[f32; 2]], inst_e: &[[f32; 2]]) -> f64 {
    let win = 882;
    let mut err = 1e-12f64;
    let mut voc = 1e-12f64;
    let n = voc_t.len().min(inst_t.len()).min(inst_e.len());
    for w0 in (0..n).step_by(win) {
        let w1 = (w0 + win).min(n);
        let mut ev = 0.0f64;
        let mut ei = 0.0f64;
        let mut ee = 0.0f64;
        for i in w0..w1 {
            for c in 0..2 {
                let v = f64::from(voc_t[i][c]);
                let it = f64::from(inst_t[i][c]);
                let ie = f64::from(inst_e[i][c]);
                ev += v * v;
                ei += it * it;
                let d = ie - it;
                ee += d * d;
            }
        }
        if ev > ei {
            voc += ev;
            err += ee;
        }
    }
    10.0 * (err / voc).log10()
}

fn load_clean_stems(dir: &Path) -> Result<[Vec<[f32; 2]>; 4]> {
    let rd = |name: &str| files::read_flac(&dir.join(format!("{name}.flac")));
    let (v, d, b, o) = (rd("vocals")?, rd("drums")?, rd("bass")?, rd("other")?);
    let n = v.len().min(d.len()).min(b.len()).min(o.len());
    let mix: Vec<[f32; 2]> = (0..n).map(|i| std::array::from_fn(|c| ((v[i][c] + d[i][c]) + b[i][c]) + o[i][c])).collect();
    let cut = |mut x: Vec<[f32; 2]>| {
        x.resize(n, [0.0; 2]);
        x
    };
    let mut stems = [cut(v), cut(d), cut(b), cut(o)];
    if dir.join("vocals_r.flac").exists() {
        stems[0] = cut(rd("vocals_r")?);
        if dir.join("drums_i.flac").exists() && dir.join("bass_i.flac").exists() {
            stems[1] = cut(rd("drums_i")?);
            stems[2] = cut(rd("bass_i")?);
        }
        stems[3] = (0..n).map(|i| std::array::from_fn(|c| ((mix[i][c] - stems[0][i][c]) - stems[1][i][c]) - stems[2][i][c])).collect();
    }
    Ok(stems)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let weights_path = PathBuf::from(args.first().map(String::as_str).unwrap_or("exports/sone-cochlea-f16.safetensors"));
    let stems_dir = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("data/stems"));
    let out_dir = PathBuf::from(args.get(2).map(String::as_str).unwrap_or("out/stems_eval"));
    let max_songs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let clip_s: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(30.0);
    std::fs::create_dir_all(&out_dir)?;

    let model = Cochlea::load(WeightFile::from_vec(std::fs::read(&weights_path).with_context(|| format!("reading {}", weights_path.display()))?)?)?;
    println!("Loaded Stems ({:.2} MB) from {}", model.weights_bytes() as f64 / 1e6, weights_path.display());

    let mut dirs: Vec<PathBuf> = HELD_OUT
        .iter()
        .map(|id| stems_dir.join(id))
        .filter(|p| p.is_dir() && p.join("vocals.flac").exists())
        .collect();
    if dirs.is_empty() {
        dirs = std::fs::read_dir(&stems_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir() && p.join("vocals.flac").exists())
            .collect();
        dirs.sort();
    }

    // Optional f32 sibling for exact FP32-vs-FP16 parity audit
    let f32_sibling = weights_path
        .to_string_lossy()
        .replace("-f16.safetensors", "-f32.safetensors");
    let model_f32 = if f32_sibling != weights_path.to_string_lossy() && Path::new(&f32_sibling).exists() {
        Some(Cochlea::load(WeightFile::from_vec(std::fs::read(&f32_sibling)?)?)?)
    } else {
        None
    };

    let mut ws = CochleaWorkspace::default();
    let hop = model.config.hop;
    let chunk = (8 * 44100 / hop) * hop;
    let overlap = (44100 / hop) * hop;

    let mut rows = Vec::new();
    let mut tested = 0usize;
    let mut max_f16_diff = 0.0f32;
    for dir in dirs {
        if tested >= max_songs {
            break;
        }
        let id = dir.file_name().unwrap().to_string_lossy().to_string();
        let tgts = load_clean_stems(&dir)?;
        let n_all = tgts[0].len();
        let start = (30 * 44100).min(n_all.saturating_sub(44100 * 10));
        let len = (((clip_s * 44100.0) as usize).min(n_all - start) / hop) * hop;
        let mut mix_l = vec![0.0f32; len];
        let mut mix_r = vec![0.0f32; len];
        let mut mix_stereo = vec![[0.0f32; 2]; len];
        for i in 0..len {
            for s in 0..4 {
                mix_l[i] += tgts[s][start + i][0];
                mix_r[i] += tgts[s][start + i][1];
            }
            mix_stereo[i] = [mix_l[i], mix_r[i]];
        }
        let mut out: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
        let t0 = Instant::now();
        model.separate(&mut ws, &mix_l, &mix_r, chunk, overlap, &mut out);
        let elapsed = t0.elapsed().as_secs_f64();
        let rtf = elapsed / (len as f64 / 44100.0);

        if tested == 0 {
            if let Some(m32) = &model_f32 {
                let n_chk = chunk.min(len);
                let mut o16 = vec![[Vec::new(), Vec::new()]; 4];
                let mut o32 = vec![[Vec::new(), Vec::new()]; 4];
                model.separate_chunk(&mut ws, &mix_l[..n_chk], &mix_r[..n_chk], &mut o16);
                m32.separate_chunk(&mut ws, &mix_l[..n_chk], &mix_r[..n_chk], &mut o32);
                for s in 0..4 {
                    for c in 0..2 {
                        for (&a, &b) in o16[s][c].iter().zip(&o32[s][c]) {
                            max_f16_diff = max_f16_diff.max((a - b).abs());
                        }
                    }
                }
            }
        }

        let est: Vec<Vec<[f32; 2]>> = (0..4)
            .map(|s| (0..len).map(|i| [out[s][0][i], out[s][1][i]]).collect())
            .collect();
        let sdrs: [f64; 4] = std::array::from_fn(|s| sdr_db(&tgts[s][start..start + len], &est[s]));
        let active_sdrs: [Option<f64>; 4] =
            std::array::from_fn(|s| active_chunk_median_sdr_db(&tgts[s][start..start + len], &est[s]));
        let voc_rms = rms_db(&tgts[0][start..start + len]);
        let voc_est_rms = rms_db(&est[0]);
        let inst_t: Vec<[f32; 2]> = (0..len)
            .map(|i| std::array::from_fn(|c| tgts[1][start + i][c] + tgts[2][start + i][c] + tgts[3][start + i][c]))
            .collect();
        let inst_e: Vec<[f32; 2]> = (0..len)
            .map(|i| std::array::from_fn(|c| est[1][i][c] + est[2][i][c] + est[3][i][c]))
            .collect();
        let ldb = leak_db(&tgts[0][start..start + len], &inst_t, &inst_e);

        // Verify mixture conservation error (sum of 4 stems vs mix)
        let sum_est: Vec<[f32; 2]> = (0..len)
            .map(|i| std::array::from_fn(|c| ((est[0][i][c] + est[1][i][c]) + est[2][i][c]) + est[3][i][c]))
            .collect();
        let recon_err_db = -sdr_db(&mix_stereo, &sum_est);

        println!(
            "{id} ({:.1}s, voc_rms={voc_rms:.1}dB) | CPU {:.2}s ({:.3}x RT) | full SDR voc={:.2} drm={:.2} bas={:.2} oth={:.2} mean={:.2} dB | active_chunk voc={} dB | leak={ldb:.2} dB | recon_err={recon_err_db:.1} dB",
            len as f64 / 44100.0,
            elapsed,
            rtf,
            sdrs[0],
            sdrs[1],
            sdrs[2],
            sdrs[3],
            sdrs.iter().sum::<f64>() / 4.0,
            active_sdrs[0].map_or_else(|| "silent".to_string(), |v| format!("{v:.2}")),
        );

        files::write_wav16(&out_dir.join(format!("{id}_mix.wav")), &mix_stereo)?;
        files::write_wav16(&out_dir.join(format!("{id}_vocals_ref.wav")), &tgts[0][start..start + len])?;
        files::write_wav16(&out_dir.join(format!("{id}_vocals.wav")), &est[0])?;
        files::write_wav16(&out_dir.join(format!("{id}_instrumental.wav")), &inst_e)?;

        rows.push(serde_json::json!({
            "id": id,
            "clip_s": len as f64 / 44100.0,
            "cpu_s": elapsed,
            "rtf": rtf,
            "voc_rms_db": voc_rms,
            "voc_est_rms_db": voc_est_rms,
            "sdr_vocals": sdrs[0],
            "sdr_drums": sdrs[1],
            "sdr_bass": sdrs[2],
            "sdr_other": sdrs[3],
            "sdr_mean": sdrs.iter().sum::<f64>() / 4.0,
            "active_sdr_vocals": active_sdrs[0],
            "active_sdr_drums": active_sdrs[1],
            "active_sdr_bass": active_sdrs[2],
            "active_sdr_other": active_sdrs[3],
            "vocal_leak_db": ldb,
            "recon_err_db": recon_err_db,
        }));
        tested += 1;
    }

    let summary = serde_json::json!({
        "weights": weights_path.display().to_string(),
        "weights_bytes": model.weights_bytes(),
        "n_fft": model.config.n_fft,
        "hop": model.config.hop,
        "bands": model.config.bands.len(),
        "dim": model.config.dim,
        "blocks": model.config.blocks,
        "time_attn": model.config.time_attn,
        "four_stems": model.config.four_stems,
        "f32_vs_f16_max_diff": max_f16_diff,
        "songs": rows,
    });
    std::fs::write(out_dir.join("cochlea_verify.json"), serde_json::to_string_pretty(&summary)?)?;
    println!("f32_vs_f16_max_diff = {max_f16_diff:.6} | saved summary -> {}", out_dir.join("cochlea_verify.json").display());
    Ok(())
}
