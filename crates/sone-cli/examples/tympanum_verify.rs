use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use sone_core::BeatTable;
use sone_infer::{Tympanum, TympanumWorkspace, WeightFile};
use sone_render::files;

fn peaks(p: &[f32], thr: f32) -> Vec<usize> {
    let n = p.len();
    let mut out = Vec::new();
    for i in 0..n {
        if p[i] < thr {
            continue;
        }
        let lo = i.saturating_sub(2);
        let hi = (i + 3).min(n);
        let mut ok = true;
        for j in lo..hi {
            if p[j] > p[i] || (j < i && p[j] >= p[i]) {
                ok = false;
                break;
            }
        }
        if ok {
            out.push(i);
        }
    }
    out
}

fn f_measure(pred: &[usize], truth_s: &[f64], fps: f64, tol_s: f64) -> f64 {
    let tol = (tol_s * fps).round() as isize;
    let truth: Vec<isize> = truth_s.iter().map(|&t| (t * fps).round() as isize).collect();
    if pred.is_empty() && truth.is_empty() {
        return 1.0;
    }
    let hits = truth
        .iter()
        .filter(|&&g| pred.iter().any(|&q| (q as isize - g).abs() <= tol))
        .count();
    2.0 * hits as f64 / (pred.len() + truth.len()).max(1) as f64
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let weights_path = PathBuf::from(args.first().map(String::as_str).unwrap_or("exports/sone-tympanum-f16.safetensors"));
    let stems_dir = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("data/stems"));
    let beats_dir = PathBuf::from(args.get(2).map(String::as_str).unwrap_or("data/analysis/beats"));

    let tympanum = Tympanum::load(WeightFile::from_vec(std::fs::read(&weights_path).with_context(|| format!("reading {}", weights_path.display()))?)?)?;
    let mut ws = TympanumWorkspace::default();
    let fps = tympanum.config.fps();

    let held_out = ["7yoqSltRmGVlJF34OdLyZG", "6wPupZx0WFmkj9wnCrCUJU", "0I4cfb7ATp8pC2w6OKk8al"];
    println!("=== SONE-TYMPANUM PURE-RUST CPU BENCHMARK ({}) ===", weights_path.display());

    for id in held_out {
        let sdir = stems_dir.join(id);
        let bpath = beats_dir.join(format!("{id}.json"));
        if !sdir.join("vocals.flac").exists() || !bpath.exists() {
            continue;
        }
        let bt = BeatTable::load(&bpath)?;
        let (v, d, b, o) = (
            files::read_flac(&sdir.join("vocals.flac"))?,
            files::read_flac(&sdir.join("drums.flac"))?,
            files::read_flac(&sdir.join("bass.flac"))?,
            files::read_flac(&sdir.join("other.flac"))?,
        );
        let n = v.len().min(d.len()).min(b.len()).min(o.len()).min(60 * 44100);
        let mono: Vec<f32> = (0..n)
            .map(|i| 0.5 * ((v[i][0] + d[i][0] + b[i][0] + o[i][0]) + (v[i][1] + d[i][1] + b[i][1] + o[i][1])))
            .collect();
        let dur_s = n as f64 / 44100.0;

        let t0 = Instant::now();
        let out = tympanum.analyze(&mut ws, &mono);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let beat_p: Vec<f32> = out.frames.iter().map(|f| f[0]).collect();
        let down_p: Vec<f32> = out.frames.iter().map(|f| f[1]).collect();
        let pred_beats = peaks(&beat_p, 0.5);
        let pred_downs = peaks(&down_p, 0.5);

        let truth_beats: Vec<f64> = bt.times.iter().copied().filter(|&t| t <= dur_s).collect();
        let truth_downs: Vec<f64> = bt.bars.iter().filter_map(|&bi| bt.times.get(bi).copied()).filter(|&t| t <= dur_s).collect();

        let bf1 = f_measure(&pred_beats, &truth_beats, fps, 0.07);
        let df1 = f_measure(&pred_downs, &truth_downs, fps, 0.07);

        println!(
            "{id} ({dur_s:.1}s) | CPU {cpu_ms:.1} ms ({:.4}x RT) | beat_F1={bf1:.3} downbeat_F1={df1:.3} | pred_key={} (true={}) | pred_tuning={:+.1}c (true={:+.1}c)",
            (cpu_ms / 1000.0) / dur_s,
            out.key_name(),
            bt.key,
            out.tuning_cents,
            bt.tuning_cents
        );
    }
    Ok(())
}
