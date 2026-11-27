//! Development aid: the bar-phase profiles of the songs of a spec file.
use std::path::PathBuf;

use anyhow::Result;
use sone_analysis::{grid, io, phase, prepare, structure};

/// Spec and side, song, anchor bar, bars the grid is fitted on, the spec's beat shift.
type Side = (String, String, i64, Option<(usize, usize)>, i64);

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (specs, tracks, stems) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]), PathBuf::from(&a[3]));
    let specs: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(specs)?)?;
    let mut extra: Vec<Side> = Vec::new();
    for s in specs.as_array().unwrap() {
        for side in ["a", "b"] {
            let d = &s[side];
            let fit = d.get("fit_bars").and_then(|f| f.as_array()).map(|f| (f[0].as_u64().unwrap() as usize, f[1].as_u64().unwrap() as usize));
            extra.push((format!("{} {}", s["id"].as_str().unwrap(), side), d["id"].as_str().unwrap().to_string(), d["anchor_bar"].as_i64().unwrap(), fit, d.get("beat_shift").and_then(|v| v.as_i64()).unwrap_or(0)));
        }
    }
    let only: Option<&String> = a.get(4);
    for (name, id, anchor, fit, spec_shift) in extra {
        if only.is_some_and(|o| !name.starts_with(o.as_str())) {
            continue;
        }
        let beats = structure::Beats::parse(&std::fs::read_to_string(tracks.join(format!("{id}.beats")))?);
        let m = io::measure_dir(&stems.join(&id), io::StemSet::Final, structure::cut_points(&beats))?;
        let p = prepare(&id, &beats, m)?;
        for shift in [0, spec_shift] {
            let g = grid::grid(&p.structure, &p.env, anchor as usize, 16, shift, fit);
            let mut prof = phase::Profile::default();
            let rng: Vec<i64> = if name.ends_with('b') { (0..32).collect() } else { (-32..0).collect() };
            prof.add(&p.m, &p.chroma_bo, &g, rng.into_iter());
            let (k, c, h) = prof.means();
            let v = phase::decide(&prof);
            let f = |x: [f64; 4]| x.iter().map(|v| format!("{v:6.1}")).collect::<Vec<_>>().join(" ");
            let g3 = |x: [f64; 4]| x.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(" ");
            println!("{name} {id} bar {anchor} fit {fit:?} shift {shift} | kick {} | clap {} | chords {} | -> {:+} conf {:.2} backbeat {:?} contrast {:.1} evidence {:.2}", f(k), f(c), g3(h), v.shift, v.confidence, v.backbeat, v.contrast, v.evidence);
            if spec_shift == 0 {
                break;
            }
        }
    }
    Ok(())
}
