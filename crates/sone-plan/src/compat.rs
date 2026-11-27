//! A `BeatTable` from the lab's per-bar analysis (`data/analysis/struct`), for
//! songs that have not been through `sone-analysis` yet. It knows the voice only
//! beat by beat and has no chroma, so lines end on beats and keys come from
//! `keys.json`.

use sone_core::{BeatTable, TrackAnalysis};

pub struct KeyInfo<'a> {
    pub key: &'a str,
    pub camelot: &'a str,
    /// Second-best key if its score is within 0.08 of the best.
    pub camelot_alt: &'a str,
    pub tuning_cents: f32,
}

pub fn table_from_analysis(a: &TrackAnalysis, key: Option<KeyInfo<'_>>) -> BeatTable {
    let down = a.downbeats();
    let n = a.beats.len();
    let mut beat_in_bar = vec![0u8; n];
    let mut bar_of = vec![usize::MAX; n];
    for (bar, w) in down.windows(2).enumerate() {
        for (k, i) in (w[0]..w[1]).enumerate() {
            beat_in_bar[i] = (k % 4) as u8;
            bar_of[i] = bar;
        }
    }
    let sung_at = |i: usize| -> bool {
        let bar = bar_of[i];
        bar != usize::MAX && bar < a.bars.len() && a.bars[bar].sung[usize::from(beat_in_bar[i]).min(3)] == 1
    };
    let mut features = Vec::with_capacity(n);
    let mut v16 = Vec::with_capacity(n);
    let mut line_ends = Vec::new();
    let mut line_starts = Vec::new();
    let top_novelty = a.novelty.iter().copied().fold(1.0f32, f32::max);
    for i in 0..n {
        let bar = bar_of[i];
        let mut f = vec![-60.0f32; sone_core::example::FEATURES.len()];
        if bar != usize::MAX && bar < a.bars.len() {
            let r = &a.bars[bar];
            let k = usize::from(beat_in_bar[i]).min(3);
            let clamp = |v: f32| v.max(-60.0);
            f[0] = clamp(r.vbeats[k]);
            f[1] = clamp(r.drums);
            f[2] = clamp(r.kick);
            f[3] = clamp(r.hats);
            f[4] = clamp(r.bass);
            f[5] = clamp(r.other);
            f[6] = clamp(r.mix);
            f[7] = f32::from(r.sung[k]);
            // A line ends where somebody sang and then is quiet for two beats.
            let ends = sung_at(i) && i + 2 < n && !sung_at(i + 1) && !sung_at(i + 2);
            f[8] = f32::from(u8::from(ends));
            f[9] = f32::from(beat_in_bar[i]) / 3.0;
            f[10] = ((bar + 8 - usize::from(a.phrase8)) % 8) as f32 / 7.0;
            f[11] = a.novelty.get(bar).copied().unwrap_or(0.0) / top_novelty;
            for c in &mut f[12..24] {
                *c = 1.0 / 12.0;
            }
            if ends {
                line_ends.push(i * 4 + 3);
            }
            let starts = sung_at(i) && (i < 2 || (!sung_at(i - 1) && !sung_at(i - 2)));
            if starts {
                line_starts.push(i * 4);
            }
            v16.push([clamp(r.vbeats[k]); 4]);
        } else {
            f[9] = 0.0;
            f[10] = 0.0;
            f[11] = 0.0;
            for c in &mut f[12..24] {
                *c = 1.0 / 12.0;
            }
            f[7] = 0.0;
            f[8] = 0.0;
            v16.push([-60.0; 4]);
        }
        features.push(f);
    }
    let (key_name, camelot, camelot_alt, tuning) = key.map_or(("", "", "", 0.0), |k| (k.key, k.camelot, k.camelot_alt, k.tuning_cents));
    BeatTable {
        id: a.id.clone(),
        duration: a.duration,
        bpm: a.bpm,
        beat_s: a.beat_s,
        times: a.beats.clone(),
        beat_in_bar,
        beat_shift: 0,
        phase_confidence: {
            let total: u32 = a.downbeat_votes.iter().sum();
            let top = a.downbeat_votes.iter().copied().max().unwrap_or(0);
            if total == 0 { 0.0 } else { top as f32 / total as f32 }
        },
        bars: down[..down.len().saturating_sub(1)].to_vec(),
        phrase8: a.phrase8,
        key: key_name.to_owned(),
        camelot: camelot.to_owned(),
        camelot_alt: camelot_alt.to_owned(),
        tuning_cents: tuning,
        loud_db: a.loud_db,
        features,
        v16,
        line_ends,
        line_starts,
    }
}

/// All songs of a directory of struct files, with the keys of `keys.json` next to it if present.
pub fn load_dir(struct_dir: &std::path::Path, keys_json: Option<&std::path::Path>) -> anyhow::Result<Vec<BeatTable>> {
    let keys: serde_json::Value = match keys_json {
        Some(p) if p.exists() => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        _ => serde_json::Value::Null,
    };
    let mut out = Vec::new();
    let mut paths: Vec<_> = std::fs::read_dir(struct_dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    paths.sort();
    for p in paths {
        let a = TrackAnalysis::load(&p)?;
        let k = keys.get(&a.id);
        let info = k.map(|k| KeyInfo {
            key: k.get("key").and_then(|v| v.as_str()).unwrap_or(""),
            camelot: k.get("camelot").and_then(|v| v.as_str()).unwrap_or(""),
            camelot_alt: {
                let best = k.get("score").and_then(serde_json::Value::as_f64).unwrap_or(1.0);
                let second = k.get("second").and_then(|s| s.get(2)).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
                if best - second < 0.08 { k.get("second").and_then(|s| s.get(1)).and_then(|v| v.as_str()).unwrap_or("") } else { "" }
            },
            tuning_cents: k.get("tuning_cents").and_then(serde_json::Value::as_f64).unwrap_or(0.0) as f32,
        });
        out.push(table_from_analysis(&a, info));
    }
    Ok(out)
}
