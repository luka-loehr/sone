//! Writes the training data: `train.jsonl.gz`, `val.jsonl.gz`, `owner.jsonl`, `stats.json`.
//!
//! Plans a transition for every ordered pair of songs (at the song's end, and
//! at `windows` random places inside it with 30 s of look-ahead, as after a
//! skip), breaks each one `broken` times, and adds the owner's verdicts.

use std::{
    collections::{BTreeMap, HashMap},
    io::Write,
    path::PathBuf,
};

use anyhow::{Context, Result};
use flate2::{Compression, write::GzEncoder};
use sone_core::{BeatTable, Example, Issue, Label, Recipe};
use sone_plan::{Lock, PlanOptions, compat, plan, tempo};

use crate::{build::example_from_spec, owner, perturb, rng::Rng};

#[derive(Debug, Clone)]
pub struct Options {
    /// The repo's `data` directory.
    pub data: PathBuf,
    pub out: PathBuf,
    pub seed: u64,
    /// Places inside song A, besides its end, at which a transition is planned.
    pub windows: usize,
    /// Broken versions per good transition.
    pub broken: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { data: "data".into(), out: "data/sim".into(), seed: 1, windows: 4, broken: 3 }
    }
}

/// Fewer digits, smaller files; the models do not hear the difference.
fn round(e: &mut Example) {
    for f in e.a_beats.iter_mut().chain(e.b_beats.iter_mut()) {
        for v in f.iter_mut() {
            *v = (*v * 100.0).round() / 100.0;
        }
    }
    for row in &mut e.lanes {
        for v in row.iter_mut() {
            *v = (*v * 1000.0).round() / 1000.0;
        }
    }
}

/// Generates and writes the data; returns the counts that also go to `stats.json`.
pub fn generate(opt: &Options) -> Result<BTreeMap<String, usize>> {
    std::fs::create_dir_all(&opt.out)?;

    let beats_dir = opt.data.join("analysis/beats");
    let tables: Vec<BeatTable> = if beats_dir.is_dir() && std::fs::read_dir(&beats_dir)?.count() >= 2 {
        let mut v = Vec::new();
        let mut paths: Vec<_> = std::fs::read_dir(&beats_dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
        paths.sort();
        for p in paths {
            v.push(BeatTable::load(&p).with_context(|| p.display().to_string())?);
        }
        eprintln!("{} songs from {}", v.len(), beats_dir.display());
        v
    } else {
        let v = compat::load_dir(&opt.data.join("analysis/struct"), Some(&opt.data.join("analysis/keys.json")))?;
        eprintln!("{} songs from the lab's per-bar analysis (no sixteenth notes, no chroma)", v.len());
        v
    };
    let by_id: HashMap<String, BeatTable> = tables.iter().map(|t| (t.id.clone(), t.clone())).collect();

    // Every fifth song (by id order) plays song A only in the validation set.
    let val_songs: Vec<&str> = tables.iter().enumerate().filter(|(i, _)| i % 5 == 4).map(|(_, t)| t.id.as_str()).collect();
    let mut rng = Rng::new(opt.seed);
    let (mut train, mut val): (Vec<Example>, Vec<Example>) = (Vec::new(), Vec::new());
    let mut stats: BTreeMap<String, usize> = BTreeMap::new();
    let count = |k: &str, stats: &mut BTreeMap<String, usize>| *stats.entry(k.into()).or_default() += 1;

    for a in &tables {
        for b in &tables {
            if a.id == b.id || a.bars.len() < 16 || b.bars.len() < 16 {
                continue;
            }
            let pair = format!("{}-{}", &a.id[..6], &b.id[..6]);
            let fit = tempo::fit(a.bpm, b.bpm, 4.0);
            let keys = sone_plan::keys::song_distance(&a.camelot, &a.camelot_alt, &b.camelot, &b.camelot_alt);
            let mut goods: Vec<Example> = Vec::new();
            // A radio queue puts songs of the same beat next to each other, so those pairs get the
            // most examples; of the pairs that only allow the tempo-gap recipe a third is kept.
            if !fit.beat_on_beat() && !rng.chance(0.33) {
                count("skipped_gap_pair", &mut stats);
                continue;
            }
            let windows = if fit.beat_on_beat() { opt.windows } else { 0 };
            let mut plans: Vec<(String, PlanOptions)> = vec![("end".into(), PlanOptions::default())];
            for w in 0..windows {
                let at = a.duration * rng.range(0.3, 0.8);
                plans.push((format!("w{w}"), PlanOptions { window: Some((at, at + 30.0)), lead_bars: if 8.0 * a.beat_s * f64::from(fit.bpb_a) > 22.0 { 4.0 } else { 8.0 }, ..PlanOptions::default() }));
            }
            if rng.chance(0.5) {
                plans.push(("talk".into(), PlanOptions { tts_seconds: Some(rng.range(3.5, 9.0)), ..PlanOptions::default() }));
            }
            for (tag, opt) in plans {
                let Ok(p) = plan(a, b, &PlanOptions { id: format!("sim-{pair}-{tag}"), ..opt }) else {
                    count("not_plannable", &mut stats);
                    continue;
                };
                let e = example_from_spec(&p.spec.id, &p.spec, a, b, p.recipe, p.fit.stretch_pct as f32, keys, Label { good: 1, issues: Vec::new(), source: "rules".into() });
                count(&format!("good_{:?}", p.recipe), &mut stats);
                goods.push(e);
            }
            let mut all = Vec::new();
            for g in &goods {
                let mut tried = 0;
                let mut made = 0;
                while made < opt.broken && tried < 12 {
                    tried += 1;
                    let (name, f) = perturb::ALL[rng.below(perturb::ALL.len())];
                    if let Some((mut e, issues)) = f(g, &mut rng) {
                        e.id = format!("{}-{name}", g.id);
                        for i in &issues {
                            count(&format!("issue_{i:?}"), &mut stats);
                        }
                        e.label = Label { good: 0, issues, source: "perturbed".into() };
                        all.push(e);
                        made += 1;
                    }
                }
            }
            // Faults of the pair itself: what must not be layered, layered.
            if keys >= 2 && fit.beat_on_beat() {
                if let Ok(p) = plan(a, b, &PlanOptions { id: format!("sim-{pair}-clash"), force: Some(Recipe::VoiceOverBeat), ..PlanOptions::default() }) {
                    let e = example_from_spec(&p.spec.id, &p.spec, a, b, p.recipe, p.fit.stretch_pct as f32, keys, Label { good: 0, issues: vec![Issue::KeyClashLayering], source: "perturbed".into() });
                    count("issue_KeyClashLayering", &mut stats);
                    all.push(e);
                }
            }
            if matches!(fit.lock, Lock::TwoToThree | Lock::ThreeToTwo) {
                if let Ok(p) = plan(a, b, &PlanOptions { id: format!("sim-{pair}-poly"), force: Some(Recipe::Sneak), ..PlanOptions::default() }) {
                    let e = example_from_spec(&p.spec.id, &p.spec, a, b, p.recipe, p.fit.stretch_pct as f32, keys, Label { good: 0, issues: vec![Issue::PolyrhythmLayering, Issue::BDrumsBeforeOne], source: "perturbed".into() });
                    count("issue_PolyrhythmLayering", &mut stats);
                    all.push(e);
                }
            }
            all.extend(goods);
            let target = if val_songs.contains(&a.id.as_str()) { &mut val } else { &mut train };
            target.extend(all);
        }
    }

    let owner = owner::examples(&opt.data, &by_id)?;
    let write_gz = |path: &std::path::Path, examples: &mut Vec<Example>| -> Result<usize> {
        let mut gz = GzEncoder::new(std::fs::File::create(path)?, Compression::new(6));
        for e in examples.iter_mut() {
            round(e);
            serde_json::to_writer(&mut gz, e)?;
            gz.write_all(b"\n")?;
        }
        gz.finish()?;
        Ok(examples.len())
    };
    let n_train = write_gz(&opt.out.join("train.jsonl.gz"), &mut train)?;
    let n_val = write_gz(&opt.out.join("val.jsonl.gz"), &mut val)?;
    let mut f = std::fs::File::create(opt.out.join("owner.jsonl"))?;
    let mut owner = owner;
    for e in &mut owner {
        round(e);
        serde_json::to_writer(&mut f, e)?;
        f.write_all(b"\n")?;
    }
    stats.insert("train".into(), n_train);
    stats.insert("val".into(), n_val);
    stats.insert("owner".into(), owner.len());
    stats.insert("owner_good".into(), owner.iter().filter(|e| e.label.good == 1).count());
    stats.insert("songs".into(), tables.len());
    stats.insert("seed".into(), opt.seed as usize);
    std::fs::write(opt.out.join("stats.json"), serde_json::to_string_pretty(&stats)?)?;
    Ok(stats)
}
