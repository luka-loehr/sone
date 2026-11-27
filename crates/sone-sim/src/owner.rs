//! The owner's verdicts as training examples: every version of a hand-built
//! transition he commented on, the ones he accepted at the end, and every
//! transition the system made that he judged (`data/owner/labelled.json`).

use std::collections::HashMap;

use anyhow::{Context, Result};
use sone_core::{BeatTable, Example, Issue, Label, Spec};
use serde::Deserialize;

use crate::build::{example_from_spec, recipe_of};

#[derive(Debug, Deserialize)]
struct Note {
    id: String,
    version: String,
    verdict: String,
    #[serde(default)]
    issues: Vec<String>,
}

pub fn examples(data: &std::path::Path, tables: &HashMap<String, BeatTable>) -> Result<Vec<Example>> {
    let notes: Vec<Note> = serde_json::from_str(&std::fs::read_to_string(data.join("labels.json"))?)?;
    let mut specs: HashMap<String, Vec<Spec>> = HashMap::new();
    for v in ["S1", "S3", "S4", "S5", "S6", "S7"] {
        specs.insert(v.into(), Spec::load_all(&data.join(format!("versions/{v}.json")))?);
    }
    let mut out = Vec::new();
    let mut add = |version: &str, id: &str, good: bool, issues: Vec<Issue>| -> Result<()> {
        let Some(spec) = specs.get(version).and_then(|v| v.iter().find(|s| s.id == id)) else {
            return Ok(());
        };
        let (a, b) = (tables.get(&spec.a.id).with_context(|| spec.a.id.clone())?, tables.get(&spec.b.id).with_context(|| spec.b.id.clone())?);
        let ratio = f64::from(spec.b.beats_per_bar) / f64::from(spec.a.beats_per_bar);
        let stretch = if spec.no_sync { 0.0 } else { ((a.bpm * ratio / b.bpm - 1.0) * 100.0) as f32 };
        let keys = sone_plan::keys::song_distance(&a.camelot, &a.camelot_alt, &b.camelot, &b.camelot_alt);
        let mut e = example_from_spec(&format!("owner-{version}-{id}"), spec, a, b, recipe_of(spec), stretch, keys, Label { good: u8::from(good), issues: issues.clone(), source: "owner".into() });
        if issues.contains(&Issue::VocalLeak) {
            e.measures.vocal_leak_db = -25.0;
        }
        out.push(e);
        Ok(())
    };
    for n in &notes {
        if n.id == "*" || n.version == "production" || n.version == "S0" {
            continue;
        }
        let issues: Vec<Issue> = n.issues.iter().filter_map(|i| serde_json::from_value(serde_json::Value::String(i.clone())).ok()).collect();
        add(&n.version, &n.id, n.verdict == "positive", issues)?;
    }
    // "All others are just amazing and perfect" (2026-09-29, about version S6).
    for k in (1..=13).filter(|&k| k != 10) {
        add("S6", &format!("h{k:02}"), true, Vec::new())?;
    }
    // His verdicts on the transitions the system made itself (`reference/learn/learn.py` writes them).
    let labelled = data.join("owner/labelled.json");
    if labelled.exists() {
        #[derive(Deserialize)]
        struct Labelled {
            id: String,
            good: u8,
            #[serde(default)]
            issues: Vec<String>,
            #[serde(default)]
            beat_offset_ms: Option<f32>,
            #[serde(default)]
            vocal_leak_db: Option<f32>,
            spec: Spec,
        }
        let items: Vec<Labelled> = serde_json::from_str(&std::fs::read_to_string(&labelled)?)?;
        for l in items {
            let (Some(a), Some(b)) = (tables.get(&l.spec.a.id), tables.get(&l.spec.b.id)) else { continue };
            let issues: Vec<Issue> = l.issues.iter().filter_map(|i| serde_json::from_value(serde_json::Value::String(i.clone())).ok()).collect();
            let ratio = f64::from(l.spec.b.beats_per_bar) / f64::from(l.spec.a.beats_per_bar);
            let stretch = if l.spec.no_sync { 0.0 } else { ((a.bpm * ratio / b.bpm - 1.0) * 100.0) as f32 };
            let keys = sone_plan::keys::song_distance(&a.camelot, &a.camelot_alt, &b.camelot, &b.camelot_alt);
            let mut ex = example_from_spec(&format!("owner-{}", l.id), &l.spec, a, b, recipe_of(&l.spec), stretch, keys, Label { good: l.good, issues: issues.clone(), source: "owner".into() });
            if let Some(bo) = l.beat_offset_ms {
                ex.measures.beat_offset_ms = bo;
            } else if issues.contains(&Issue::BeatsOffset) {
                ex.measures.beat_offset_ms = 12.0;
            }
            if let Some(vl) = l.vocal_leak_db {
                ex.measures.vocal_leak_db = vl;
            } else if issues.contains(&Issue::VocalLeak) {
                ex.measures.vocal_leak_db = -25.0;
            }
            out.push(ex);
        }
    }
    Ok(out)
}
