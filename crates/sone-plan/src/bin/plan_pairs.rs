//! `plan_pairs <data dir> <out.json> <a id> <b id> [<a id> <b id> …]`: plans the pairs and writes
//! the specs as one JSON file the renderers read.

use std::collections::HashMap;

use anyhow::{Context, Result};
use sone_core::BeatTable;
use sone_plan::{PlanOptions, compat, plan};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = std::path::PathBuf::from(&args[1]);
    let beats_dir = data.join("analysis/beats");
    let tables: Vec<BeatTable> = if beats_dir.is_dir() && std::fs::read_dir(&beats_dir)?.count() >= 2 {
        std::fs::read_dir(&beats_dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).map(|p| BeatTable::load(&p)).collect::<Result<_>>()?
    } else {
        compat::load_dir(&data.join("analysis/struct"), Some(&data.join("analysis/keys.json")))?
    };
    let by_id: HashMap<&str, &BeatTable> = tables.iter().map(|t| (t.id.as_str(), t)).collect();
    let mut specs = Vec::new();
    for (k, pair) in args[3..].chunks(2).enumerate() {
        let a = by_id.get(pair[0].as_str()).with_context(|| pair[0].clone())?;
        let b = by_id.get(pair[1].as_str()).with_context(|| pair[1].clone())?;
        let p = plan(a, b, &PlanOptions { id: format!("p{:02}", k + 1), ..PlanOptions::default() })?;
        eprintln!("p{:02} {:?} confidence {:.2}", k + 1, p.recipe, p.confidence);
        for n in &p.notes {
            eprintln!("    · {n}");
        }
        specs.push(p.spec);
    }
    std::fs::write(&args[2], serde_json::to_string_pretty(&specs)?)?;
    Ok(())
}
