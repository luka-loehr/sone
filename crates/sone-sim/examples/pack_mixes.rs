use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Result;
use sone_core::{BeatTable, Label, Spec};
use sone_plan::keys::distance as key_distance;
use sone_sim::build::{example_from_spec, recipe_of};

fn main() -> Result<()> {
    let mixes_dir = Path::new("/home/user/bot/mixes");
    let beats_dir = Path::new("/home/user/bot/sone/data/analysis/beats");
    let out_path = mixes_dir.join("mixes_1_6.jsonl");

    let mut specs_paths: Vec<_> = std::fs::read_dir(mixes_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".spec.json")))
        .collect();
    specs_paths.sort();

    let mut out = BufWriter::new(File::create(&out_path)?);
    for sp in specs_paths {
        let specs = Spec::load_all(&sp)?;
        let spec = &specs[0];
        let rep_path = sp.to_string_lossy().replace(".spec.json", ".report.json");
        let rep: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&rep_path)?)?;
        let stretch_pct = rep["stretch_pct"].as_f64().unwrap_or(0.0) as f32;
        let beat_offset_ms = rep["align"]["all"][0].as_f64().unwrap_or(0.0).abs() as f32;

        let a: BeatTable = serde_json::from_str(&std::fs::read_to_string(beats_dir.join(format!("{}.json", spec.a.id)))?)?;
        let b: BeatTable = serde_json::from_str(&std::fs::read_to_string(beats_dir.join(format!("{}.json", spec.b.id)))?)?;
        let kd = key_distance(&a.camelot, &b.camelot);
        let rec = recipe_of(spec);
        let mut ex = example_from_spec(
            &spec.id,
            spec,
            &a,
            &b,
            rec,
            stretch_pct,
            kd,
            Label { good: 1, issues: vec![], source: "rules".into() },
        );
        ex.measures.beat_offset_ms = beat_offset_ms;
        writeln!(out, "{}", serde_json::to_string(&ex)?)?;
        println!("Packed {} (recipe={:?}, kd={}, stretch={:.2}%, offset={:.2}ms)", ex.id, rec, kd, stretch_pct, beat_offset_ms);
    }
    println!("Wrote {}", out_path.display());
    Ok(())
}
