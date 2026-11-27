use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use sone_core::Example;
use sone_infer::{BeatNet, BeatWorkspace, WeightFile};

fn load_net(path: &Path) -> Result<BeatNet> {
    BeatNet::load(WeightFile::from_vec(std::fs::read(path)?)?)
}

fn load_jsonl(path: &Path) -> Result<Vec<Example>> {
    let f = File::open(path)?;
    let reader: Box<dyn BufRead> = if path.to_string_lossy().ends_with(".gz") {
        Box::new(BufReader::new(flate2::read::GzDecoder::new(f)))
    } else {
        Box::new(BufReader::new(f))
    };
    let mut out = Vec::new();
    for line in reader.lines() {
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&l)?);
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exports = Path::new(args.first().map(String::as_str).unwrap_or("exports"));
    let owner_path = Path::new(args.get(1).map(String::as_str).unwrap_or("data/sim/owner.jsonl"));
    let extra_path = args.get(2).map(Path::new);

    let pick = |new_name: &str, old_name: &str| {
        let p = exports.join(new_name);
        if p.exists() { p } else { exports.join(old_name) }
    };
    let plan_f32 = load_net(&pick("sone-cortex-f32.safetensors", "plan-best-f32.safetensors"))?;
    let plan_f16 = load_net(&pick("sone-cortex-f16.safetensors", "plan-best-f16.safetensors"))?;
    let crit_f32 = load_net(&pick("sone-cingulate-f32.safetensors", "critic-best-f32.safetensors"))?;
    let crit_f16 = load_net(&pick("sone-cingulate-f16.safetensors", "critic-best-f16.safetensors"))?;

    println!("Loaded all 4 safetensors cleanly via sone_infer::BeatNet::load!");
    println!("  plan  config: {:?}", plan_f16.config);
    println!("  critic config: {:?}", crit_f16.config);

    let owner = load_jsonl(owner_path)?;
    println!("\nLoaded {} owner examples from {}", owner.len(), owner_path.display());

    let mut ws = BeatWorkspace::default();

    // Benchmark single-threaded CPU latency (20 reps on first example)
    let ex0 = &owner[0];
    for _ in 0..3 {
        let _ = plan_f16.cortex(&mut ws, ex0);
        let _ = crit_f16.cingulate(&mut ws, ex0);
    }
    let reps = 20;
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = plan_f16.cortex(&mut ws, ex0);
    }
    let plan_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    let t1 = Instant::now();
    for _ in 0..reps {
        let _ = crit_f16.cingulate(&mut ws, ex0);
    }
    let crit_ms = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    println!("\nSingle-thread CPU latency (f16): Plan = {:.2} ms/transition, Critic = {:.2} ms/transition", plan_ms, crit_ms);

    // Evaluate on owner.jsonl & check f32 vs f16 parity
    let mut plan_n = 0usize;
    let mut plan_rec_ok = 0usize;
    let mut plan_bars_err = 0.0f64;
    let mut plan_max_f16_diff = 0.0f32;

    let mut crit_ok = 0usize;
    let mut crit_tp = 0usize;
    let mut crit_fp = 0usize;
    let mut crit_fn = 0usize;
    let mut crit_tn = 0usize;
    let mut crit_max_f16_diff = 0.0f32;

    for ex in &owner {
        let c32 = crit_f32.cingulate(&mut ws, ex);
        let c16 = crit_f16.cingulate(&mut ws, ex);
        crit_max_f16_diff = crit_max_f16_diff.max((c32.good - c16.good).abs());
        let pred_good = c16.good >= 0.5;
        let true_good = ex.label.good == 1;
        if pred_good == true_good {
            crit_ok += 1;
        }
        match (pred_good, true_good) {
            (true, true) => crit_tp += 1,
            (true, false) => crit_fp += 1,
            (false, true) => crit_fn += 1,
            (false, false) => crit_tn += 1,
        }

        if true_good {
            let p32 = plan_f32.cortex(&mut ws, ex);
            let p16 = plan_f16.cortex(&mut ws, ex);
            plan_max_f16_diff = plan_max_f16_diff
                .max((p32.start_bars - p16.start_bars).abs())
                .max((p32.swap_bars - p16.swap_bars).abs());
            plan_n += 1;
            if p16.recipe == ex.recipe {
                plan_rec_ok += 1;
            }
            plan_bars_err += ((p16.start_bars - ex.start_bars).abs() + (p16.swap_bars - ex.swap_bars).abs()) as f64 / 2.0;
        }
    }

    println!("\nHeld-out Owner Evaluation (pure Rust sone-infer f16):");
    println!(
        "  Plan   (n={}): recipe_acc = {:.1}% ({}/{}), bars_mae = {:.2} bars, f32-vs-f16 max diff = {:.5}",
        plan_n,
        100.0 * plan_rec_ok as f64 / plan_n.max(1) as f64,
        plan_rec_ok,
        plan_n,
        plan_bars_err / plan_n.max(1) as f64,
        plan_max_f16_diff
    );
    println!(
        "  Critic (n={}): good_acc = {:.1}% ({}/{}), TP={} FP={} FN={} TN={}, f32-vs-f16 max diff = {:.5}",
        owner.len(),
        100.0 * crit_ok as f64 / owner.len().max(1) as f64,
        crit_ok,
        owner.len(),
        crit_tp,
        crit_fp,
        crit_fn,
        crit_tn,
        crit_max_f16_diff
    );

    if let Some(ep) = extra_path {
        let extras = load_jsonl(ep)?;
        println!("\nScoring {} transitions from {}:", extras.len(), ep.display());
        for ex in &extras {
            let c = crit_f16.cingulate(&mut ws, ex);
            let p = plan_f16.cortex(&mut ws, ex);
            let mut top_issues: Vec<_> = c.issues.iter().filter(|(_, prob)| *prob >= 0.15).collect();
            top_issues.sort_by(|a, b| b.1.total_cmp(&a.1));
            let iss_str: Vec<String> = top_issues.iter().take(5).map(|(i, pr)| format!("{:?}:{:.2}", i, pr)).collect();
            println!(
                "  {} | Critic good={:.3} issues=[{}] | Rule: {:?} start={:.1} swap={:.1} | AI Plan: {:?} (p={:.2}) start={:.1} swap={:.1}",
                ex.id,
                c.good,
                iss_str.join(", "),
                ex.recipe,
                ex.start_bars,
                ex.swap_bars,
                p.recipe,
                p.recipe_probs.iter().cloned().fold(0.0f32, f32::max),
                p.start_bars,
                p.swap_bars,
            );
        }
    }
    Ok(())
}
