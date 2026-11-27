//! `simulate --data <dir> --out <dir> [--seed n] [--windows n] [--broken n]`
//! (defaults: seed 1, 4 windows, 3 broken versions per good one)
//!
//! Plans a transition for every ordered pair of songs (at the song's end, and
//! at `--windows` random places inside it with 30 s of look-ahead, as after a
//! skip), breaks each one `--broken` times, and adds the owner's verdicts.
//! Writes `train.jsonl.gz`, `val.jsonl.gz`, `owner.jsonl` and `stats.json`.

use anyhow::Result;
use sone_sim::generate::{Options, generate};

fn arg(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.into())
}

fn main() -> Result<()> {
    let opt = Options { data: arg("--data", "data").into(), out: arg("--out", "data/sim").into(), seed: arg("--seed", "1").parse()?, windows: arg("--windows", "4").parse()?, broken: arg("--broken", "3").parse()? };
    let stats = generate(&opt)?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}
