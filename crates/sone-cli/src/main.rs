//! `sone`: Sone-6.5M in one binary.
//!
//! ```text
//! sone analyse  <id>… --tracks DIR --stems DIR --out DIR [--model FILE]
//! sone plan     <A> <B> --beats DIR [--talk SECONDS] [--lead BARS] [--window FROM,TO] [--out FILE]
//! sone render   <specs.json> [id…] --stems DIR (--beats DIR | --struct DIR) --out DIR
//! sone mix      <A> <B> --stems DIR --beats DIR --out DIR [--talk SECONDS]
//! sone live     <A> <B> [<C>…] --stems DIR --beats DIR --out FILE.wav [--events FILE.jsonl]
//!               [--from SECONDS] [--queue-at SECONDS] [--skip] [--talk SECONDS] [--voice FILE.wav]
//!               [--after SECONDS] [--block FRAMES] [--lookahead SECONDS] [--loudness LUFS]
//! sone simulate [--data DIR] [--out DIR] [--seed N] [--windows N] [--broken N]
//! ```
//!
//! Songs are named by their id. `--stems` holds `<id>/{vocals,drums,bass,other}.flac`
//! (and `vocals_r`, `drums_i`, `bass_i` when the song was separated a second time),
//! `--beats` the beat tables `<id>.json` that `sone analyse` writes.

mod adapters;
mod args;
mod live;
mod songs;

use std::path::Path;

use anyhow::{Context, Result, bail};
use sone_core::Spec;
use sone_plan::{Plan, PlanOptions};
use sone_render::files::{load_song, write_wav16};

use crate::adapters::{beat_offset_ms, level_dip_db, render_placed};
use crate::args::Args;
use crate::songs::{Library, STATION_LUFS};

const USAGE: &str = "sone <analyse|plan|render|mix|live|simulate> … (see the top of crates/sone-cli/src/main.rs)";

fn main() -> Result<()> {
    let mut argv = std::env::args().skip(1);
    let Some(command) = argv.next() else { bail!(USAGE) };
    match command.as_str() {
        "analyse" => analyse(Args::parse(argv, &["original-stems", "print"], &["tracks", "stems", "out", "model"])?),
        "plan" => plan(Args::parse(argv, &[], &["beats", "talk", "lead", "window", "out", "id", "owner"])?),
        "render" => render(Args::parse(argv, &[], &["stems", "beats", "struct", "out"])?),
        "mix" => mix(Args::parse(argv, &[], &["stems", "beats", "out", "talk", "lead", "id", "loudness", "owner"])?),
        "live" => live::run(Args::parse(argv, &["skip"], &["stems", "beats", "out", "events", "from", "queue-at", "talk", "voice", "after", "block", "loudness", "lookahead"])?),
        "simulate" => simulate(Args::parse(argv, &[], &["data", "out", "seed", "windows", "broken"])?),
        _ => bail!(USAGE),
    }
}

/// Beats, bars, levels, lines, key of every song; writes `struct_final/<id>.json` and `beats/<id>.json`.
fn analyse(args: Args) -> Result<()> {
    use sone_analysis::{io, structure};
    let (tracks, stems, out) = (args.path("tracks")?, args.path("stems")?, args.path("out")?);
    let model = args.opt_path("model").map(|p| sone_beats::load_model_file(&p)).transpose()?;
    let set = if args.flag("original-stems") { io::StemSet::Original } else { io::StemSet::Final };
    let struct_dir = out.join(if args.flag("original-stems") { "struct" } else { "struct_final" });
    std::fs::create_dir_all(&struct_dir)?;
    std::fs::create_dir_all(out.join("beats"))?;
    for id in &args.words {
        let started = std::time::Instant::now();
        let beats = match std::fs::read_to_string(tracks.join(format!("{id}.beats"))) {
            Ok(text) => structure::Beats::parse(&text),
            Err(_) => {
                let path = tracks.join(format!("{id}.f32"));
                let bytes = std::fs::read(&path).with_context(|| format!("neither {id}.beats nor {}", path.display()))?;
                let audio: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
                let flagged = sone_beats::track(model.as_ref().unwrap_or(sone_beats::small()), &audio, 2)?;
                structure::Beats::from_flagged(&flagged)
            }
        };
        let m = io::measure_dir(&stems.join(id), set, structure::cut_points(&beats))?;
        let song = sone_analysis::analyse(id, &beats, m)?;
        std::fs::write(struct_dir.join(format!("{id}.json")), serde_json::to_string(&song.track)?)?;
        if !args.flag("original-stems") {
            std::fs::write(out.join("beats").join(format!("{id}.json")), serde_json::to_string(&song.table)?)?;
        }
        if args.flag("print") {
            print!("{}", sone_analysis::report::bar_map(&song.track, ""));
        }
        let t = &song.table;
        eprintln!("{id}: {:.2} BPM, {} bars, key {} ({}), {} line ends, {:.1} s", t.bpm, t.bars.len(), t.key, t.camelot, t.line_ends.len(), started.elapsed().as_secs_f64());
    }
    Ok(())
}

/// The owner's decisions for single pairs (`--owner FILE`, by default `data/owner/pair_rules.json`).
fn owner_rules(args: &Args) -> Result<sone_plan::PairRules> {
    sone_plan::PairRules::load(&args.path_or("owner", "data/owner/pair_rules.json"))
}

fn plan_options(args: &Args, default_id: String) -> Result<PlanOptions> {
    let window = match args.get::<String>("window")? {
        Some(w) => {
            let (from, to) = w.split_once(',').context("--window FROM,TO (seconds of song A)")?;
            Some((from.parse()?, to.parse()?))
        }
        None => None,
    };
    Ok(PlanOptions { id: args.get_or("id", default_id)?, tts_seconds: args.get("talk")?, lead_bars: args.get_or("lead", 8.0)?, window, ..PlanOptions::default() })
}

fn print_plan(p: &Plan) {
    eprintln!("{}: {:?}, the one at {:.2} s of song A, confidence {:.2}", p.spec.id, p.recipe, p.one_s, p.confidence);
    for n in &p.notes {
        eprintln!("  · {n}");
    }
    if let Some(t) = &p.tts {
        eprintln!("  · the DJ may talk from {:+.2} to {:+.2} bars ({:.1} s), over {}", t.from_u, t.post_u, t.max_seconds(p.bar_s), t.over);
    }
}

/// The transition from A to B as a spec.
fn plan(args: Args) -> Result<()> {
    let [a, b] = args.words.as_slice() else { bail!("sone plan <A> <B> --beats DIR") };
    let lib = Library::new(Path::new("."), &args.path("beats")?, STATION_LUFS);
    let (ta, tb) = (lib.table(a)?, lib.table(b)?);
    let opt = owner_rules(&args)?.apply(a, b, plan_options(&args, format!("{}-{}", &a[..a.len().min(6)], &b[..b.len().min(6)]))?);
    let p = sone_plan::plan(&ta, &tb, &opt)?;
    print_plan(&p);
    let json = serde_json::to_string_pretty(&[&p.spec])?;
    match args.opt_path("out") {
        Some(path) => std::fs::write(path, json)?,
        None => println!("{json}"),
    }
    Ok(())
}

fn write_render(out: &Path, placed: &adapters::Placed) -> Result<()> {
    let id = &placed.spec.id;
    let r = &placed.rendered;
    write_wav16(&out.join(format!("{id}.wav")), &r.audio)?;
    std::fs::write(out.join(format!("{id}.report.json")), serde_json::to_string(&r.report)?)?;
    std::fs::write(out.join(format!("{id}.lanes.json")), serde_json::to_string(&r.lanes)?)?;
    std::fs::write(out.join(format!("{id}.spec.json")), serde_json::to_string_pretty(&[&placed.spec])?)?;
    eprintln!(
        "{id}: {:.1} s of audio in {:.2} s; tempo {:.2} → {:.2} BPM (stretch {:+.2} %), beats {:+.1} ms apart, level dip {:.1} dB, limiter {:.1} dB",
        r.audio.len() as f64 / sone_dsp::SR,
        placed.render_seconds,
        r.report.tempo_b,
        r.report.tempo_a,
        r.report.stretch_pct,
        beat_offset_ms(&r.report),
        level_dip_db(&r.report),
        r.report.limiter_min_gain_db,
    );
    Ok(())
}

/// Specs to audio. With `--struct` the bars are counted as in the lab (the hand-built specs),
/// with `--beats` as in the beat tables (the planner's specs).
fn render(args: Args) -> Result<()> {
    let Some((specs, ids)) = args.words.split_first() else { bail!("sone render <specs.json> [id…] --stems DIR (--beats DIR | --struct DIR) --out DIR") };
    let (stems, out) = (args.path("stems")?, args.path("out")?);
    std::fs::create_dir_all(&out)?;
    let specs = Spec::load_all(Path::new(specs))?;
    for spec in specs.iter().filter(|s| ids.is_empty() || ids.contains(&s.id)) {
        let started = std::time::Instant::now();
        let rendered = match (args.opt_path("beats"), args.opt_path("struct")) {
            (Some(beats), _) => {
                let mut lib = Library::new(&stems, &beats, STATION_LUFS);
                let (a, b) = (lib.song(&spec.a.id)?, lib.song(&spec.b.id)?);
                sone_render::render_with(spec, &a.stems, &b.stems, &sone_render::RenderOptions::default())?
            }
            (None, Some(dir)) => {
                let (a, b) = (load_song(&stems, &dir, &spec.a.id)?, load_song(&stems, &dir, &spec.b.id)?);
                sone_render::render(spec, &a, &b)?
            }
            (None, None) => bail!("--beats DIR or --struct DIR"),
        };
        let placed = adapters::Placed { a_from: 0, b_zero: rendered.b_zero_s, spec: spec.clone(), rendered, render_seconds: started.elapsed().as_secs_f64() };
        write_render(&out, &placed)?;
    }
    Ok(())
}

/// Plan and render in one go.
fn mix(args: Args) -> Result<()> {
    let [a, b] = args.words.as_slice() else { bail!("sone mix <A> <B> --stems DIR --beats DIR --out DIR") };
    let out = args.path("out")?;
    std::fs::create_dir_all(&out)?;
    let mut lib = Library::new(&args.path("stems")?, &args.path("beats")?, args.get_or("loudness", STATION_LUFS)?);
    let (sa, sb) = (lib.song(a)?, lib.song(b)?);
    let opt = owner_rules(&args)?.apply(a, b, plan_options(&args, format!("{}-{}", &a[..a.len().min(6)], &b[..b.len().min(6)]))?);
    // A blend whose drums drift apart is tried shorter; if that drifts too, there is no transition.
    let mut lead = opt.lead_bars;
    loop {
        let p = sone_plan::plan(&sa.table, &sb.table, &PlanOptions { lead_bars: lead, ..opt.clone() })?;
        print_plan(&p);
        match render_placed(&p.spec, &sa, &sb) {
            Ok(placed) => return write_render(&out, &placed),
            Err(e) if lead > 4.0 => {
                eprintln!("{}: {e:#}; trying a blend of {} bars", p.spec.id, lead / 2.0);
                lead /= 2.0;
            }
            Err(e) => bail!("{}: {e:#}: no transition", p.spec.id),
        }
    }
}

fn simulate(args: Args) -> Result<()> {
    let d = sone_sim::generate::Options::default();
    let opt = sone_sim::generate::Options {
        data: args.path_or("data", "data"),
        out: args.path_or("out", "data/sim"),
        seed: args.get_or("seed", d.seed)?,
        windows: args.get_or("windows", d.windows)?,
        broken: args.get_or("broken", d.broken)?,
    };
    let stats = sone_sim::generate::generate(&opt)?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}
