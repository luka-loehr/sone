//! render <specs.json> [ids…] --stems <dir> (--struct <dir> | --beats <dir>) --out <dir>
//!        [--tracks <dir>] [--nudge-from <dir>] [--stretched-b-from <dir>]
//!        [--dump] [--aac] [--pv | --splice]
//!
//! Renders transitions like `handmix.py`: writes `<id>.wav`, `<id>.report.json`
//! and `<id>.json` (lanes for the site) into `--out` and prints the report.
//! `--struct` is the beat analysis (`<id>.json`); without it, `--tracks <dir>`
//! points at `<dir>/../hand/struct` (the layout on atlas). `--beats <dir>` takes
//! the bar lines from the beat tables (`<dir>/<id>.json`) instead, as the planner
//! counts bars (for its specs; the hand specs use `--struct`).
//! `--nudge-from <dir>` takes B's nudge from `<dir>/<id>.meta.json` (a reference
//! dump) instead of finding it; `--stretched-b-from <dir>` takes deck B after the
//! stretch from `<dir>/<id>.deck_b_stretched.f32` (to check the later stages
//! alone); `--dump` also writes every stage as raw float32 and `<id>.meta.json`;
//! `--aac` encodes `<id>.m4a` with ffmpeg; `--pv` / `--splice` choose the stretch engine.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
use sone_core::Spec;
use sone_render::files::{load_song, load_song_with_table, write_f32, write_wav16};
use sone_render::{MemoryStems, RenderOptions, meter_table, render_with};

/// JSON the way Python's json.dumps writes it (", " and ": " between items), fields in
/// declaration order.
fn py_json<T: serde::Serialize>(v: &T) -> anyhow::Result<String> {
    let compact = serde_json::to_string(v)?;
    let mut out = String::with_capacity(compact.len() * 11 / 10);
    let (mut in_str, mut escaped) = (false, false);
    for ch in compact.chars() {
        out.push(ch);
        if in_str {
            match ch {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_str = false,
                _ => {}
            }
        } else if ch == '"' {
            in_str = true;
        } else if ch == ',' || ch == ':' {
            out.push(' ');
        }
    }
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let mut opts: HashMap<&str, String> = HashMap::new();
    let mut flags = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--stems" | "--struct" | "--beats" | "--out" | "--tracks" | "--nudge-from" | "--stretched-b-from" => {
                opts.insert(a, args.get(i + 1).context("missing value")?.clone());
                i += 2;
            }
            "--dump" | "--aac" | "--pv" | "--splice" => {
                flags.push(a);
                i += 1;
            }
            _ if a.starts_with("--") => bail!("unknown option {a}"),
            _ => {
                pos.push(a.to_string());
                i += 1;
            }
        }
    }
    let Some((specs_path, ids)) = pos.split_first() else {
        bail!("usage: render <specs.json> [ids…] --stems <dir> --struct <dir> --out <dir> [--nudge-from <dir>] [--stretched-b-from <dir>] [--dump] [--aac]");
    };
    let stems = PathBuf::from(opts.get("--stems").context("--stems")?);
    let beats_dir = opts.get("--beats").map(PathBuf::from);
    let struct_dir = match (opts.get("--struct"), opts.get("--tracks"), &beats_dir) {
        (Some(s), _, _) => PathBuf::from(s),
        (None, Some(t), _) => Path::new(t).join("../hand/struct"),
        (None, None, Some(_)) => PathBuf::new(),
        (None, None, None) => bail!("--struct, --tracks or --beats"),
    };
    let out = PathBuf::from(opts.get("--out").context("--out")?);
    std::fs::create_dir_all(&out)?;
    let specs = Spec::load_all(Path::new(specs_path))?;
    let mut songs: HashMap<String, MemoryStems> = HashMap::new();
    for spec in specs.iter().filter(|s| ids.is_empty() || ids.contains(&s.id)) {
        let t0 = Instant::now();
        for id in [&spec.a.id, &spec.b.id] {
            if !songs.contains_key(id) {
                let song = match &beats_dir {
                    Some(dir) => load_song_with_table(&stems, dir, id)?,
                    None => load_song(&stems, &struct_dir, id)?,
                };
                songs.insert(id.clone(), song);
            }
        }
        let t_load = t0.elapsed().as_secs_f64();
        let mut opt = RenderOptions { keep_signals: flags.contains(&"--dump"), ..Default::default() };
        if flags.contains(&"--pv") {
            opt.method = sone_dsp::stretch::Method::PhaseVocoder;
        }
        if flags.contains(&"--splice") {
            opt.method = sone_dsp::stretch::Method::Splice;
        }
        if let Some(dir) = opts.get("--nudge-from") {
            let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(Path::new(dir).join(format!("{}.meta.json", spec.id)))?)?;
            opt.nudge_s = Some(meta["nudge_s"].as_f64().context("nudge_s")?);
        }
        if let Some(dir) = opts.get("--stretched-b-from") {
            let bytes = std::fs::read(Path::new(dir).join(format!("{}.deck_b_stretched.f32", spec.id)))?;
            let v: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            opt.stretched_b = Some(v.as_chunks::<2>().0.iter().map(|c| [c[0], c[1]]).collect());
        }
        let t1 = Instant::now();
        let r = render_with(spec, &songs[&spec.a.id], &songs[&spec.b.id], &opt)?;
        let t_render = t1.elapsed().as_secs_f64();
        let base = out.join(&spec.id);
        write_wav16(&base.with_extension("wav"), &r.audio)?;
        std::fs::write(out.join(format!("{}.report.json", spec.id)), py_json(&r.report)?)?;
        std::fs::write(out.join(format!("{}.json", spec.id)), py_json(&r.lanes)?)?;
        if let Some(sig) = &r.signals {
            for (name, x) in [
                ("deck_a", &sig.deck_a),
                ("deck_b", &sig.deck_b),
                ("deck_b_stretched", &sig.deck_b_stretched),
                ("mix_unlimited", &sig.mix_unlimited),
                ("mix", &r.audio),
            ] {
                write_f32(&out.join(format!("{}.{name}.f32", spec.id)), x)?;
            }
            let meta = serde_json::json!({
                "nudge_s": sig.nudge_s, "ratio": r.ratio,
                "grid_a": [sig.grid_a.t, sig.grid_a.beat, sig.grid_a.shift],
                "grid_b": [sig.grid_b.t, sig.grid_b.beat, sig.grid_b.shift],
                "lufs": [songs[&spec.a.id].lufs, songs[&spec.b.id].lufs],
                "seconds_load": t_load, "seconds_render": t_render,
            });
            std::fs::write(out.join(format!("{}.meta.json", spec.id)), serde_json::to_string_pretty(&meta)?)?;
        }
        if flags.contains(&"--aac") {
            let wav = base.with_extension("wav");
            let status = std::process::Command::new("ffmpeg")
                .args(["-loglevel", "error", "-y", "-i"])
                .arg(&wav)
                .args(["-c:a", "aac", "-b:a", "256k"])
                .arg(base.with_extension("m4a"))
                .status();
            if !status.is_ok_and(|s| s.success()) {
                eprintln!("{}: ffmpeg failed or is not installed, no m4a", spec.id);
            }
        }
        let head = sone_render::Report { bars: Vec::new(), ..r.report.clone() };
        let head = py_json(&head)?.replace(", \"bars\": []", "");
        println!("{} {}", spec.id, head);
        print!("{}", meter_table(&r.report));
        eprintln!("{}: load {t_load:.2}s render {t_render:.2}s", spec.id);
    }
    Ok(())
}
