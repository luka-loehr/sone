//! `analyse <id>… --tracks <dir> --stems <dir> --out <dir> [--original-stems] [--print] [--model <file>]`
//!
//! Per song: the beats from `<tracks>/<id>.beats` (the Python reference's
//! Beat This final0), or, when that file is missing, from `sone-beats` on
//! `<tracks>/<id>.f32` (the `final0` weights given with `--model`, else the
//! embedded `small0`); the stems from `<stems>/<id>/`.
//!
//! Writes `<out>/struct/<id>.json` from the original Demucs stems with
//! `--original-stems`, else `<out>/struct_final/<id>.json` and
//! `<out>/beats/<id>.json` (the `BeatTable`) from the final stems. `--print`
//! prints the bar map; a summary line per song goes to stderr.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use sone_analysis::{io, report, structure};

fn beats_for(id: &str, tracks: &Path, model: Option<&Path>) -> Result<structure::Beats> {
    let path = tracks.join(format!("{id}.beats"));
    if let Ok(text) = std::fs::read_to_string(&path) {
        return Ok(structure::Beats::parse(&text));
    }
    let bytes = std::fs::read(tracks.join(format!("{id}.f32"))).with_context(|| format!("neither {} nor {id}.f32", path.display()))?;
    let audio: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
    drop(bytes);
    let flagged = match model {
        Some(file) => sone_beats::track(&sone_beats::load_model_file(file)?, &audio, 2)?,
        None => sone_beats::track(sone_beats::small(), &audio, 2)?,
    };
    Ok(structure::Beats::from_flagged(&flagged))
}

fn main() -> Result<()> {
    let mut ids = Vec::new();
    let (mut tracks, mut stems, mut out) = (PathBuf::from("."), PathBuf::from("."), PathBuf::from("."));
    let (mut original, mut print) = (false, false);
    let mut model: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tracks" => tracks = args.next().context("--tracks <dir>")?.into(),
            "--stems" => stems = args.next().context("--stems <dir>")?.into(),
            "--out" => out = args.next().context("--out <dir>")?.into(),
            "--model" => model = Some(args.next().context("--model <file>")?.into()),
            "--original-stems" => original = true,
            "--print" => print = true,
            s if s.starts_with("--") => bail!("unknown option {s}"),
            _ => ids.push(a),
        }
    }
    let set = if original { io::StemSet::Original } else { io::StemSet::Final };
    let struct_dir = out.join(if original { "struct" } else { "struct_final" });
    let beats_dir = out.join("beats");
    std::fs::create_dir_all(&struct_dir)?;
    if !original {
        std::fs::create_dir_all(&beats_dir)?;
    }
    for id in &ids {
        let beats = beats_for(id, &tracks, model.as_deref())?;
        let started = Instant::now();
        let (m, measure_secs) = io::measure_dir_timed(&stems.join(id), set, structure::cut_points(&beats))?;
        let analysed = Instant::now();
        let song = sone_analysis::analyse(id, &beats, m)?;
        let analysis_secs = measure_secs + analysed.elapsed().as_secs_f64();
        let secs = started.elapsed().as_secs_f64();
        std::fs::write(struct_dir.join(format!("{id}.json")), serde_json::to_string(&song.track)?)?;
        if !original {
            std::fs::write(beats_dir.join(format!("{id}.json")), serde_json::to_string(&song.table)?)?;
        }
        let ph = &song.phase;
        let sections: Vec<String> = ph.sections.iter().map(|(a, b, v)| format!("{a}-{b}:{:+}/{:.2}", v.shift, v.confidence)).collect();
        let windows: Vec<String> = ph.windows.iter().map(|(w, p)| format!("{w}:{:+}/{:.2}", p.shift, p.confidence)).collect();
        eprintln!(
            "{id} bars {} key {} {} ({:.3}; 2nd {} {} {:.3}) tuning {:+.1} | phase {:+} (verdict {:+} conf {:.2} backbeat {:?} contrast {:.1} evidence {:.2}) sections [{}] | windows {} | lines {} | {secs:.2} s ({analysis_secs:.2} s without decoding)",
            song.track.bars.len(),
            song.key.key,
            song.key.camelot,
            song.key.score,
            song.key.second.0,
            song.key.second.1,
            song.key.second.2,
            song.key.tuning_cents,
            ph.shift,
            ph.song.shift,
            ph.song.confidence,
            ph.song.backbeat,
            ph.song.contrast,
            ph.song.evidence,
            sections.join(" "),
            windows.join(" "),
            song.table.line_ends.len(),
        );
        if std::env::var_os("ANALYSE_CHROMA").is_some() {
            eprintln!("{id} chroma {:?}", song.key.chroma);
        }
        if print {
            print!("{}", report::bar_map(&song.track, ""));
        }
    }
    Ok(())
}
