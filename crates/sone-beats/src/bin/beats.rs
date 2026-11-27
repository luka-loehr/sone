//! `beats <id>... --tracks <dir> --model <file.safetensors> [--out <dir>] [--focus mix-out|entry] [--chunks N]`
//!
//! Tracks `<tracks>/<id>.f32` (interleaved stereo f32, 44.1 kHz), writes
//! `<out>/<id>.beats` ("seconds flag" lines, as the Python reference) and,
//! when `<tracks>/<id>.beats` exists, prints the F-measure against it at
//! ±70 ms for beats and downbeats (inside the analysed span for `--focus`).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use sone_beats::{BeatThis, Focus, FocusOptions, f_measure};

/// Beats of an interleaved stereo f32 file, read a second at a time.
fn stream_track(model: &BeatThis, path: &std::path::Path) -> Result<Vec<(f64, bool)>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).with_context(|| format!("{}", path.display()))?;
    let seconds = file.metadata()?.len() as f32 / 8.0 / 44_100.0;
    let mut analyzer = sone_beats::Analyzer::new(44_100)?;
    analyzer.reserve_seconds(seconds);
    let mut buf = vec![0u8; 44_100 * 8];
    let mut mono = Vec::with_capacity(44_100);
    loop {
        let mut got = 0;
        while got < buf.len() {
            let n = file.read(&mut buf[got..])?;
            if n == 0 {
                break;
            }
            got += n;
        }
        if got < 8 {
            break;
        }
        mono.clear();
        mono.extend(buf[..got - got % 8].as_chunks::<8>().0.iter().map(|f| {
            let l = f32::from_le_bytes([f[0], f[1], f[2], f[3]]);
            let r = f32::from_le_bytes([f[4], f[5], f[6], f[7]]);
            (l + r) / 2.0
        }));
        analyzer.push(&mono);
        if got < buf.len() {
            break;
        }
    }
    Ok(analyzer.finish(model).flagged())
}

fn main() -> Result<()> {
    let mut ids = Vec::new();
    let mut tracks = PathBuf::from(".");
    let mut out: Option<PathBuf> = None;
    let mut model_arg: Option<PathBuf> = None;
    let mut focus: Option<Focus> = None;
    let mut chunks = 1usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tracks" => tracks = args.next().context("--tracks <dir>")?.into(),
            "--out" => out = Some(args.next().context("--out <dir>")?.into()),
            "--model" => model_arg = Some(args.next().context("--model <file.safetensors>")?.into()),
            "--focus" => {
                focus = Some(match args.next().as_deref() {
                    Some("mix-out") => Focus::MixOut,
                    Some("entry") => Focus::Entry,
                    other => bail!("--focus mix-out|entry, not {other:?}"),
                })
            }
            "--chunks" => chunks = args.next().context("--chunks N")?.parse()?,
            s if s.starts_with("--") => bail!("unknown option {s}"),
            _ => ids.push(a),
        }
    }
    let model_path = model_arg.context("--model <beat_this.safetensors> is required (or use `sone transition` for Tympanum)")?;
    let loaded = sone_beats::load_model_file(&model_path)?;
    let model: &BeatThis = &loaded;
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir)?;
    }
    let (mut sum_b, mut sum_d, mut n) = (0.0, 0.0, 0);
    for id in &ids {
        let path = tracks.join(format!("{id}.f32"));
        let started = Instant::now();
        let (beats, span) = match focus {
            // the whole song streams from the file: only the spectrogram is kept
            None => (stream_track(model, &path)?, None),
            Some(f) => {
                let bytes = std::fs::read(&path).with_context(|| format!("{id}.f32"))?;
                let audio: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
                drop(bytes);
                let options = FocusOptions { max_chunks: chunks, ..FocusOptions::default_for(f) };
                let b = sone_beats::track_focus(model, &audio, 2, f, options)?;
                let span = (b.first().map_or(0.0, |b| b.0), b.last().map_or(0.0, |b| b.0));
                (b, Some(span))
            }
        };
        let secs = started.elapsed().as_secs_f64();
        if let Some(dir) = &out {
            let text: String = beats.iter().map(|&(t, d)| format!("{t:.4} {}\n", u8::from(d))).collect();
            std::fs::write(dir.join(format!("{id}.beats")), text)?;
        }
        let reference = tracks.join(format!("{id}.beats"));
        if let Ok(text) = std::fs::read_to_string(&reference) {
            let mut rb = Vec::new();
            let mut rd = Vec::new();
            for line in text.lines() {
                let mut p = line.split_whitespace();
                let Some(t) = p.next().and_then(|v| v.parse::<f64>().ok()) else { continue };
                let flag = p.next().is_some_and(|v| v == "1");
                // a focus run is compared inside its own span only
                if let Some((a, b)) = span
                    && (t < a - 0.035 || t > b + 0.035)
                {
                    continue;
                }
                rb.push(t);
                if flag {
                    rd.push(t);
                }
            }
            let eb: Vec<f64> = beats.iter().map(|b| b.0).collect();
            let ed: Vec<f64> = beats.iter().filter(|b| b.1).map(|b| b.0).collect();
            let (fb, fd) = (f_measure(&rb, &eb, 0.07), f_measure(&rd, &ed, 0.07));
            sum_b += fb;
            sum_d += fd;
            n += 1;
            println!("{id} beats {:4} ref {:4} F {fb:.4} down F {fd:.4} secs {secs:.2}", eb.len(), rb.len());
        } else {
            println!("{id} beats {:4} secs {secs:.2}", beats.len());
        }
    }
    if n > 1 {
        println!("mean F beats {:.4} downbeats {:.4} over {n}", sum_b / n as f64, sum_d / n as f64);
    }
    Ok(())
}
