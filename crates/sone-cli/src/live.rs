//! `sone live`: a station in one process. Song A is on air, the others are queued one after
//! the other; audio is pulled block by block as a sound card would, the events go to a file
//! as JSON lines.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use sone_core::{Event, SAMPLE_RATE};
use sone_live::{Config, Live, When};

use crate::adapters::{KeyLock, StemRenderer};
use crate::args::Args;
use crate::songs::{Library, OnAir, STATION_LUFS};

const RATE: f64 = SAMPLE_RATE as f64;

/// A WAV file as stereo at the station's rate (other rates are resampled linearly).
fn read_voice(path: &std::path::Path) -> Result<Vec<[f32; 2]>> {
    let mut r = hound::WavReader::open(path).with_context(|| path.display().to_string())?;
    let spec = r.spec();
    let flat: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<std::result::Result<_, _>>()?
        }
    };
    let ch = usize::from(spec.channels.max(1));
    let frames: Vec<[f32; 2]> = flat.chunks_exact(ch).map(|c| [c[0], c[ch.min(2) - 1]]).collect();
    if spec.sample_rate == SAMPLE_RATE {
        return Ok(frames);
    }
    let step = f64::from(spec.sample_rate) / RATE;
    let n = (frames.len() as f64 / step) as usize;
    Ok((0..n)
        .map(|i| {
            let p = i as f64 * step;
            let (k, t) = (p as usize, (p - p.floor()) as f32);
            let (a, b) = (frames[k], frames[(k + 1).min(frames.len() - 1)]);
            [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
        })
        .collect())
}

pub fn run(args: Args) -> Result<()> {
    if args.words.len() < 2 {
        bail!("sone live <A> <B> [<C>…] --stems DIR --beats DIR --out FILE.wav");
    }
    let mut lib = Library::new(&args.path("stems")?, &args.path("beats")?, args.get_or("loudness", STATION_LUFS)?);
    let out_path = args.path("out")?;
    let from_s: f64 = args.get_or("from", 0.0)?;
    let queue_at: f64 = args.get_or("queue-at", 2.0)?;
    let after: f64 = args.get_or("after", 25.0)?;
    let block_frames: usize = args.get_or("block", 1024)?;
    let talk: Option<f64> = args.get("talk")?;
    let voice = args.opt_path("voice").map(|p| read_voice(&p)).transpose()?;
    let talk = talk.or(voice.as_ref().map(|v| v.len() as f64 / RATE));
    // `--skip`: the listener asks for the next song while this one plays; without it the song plays to its end.
    let when = if args.flag("skip") { When::Now } else { When::AtEnd };
    let cfg = Config { lookahead_s: args.get_or("lookahead", 30.0)?, ..Config::default() };

    let first = lib.song(&args.words[0])?;
    let mut renderer = StemRenderer::default();
    renderer.add(first.clone());
    let mut live = Live::new(Box::new(OnAir(first)), (from_s * RATE) as usize, renderer, KeyLock::default(), cfg);

    let mut wav = hound::WavWriter::create(&out_path, hound::WavSpec { channels: 2, sample_rate: SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int })?;
    let mut events_file = args.opt_path("events").map(std::fs::File::create).transpose()?;
    let started = Instant::now();
    let mut cpu = || started.elapsed();

    let mut queue = args.words[1..].iter();
    let mut next_queue_at = Some(queue_at);
    let mut stop_at: Option<f64> = None;
    let mut block = vec![[0.0f32; 2]; block_frames];
    let (mut slowest, mut spent, mut slow_blocks) = (Duration::ZERO, Duration::ZERO, 0usize);
    let block_s = block_frames as f64 / RATE;
    let mut shown = 0;
    loop {
        let now = live.now();
        if next_queue_at.is_some_and(|t| now >= t) {
            next_queue_at = None;
            match queue.next() {
                Some(id) => {
                    let song = lib.song(id)?;
                    live.renderer_mut().add(song.clone());
                    live.queue(Box::new(OnAir(song)), talk, when);
                    if let Some(v) = &voice {
                        live.say(v.clone());
                    }
                    eprintln!("{now:8.2} s  queued {id}");
                }
                None => stop_at = Some(now + after),
            }
        }
        if stop_at.is_some_and(|t| now >= t) {
            break;
        }
        let t0 = Instant::now();
        let events = live.pull(&mut block, &mut cpu);
        let took = t0.elapsed();
        spent += took;
        slowest = slowest.max(took);
        slow_blocks += usize::from(took.as_secs_f64() > block_s);
        for f in &block {
            for s in f {
                wav.write_sample((s * 32767.0).round().clamp(-32768.0, 32767.0) as i16)?;
            }
        }
        for e in &events {
            if let Some(f) = &mut events_file {
                serde_json::to_writer(&mut *f, e)?;
                f.write_all(b"\n")?;
            }
            match e {
                Event::Plan { t, recipe, starts_at, one_at, ends_at, tts, tempo, key, confidence, .. } => {
                    eprintln!("{t:8.2} s  plan: {recipe:?}, starts {starts_at:.2} s, the one {one_at:.2} s, ends {ends_at:.2} s; tempo {:.2} → {:.2} BPM ({}, {:+.2} %), keys {} → {} ({}), confidence {confidence:.2}", tempo.a, tempo.b, tempo.ratio, tempo.stretch_pct, key.a, key.b, key.fit);
                    if let Some(s) = tts {
                        eprintln!("             the DJ may talk from {:.2} s, has to end by {:.2} s ({:.1} s at most), over {}", s.slot_from, s.must_end_by, s.max_seconds, s.over);
                    }
                }
                Event::TransitionStarted { t } => eprintln!("{t:8.2} s  transition started"),
                Event::TheOne { t } => eprintln!("{t:8.2} s  the one"),
                Event::TransitionEnded { t } => {
                    eprintln!("{t:8.2} s  transition ended");
                    // The next song is asked for a few seconds into this one.
                    next_queue_at = Some(*t + 6.0);
                }
                Event::PlanDropped { t, reason } => eprintln!("{t:8.2} s  plan dropped: {reason}"),
                Event::Stats { .. } => {}
            }
        }
        for line in &live.log[shown..] {
            eprintln!("             {line}");
        }
        shown = live.log.len();
    }
    wav.finalize()?;
    let seconds = live.now();
    eprintln!("\n{seconds:.1} s of audio in {:.2} s ({:.2} % of one core); slowest block {:.0} ms, {slow_blocks} of {} blocks took longer than they last", spent.as_secs_f64(), 100.0 * spent.as_secs_f64() / seconds, slowest.as_secs_f64() * 1000.0, (seconds / block_s) as usize);
    for (spec, report, took, audio_s) in &live.renderer().done {
        eprintln!("rendered {}: {audio_s:.1} s of audio in {took:.2} s, beats {:+.1} ms apart, stretch {:+.2} %", spec.id, crate::adapters::beat_offset_ms(report), report.stretch_pct);
    }
    Ok(())
}
