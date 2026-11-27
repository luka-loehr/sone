//! stretch <in.wav> <out.wav> <ratio> [--stream <chunk>] [--splice]
//! Key-locked time stretch of a stereo 44.1 kHz WAV (float or 16/24-bit),
//! written as 32-bit float, with the phase vocoder (default) or the splicing engine
//! (`--splice`). Prints the engine's counters and the CPU time.

use anyhow::{Context, bail};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: stretch <in.wav> <out.wav> <ratio> [--stream <chunk frames>] [--splice]");
    }
    let ratio: f64 = args[3].parse().context("ratio")?;
    let stream: Option<usize> = args.iter().position(|a| a == "--stream").map(|i| args[i + 1].parse()).transpose()?;
    let method = if args.iter().any(|a| a == "--splice") { sone_dsp::stretch::Method::Splice } else { sone_dsp::stretch::Method::PhaseVocoder };
    let mut r = hound::WavReader::open(&args[1])?;
    let spec = r.spec();
    if spec.channels != 2 || spec.sample_rate != 44100 {
        bail!("need stereo 44.1 kHz, got {} ch {} Hz", spec.channels, spec.sample_rate);
    }
    let flat: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    let x: Vec<[f32; 2]> = flat.as_chunks::<2>().0.iter().map(|c| [c[0], c[1]]).collect();
    let t0 = std::time::Instant::now();
    let (y, stats) = match stream {
        None => sone_dsp::stretch::stretch_with(&x, ratio, method),
        Some(chunk) => {
            let mut s = sone_dsp::Stretcher::with_method(ratio, method);
            let mut out = Vec::new();
            for c in flat.chunks(chunk * 2) {
                out.extend(s.process(c));
            }
            out.extend(s.flush());
            (out.as_chunks::<2>().0.iter().map(|c| [c[0], c[1]]).collect(), s.stats())
        }
    };
    let secs = t0.elapsed().as_secs_f64();
    let mut w = hound::WavWriter::create(
        &args[2],
        hound::WavSpec { channels: 2, sample_rate: 44100, bits_per_sample: 32, sample_format: hound::SampleFormat::Float },
    )?;
    for v in &y {
        w.write_sample(v[0])?;
        w.write_sample(v[1])?;
    }
    w.finalize()?;
    let dur = x.len() as f64 / 44100.0;
    println!(
        "in {} out {} (want {}) cpu {:.3}s realtime x{:.0} {:?}",
        x.len(),
        y.len(),
        (x.len() as f64 * ratio).round(),
        secs,
        dur / secs,
        stats
    );
    Ok(())
}
