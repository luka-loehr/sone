//! onsets <in.wav> [<beats file> <start seconds>]
//! What the stretcher's attack detector finds in a stereo 44.1 kHz WAV: attacks
//! per second, strong ones per second, how much of the time protected zones
//! cover, and (with the song's beat list) how the strong ones sit on the beats.

use anyhow::bail;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        bail!("usage: onsets <in.wav> [<beats file> <start seconds>]");
    }
    let mut r = hound::WavReader::open(&args[1])?;
    let flat: Vec<f32> = r.samples::<f32>().collect::<Result<_, _>>()?;
    let x: Vec<[f32; 2]> = flat.as_chunks::<2>().0.iter().map(|c| [c[0], c[1]]).collect();
    let on = sone_dsp::stretch::detect_onsets(&x);
    let secs = x.len() as f64 / 44100.0;
    let strong: Vec<f64> = on.iter().filter(|o| o.strong).map(|o| o.pos as f64 / 44100.0).collect();
    // Time covered by protected zones (5 ms before, 30 ms after).
    let mut covered = 0i64;
    let mut end = i64::MIN;
    for o in &on {
        let (a, b) = (o.pos - 220, o.pos + 1323);
        covered += b - a.max(end).min(b);
        end = end.max(b);
    }
    print!(
        "onsets/s {:.1} strong/s {:.1} covered {:.0}%",
        on.len() as f64 / secs,
        strong.len() as f64 / secs,
        100.0 * covered as f64 / x.len() as f64
    );
    if args.len() >= 4 {
        let start: f64 = args[3].parse()?;
        let beats: Vec<f64> = std::fs::read_to_string(&args[2])?
            .lines()
            .filter_map(|l| l.split_whitespace().next()?.parse::<f64>().ok())
            .map(|t| t - start)
            .filter(|&t| t > 0.1 && t < secs - 0.1)
            .collect();
        let near = |t: f64, set: &[f64]| set.iter().any(|&s| (s - t).abs() < 0.025);
        let hit = beats.iter().filter(|&&b| near(b, &strong)).count();
        let mut grid = beats.clone();
        grid.extend(beats.windows(2).map(|w| (w[0] + w[1]) / 2.0));
        let on_grid = strong.iter().filter(|&&s| near(s, &grid)).count();
        print!(
            " beats {} with strong {:.0}% strong on 8th grid {:.0}%",
            beats.len(),
            100.0 * hit as f64 / beats.len().max(1) as f64,
            100.0 * on_grid as f64 / strong.len().max(1) as f64
        );
    }
    let mut st: Vec<f64> = on.iter().map(|o| o.strength).collect();
    st.sort_by(f64::total_cmp);
    let pct = |p: f64| st.get(((st.len() as f64 - 1.0) * p) as usize).copied().unwrap_or(0.0);
    println!(" strength p10 {:.1} p50 {:.1} p75 {:.1} p90 {:.1} p97 {:.1}", pct(0.1), pct(0.5), pct(0.75), pct(0.9), pct(0.97));
    if std::env::var_os("ONSET_LIST").is_some() {
        for o in on.iter().take(80) {
            println!("{:.3} {:.1} {}", o.pos as f64 / 44100.0, o.strength, if o.strong { "S" } else { "" });
        }
    }
    Ok(())
}
