//! Development aid: vocal levels per sixteenth on the analysis' bars (as
//! vmap.py prints them) with the detected line ends (`]`) and starts (`[`).
use std::path::PathBuf;

use anyhow::Result;
use sone_analysis::measure::Lane;
use sone_analysis::{io, prepare, stats, structure, voice};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (tracks, stems, id) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]), &a[3]);
    let beats = structure::Beats::parse(&std::fs::read_to_string(tracks.join(format!("{id}.beats")))?);
    let m = io::measure_dir(&stems.join(id), io::StemSet::Final, structure::cut_points(&beats))?;
    let p = prepare(id, &beats, m)?;
    let bars = &p.structure.bar_times;
    let bar_db: Vec<f64> = bars.windows(2).map(|w| p.m.seg_db(Lane::Vocals, w[0], w[1])).collect();
    let sung = stats::percentile(&bar_db, 85.0);
    // quarter-bar "beats" as vmap splits the bar
    let mut times = Vec::new();
    for w in bars.windows(2) {
        for j in 0..4 {
            times.push(w[0] + (w[1] - w[0]) * j as f64 / 4.0);
        }
    }
    times.push(*bars.last().unwrap());
    let v16 = voice::sixteenths(&p.m, &times, sung);
    let (ends, starts) = voice::lines(&v16);
    println!("{id} sung ref {sung:.1} dB");
    for pair in a[4..].chunks(2) {
        let (b0, b1): (usize, usize) = (pair[0].parse()?, pair[1].parse()?);
        for (b, bar_t) in bars.iter().enumerate().take(b1 + 1).skip(b0) {
            let mut digits = String::new();
            let mut dbs = String::new();
            for k in 0..16 {
                let i = b * 16 + k;
                let d = v16[i / 4][i % 4];
                let c = if d < -30.0 { '.' } else { char::from_digit(((d + 30.0) / 3.4).clamp(0.0, 9.0) as u32, 10).unwrap() };
                if starts.contains(&i) {
                    digits.push('[');
                }
                digits.push(c);
                if ends.contains(&i) {
                    digits.push(']');
                }
                if k % 4 == 3 {
                    digits.push('|');
                }
                dbs.push_str(&format!("{d:5.0}"));
            }
            println!("{b:4} {bar_t:7.2}  {digits:28} {dbs}");
        }
    }
    Ok(())
}
