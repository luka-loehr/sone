//! The readable bar map of structure.py's `show`.

use std::fmt::Write;

use sone_core::TrackAnalysis;

/// 0..9 for −36..0 dB, '·' below.
fn lvl(v: f32) -> char {
    if v < -36.0 { '·' } else { char::from_digit(((v + 36.0) / 4.0).clamp(0.0, 9.0) as u32, 10).unwrap_or('9') }
}

/// One line per bar: phrase mark, time, mix, the vocal per beat, drums,
/// kick, hats, bass, other, novelty (structure.py's format).
pub fn bar_map(o: &TrackAnalysis, name: &str) -> String {
    let mut s = String::new();
    let lb = &o.local_bpm;
    let (lo, hi) = (lb.iter().copied().fold(f32::INFINITY, f32::min), lb.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let _ = writeln!(
        s,
        "== {} {} | bpm {} (local {}-{}) dur {} steady {:?} resid {}/{} ms drumd {}ms votes {:?} phrase8 {} {:?}",
        o.id,
        name,
        o.bpm,
        if lb.is_empty() { 0.0 } else { lo },
        if lb.is_empty() { 0.0 } else { hi },
        o.duration,
        o.steady,
        o.grid_resid_ms.rms,
        o.grid_resid_ms.max,
        o.drum_delta_ms,
        o.downbeat_votes,
        o.phrase8,
        o.phrase8_scores
    );
    let _ = writeln!(s, "   bar    time  M  vocal  D K H B O  nov   (levels 0-9 = -36..0 dB rel. loud, per-beat vocal digits)");
    for r in &o.bars {
        let mark = if r.bar % 8 == o.phrase8 as usize { '#' } else if r.bar % 4 == (o.phrase8 % 4) as usize { '+' } else { ' ' };
        let vb: String = r.vbeats.iter().map(|&v| lvl(v)).collect();
        let _ = writeln!(
            s,
            "{mark}{:4} {:7.2}  {}  {vb}   {} {} {} {} {}  {:4.0}",
            r.bar,
            r.t,
            lvl(r.mix),
            lvl(r.drums),
            lvl(r.kick),
            lvl(r.hats),
            lvl(r.bass),
            lvl(r.other),
            o.novelty.get(r.bar).copied().unwrap_or(0.0)
        );
    }
    s
}
