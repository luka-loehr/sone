//! The voice at sixteenth-note resolution and where sung lines end and begin
//! (RULES.md rule 13: a voice is never cut inside a word; leave on the end
//! of a line, enter on the start of one).
//!
//! Level: per sixteenth note of the beat grid, the vocal stem's RMS in dB
//! relative to the song's sung level, which is the 85th percentile of the
//! bars' vocal levels (as `vmap.py`; 0 dB is a typical sung bar, a bar
//! without voice sits far below).
//!
//! Lines, with thresholds read off the owner's vocal maps (`vmap.py`, one
//! digit per 3.4 dB from −30 dB):
//!
//! - a sixteenth is *voiced* when its level is above −20 dB (vmap digits 3
//!   and up); a word's decaying tail below that (digits 0–2) does not count;
//! - voiced sixteenths separated by at most one unvoiced one belong to the
//!   same stretch (a consonant, a breath inside a word), unless that one is
//!   silent, below −30 dB (vmap's '·'): silence always breaks a stretch;
//! - a stretch that is a single sixteenth below −14 dB is a blip (a breath,
//!   a hi-hat leaking into the stem, reverb) and is ignored;
//! - a *line end* is the last voiced sixteenth of a stretch followed by at
//!   least two unvoiced sixteenths (an eighth note of silence or more); a
//!   *line start* is the first voiced sixteenth of a stretch preceded by at
//!   least two unvoiced ones (or the song's start).

use crate::measure::{Lane, Measurements};

/// Level above which a sixteenth is voiced, dB relative to the sung level.
pub const VOICED_DB: f32 = -20.0;
/// A lone voiced sixteenth quieter than this is a blip.
pub const BLIP_DB: f32 = -14.0;
/// Below this a sixteenth is silent and breaks a stretch.
pub const SILENT_DB: f32 = -30.0;
/// Unvoiced sixteenths that end a line.
pub const GAP: usize = 2;

/// Vocal level per sixteenth of every beat, dB relative to `sung_db`
/// (clamped at −60). `times` are the beat times; the last beat is as long
/// as the one before it.
pub fn sixteenths(m: &Measurements, times: &[f64], sung_db: f64) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(times.len());
    for i in 0..times.len() {
        let t0 = times[i];
        let len = if i + 1 < times.len() { times[i + 1] - t0 } else if i > 0 { t0 - times[i - 1] } else { 0.5 };
        let mut v = [0.0f32; 4];
        for (j, x) in v.iter_mut().enumerate() {
            let a = t0 + len * j as f64 / 4.0;
            *x = ((m.level_db(Lane::Vocals, a, a + len / 4.0) - sung_db).max(-60.0)) as f32;
        }
        out.push(v);
    }
    out
}

/// Line ends and line starts, as sixteenth indices (`beat * 4 + j`).
pub fn lines(v16: &[[f32; 4]]) -> (Vec<usize>, Vec<usize>) {
    let flat: Vec<f32> = v16.iter().flatten().copied().collect();
    let n = flat.len();
    let voiced: Vec<bool> = flat.iter().map(|&v| v > VOICED_DB).collect();
    // stretches of voiced sixteenths, single unvoiced ones bridged
    let mut stretches: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if !voiced[i] {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        let mut j = i + 1;
        while j < n {
            if voiced[j] {
                end = j;
                j += 1;
            } else if j + 1 < n && voiced[j + 1] && flat[j] >= SILENT_DB {
                j += 1;
            } else {
                break;
            }
        }
        let blip = end == start && flat[start] < BLIP_DB;
        if !blip {
            stretches.push((start, end));
        }
        i = end + 1;
    }
    // merge stretches closer than the gap (a blip between them was dropped)
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for s in stretches {
        match merged.last_mut() {
            Some(last) if s.0 <= last.1 + GAP => last.1 = s.1,
            _ => merged.push(s),
        }
    }
    let ends = merged.iter().filter(|s| s.1 + GAP < n).map(|s| s.1).collect();
    let starts = merged.iter().map(|s| s.0).collect();
    (ends, starts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// vmap digits to dB (the middle of each 3.4 dB step; '.' is silence).
    fn from_map(map: &str) -> Vec<[f32; 4]> {
        let d: Vec<f32> = map
            .chars()
            .filter(|c| *c != '|' && *c != ' ')
            .map(|c| c.to_digit(10).map_or(-45.0, |d| d as f32 * 3.4 - 30.0 + 1.7))
            .collect();
        d.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
    }

    #[test]
    fn a_line_ends_before_a_gap_and_the_next_begins_after_it() {
        // "Don't Stop The Music", bars 1-3 of its intro (vmap.py)
        let v = from_map("3343|4410|1111|22··|····|·0··|····|····|····|···4|7787|8577|");
        let (ends, starts) = lines(&v);
        // the line ends on the second sixteenth of beat 2 (index 5), the
        // next begins on the last sixteenth of bar 3's beat 2 (index 16 + 16 + ... )
        assert_eq!(ends, vec![5]);
        assert_eq!(starts, vec![0, 39]);
    }

    #[test]
    fn a_quiet_sixteenth_after_silence_does_not_extend_the_line() {
        // "Mr. Saxobeat", bars 99-100: the line ends on beat 3 of bar 99
        let v = from_map("7674|6253|424·|3···|8999|5888|");
        let (ends, starts) = lines(&v);
        assert_eq!(ends, vec![10]);
        assert_eq!(starts, vec![0, 16]);
    }

    #[test]
    fn a_lone_quiet_sixteenth_is_not_a_line() {
        let v = from_map("8888|8888|····|··4·|····|····|8888|8888|");
        let (ends, starts) = lines(&v);
        assert_eq!(ends, vec![7]);
        assert_eq!(starts, vec![0, 24]);
    }
}
