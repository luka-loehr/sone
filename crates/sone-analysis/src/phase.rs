//! Is the one really the one? (RULES.md rule 3; `phasecheck.py`.) Beat
//! trackers put the bar line a beat off now and then; kicks still line up
//! then, so only the pattern inside the bar tells. Per position in the bar
//! (as the analysis counts it) three things are measured on every beat:
//!
//! - the backbeat band (drum stem, 1.5–6 kHz: claps and snares), level of the
//!   first half of the beat;
//! - the kick band (drum stem below 120 Hz), the same way;
//! - how much the harmony changes on the beat: 1 − cosine between the chroma
//!   of bass + other over the beat before and the beat after.
//!
//! The decision, in the analysis' positions 1–4 (0–3 here):
//!
//! 1. Parity, from the backbeat band. A two-and-four backbeat is clear when
//!    both of 2 and 4 are at least 1.5 dB above both of 1 and 3: the bar line
//!    has the right parity. When instead 1 and 3 are both 1.5 dB above 2 and
//!    4, the bar line is a beat off (−1 or +1). Anything else, including a
//!    song counted in half time (one snare per bar, on 3) and songs without a
//!    backbeat, leaves the parity untested: such songs are not "corrected"
//!    by a beat.
//! 2. Which of the two candidate ones, from the chords and the kick: the
//!    harmony change (normalised difference between the two candidates) and
//!    the kick band (difference in dB / 6, clamped to ±1), weighted 0.7 and
//!    0.3. A beat off: the candidate with more evidence wins. Right parity (or
//!    untested): the bar line moves by 2 only when the chords change at least
//!    twice as much on 3 as on 1 and the kick on 3 is not weaker by more than
//!    1 dB, because the tracker's downbeats are rarely wrong by half a bar
//!    and many songs change chords on 1 and 3.
//!
//! Confidence: `(1 − exp(−contrast / 3)) · (0.5 + 0.5·|evidence|)` with a
//! clear backbeat, where `contrast` is its margin in dB and `evidence` the
//! downbeat evidence of step 2 (clamped to 1); without one, a flat 0.25 ×
//! the second factor.

use crate::chroma::{Chroma, unit};
use crate::grid::Grid;
use crate::measure::{Lane, Measurements, SR};

/// Measurements per position in the bar (0 = the analysis' one).
#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub kick: [f64; 4],
    pub clap: [f64; 4],
    pub chords: [f64; 4],
    pub n: [usize; 4],
    pub n_chords: [usize; 4],
}

impl Profile {
    /// Add the ruler beats `ks` of `grid` (beat `k` at `grid.at(k)`; position
    /// `k mod 4`).
    pub fn add(&mut self, m: &Measurements, chroma: &Chroma, grid: &Grid, ks: impl Iterator<Item = i64>) {
        let step = grid.period;
        for k in ks {
            let t = grid.at(k as f64);
            let (i0, i1) = ((t * SR).floor(), ((t + step * 0.5) * SR).floor());
            if i0 < 0.0 || i1 > m.len as f64 {
                continue;
            }
            let (i0, i1) = (i0 as usize, i1 as usize);
            let db = |lane| 10.0 * (m.mean_square(lane, i0, i1).unwrap_or(0.0) + 1e-10).log10();
            let p = k.rem_euclid(4) as usize;
            self.kick[p] += db(Lane::Kick120);
            self.clap[p] += db(Lane::Clap);
            self.n[p] += 1;
            if let (Some(c0), Some(c1)) = (chroma.mean(t - step, t), chroma.mean(t, t + step)) {
                let (c0, c1) = (unit(c0), unit(c1));
                self.chords[p] += 1.0 - c0.iter().zip(&c1).map(|(a, b)| a * b).sum::<f64>();
                self.n_chords[p] += 1;
            }
        }
    }

    /// Means per position.
    pub fn means(&self) -> ([f64; 4], [f64; 4], [f64; 4]) {
        let mean = |v: &[f64; 4], n: &[usize; 4]| -> [f64; 4] { std::array::from_fn(|p| if n[p] > 0 { v[p] / n[p] as f64 } else { f64::NAN }) };
        (mean(&self.kick, &self.n), mean(&self.clap, &self.n), mean(&self.chords, &self.n_chords))
    }
}

/// The verdict of the check.
#[derive(Debug, Clone, PartialEq)]
pub struct Phase {
    /// Beats to move the bar line by: 0, −1, 1 or 2 (the new one is the old
    /// position `shift mod 4`).
    pub shift: i8,
    pub confidence: f32,
    /// How the backbeat reads: `Some(true)` on 2 and 4, `Some(false)` on 1
    /// and 3, `None` unclear (no backbeat, or half time).
    pub backbeat: Option<bool>,
    /// Margin of the clear pattern in dB (0 when unclear).
    pub contrast: f32,
    /// Downbeat evidence for the chosen one against its alternative, −1..1.
    pub evidence: f32,
}

/// Evidence (−1..1) that position `a` rather than `b` is the one.
fn downbeat_evidence(kick: &[f64; 4], chords: &[f64; 4], a: usize, b: usize) -> f64 {
    let (ha, hb) = (chords[a], chords[b]);
    let chord = if ha.is_finite() && hb.is_finite() && ha + hb > 1e-9 { (ha - hb) / (ha + hb) } else { 0.0 };
    let k = ((kick[a] - kick[b]) / 6.0).clamp(-1.0, 1.0);
    0.7 * chord + 0.3 * k
}

pub fn decide(profile: &Profile) -> Phase {
    let (kick, clap, chords) = profile.means();
    if profile.n.contains(&0) {
        return Phase { shift: 0, confidence: 0.0, backbeat: None, contrast: 0.0, evidence: 0.0 };
    }
    let even = clap[1].min(clap[3]) - clap[0].max(clap[2]);
    let odd = clap[0].min(clap[2]) - clap[1].max(clap[3]);
    let (backbeat, contrast) = if even >= 1.5 {
        (Some(true), even)
    } else if odd >= 1.5 {
        (Some(false), odd)
    } else {
        (None, 0.0)
    };
    let (shift, evidence) = if backbeat == Some(false) {
        // a beat off: the one is old position 3 (−1) or 1 (+1)
        let e = downbeat_evidence(&kick, &chords, 3, 1);
        if e >= 0.0 { (-1, e) } else { (1, -e) }
    } else {
        let e = downbeat_evidence(&kick, &chords, 0, 2);
        let flip = chords[2].is_finite() && chords[0].is_finite() && chords[2] > 2.0 * chords[0] && kick[2] >= kick[0] - 1.0;
        if flip { (2, -e) } else { (0, e) }
    };
    let second = 0.5 + 0.5 * evidence.clamp(-1.0, 1.0).abs();
    let confidence = match backbeat {
        Some(_) => (1.0 - (-contrast / 3.0).exp()) * second,
        None => 0.25 * second,
    };
    Phase { shift, confidence: confidence as f32, backbeat, contrast: contrast as f32, evidence: evidence as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(kick: [f64; 4], clap: [f64; 4], chords: [f64; 4]) -> Profile {
        Profile { kick, clap, chords, n: [8; 4], n_chords: [8; 4] }.scaled()
    }

    impl Profile {
        /// Sums of eight beats per position with these means.
        fn scaled(mut self) -> Self {
            for p in 0..4 {
                self.kick[p] *= 8.0;
                self.clap[p] *= 8.0;
                self.chords[p] *= 8.0;
            }
            self
        }
    }

    #[test]
    fn backbeat_on_two_and_four_keeps_the_bar_line() {
        let v = decide(&profile([-20.0; 4], [-45.0, -35.0, -45.0, -35.0], [0.2, 0.02, 0.1, 0.03]));
        assert_eq!((v.shift, v.backbeat), (0, Some(true)));
        assert!(v.confidence > 0.5, "{v:?}");
    }

    #[test]
    fn backbeat_on_one_and_three_moves_it_to_where_the_chords_change() {
        // "Blessings", bars 21-34, as the analysis counts them
        let v = decide(&profile([-19.0, -26.2, -23.7, -18.7], [-34.4, -46.2, -34.4, -41.1], [0.029, 0.032, 0.025, 0.061]));
        assert_eq!((v.shift, v.backbeat), (-1, Some(false)));
        let v = decide(&profile([-19.0, -18.7, -23.7, -26.2], [-34.4, -41.1, -34.4, -46.2], [0.029, 0.061, 0.025, 0.032]));
        assert_eq!(v.shift, 1);
    }

    #[test]
    fn half_time_and_no_backbeat_are_not_corrected_by_a_beat() {
        // one snare per bar, on 3: a song counted in half time
        let v = decide(&profile([-20.0, -40.0, -30.0, -40.0], [-50.0, -50.0, -30.0, -50.0], [0.2, 0.02, 0.05, 0.03]));
        assert_eq!((v.shift, v.backbeat), (0, None));
        // one snare per bar on 4 ("Sprinter"): unclear, left alone
        let v = decide(&profile([-53.9, -87.3, -79.2, -39.6], [-46.5, -67.6, -50.9, -33.8], [0.151, 0.022, 0.13, 0.084]));
        assert_eq!(v.shift, 0);
        assert!(v.confidence < 0.3);
    }

    #[test]
    fn the_bar_line_moves_by_two_only_on_clear_chords_and_kick() {
        let clap = [-45.0, -35.0, -45.0, -35.0];
        // chords change on 1 and 3 alike: stay
        assert_eq!(decide(&profile([-19.0, -19.0, -19.0, -25.5], clap, [0.140, 0.004, 0.164, 0.035])).shift, 0);
        // chords change on 3 only, the kick there as strong: move by 2
        assert_eq!(decide(&profile([-20.0, -30.0, -20.0, -30.0], clap, [0.03, 0.01, 0.2, 0.01])).shift, 2);
    }
}
