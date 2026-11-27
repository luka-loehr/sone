//! Attack detector of the stretcher.

use std::collections::VecDeque;

use crate::SR;
use crate::filter::{Band, Sos, SosState, butter};

/// Hop of the detector (1.45 ms).
pub(crate) const HOP: usize = 64;
/// Peak picking window (hops each side, 17 ms).
pub(crate) const PEAK_HOPS: usize = 12;
/// Reference window of the rise: hops 4 to 14 before the current one (6 to 20 ms).
const REF_FROM: usize = 14;
const REF_TO: usize = 4;
/// Thresholds (dB of rise summed over the bands): a peak counts as an attack above
/// WEAK_DB and WEAK_SHARE of the song's recent big hits (the 90th percentile of its
/// peaks over the last ~6 s), as a strong one above STRONG_DB and STRONG_SHARE.
const PEAK_MIN_DB: f64 = 6.0;
const WEAK_DB: f64 = 8.0;
const WEAK_SHARE: f64 = 0.3;
const STRONG_DB: f64 = 12.0;
const STRONG_SHARE: f64 = 0.5;
const PAST_PEAKS: usize = 120;

/// An attack found in the input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Onset {
    /// Input position (samples) of the attack's start, give or take a hop.
    pub pos: i64,
    /// Detection value (dB of rise summed over the bands).
    pub strength: f64,
    /// A hit whose timing matters most (kick, snare, clap).
    pub strong: bool,
}

/// Per hop the energy in three bands (below 200 Hz, 200 Hz to 3 kHz, above 3 kHz);
/// an attack is a rise of the last 3 ms over the 6 to 20 ms before, in dB, summed
/// over the bands. A band far below the music's level cannot rise by much (a floor
/// 30 dB under the total level), so a quiet shaker does not count like a snare.
/// Peaks are judged against the size of the song's recent big hits, so dense
/// hi-hats and textures do not flood the stretcher with protected zones.
pub(super) struct Detector {
    bands: [Sos; 3],
    state: [[SosState; 2]; 3],
    acc: [f64; 3],
    fill: usize,
    e: VecDeque<[f64; 3]>,
    /// Slow average energy of all bands together.
    level: f64,
    /// Detection values not yet peak-picked, with their hop index.
    odf: VecDeque<(i64, f64)>,
    /// Recent peak values, and the same sorted.
    past: VecDeque<f64>,
    sorted: Vec<f64>,
    hop_index: i64,
}

impl Detector {
    pub(super) fn new() -> Self {
        let mk = || [SosState::new(1), SosState::new(1)];
        Detector {
            bands: [
                butter(2, Band::Lowpass(200.0), SR),
                butter(1, Band::Bandpass(200.0, 3000.0), SR),
                butter(2, Band::Highpass(3000.0), SR),
            ],
            state: [mk(), mk(), mk()],
            acc: [0.0; 3],
            fill: 0,
            e: VecDeque::new(),
            level: 0.0,
            odf: VecDeque::new(),
            past: VecDeque::new(),
            sorted: Vec::new(),
            hop_index: 0,
        }
    }

    /// Feeds samples; appends the onsets it is sure of.
    pub(super) fn push(&mut self, x: &[[f32; 2]], out: &mut Vec<Onset>) {
        for v in x {
            for b in 0..3 {
                let mut e = 0.0;
                for (st, &x) in self.state[b].iter_mut().zip(v) {
                    let y = self.bands[b].tick(st, f64::from(x));
                    e += y * y;
                }
                self.acc[b] += e;
            }
            self.fill += 1;
            if self.fill == HOP {
                self.end_hop(out);
            }
        }
    }

    /// Pushes enough silence to report the last onsets.
    pub(super) fn finish(&mut self, out: &mut Vec<Onset>) {
        self.push(&vec![[0.0f32; 2]; HOP * (2 * PEAK_HOPS + 3)], out);
    }

    fn end_hop(&mut self, out: &mut Vec<Onset>) {
        let e: [f64; 3] = std::array::from_fn(|b| self.acc[b] / HOP as f64);
        self.acc = [0.0; 3];
        self.fill = 0;
        let total: f64 = e.iter().sum();
        // About 1.5 s to follow the music's level; starts at the first hop's level.
        self.level = if self.hop_index == 0 { total } else { self.level + (total - self.level) / 1000.0 };
        self.e.push_back(e);
        if self.e.len() > REF_FROM + 1 {
            self.e.pop_front();
        }
        let mut d = 0.0;
        if self.e.len() == REF_FROM + 1 {
            let n = self.e.len();
            let floor = 1e-3 * self.level + 1e-12;
            for b in 0..3 {
                let fast = (self.e[n - 1][b] + self.e[n - 2][b]) / 2.0;
                let refe = (n - 1 - REF_FROM..=n - 1 - REF_TO).map(|k| self.e[k][b]).sum::<f64>()
                    / (REF_FROM - REF_TO + 1) as f64;
                d += (10.0 * ((fast + floor) / (refe + floor)).log10()).max(0.0);
            }
        }
        self.odf.push_back((self.hop_index, d));
        self.hop_index += 1;
        // Peak-pick the value PEAK_HOPS back once its right neighbours exist.
        while self.odf.len() > 2 * PEAK_HOPS {
            let (i, v) = self.odf[PEAK_HOPS];
            let is_peak = v >= PEAK_MIN_DB
                && self.odf.iter().enumerate().all(|(k, &(_, w))| if k < PEAK_HOPS { w < v } else { w <= v });
            if is_peak {
                self.past.push_back(v);
                let at = self.sorted.partition_point(|&x| x < v);
                self.sorted.insert(at, v);
                if self.past.len() > PAST_PEAKS {
                    let old = self.past.pop_front().unwrap();
                    let k = self.sorted.partition_point(|&x| x < old);
                    self.sorted.remove(k);
                }
                let hits = self.sorted[(self.sorted.len() - 1) * 9 / 10];
                if v >= WEAK_DB.max(WEAK_SHARE * hits) {
                    // The rise is complete in hop i; the attack started about a hop earlier.
                    out.push(Onset {
                        pos: (i - 1).max(0) * HOP as i64,
                        strength: v,
                        strong: v >= STRONG_DB.max(STRONG_SHARE * hits),
                    });
                }
            }
            self.odf.pop_front();
        }
    }

    /// Input position up to which every onset has been reported.
    pub(super) fn done_until(&self) -> i64 {
        (self.hop_index - 2 * PEAK_HOPS as i64 - 1).max(0) * HOP as i64
    }
}

/// The attacks the stretcher sees in a signal (for diagnostics).
pub fn detect_onsets(x: &[[f32; 2]]) -> Vec<Onset> {
    let mut d = Detector::new();
    let mut out = Vec::new();
    d.push(x, &mut out);
    d.finish(&mut out);
    out
}
