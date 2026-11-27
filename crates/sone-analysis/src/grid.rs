//! The local constant-tempo grid of `reference/python/handmix.py`
//! (`Track.grid`): a line through the tracker's beats around a bar (beats off
//! the line dropped), its phase moved onto the drum stem's attacks.

use crate::measure::{HOP, Lane, Measurements, SR};
use crate::stats::lstsq_line;
use crate::structure::Structure;

/// Attack envelope of the drum stem above 1.5 kHz at 32-sample hops
/// (handmix.py's `onset_env`): the rise of the log energy, smoothed over six
/// hops. There a hit is a sharp click; the low end of a kick swings for 20 ms
/// and would smear the timing.
pub fn attack_envelope(m: &Measurements) -> Vec<f32> {
    let n = m.len / HOP;
    let e: Vec<f64> = (0..n).map(|h| m.hop_energy(Lane::Attack, h) / HOP as f64).collect();
    // scipy uniform_filter1d(size=6, mode="reflect"): e[i-3..=i+2]
    let at = |i: i64| -> f64 {
        let n = n as i64;
        let j = if i < 0 { -i - 1 } else if i >= n { 2 * n - i - 1 } else { i };
        e[j.clamp(0, n - 1) as usize]
    };
    let l: Vec<f64> = (0..n as i64).map(|i| ((i - 3..=i + 2).map(at).sum::<f64>() / 6.0 + 1e-6).log10()).collect();
    (0..n).map(|i| if i == 0 { 0.0 } else { (l[i] - l[i - 1]).max(0.0) as f32 }).collect()
}

/// A local grid: beat `k` (counted from the anchor bar's downbeat) at
/// `t0 + k * period`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub t0: f64,
    pub period: f64,
    /// How far the drum attacks moved the line's phase, seconds.
    pub nudge: f64,
}

impl Grid {
    pub fn at(&self, k: f64) -> f64 {
        self.t0 + k * self.period
    }
}

/// handmix.py's `grid(bar, span_bars, beat_shift, fit_bars)`.
pub fn grid(s: &Structure, env: &[f32], bar: usize, span_bars: i64, beat_shift: i64, fit_bars: Option<(usize, usize)>) -> Grid {
    let beats = &s.json.beats;
    let idx = &s.idx;
    let down = &s.down;
    let k0 = idx[down[bar.min(down.len() - 1)]] + beat_shift;
    let sel: Vec<usize> = match fit_bars {
        Some((lo, hi)) => {
            let (a, b) = (idx[down[lo.min(down.len() - 1)]], idx[down[hi.min(down.len() - 1)]]);
            (0..beats.len()).filter(|&i| idx[i] >= a && idx[i] <= b).collect()
        }
        None => (0..beats.len()).filter(|&i| (idx[i] - k0).abs() <= span_bars * 4).collect(),
    };
    let mut b: Vec<f64> = sel.iter().map(|&i| beats[i]).collect();
    let mut k: Vec<f64> = sel.iter().map(|&i| (idx[i] - k0) as f64).collect();
    // drop beats off the line (fills, breaks)
    let (mut period, mut t0) = (s.beat_s, beats[down[bar.min(down.len() - 1)]]);
    for _ in 0..3 {
        if b.len() < 2 {
            break;
        }
        (period, t0) = lstsq_line(&k, &b);
        let ok: Vec<bool> = k.iter().zip(&b).map(|(&kk, &bb)| (bb - (period * kk + t0)).abs() < 0.03).collect();
        let n_ok = ok.iter().filter(|&&o| o).count();
        if n_ok == ok.len() || n_ok < 8 {
            break;
        }
        let keep = |v: &[f64]| v.iter().zip(&ok).filter(|p| *p.1).map(|p| *p.0).collect::<Vec<f64>>();
        b = keep(&b);
        k = keep(&k);
    }
    let ks: Vec<i64> = match fit_bars {
        Some(_) if !k.is_empty() => {
            let lo = k.iter().copied().fold(f64::INFINITY, f64::min) as i64;
            let hi = k.iter().copied().fold(f64::NEG_INFINITY, f64::max) as i64;
            (lo..=hi).collect()
        }
        _ => (-span_bars * 4..span_bars * 4).collect(),
    };
    let mut best = (-1.0f64, 0.0f64);
    for step in 0..=180 {
        let d = -0.045 + step as f64 * 0.0005;
        let v: f64 = ks
            .iter()
            .filter_map(|&kk| {
                let i = ((t0 + kk as f64 * period + d) * SR / HOP as f64).round_ties_even() as i64;
                (i >= 0 && (i as usize) < env.len()).then(|| f64::from(env[i as usize]))
            })
            .sum();
        if v > best.0 {
            best = (v, d);
        }
    }
    Grid { t0: t0 + best.1, period, nudge: best.1 }
}
