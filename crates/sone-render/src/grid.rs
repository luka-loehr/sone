//! The local beat grid of a song around one bar (`Track.grid` of the reference).

use sone_core::Stem;
use sone_dsp::np::pairwise_sum_f64;
use sone_dsp::{SR, onset_env};

use crate::source::StemSource;

/// Where a bar's downbeat is and how long a beat lasts there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    /// Time of the bar's downbeat, seconds.
    pub t: f64,
    /// Beat period, seconds.
    pub beat: f64,
    /// Phase correction found on the drum stem's onsets, seconds.
    pub shift: f64,
}

/// Least-squares line b = T k + t0 (what `np.polyfit(k, b, 1)` computes).
fn polyfit1(k: &[f64], b: &[f64]) -> (f64, f64) {
    let n = k.len() as f64;
    let mk = k.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (x, y) in k.iter().zip(b) {
        sxy += (x - mk) * (y - mb);
        sxx += (x - mk) * (x - mk);
    }
    let t = sxy / sxx;
    (t, mb - t * mk)
}

/// Mono drum stem (`drums.mean(1)` in float32) over [lo, hi).
fn mono_drums(src: &dyn StemSource, lo: usize, hi: usize) -> Vec<f32> {
    let mut buf = vec![[0.0f32; 2]; hi.saturating_sub(lo)];
    src.read(Stem::Drums, lo as i64, &mut buf);
    buf.iter().map(|v| (v[0] + v[1]) / 2.0).collect()
}

/// Local constant-tempo grid around `bar`: a least-squares line through the beats
/// within `span_bars` (or only those of `fit_bars`), dropping beats more than 30 ms
/// off (fills, breaks) for up to two rounds, then the phase moved in 0.5 ms steps
/// within ±45 ms to where the drum stem's attacks sit best. `beat_shift` moves the
/// bar line by whole beats. A `fit_bars` window with fewer than 8 beats is ignored
/// (a line through so few beats is no tempo); an error for a bar the song does not
/// have or a grid that comes out not finite.
pub fn grid(
    src: &dyn StemSource,
    bar: usize,
    span_bars: i64,
    beat_shift: i32,
    fit_bars: Option<[usize; 2]>,
) -> anyhow::Result<Grid> {
    let g = src.beats();
    let down = g.downbeats();
    let idx = &g.beat_idx;
    let Some(&anchor) = down.get(bar) else {
        anyhow::bail!("{}: bar {bar} does not exist ({} bars)", src.id(), down.len());
    };
    let k0 = idx[anchor] + i64::from(beat_shift);
    let around = || -> Vec<usize> { (0..idx.len()).filter(|&i| (idx[i] - k0).abs() <= span_bars * 4).collect() };
    let window = match fit_bars {
        Some([f0, f1]) if f0 < down.len() => {
            let lo = idx[down[f0]];
            let hi = idx[down[f1.min(down.len() - 1)]];
            let sel: Vec<usize> = (0..idx.len()).filter(|&i| idx[i] >= lo && idx[i] <= hi).collect();
            (sel.len() >= 8).then_some(sel)
        }
        _ => None,
    };
    let fit_bars = if window.is_some() { fit_bars } else { None };
    let sel = window.unwrap_or_else(around);
    if sel.len() < 2 {
        anyhow::bail!("{}: too few beats around bar {bar} for a grid", src.id());
    }
    let mut b: Vec<f64> = sel.iter().map(|&i| g.beats[i]).collect();
    let mut k: Vec<f64> = sel.iter().map(|&i| (idx[i] - k0) as f64).collect();
    let (mut t_beat, mut t0) = (0.0, 0.0);
    for _ in 0..3 {
        (t_beat, t0) = polyfit1(&k, &b);
        let ok: Vec<bool> = k.iter().zip(&b).map(|(x, y)| (y - (t_beat * x + t0)).abs() < 0.03).collect();
        let n_ok = ok.iter().filter(|&&v| v).count();
        if n_ok == ok.len() || n_ok < 8 {
            break;
        }
        b = b.iter().zip(&ok).filter(|(_, o)| **o).map(|(v, _)| *v).collect();
        k = k.iter().zip(&ok).filter(|(_, o)| **o).map(|(v, _)| *v).collect();
    }
    let n = src.len() as f64;
    let (lo, hi, ks): (usize, usize, Vec<i64>) = if fit_bars.is_some() {
        let bmin = b.iter().copied().fold(f64::INFINITY, f64::min);
        let bmax = b.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let kmin = k.iter().copied().fold(f64::INFINITY, f64::min) as i64;
        let kmax = k.iter().copied().fold(f64::NEG_INFINITY, f64::max) as i64;
        (((bmin - 0.1).max(0.0) * SR) as usize, ((bmax + 0.5) * SR).min(n) as usize, (kmin..=kmax).collect())
    } else {
        let s = (span_bars * 4) as f64;
        (
            ((t0 - s * t_beat).max(0.0) * SR) as usize,
            ((t0 + s * t_beat) * SR).min(n) as usize,
            (-span_bars * 4..span_bars * 4).collect(),
        )
    };
    let hop = 32usize;
    let env = onset_env(&mono_drums(src, lo, hi), hop);
    // np.arange(-0.045, 0.0451, 0.0005): numpy fills start + i * (second - first).
    let start = -0.045f64;
    let delta = (start + 0.0005) - start;
    let count = ((0.0451 - start) / 0.0005).ceil() as usize;
    let (mut best, mut bd) = (-1.0f64, 0.0f64);
    let mut vals = Vec::with_capacity(ks.len());
    for i in 0..count {
        let d = match i {
            0 => start,
            1 => start + 0.0005,
            _ => start + i as f64 * delta,
        };
        vals.clear();
        for &kk in &ks {
            let pos = (((t0 + kk as f64 * t_beat) + d) * SR - lo as f64) / hop as f64;
            let j = pos.round_ties_even() as i64;
            if j >= 0 && (j as usize) < env.len() {
                vals.push(env[j as usize]);
            }
        }
        let v = pairwise_sum_f64(&vals);
        if v > best {
            best = v;
            bd = d;
        }
    }
    if !(t0.is_finite() && t_beat.is_finite() && t_beat > 0.0) {
        anyhow::bail!("{}: no usable grid around bar {bar} (beat period {t_beat})", src.id());
    }
    Ok(Grid { t: t0 + bd, beat: t_beat, shift: bd })
}
