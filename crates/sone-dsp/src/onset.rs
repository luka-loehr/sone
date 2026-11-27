//! Drum attack envelope and the beat lag between two drum signals.

use std::sync::OnceLock;

use crate::SR;
use crate::filter::{Band, Sos, butter};
use crate::np::{Edge, pairwise_sum_f64, uniform_filter1d};

fn hp1500() -> &'static Sos {
    static HP: OnceLock<Sos> = OnceLock::new();
    HP.get_or_init(|| butter(4, Band::Highpass(1500.0), SR))
}

/// Attack envelope of a mono drum signal, one value per `hop` samples. Only the
/// band above 1.5 kHz counts: there a hit is a sharp click, while the low end of
/// a kick swings for 20 ms and would smear the timing.
pub fn onset_env(x: &[f32], hop: usize) -> Vec<f64> {
    let y = hp1500().filter_f32(x);
    let n = y.len() / hop;
    let e: Vec<f64> = (0..n)
        .map(|i| {
            let sq: Vec<f64> = y[i * hop..(i + 1) * hop].iter().map(|v| v * v).collect();
            pairwise_sum_f64(&sq) / hop as f64
        })
        .collect();
    let e = uniform_filter1d(&e, 6, Edge::Reflect);
    let l: Vec<f64> = e.iter().map(|v| (v + 1e-6).log10()).collect();
    (0..n).map(|i| if i == 0 { 0.0 } else { (l[i] - l[i - 1]).max(0.0) }).collect()
}

/// Lag between two drum signals from their onset envelopes, searched over ±`max_ms`.
/// Returns (lag in ms, sharpness). Positive lag: `b` is early (A's events come later).
/// Sharpness is the peak over the mean absolute correlation; below about 2 the
/// lag means little.
pub fn xcorr_lag(a: &[f32], b: &[f32], max_ms: f64, hop: usize) -> (f64, f64) {
    let ea = onset_env(a, hop);
    let eb = onset_env(b, hop);
    let n = ea.len().min(eb.len());
    let ma = pairwise_sum_f64(&ea[..n]) / n as f64;
    let mb = pairwise_sum_f64(&eb[..n]) / n as f64;
    let ea: Vec<f64> = ea[..n].iter().map(|v| v - ma).collect();
    let eb: Vec<f64> = eb[..n].iter().map(|v| v - mb).collect();
    let m = (max_ms / 1000.0 * SR / hop as f64) as isize;
    let mut best = -1e9;
    let mut bl = 0isize;
    let mut vals = Vec::with_capacity((2 * m + 1) as usize);
    let mut prod = Vec::with_capacity(n);
    for l in -m..=m {
        prod.clear();
        if l >= 0 {
            let l = l as usize;
            if l <= n {
                prod.extend(ea[l..].iter().zip(&eb[..n - l]).map(|(x, y)| x * y));
            }
        } else {
            let k = (-l) as usize;
            if k <= n {
                prod.extend(ea[..n - k].iter().zip(&eb[k..]).map(|(x, y)| x * y));
            }
        }
        let v = pairwise_sum_f64(&prod);
        vals.push(v);
        if v > best {
            best = v;
            bl = l;
        }
    }
    let absmean = pairwise_sum_f64(&vals.iter().map(|v| v.abs()).collect::<Vec<_>>()) / vals.len() as f64;
    let sharp = best / (absmean + 1e-9);
    (bl as f64 * hop as f64 / SR * 1000.0, sharp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clicks(n: usize, every: usize, offset: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; n];
        let mut i = offset;
        while i + 200 < n {
            for k in 0..200 {
                // A decaying noise burst: bright attack.
                x[i + k] = (((k * 7919 + i) % 97) as f32 / 48.0 - 1.0) * (-(k as f32) / 40.0).exp();
            }
            i += every;
        }
        x
    }

    #[test]
    fn lag_of_shifted_clicks() {
        let a = clicks(44100 * 4, 11025, 1000 + 320);
        let b = clicks(44100 * 4, 11025, 1000);
        let (lag, sharp) = xcorr_lag(&a, &b, 40.0, 32);
        assert!((lag - 320.0 / 44.1).abs() < 0.8, "{lag}");
        assert!(sharp > 2.0);
    }
}
