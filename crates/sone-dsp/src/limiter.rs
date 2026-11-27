//! Look-ahead peak limiter of the master bus.

use crate::SR;
use crate::np::{Edge, minimum_filter1d, uniform_filter1d};

/// The gain is down 5 ms before a peak arrives and held 60 ms after it, then
/// smoothed over 5 ms. Returns the limited signal and the smallest gain applied.
pub fn limiter(x: &[[f32; 2]], ceiling: f32) -> (Vec<[f32; 2]>, f64) {
    const LOOK_MS: f64 = 5.0;
    const HOLD_MS: f64 = 60.0;
    let need: Vec<f32> = x
        .iter()
        .map(|v| {
            let peak = v[0].abs().max(v[1].abs());
            (ceiling / peak.max(1e-9)).min(1.0)
        })
        .collect();
    if need.iter().all(|&g| g >= 1.0) {
        return (x.to_vec(), 1.0);
    }
    let k = (LOOK_MS * SR / 1000.0) as usize;
    let h = (HOLD_MS * SR / 1000.0) as usize;
    let g = minimum_filter1d(&need, 2 * k + 1, Edge::Reflect, 0);
    let g = minimum_filter1d(&g, h, Edge::Reflect, -((h / 2) as isize) + 1);
    let g64: Vec<f64> = g.iter().map(|&v| f64::from(v)).collect();
    let g: Vec<f32> = uniform_filter1d(&g64, k, Edge::Nearest).into_iter().map(|v| (v as f32).min(1.0)).collect();
    let min = g.iter().copied().fold(f32::INFINITY, f32::min);
    let y = x.iter().zip(&g).map(|(v, &g)| [v[0] * g, v[1] * g]).collect();
    (y, f64::from(min))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_signal_passes_untouched() {
        let x = vec![[0.5f32, -0.5]; 1000];
        let (y, g) = limiter(&x, 0.89);
        assert_eq!(g, 1.0);
        assert_eq!(x, y);
    }

    #[test]
    fn peak_is_held_under_the_ceiling_ahead_of_time() {
        let mut x = vec![[0.5f32, 0.5]; 20000];
        x[10000] = [1.5, -1.2];
        let (y, g) = limiter(&x, 0.89);
        assert!(y.iter().all(|v| v[0].abs() <= 0.8901 && v[1].abs() <= 0.8901));
        assert!((g - 0.89 / 1.5).abs() < 1e-3);
        // Already fully down 3 ms before the peak, and back up well after the hold.
        assert!((y[10000 - 130][0] / 0.5 - g as f32).abs() < 1e-3);
        assert_eq!(y[19999][0], 0.5);
    }
}
