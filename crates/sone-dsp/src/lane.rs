//! Gain lanes: breakpoints on the bar ruler turned into a per-sample gain.

use sone_core::Shape;

use crate::SR;
use crate::np::{Edge, interp, segment_index, uniform_filter1d};

fn smoothstep(x: f64) -> f64 {
    x * x * (3.0 - 2.0 * x)
}

/// Gain at every ruler position `u` from breakpoints `[u, amplitude]`, then a
/// moving average of `smooth_ms` so a step never clicks.
/// Returns `None` for an empty lane, like the reference.
pub fn lane(points: &[[f64; 2]], u: &[f64], smooth_ms: f64, shape: Shape) -> Option<Vec<f32>> {
    if points.is_empty() {
        return None;
    }
    let pu: Vec<f64> = points.iter().map(|p| p[0]).collect();
    let pv: Vec<f64> = points.iter().map(|p| p[1]).collect();
    let g: Vec<f64> = match shape {
        Shape::Lin => u.iter().map(|&x| interp(x, &pu, &pv)).collect(),
        Shape::Db => {
            // Straight lines in dB, silence is -70 dB, and the last 4 dB above it are silence too.
            let floor = 10f64.powf(-70.0 / 20.0);
            let pd: Vec<f64> = pv.iter().map(|&v| 20.0 * v.max(floor).log10()).collect();
            let on: Vec<f64> = pv.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
            let cut = 10f64.powf(-66.0 / 20.0);
            u.iter()
                .map(|&x| {
                    let mut g = 10f64.powf(interp(x, &pu, &pd) / 20.0);
                    if interp(x, &pu, &on) <= 0.0 {
                        g = 0.0;
                    }
                    if g < cut { 0.0 } else { g }
                })
                .collect()
        }
        Shape::S => {
            let n = pu.len();
            u.iter()
                .map(|&x| {
                    if x <= pu[0] {
                        return pv[0];
                    }
                    if x >= pu[n - 1] {
                        return pv[n - 1];
                    }
                    let seg = segment_index(&pu, x).clamp(0, n as isize - 2) as usize;
                    let w = ((x - pu[seg]) / (pu[seg + 1] - pu[seg]).max(1e-9)).clamp(0.0, 1.0);
                    pv[seg] + (pv[seg + 1] - pv[seg]) * smoothstep(w)
                })
                .collect()
        }
    };
    let k = ((smooth_ms * SR / 1000.0) as usize).max(1);
    Some(uniform_filter1d(&g, k, Edge::Nearest).into_iter().map(|v| v as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ruler(n: usize, from: f64, per_sample: f64) -> Vec<f64> {
        (0..n).map(|i| from + i as f64 * per_sample).collect()
    }

    #[test]
    fn linear_fade_and_ends() {
        // One bar = 88200 samples; fade 1 → 0 over bar 1.
        let u = ruler(4 * 88200, 0.0, 1.0 / 88200.0);
        let g = lane(&[[1.0, 1.0], [2.0, 0.0]], &u, 6.0, Shape::Lin).unwrap();
        assert_eq!(g[0], 1.0);
        assert!((g[88200 + 44100] - 0.5).abs() < 1e-4);
        assert!(g.last().unwrap().abs() < 1e-9);
        // The 6 ms average only touches the corners.
        assert!((g[88200 - 200] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn smoothstep_is_symmetric() {
        let u = ruler(88200, 0.0, 1.0 / 88200.0);
        let g = lane(&[[0.0, 0.0], [1.0, 1.0]], &u, 6.0, Shape::S).unwrap();
        assert!((g[44100] - 0.5).abs() < 1e-4);
        assert!(g[8820] < 0.1 * 0.5); // slower start than a straight line
        let s = |i: usize| g[i] + g[88200 - i];
        assert!((s(20000) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn db_shape_reaches_silence() {
        let u = ruler(88200, 0.0, 1.0 / 88200.0);
        let g = lane(&[[0.0, 1.0], [1.0, 0.0]], &u, 6.0, Shape::Db).unwrap();
        // Halfway in dB between 0 and -70 dB is -35 dB.
        assert!((20.0 * f64::from(g[44100]).log10() + 35.0).abs() < 0.05);
        // Below -66 dB it is cut to zero.
        assert!(g[88200 - 300].abs() < 1e-9);
    }

    #[test]
    fn step_is_smoothed_over_6_ms() {
        let u = ruler(10000, 0.0, 1.0 / 10000.0);
        let g = lane(&[[0.5, 0.0], [0.5000001, 1.0]], &u, 6.0, Shape::Lin).unwrap();
        let rising = g.iter().filter(|&&v| v > 0.001 && v < 0.999).count();
        assert!((260..=266).contains(&rising), "{rising}");
    }

    #[test]
    fn empty_lane_is_none() {
        assert!(lane(&[], &[0.0], 6.0, Shape::Lin).is_none());
    }
}
