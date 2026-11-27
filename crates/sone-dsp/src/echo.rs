//! Tempo echo (wet only).

use crate::SR;
use crate::filter::{Band, butter};

/// Each repeat is the previous one band-passed once more and `feedback` quieter,
/// and repeats alternate sides (ping-pong, the far side at 0.8).
pub fn echo(x: &[[f32; 2]], delay_s: f64, feedback: f64, repeats: usize, hp: f64, lp: f64) -> Vec<[f32; 2]> {
    let n = x.len();
    let d = crate::np::iround(delay_s * SR).max(0) as usize;
    let mut wet = vec![[0.0f32; 2]; n];
    // An IIR filter from rest stays exactly zero until its input starts, so the
    // silent lead-in of the send can be skipped.
    let Some(first) = x.iter().position(|v| v[0] != 0.0 || v[1] != 0.0) else {
        return wet;
    };
    let sos = butter(2, Band::Bandpass(hp, lp), SR);
    let mut cur: [Vec<f64>; 2] = [
        x[first..].iter().map(|v| f64::from(v[0])).collect(),
        x[first..].iter().map(|v| f64::from(v[1])).collect(),
    ];
    for k in 1..=repeats {
        for ch in cur.iter_mut() {
            let mut st = crate::filter::SosState::new(sos.sections());
            sos.filter_in_place(ch, &mut st);
        }
        let amp = feedback.powi(k as i32 - 1);
        if k * d >= n {
            break;
        }
        // Source sample i (absolute) lands at i + k*d; only i < n - k*d fits.
        let limit = n - k * d;
        if first >= limit {
            continue;
        }
        let (l, r) = if k % 2 == 0 { (1, 0) } else { (0, 1) };
        for i in first..limit {
            let j = i - first;
            let o = &mut wet[i + k * d];
            o[0] = (f64::from(o[0]) + cur[l][j] * amp * 1.0) as f32;
            o[1] = (f64::from(o[1]) + cur[r][j] * amp * 0.8) as f32;
        }
    }
    wet
}
