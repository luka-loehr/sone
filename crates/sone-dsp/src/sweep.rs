//! Filter sweeps: a 4th-order Butterworth high-pass or low-pass whose cutoff
//! follows a lane on the ruler.

use crate::SR;
use crate::filter::{Band, SosState, butter};
use crate::np::interp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepKind {
    Highpass,
    Lowpass,
}

/// The cutoff is re-designed every 64 samples (read at the middle of the block,
/// interpolated in log frequency) while the filter state runs on, so a moving
/// cutoff neither clicks nor restarts the filter.
pub fn sweep(x: &[[f32; 2]], points: &[[f64; 2]], u: &[f64], kind: SweepKind) -> Vec<[f32; 2]> {
    const B: usize = 64;
    let pu: Vec<f64> = points.iter().map(|p| p[0]).collect();
    let pf: Vec<f64> = points.iter().map(|p| p[1].ln()).collect();
    let mut out = vec![[0.0f32; 2]; x.len()];
    let mut st = [SosState::new(2), SosState::new(2)];
    let mut i = 0;
    while i < x.len() {
        let at = (i + B / 2).min(u.len() - 1);
        let f = interp(u[at], &pu, &pf).exp().clamp(15.0, 20000.0);
        let sos = butter(4, if kind == SweepKind::Highpass { Band::Highpass(f) } else { Band::Lowpass(f) }, SR);
        let end = (i + B).min(x.len());
        for j in i..end {
            for (c, s) in st.iter_mut().enumerate() {
                out[j][c] = sos.tick(s, f64::from(x[j][c])) as f32;
            }
        }
        i = end;
    }
    out
}
