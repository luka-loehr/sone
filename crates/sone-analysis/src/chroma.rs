//! Pitch-class profiles per chroma frame, like librosa's `chroma_cqt`: the
//! tuned profile of each frame scaled so its strongest class is 1.

use crate::measure::{C_HOP, Measurements, SR};

pub struct Chroma {
    /// Per frame (centred on sample `j * C_HOP`), 12 values, the maximum 1
    /// (all 0 for a silent frame).
    pub frames: Vec<[f32; 12]>,
    /// Frame energy (the sum before scaling), to tell silence.
    pub weight: Vec<f32>,
}

impl Chroma {
    pub fn new(m: &Measurements, fine: &[f32], tuning_cents: f64) -> Self {
        let n = fine.len() / crate::measure::FINE;
        let mut frames = Vec::with_capacity(n);
        let mut weight = Vec::with_capacity(n);
        let mut c = [0.0; 12];
        for j in 0..n {
            m.chroma_frame(fine, j, tuning_cents, &mut c);
            let max = c.iter().copied().fold(0.0, f64::max);
            let sum: f64 = c.iter().sum();
            weight.push(sum as f32);
            frames.push(if max > 1e-9 { c.map(|v| (v / max) as f32) } else { [0.0; 12] });
        }
        Self { frames, weight }
    }

    /// Frames whose centre lies in `t0..t1` seconds.
    pub fn range(&self, t0: f64, t1: f64) -> std::ops::Range<usize> {
        let a = (t0 * SR / C_HOP as f64).ceil().max(0.0) as usize;
        let b = ((t1 * SR / C_HOP as f64).ceil().max(0.0) as usize).min(self.frames.len());
        a.min(b)..b
    }

    /// Mean of the scaled frames in `t0..t1`, or `None` without frames.
    pub fn mean(&self, t0: f64, t1: f64) -> Option<[f64; 12]> {
        let r = self.range(t0, t1);
        if r.is_empty() {
            return None;
        }
        let mut out = [0.0; 12];
        for f in &self.frames[r.clone()] {
            for (o, &v) in out.iter_mut().zip(f) {
                *o += f64::from(v);
            }
        }
        Some(out.map(|v| v / r.len() as f64))
    }

    /// Sum of the raw (unscaled) profile in `t0..t1`: the weight of each
    /// pitch class in that stretch.
    pub fn energy(&self, t0: f64, t1: f64) -> [f64; 12] {
        let mut out = [0.0; 12];
        for j in self.range(t0, t1) {
            let max = self.frames[j].iter().copied().fold(0.0f32, f32::max);
            if max <= 0.0 {
                continue;
            }
            // undo the scaling: raw = scaled * raw_max, raw_max = weight / sum(scaled)
            let scaled_sum: f32 = self.frames[j].iter().sum();
            let raw_max = f64::from(self.weight[j] / scaled_sum);
            for (o, &v) in out.iter_mut().zip(&self.frames[j]) {
                *o += f64::from(v) * raw_max;
            }
        }
        out
    }
}

/// `v` scaled to unit length.
pub fn unit(v: [f64; 12]) -> [f64; 12] {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.map(|x| x / (n + 1e-9))
}
