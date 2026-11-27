//! Fourth-order Butterworth filters as scipy's `butter(4, …, output="sos")`
//! and `sosfilt` (zero initial state), so band levels match the Python
//! references. Designed through the analog prototype and the bilinear
//! transform with prewarping; run in f64.

use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
struct C(f64, f64);

impl C {
    fn add(self, o: C) -> C {
        C(self.0 + o.0, self.1 + o.1)
    }
    fn sub(self, o: C) -> C {
        C(self.0 - o.0, self.1 - o.1)
    }
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn div(self, o: C) -> C {
        let d = o.0 * o.0 + o.1 * o.1;
        C((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
    fn scale(self, k: f64) -> C {
        C(self.0 * k, self.1 * k)
    }
    fn sqrt(self) -> C {
        let r = (self.0 * self.0 + self.1 * self.1).sqrt();
        let re = ((r + self.0) / 2.0).max(0.0).sqrt();
        let im = ((r - self.0) / 2.0).max(0.0).sqrt();
        C(re, if self.1 < 0.0 { -im } else { im })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Band {
    Lowpass(f64),
    Highpass(f64),
    Bandpass(f64, f64),
}

/// A cascade of biquads (b0, b1, b2, a1, a2) with their states.
#[derive(Debug, Clone)]
pub struct Sos {
    sections: Vec<[f64; 5]>,
    state: Vec<[f64; 2]>,
}

const ORDER: usize = 4;

impl Sos {
    pub fn butter4(band: Band, fs: f64) -> Self {
        let fs2 = 2.0 * fs;
        let warp = |f: f64| fs2 * (PI * f / fs).tan();
        // analog prototype poles (scipy buttap), gain 1
        let proto: Vec<C> = (0..ORDER)
            .map(|k| {
                let m = -(ORDER as f64) + 1.0 + 2.0 * k as f64;
                let a = PI * m / (2.0 * ORDER as f64);
                C(-a.cos(), -a.sin())
            })
            .collect();
        let (poles, zero_at, gain): (Vec<C>, Vec<f64>, f64) = match band {
            Band::Lowpass(f) => {
                let w = warp(f);
                (proto.iter().map(|p| p.scale(w)).collect(), vec![], w.powi(ORDER as i32))
            }
            Band::Highpass(f) => {
                let w = warp(f);
                (proto.iter().map(|p| C(w, 0.0).div(*p)).collect(), vec![0.0; ORDER], 1.0)
            }
            Band::Bandpass(lo, hi) => {
                let (w1, w2) = (warp(lo), warp(hi));
                let bw = w2 - w1;
                let w0 = (w1 * w2).sqrt();
                let mut ps = Vec::new();
                for p in &proto {
                    let pl = p.scale(bw / 2.0);
                    let r = pl.mul(pl).sub(C(w0 * w0, 0.0)).sqrt();
                    ps.push(pl.add(r));
                    ps.push(pl.sub(r));
                }
                (ps, vec![0.0; ORDER], bw.powi(ORDER as i32))
            }
        };
        // bilinear transform; zeros at infinity land on z = -1
        let zd: Vec<f64> = zero_at.iter().map(|&z| (fs2 + z) / (fs2 - z)).collect();
        let pd: Vec<C> = poles.iter().map(|&p| C(fs2, 0.0).add(p).div(C(fs2, 0.0).sub(p))).collect();
        let mut num = C(1.0, 0.0);
        for &z in &zero_at {
            num = num.mul(C(fs2 - z, 0.0));
        }
        let mut den = C(1.0, 0.0);
        for &p in &poles {
            den = den.mul(C(fs2, 0.0).sub(p));
        }
        let k = gain * num.div(den).0;
        let mut zeros = zd;
        zeros.resize(pd.len(), -1.0);
        // one conjugate pair per section; the upper half-plane poles
        let mut upper: Vec<C> = pd.iter().copied().filter(|p| p.1 > 0.0).collect();
        upper.sort_by(|a, b| (a.0 * a.0 + a.1 * a.1).total_cmp(&(b.0 * b.0 + b.1 * b.1)));
        zeros.sort_by(f64::total_cmp);
        let n = upper.len();
        let mut sections = Vec::with_capacity(n);
        for (i, p) in upper.iter().enumerate() {
            // a band-pass pairs one zero at +1 with one at -1
            let (z1, z2) = if matches!(band, Band::Bandpass(..)) { (zeros[i], zeros[zeros.len() - 1 - i]) } else { (zeros[2 * i], zeros[2 * i + 1]) };
            let g = if i == 0 { k } else { 1.0 };
            sections.push([g, -g * (z1 + z2), g * z1 * z2, -2.0 * p.0, p.0 * p.0 + p.1 * p.1]);
        }
        Self { state: vec![[0.0; 2]; n], sections }
    }

    #[inline]
    pub fn tick(&mut self, x: f64) -> f64 {
        let mut v = x;
        for (s, st) in self.sections.iter().zip(self.state.iter_mut()) {
            let y = s[0] * v + st[0];
            st[0] = s[1] * v - s[3] * y + st[1];
            st[1] = s[2] * v - s[4] * y;
            v = y;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn impulse(band: Band) -> Vec<f64> {
        let mut f = Sos::butter4(band, 44_100.0);
        (0..64).map(|i| f.tick(if i == 0 { 1.0 } else { 0.0 })).collect()
    }

    fn close(got: &[f64], idx: &[usize], want: &[f64]) {
        for (&i, &w) in idx.iter().zip(want) {
            assert!((got[i] - w).abs() <= 1e-9 * (1.0 + w.abs()) + 1e-12, "sample {i}: {} vs {w}", got[i]);
        }
    }

    // impulse responses from scipy.signal.sosfilt(butter(4, …, fs=44100, output="sos"))
    #[test]
    #[allow(clippy::excessive_precision)]
    fn matches_scipy() {
        let idx = [0, 1, 2, 5, 10, 30, 63];
        close(&impulse(Band::Lowpass(150.0)), &idx, &[1.2680874150452645e-08, 1.0073882233148770e-07, 3.9943427605567243e-07, 4.3413429863617450e-06, 3.0660117043524213e-05, 6.0684077535369096e-04, 3.3092986260956775e-03]);
        close(&impulse(Band::Bandpass(1500.0, 6000.0)), &idx, &[0.00516740384148385, 0.02969332205448673, 0.07298491233844048, -0.05709639311919804, 0.03956218523106969, -0.02035269730720528, 0.00057687438324183]);
        close(&impulse(Band::Highpass(5000.0)), &idx, &[3.8482177972162307e-01, -7.1019458580750250e-01, 1.3253631417016770e-02, 6.5902019824472455e-02, 1.4431964948149740e-04, 2.4220705172525031e-04, -6.1466070495685446e-08]);
    }
}
