//! Second-order sections and the Butterworth design of
//! `scipy.signal.butter(N, f, btype, fs, output="sos")`, followed step by step
//! (analog prototype, frequency transform, bilinear transform, scipy's
//! "nearest" pole/zero pairing and section order) so the coefficients agree
//! with scipy's to rounding.

/// One section `b0 b1 b2 / 1 a1 a2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Biquad {
    pub b: [f64; 3],
    pub a: [f64; 2],
}

/// A cascade of sections (scipy's `sos` array without the leading 1 of `a`).
#[derive(Debug, Clone, PartialEq)]
pub struct Sos(pub Vec<Biquad>);

/// Filter state of one channel: direct form II transposed, two values per section,
/// the same state layout as scipy's `zi`, so it survives a coefficient change.
#[derive(Debug, Clone, PartialEq)]
pub struct SosState(pub Vec<[f64; 2]>);

impl SosState {
    pub fn new(sections: usize) -> Self {
        SosState(vec![[0.0; 2]; sections])
    }
}

impl Sos {
    pub fn sections(&self) -> usize {
        self.0.len()
    }

    /// One sample through the cascade.
    #[inline]
    pub fn tick(&self, st: &mut SosState, x: f64) -> f64 {
        let mut x = x;
        for (s, z) in self.0.iter().zip(st.0.iter_mut()) {
            let y = s.b[0] * x + z[0];
            z[0] = s.b[1] * x - s.a[0] * y + z[1];
            z[1] = s.b[2] * x - s.a[1] * y;
            x = y;
        }
        x
    }

    /// `scipy.signal.sosfilt` from a zero state.
    pub fn filter(&self, x: &[f64]) -> Vec<f64> {
        let mut st = SosState::new(self.sections());
        x.iter().map(|&v| self.tick(&mut st, v)).collect()
    }

    pub fn filter_f32(&self, x: &[f32]) -> Vec<f64> {
        let mut st = SosState::new(self.sections());
        x.iter().map(|&v| self.tick(&mut st, f64::from(v))).collect()
    }

    /// Filters in place, section by section like scipy (same numbers as `tick`).
    pub fn filter_in_place(&self, x: &mut [f64], st: &mut SosState) {
        for v in x.iter_mut() {
            *v = self.tick(st, *v);
        }
    }

    /// As a flat scipy `sos` row list `[b0 b1 b2 1 a1 a2]`.
    pub fn to_rows(&self) -> Vec<[f64; 6]> {
        self.0.iter().map(|s| [s.b[0], s.b[1], s.b[2], 1.0, s.a[0], s.a[1]]).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Band {
    Lowpass(f64),
    Highpass(f64),
    Bandpass(f64, f64),
}

// ---------------------------------------------------------------------------
// Complex numbers, only what the design needs.

#[derive(Debug, Clone, Copy, PartialEq)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    fn real(re: f64) -> C {
        C { re, im: 0.0 }
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn scale(self, k: f64) -> C {
        C::new(self.re * k, self.im * k)
    }
    fn div(self, o: C) -> C {
        // Smith's algorithm, as numpy does it.
        if o.re.abs() >= o.im.abs() {
            if o.re == 0.0 && o.im == 0.0 {
                return C::new(self.re / o.re.abs(), self.im / o.re.abs());
            }
            let rat = o.im / o.re;
            let scl = 1.0 / (o.re + o.im * rat);
            C::new((self.re + self.im * rat) * scl, (self.im - self.re * rat) * scl)
        } else {
            let rat = o.re / o.im;
            let scl = 1.0 / (o.im + o.re * rat);
            C::new((self.re * rat + self.im) * scl, (self.im * rat - self.re) * scl)
        }
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn conj(self) -> C {
        C::new(self.re, -self.im)
    }
    fn is_real(self) -> bool {
        self.im == 0.0
    }
    fn sqrt(self) -> C {
        // Principal branch (like numpy's csqrt).
        if self.re == 0.0 && self.im == 0.0 {
            return C::new(0.0, self.im);
        }
        let t = ((self.re.abs() + self.abs()) / 2.0).sqrt();
        if self.re >= 0.0 {
            C::new(t, self.im / (2.0 * t))
        } else {
            C::new(self.im.abs() / (2.0 * t), t.copysign(self.im))
        }
    }
    fn exp_i(theta: f64) -> C {
        C::new(theta.cos(), theta.sin())
    }
}

fn prod(v: &[C]) -> C {
    v.iter().fold(C::real(1.0), |a, &b| a.mul(b))
}

/// `np.poly`: coefficients of the monic polynomial with these roots.
fn poly(roots: &[C]) -> Vec<C> {
    let mut a = vec![C::real(1.0)];
    for &r in roots {
        let mut next = vec![C::real(0.0); a.len() + 1];
        for (i, &c) in a.iter().enumerate() {
            next[i] = next[i].add(c);
            next[i + 1] = next[i + 1].sub(c.mul(r));
        }
        a = next;
    }
    a
}

/// Butterworth design of order `n` at sample rate `fs`, as scipy's `butter(..., output="sos")`.
pub fn butter(n: usize, band: Band, fs: f64) -> Sos {
    // Analog prototype (buttap).
    let nn = n as f64;
    let p: Vec<C> = (0..n)
        .map(|i| {
            let m = -(nn) + 1.0 + 2.0 * i as f64;
            C::exp_i(std::f64::consts::PI * m / (2.0 * nn)).scale(-1.0)
        })
        .collect();
    let z: Vec<C> = Vec::new();
    let k = 1.0f64;
    // Pre-warp with scipy's normalisation (fs = 2 after dividing by Nyquist).
    let warp = |f: f64| {
        let wn = f / (fs / 2.0);
        2.0 * 2.0 * (std::f64::consts::PI * wn / 2.0).tan()
    };
    let (z, p, k) = match band {
        Band::Lowpass(f) => {
            let wo = warp(f);
            let degree = p.len() - z.len();
            let z: Vec<C> = z.iter().map(|v| v.scale(wo)).collect();
            let p: Vec<C> = p.iter().map(|v| v.scale(wo)).collect();
            (z, p, k * wo.powi(degree as i32))
        }
        Band::Highpass(f) => {
            let wo = warp(f);
            let degree = p.len() - z.len();
            let mut zh: Vec<C> = z.iter().map(|v| C::real(wo).div(*v)).collect();
            let ph: Vec<C> = p.iter().map(|v| C::real(wo).div(*v)).collect();
            zh.extend(std::iter::repeat_n(C::real(0.0), degree));
            let nz: Vec<C> = z.iter().map(|v| v.scale(-1.0)).collect();
            let np_: Vec<C> = p.iter().map(|v| v.scale(-1.0)).collect();
            (zh, ph, k * prod(&nz).div(prod(&np_)).re)
        }
        Band::Bandpass(lo, hi) => {
            let (w0, w1) = (warp(lo), warp(hi));
            let bw = w1 - w0;
            let wo = (w0 * w1).sqrt();
            let degree = p.len() - z.len();
            let zl: Vec<C> = z.iter().map(|v| v.scale(bw / 2.0)).collect();
            let pl: Vec<C> = p.iter().map(|v| v.scale(bw / 2.0)).collect();
            let wo2 = C::real(wo * wo);
            let mut zb: Vec<C> = zl.iter().map(|v| v.add(v.mul(*v).sub(wo2).sqrt())).collect();
            zb.extend(zl.iter().map(|v| v.sub(v.mul(*v).sub(wo2).sqrt())));
            let mut pb: Vec<C> = pl.iter().map(|v| v.add(v.mul(*v).sub(wo2).sqrt())).collect();
            pb.extend(pl.iter().map(|v| v.sub(v.mul(*v).sub(wo2).sqrt())));
            zb.extend(std::iter::repeat_n(C::real(0.0), degree));
            (zb, pb, k * bw.powi(degree as i32))
        }
    };
    // Bilinear transform with fs = 2.
    let fs2 = C::real(4.0);
    let degree = p.len() - z.len();
    let mut zz: Vec<C> = z.iter().map(|v| fs2.add(*v).div(fs2.sub(*v))).collect();
    let pz: Vec<C> = p.iter().map(|v| fs2.add(*v).div(fs2.sub(*v))).collect();
    zz.extend(std::iter::repeat_n(C::real(-1.0), degree));
    let dz: Vec<C> = z.iter().map(|v| fs2.sub(*v)).collect();
    let dp: Vec<C> = p.iter().map(|v| fs2.sub(*v)).collect();
    let kz = k * prod(&dz).div(prod(&dp)).re;
    zpk2sos(zz, pz, kz)
}

/// scipy's `_cplxreal`: one of each conjugate pair (sorted by real part), then the reals.
fn cplxreal(z: &[C]) -> Vec<C> {
    if z.is_empty() {
        return Vec::new();
    }
    let tol = 100.0 * f64::EPSILON;
    let mut v = z.to_vec();
    v.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.abs().total_cmp(&b.im.abs())));
    let (reals, cplx): (Vec<C>, Vec<C>) = v.iter().partition(|c| c.im.abs() <= tol * c.abs());
    let zr: Vec<C> = reals.iter().map(|c| C::real(c.re)).collect();
    let zp: Vec<C> = cplx.iter().filter(|c| c.im > 0.0).copied().collect();
    let zn: Vec<C> = cplx.iter().filter(|c| c.im < 0.0).copied().collect();
    assert_eq!(zp.len(), zn.len(), "complex value without its conjugate");
    // Butterworth pairs never share a real part, so scipy's run re-sorting does not apply.
    let mut out: Vec<C> = zp.iter().zip(&zn).map(|(a, b)| a.add(b.conj()).scale(0.5)).collect();
    out.extend(zr);
    out
}

fn idx_worst(p: &[C]) -> usize {
    let mut best = 0;
    let mut bv = f64::INFINITY;
    for (i, c) in p.iter().enumerate() {
        let v = (1.0 - c.abs()).abs();
        if v < bv {
            bv = v;
            best = i;
        }
    }
    best
}

#[derive(PartialEq)]
enum Which {
    Real,
    Complex,
    Any,
}

fn nearest_idx(from: &[C], to: C, which: Which) -> usize {
    let mut order: Vec<usize> = (0..from.len()).collect();
    order.sort_by(|&a, &b| from[a].sub(to).abs().total_cmp(&from[b].sub(to).abs()));
    match which {
        Which::Any => order[0],
        Which::Real => *order.iter().find(|&&i| from[i].is_real()).expect("no real zero left"),
        Which::Complex => *order.iter().find(|&&i| !from[i].is_real()).expect("no complex zero left"),
    }
}

fn single_section(z: &[C], p: &[C]) -> [f64; 6] {
    let b = poly(z);
    let a = poly(p);
    let mut s = [0.0; 6];
    for (i, c) in b.iter().enumerate() {
        s[3 - b.len() + i] = c.re;
    }
    for (i, c) in a.iter().enumerate() {
        s[6 - a.len() + i] = c.re;
    }
    s
}

/// scipy's `zpk2sos(..., pairing="nearest")` for a digital filter.
fn zpk2sos(z: Vec<C>, p: Vec<C>, k: f64) -> Sos {
    let mut z = z;
    let mut p = p;
    if z.len() < p.len() {
        z.extend(std::iter::repeat_n(C::real(0.0), p.len() - z.len()));
    }
    if p.len() < z.len() {
        p.extend(std::iter::repeat_n(C::real(0.0), z.len() - p.len()));
    }
    let n_sections = p.len().max(z.len()).div_ceil(2);
    if p.len() % 2 == 1 {
        p.push(C::real(0.0));
        z.push(C::real(0.0));
    }
    let mut z = cplxreal(&z);
    let mut p = cplxreal(&p);
    let mut rows = vec![[0.0; 6]; n_sections];
    for si in (0..n_sections).rev() {
        let i1 = idx_worst(&p);
        let p1 = p.remove(i1);
        let n_real_p = p.iter().filter(|c| c.is_real()).count();
        let n_real_z = z.iter().filter(|c| c.is_real()).count();
        if p1.is_real() && n_real_p == 0 {
            let iz = nearest_idx(&z, p1, Which::Real);
            let z1 = z.remove(iz);
            rows[si] = single_section(&[z1, C::real(0.0)], &[p1, C::real(0.0)]);
        } else if p.len() + 1 == z.len() && !p1.is_real() && n_real_p == 1 && n_real_z == 1 {
            let iz = nearest_idx(&z, p1, Which::Complex);
            let z1 = z.remove(iz);
            rows[si] = single_section(&[z1, z1.conj()], &[p1, p1.conj()]);
        } else {
            let p2 = if p1.is_real() {
                let reals: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_real()).collect();
                let sub: Vec<C> = reals.iter().map(|&i| p[i]).collect();
                let i2 = reals[idx_worst(&sub)];
                p.remove(i2)
            } else {
                p1.conj()
            };
            if !z.is_empty() {
                let iz = nearest_idx(&z, p1, Which::Any);
                let z1 = z.remove(iz);
                if !z1.is_real() {
                    rows[si] = single_section(&[z1, z1.conj()], &[p1, p2]);
                } else if !z.is_empty() {
                    let iz2 = nearest_idx(&z, p1, Which::Real);
                    let z2 = z.remove(iz2);
                    rows[si] = single_section(&[z1, z2], &[p1, p2]);
                } else {
                    rows[si] = single_section(&[z1], &[p1, p2]);
                }
            } else {
                rows[si] = single_section(&[], &[p1, p2]);
            }
        }
    }
    for v in rows[0][..3].iter_mut() {
        *v *= k;
    }
    Sos(rows.iter().map(|r| Biquad { b: [r[0], r[1], r[2]], a: [r[4] / r[3], r[5] / r[3]] }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Sos, want: &[[f64; 6]]) {
        let rows = a.to_rows();
        assert_eq!(rows.len(), want.len());
        for (r, w) in rows.iter().zip(want) {
            for (x, y) in r.iter().zip(w) {
                assert!((x - y).abs() <= 1e-9 * y.abs() + 1e-20, "{r:?} vs {w:?}");
            }
        }
    }

    // Reference values: scipy 1.17.1, `signal.butter(..., fs=44100, output="sos")`, printed with repr().
    #[test]
    fn highpass_1500_matches_scipy() {
        close(&butter(4, Band::Highpass(1500.0), 44100.0), &SCIPY_HP1500);
    }

    #[test]
    fn lowpass_150_matches_scipy() {
        close(&butter(4, Band::Lowpass(150.0), 44100.0), &SCIPY_LP150);
    }

    #[test]
    fn highpass_order2_100_matches_scipy() {
        close(&butter(2, Band::Highpass(100.0), 44100.0), &SCIPY_HP100_2);
    }

    #[test]
    fn bandpass_350_4500_matches_scipy() {
        close(&butter(2, Band::Bandpass(350.0, 4500.0), 44100.0), &SCIPY_BP350_4500);
    }

    #[test]
    fn lowpass_18000_matches_scipy() {
        close(&butter(4, Band::Lowpass(18000.0), 44100.0), &SCIPY_LP18000);
    }

    #[test]
    fn highpass_20_matches_scipy() {
        close(&butter(4, Band::Highpass(20.0), 44100.0), &SCIPY_HP20);
    }

    #[test]
    fn bandpass_300_6500_matches_scipy() {
        close(&butter(2, Band::Bandpass(300.0, 6500.0), 44100.0), &SCIPY_BP300_6500);
    }

    include!("filter_ref.rs");
}
