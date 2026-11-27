//! FFT, STFT and log-mel, defined once so that training (which computes the
//! same transforms as matrix products in Burn) and inference agree:
//!
//! - window: periodic Hann, `w[n] = 0.5 - 0.5·cos(2πn/N)`;
//! - framing: centred and zero-padded, frame `t` covers samples
//!   `t·hop - N/2 .. t·hop + N/2`, so a signal of `len` samples (a multiple
//!   of the hop) has `len / hop + 1` frames;
//! - forward: `X_t[f] = Σ_n w[n]·x[t·hop + n - N/2]·e^{-2πi·fn/N}`, `f < N/2 + 1`;
//! - inverse: each frame's real inverse DFT (the imaginary parts of bins 0
//!   and N/2 are ignored) times the window, overlap-added, divided by the
//!   overlap-added squared window.

use std::f64::consts::PI;

/// Frames of a signal of `len` samples.
pub fn frames(len: usize, hop: usize) -> usize {
    len / hop + 1
}

/// The periodic Hann window.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()) as f32).collect()
}

/// In-place radix-2 complex FFT.
pub struct Fft {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    rev: Vec<u32>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2, "FFT size must be a power of two");
        let bits = n.trailing_zeros();
        let rev = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
        let cos = (0..n / 2).map(|k| (2.0 * PI * k as f64 / n as f64).cos() as f32).collect();
        let sin = (0..n / 2).map(|k| (2.0 * PI * k as f64 / n as f64).sin() as f32).collect();
        Self { n, cos, sin, rev }
    }

    /// `X[k] = Σ x[n]·e^{∓2πi·kn/N}` (minus sign unless `inverse`), unnormalised.
    pub fn run(&self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i] as usize;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut size = 2;
        while size <= n {
            let half = size / 2;
            let step = n / size;
            for start in (0..n).step_by(size) {
                for k in 0..half {
                    let (c, s) = (self.cos[k * step], sign * self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * c - im[b] * s;
                    let ti = re[b] * s + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
    }
}

/// STFT and its inverse for one `(n_fft, hop)`, with its own scratch.
pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    window: Vec<f32>,
    fft: Fft,
    re: Vec<f32>,
    im: Vec<f32>,
    env: Vec<f32>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        Self { n_fft, hop, window: hann(n_fft), fft: Fft::new(n_fft), re: vec![0.0; n_fft], im: vec![0.0; n_fft], env: Vec::new() }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Windowed frame `t` of `x` into `out` (zero outside the signal).
    fn frame(&self, x: &[f32], t: usize, out: &mut [f32]) {
        let n = self.n_fft;
        let first = (t * self.hop) as isize - (n / 2) as isize;
        for (i, o) in out[..n].iter_mut().enumerate() {
            let at = first + i as isize;
            *o = if at >= 0 && (at as usize) < x.len() { x[at as usize] * self.window[i] } else { 0.0 };
        }
    }

    /// Both channels at once (one complex FFT carries two real ones):
    /// `out (frames, bins, 4)` = `[re_l, im_l, re_r, im_r]` per bin.
    pub fn forward_stereo(&mut self, l: &[f32], r: &[f32], out: &mut Vec<f32>) {
        assert_eq!(l.len(), r.len());
        let (n, bins, t_n) = (self.n_fft, self.bins(), frames(l.len(), self.hop));
        out.clear();
        out.resize(t_n * bins * 4, 0.0);
        let (mut re, mut im) = (std::mem::take(&mut self.re), std::mem::take(&mut self.im));
        for t in 0..t_n {
            self.frame(l, t, &mut re);
            self.frame(r, t, &mut im);
            self.fft.run(&mut re, &mut im, false);
            let row = &mut out[t * bins * 4..(t + 1) * bins * 4];
            for k in 0..bins {
                // Z = L + iR: L[k] = (Z[k] + conj Z[N-k]) / 2, R[k] = (Z[k] - conj Z[N-k]) / 2i
                let m = (n - k) % n;
                let (zr, zi, cr, ci) = (re[k], im[k], re[m], -im[m]);
                row[4 * k] = 0.5 * (zr + cr);
                row[4 * k + 1] = 0.5 * (zi + ci);
                row[4 * k + 2] = 0.5 * (zi - ci);
                row[4 * k + 3] = -0.5 * (zr - cr);
            }
        }
        self.re = re;
        self.im = im;
    }

    /// Magnitudes of a mono signal, `out (frames, bins)`.
    pub fn magnitude(&mut self, x: &[f32], out: &mut Vec<f32>) {
        let (bins, t_n) = (self.bins(), frames(x.len(), self.hop));
        out.clear();
        out.resize(t_n * bins, 0.0);
        let (mut re, mut im) = (std::mem::take(&mut self.re), std::mem::take(&mut self.im));
        for t in 0..t_n {
            self.frame(x, t, &mut re);
            im.fill(0.0);
            self.fft.run(&mut re, &mut im, false);
            for k in 0..bins {
                out[t * bins + k] = (re[k] * re[k] + im[k] * im[k]).sqrt();
            }
        }
        self.re = re;
        self.im = im;
    }

    /// The inverse of [`Stft::forward_stereo`] for a signal of `len` samples;
    /// `l` and `r` are overwritten.
    pub fn inverse_stereo(&mut self, spec: &[f32], len: usize, l: &mut [f32], r: &mut [f32]) {
        let (n, bins, hop) = (self.n_fft, self.bins(), self.hop);
        let t_n = frames(len, hop);
        assert!(spec.len() >= t_n * bins * 4);
        l[..len].fill(0.0);
        r[..len].fill(0.0);
        self.envelope(len);
        let (mut re, mut im) = (std::mem::take(&mut self.re), std::mem::take(&mut self.im));
        let scale = 1.0 / n as f32;
        for t in 0..t_n {
            let row = &spec[t * bins * 4..(t + 1) * bins * 4];
            // Z = L + iR with both spectra made Hermitian: the real part of
            // the inverse is l, the imaginary part r.
            for k in 0..bins {
                let (mut lr, mut li, mut rr, mut ri) = (row[4 * k], row[4 * k + 1], row[4 * k + 2], row[4 * k + 3]);
                if k == 0 || k == n / 2 {
                    li = 0.0;
                    ri = 0.0;
                }
                re[k] = lr - ri;
                im[k] = li + rr;
                if k != 0 && k != n / 2 {
                    // bin N-k: conj(L) + i·conj(R)
                    lr = row[4 * k];
                    li = -row[4 * k + 1];
                    rr = row[4 * k + 2];
                    ri = -row[4 * k + 3];
                    re[n - k] = lr - ri;
                    im[n - k] = li + rr;
                }
            }
            self.fft.run(&mut re, &mut im, true);
            let first = (t * hop) as isize - (n / 2) as isize;
            for i in 0..n {
                let at = first + i as isize;
                if at >= 0 && (at as usize) < len {
                    let w = self.window[i] * scale;
                    l[at as usize] += re[i] * w;
                    r[at as usize] += im[i] * w;
                }
            }
        }
        for i in 0..len {
            let e = 1.0 / self.env[i];
            l[i] *= e;
            r[i] *= e;
        }
        self.re = re;
        self.im = im;
    }

    /// Overlap-added squared window over `len` samples (floored, so an
    /// uncovered sample cannot divide by zero).
    fn envelope(&mut self, len: usize) {
        if self.env.len() == len {
            return;
        }
        self.env = envelope(self.n_fft, self.hop, len);
    }
}

/// Σ_t w²[n - t·hop + N/2] for every sample of a `len`-sample signal, floored at 1e-8.
pub fn envelope(n_fft: usize, hop: usize, len: usize) -> Vec<f32> {
    let w = hann(n_fft);
    let mut env = vec![0.0f32; len];
    for t in 0..frames(len, hop) {
        let first = (t * hop) as isize - (n_fft / 2) as isize;
        for (i, wv) in w.iter().enumerate() {
            let at = first + i as isize;
            if at >= 0 && (at as usize) < len {
                env[at as usize] += wv * wv;
            }
        }
    }
    env.iter_mut().for_each(|e| *e = e.max(1e-8));
    env
}

/// Triangular mel filters (HTK mel scale, no area normalisation) over the
/// bins of an `n_fft`-point spectrum: `(n_mels, bins)`, row-major.
pub fn mel_filters(sample_rate: u32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (m0, m1) = (mel(fmin as f64), mel(fmax as f64));
    let points: Vec<f64> = (0..n_mels + 2).map(|i| hz(m0 + (m1 - m0) * i as f64 / (n_mels + 1) as f64)).collect();
    let mut out = vec![0.0f32; n_mels * bins];
    for m in 0..n_mels {
        let (lo, mid, hi) = (points[m], points[m + 1], points[m + 2]);
        for k in 0..bins {
            let f = k as f64 * sample_rate as f64 / n_fft as f64;
            let v = if f > lo && f < mid {
                (f - lo) / (mid - lo)
            } else if f >= mid && f < hi {
                (hi - f) / (hi - mid)
            } else {
                0.0
            };
            out[m * bins + k] = v as f32;
        }
    }
    out
}

/// The Ear's input: `ln(1 + 1000·mel(|STFT(mono)|))`, `(frames, n_mels)`.
pub struct LogMel {
    stft: Stft,
    filters: Vec<f32>,
    n_mels: usize,
    mag: Vec<f32>,
}

impl LogMel {
    pub fn new(sample_rate: u32, n_fft: usize, hop: usize, n_mels: usize, fmin: f32, fmax: f32) -> Self {
        Self { stft: Stft::new(n_fft, hop), filters: mel_filters(sample_rate, n_fft, n_mels, fmin, fmax), n_mels, mag: Vec::new() }
    }

    pub fn for_tympanum(c: &crate::config::TympanumConfig) -> Self {
        Self::new(c.sample_rate, c.n_fft, c.hop, c.n_mels, c.fmin, c.fmax)
    }

    /// Returns the number of frames.
    pub fn compute(&mut self, mono: &[f32], out: &mut Vec<f32>) -> usize {
        let mut mag = std::mem::take(&mut self.mag);
        self.stft.magnitude(mono, &mut mag);
        let bins = self.stft.bins();
        let t_n = mag.len() / bins;
        out.clear();
        out.resize(t_n * self.n_mels, 0.0);
        // (frames, bins) · filtersᵀ (filters: (mels, bins))
        crate::ops::gemm(t_n, bins, self.n_mels, &mag, bins, 1, &self.filters, 1, bins, 0.0, out, self.n_mels);
        for v in out.iter_mut() {
            *v = (1.0 + 1000.0 * *v).ln();
        }
        self.mag = mag;
        t_n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft() {
        let n = 64;
        let fft = Fft::new(n);
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin() + (i as f32 * 0.13).cos()).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; n]);
        fft.run(&mut re, &mut im, false);
        for k in 0..n {
            let (mut r, mut i) = (0.0f64, 0.0f64);
            for (t, &v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * t) as f64 / n as f64;
                r += v as f64 * a.cos();
                i += v as f64 * a.sin();
            }
            assert!((re[k] as f64 - r).abs() < 1e-3 && (im[k] as f64 - i).abs() < 1e-3, "{k}");
        }
    }

    #[test]
    fn stereo_stft_matches_dft_and_inverts() {
        let (n, hop, len) = (256, 64, 64 * 20);
        let l: Vec<f32> = (0..len).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let r: Vec<f32> = (0..len).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * 0.3).collect();
        let mut stft = Stft::new(n, hop);
        let mut spec = Vec::new();
        stft.forward_stereo(&l, &r, &mut spec);
        let bins = n / 2 + 1;
        // direct DFT of frame 3, right channel, bin 5
        let (t, k) = (3, 5);
        let w = hann(n);
        let (mut sr, mut si) = (0.0f64, 0.0f64);
        for i in 0..n {
            let at = (t * hop + i) as isize - (n / 2) as isize;
            if at >= 0 && (at as usize) < len {
                let a = -2.0 * PI * (k * i) as f64 / n as f64;
                let v = (r[at as usize] * w[i]) as f64;
                sr += v * a.cos();
                si += v * a.sin();
            }
        }
        let at = (t * bins + k) * 4;
        assert!((spec[at + 2] as f64 - sr).abs() < 1e-3 && (spec[at + 3] as f64 - si).abs() < 1e-3);
        let (mut l2, mut r2) = (vec![0.0; len], vec![0.0; len]);
        stft.inverse_stereo(&spec, len, &mut l2, &mut r2);
        for i in 0..len {
            assert!((l[i] - l2[i]).abs() < 1e-4 && (r[i] - r2[i]).abs() < 1e-4, "{i}");
        }
    }
}
