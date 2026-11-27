//! The STFT of `sone_infer::dsp` as matrix products (a four-step FFT), so
//! that it is differentiable on every backend (Burn's own `rfft` has no
//! autodiff yet).
//! Same window, framing and inverse; the parity tests hold the two together.

use std::f64::consts::PI;

use burn::prelude::*;

/// A length-N DFT as two small ones (Cooley-Tukey "four-step", N = N1·N2):
/// DFTs of length N1 over the columns, a twiddle, DFTs of length N2 over the
/// rows, each a matrix product. It is differentiable like any product and
/// costs N·(N1 + N2) complex multiply-adds per frame instead of N².
#[derive(Debug, Clone)]
struct FourStep<B: Backend> {
    n1: usize,
    n2: usize,
    /// `(N1, N1)`: e^{∓2πi·ab/N1}, real and imaginary parts.
    a: [Tensor<B, 2>; 2],
    /// `(1, N2, N1)`: e^{∓2πi·n2·k1/N}.
    tw: [Tensor<B, 3>; 2],
    /// `(N2, N2)`: e^{∓2πi·ab/N2}.
    b: [Tensor<B, 2>; 2],
}

impl<B: Backend> FourStep<B> {
    fn new(n: usize, inverse: bool, device: &B::Device) -> Self {
        let n1 = 1 << (n.trailing_zeros() / 2);
        let n2 = n / n1;
        let sign = if inverse { 1.0 } else { -1.0 };
        let mat = |rows: usize, cols: usize, period: usize, f: &dyn Fn(usize, usize) -> usize| -> [Vec<f32>; 2] {
            let mut re = vec![0.0f32; rows * cols];
            let mut im = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    let a = 2.0 * PI * (f(r, c) % period) as f64 / period as f64;
                    re[r * cols + c] = a.cos() as f32;
                    im[r * cols + c] = (sign * a.sin()) as f32;
                }
            }
            [re, im]
        };
        let t2 = |v: Vec<f32>, r: usize, c: usize| Tensor::<B, 2>::from_data(TensorData::new(v, [r, c]), device);
        let [ar, ai] = mat(n1, n1, n1, &|x, y| x * y);
        let [tr, ti] = mat(n2, n1, n, &|x, y| x * y);
        let [br, bi] = mat(n2, n2, n2, &|x, y| x * y);
        Self {
            n1,
            n2,
            a: [t2(ar, n1, n1), t2(ai, n1, n1)],
            tw: [t2(tr, n2, n1).unsqueeze(), t2(ti, n2, n1).unsqueeze()],
            b: [t2(br, n2, n2), t2(bi, n2, n2)],
        }
    }

    /// DFT of `m` rows of length N (`im` None: real input). With
    /// `real_out`, only the real part of the result is computed.
    fn run(&self, re: Tensor<B, 2>, im: Option<Tensor<B, 2>>, real_out: bool) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
        let [m, n] = re.dims();
        let (n1, n2) = (self.n1, self.n2);
        // index n = N2·n1 + n2: contract over n1 first
        let cols = |x: Tensor<B, 2>| x.reshape([m, n1, n2]).swap_dims(1, 2).reshape([m * n2, n1]);
        let [ar, ai] = self.a.clone();
        let (xr, xi) = (cols(re), im.map(cols));
        let (yr, yi) = match xi {
            None => (xr.clone().matmul(ar), xr.matmul(ai)),
            Some(xi) => (xr.clone().matmul(ar.clone()) - xi.clone().matmul(ai.clone()), xr.matmul(ai) + xi.matmul(ar)),
        };
        let (yr, yi) = (yr.reshape([m, n2, n1]), yi.reshape([m, n2, n1]));
        let [tr, ti] = self.tw.clone();
        let zr = yr.clone() * tr.clone() - yi.clone() * ti.clone();
        let zi = yr * ti + yi * tr;
        // then over n2; the result is indexed (k1, k2) with k = k1 + N1·k2
        let rows = |z: Tensor<B, 3>| z.swap_dims(1, 2).reshape([m * n1, n2]);
        let (zr, zi) = (rows(zr), rows(zi));
        let [br, bi] = self.b.clone();
        let order = |x: Tensor<B, 2>| x.reshape([m, n1, n2]).swap_dims(1, 2).reshape([m, n]);
        let out_r = order(zr.clone().matmul(br.clone()) - zi.clone().matmul(bi.clone()));
        if real_out {
            return (out_r, None);
        }
        (out_r, Some(order(zr.matmul(bi) + zi.matmul(br))))
    }
}

/// STFT and inverse for one `(n_fft, hop)`; `n_fft` a power of two and a
/// multiple of `hop`.
#[derive(Debug, Clone)]
pub struct Stft<B: Backend> {
    pub n_fft: usize,
    pub hop: usize,
    window: Tensor<B, 2>,
    fwd: FourStep<B>,
    inv: FourStep<B>,
    /// `(1, bins)`: 0 at bin 0 and N/2 (their imaginary parts are ignored), else 1.
    imag_mask: Tensor<B, 2>,
}

impl<B: Backend> Stft<B> {
    pub fn new(n_fft: usize, hop: usize, device: &B::Device) -> Self {
        assert!(n_fft.is_power_of_two() && n_fft % hop == 0, "n_fft must be a power of two and a multiple of the hop");
        let bins = n_fft / 2 + 1;
        let w = sone_infer::dsp::hann(n_fft);
        let mask: Vec<f32> = (0..bins).map(|k| if k == 0 || k == n_fft / 2 { 0.0 } else { 1.0 }).collect();
        Self {
            n_fft,
            hop,
            window: Tensor::<B, 1>::from_data(TensorData::new(w, [n_fft]), device).unsqueeze(),
            fwd: FourStep::new(n_fft, false, device),
            inv: FourStep::new(n_fft, true, device),
            imag_mask: Tensor::<B, 1>::from_data(TensorData::new(mask, [bins]), device).unsqueeze(),
        }
    }

    /// Windowed frames `(rows, n_fft)` → `(re, im)`, each `(rows, bins)`.
    fn rfft(&self, frames: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let bins = self.bins();
        let (re, im) = self.fwd.run(frames * self.window.clone(), None, false);
        (re.narrow(1, 0, bins), im.expect("complex output").narrow(1, 0, bins))
    }

    /// `(re, im)` `(rows, bins)` → real frames `(rows, n_fft)`, times the window.
    fn irfft(&self, re: Tensor<B, 2>, im: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = self.n_fft;
        let im = im * self.imag_mask.clone();
        // Hermitian completion: bins N/2+1..N-1 are the conjugates of N/2-1..1
        let tail = |x: Tensor<B, 2>| x.narrow(1, 1, n / 2 - 1).flip([1]);
        let full_re = Tensor::cat(vec![re.clone(), tail(re)], 1);
        let full_im = Tensor::cat(vec![im.clone(), tail(im).neg()], 1);
        let (x, _) = self.inv.run(full_re, Some(full_im), true);
        x * self.window.clone() * (1.0 / n as f64)
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// `audio (batch, channels, len)`, `len` a multiple of the hop →
    /// `(batch, frames, bins, 2·channels)`, last index `c·2 + re/im`.
    pub fn forward(&self, audio: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, c, len] = audio.dims();
        assert!(len % self.hop == 0, "length must be a multiple of the hop");
        let (n, hop, bins) = (self.n_fft, self.hop, self.bins());
        let r = n / hop;
        let t = len / hop + 1;
        let padded = audio.pad([(n / 2, n / 2)], burn::tensor::ops::PadMode::Constant(0.0));
        let blocks = padded.reshape([b, c, (len + n) / hop, hop]);
        let frames = Tensor::cat((0..r).map(|q| blocks.clone().narrow(2, q, t)).collect(), 3);
        let (re, im) = self.rfft(frames.reshape([b * c * t, n]));
        let spec = Tensor::cat(vec![re, im], 1);
        spec.reshape([b, c, t, 2, bins]).permute([0, 2, 4, 1, 3]).reshape([b, t, bins, 2 * c])
    }

    /// Magnitude `(batch, frames, bins, channels)` with a small floor inside
    /// the square root (its gradient is finite at zero).
    pub fn magnitude(&self, audio: Tensor<B, 3>) -> Tensor<B, 4> {
        let spec = self.forward(audio);
        let [b, t, f, c2] = spec.dims();
        let pairs = spec.reshape([b, t, f, c2 / 2, 2]);
        (pairs.powi_scalar(2).sum_dim(4) + 1e-8).sqrt().reshape([b, t, f, c2 / 2])
    }

    /// The inverse of [`Stft::forward`] for `len` samples: `(batch, channels, len)`.
    pub fn inverse(&self, spec: Tensor<B, 4>, len: usize) -> Tensor<B, 3> {
        let [b, t, bins, c2] = spec.dims();
        let c = c2 / 2;
        let (n, hop) = (self.n_fft, self.hop);
        let r = n / hop;
        assert_eq!(t, len / hop + 1);
        let flat = spec.reshape([b, t, bins, c, 2]).permute([0, 3, 1, 4, 2]).reshape([b * c * t, 2 * bins]);
        let frames = self.irfft(flat.clone().narrow(1, 0, bins), flat.narrow(1, bins, bins)).reshape([b, c, t, r, hop]);
        let mut sum: Option<Tensor<B, 4>> = None;
        for q in 0..r {
            let part = frames.clone().narrow(3, q, 1).reshape([b, c, t, hop]);
            let part = part.pad([(q, r - 1 - q), (0, 0)], burn::tensor::ops::PadMode::Constant(0.0));
            sum = Some(match sum {
                Some(s) => s + part,
                None => part,
            });
        }
        let wave = sum.expect("r ≥ 1").reshape([b, c, len + n]).narrow(2, n / 2, len);
        let env: Vec<f32> = sone_infer::dsp::envelope(n, hop, len).iter().map(|e| 1.0 / e).collect();
        wave * Tensor::<B, 1>::from_data(TensorData::new(env, [len]), &self.window.device()).reshape([1, 1, len])
    }
}
