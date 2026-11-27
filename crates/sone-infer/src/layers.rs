//! Layers shared by the networks: pre-norm feed-forward, pre-norm
//! self-attention and the depthwise time-convolution block. Activations are
//! row-major `(rows, dim)`; they are processed in groups of rows so that the
//! scratch stays a few MB whatever the input length.

use anyhow::Result;

use crate::ops::{self, EPS, gemm, gelu, linear, linear_add, rms_norm_copy, softmax_unnormalised};
use crate::st::{Mat, WeightFile};

/// Rows processed together by the feed-forward and attention blocks.
pub const GROUP_ROWS: usize = 512;

/// Reusable buffers (wasm memory never shrinks: sizes are bounded by `GROUP_ROWS`).
#[derive(Default)]
pub struct Scratch {
    pub norm: Vec<f32>,
    pub inner: Vec<f32>,
    pub qkv: Vec<f32>,
    pub heads: Vec<f32>,
    pub scores: Vec<f32>,
    pub sums: Vec<f32>,
    pub rows: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
}

pub struct Linear {
    pub w: Mat,
    pub b: Vec<f32>,
}

impl Linear {
    pub fn load(t: &mut WeightFile, p: &str, n_out: usize, n_in: usize) -> Result<Self> {
        Ok(Self { w: t.mat(&format!("{p}.w"), n_out, n_in)?, b: t.vec(&format!("{p}.b"), &[n_out])? })
    }

    pub fn n_out(&self) -> usize {
        self.w.rows
    }
}

/// `x += l2(gelu(l1(norm(x))))`.
pub struct FeedForward {
    pub norm: Vec<f32>,
    pub l1: Linear,
    pub l2: Linear,
    pub dim: usize,
    pub inner: usize,
}

impl FeedForward {
    pub fn load(t: &mut WeightFile, p: &str, dim: usize, inner: usize) -> Result<Self> {
        Ok(Self {
            norm: t.vec(&format!("{p}.norm"), &[dim])?,
            l1: Linear::load(t, &format!("{p}.l1"), inner, dim)?,
            l2: Linear::load(t, &format!("{p}.l2"), dim, inner)?,
            dim,
            inner,
        })
    }

    pub fn apply(&self, file: &WeightFile, x: &mut [f32], rows: usize, s: &mut Scratch) {
        self.l1.w.load(file, &mut s.w1);
        self.l2.w.load(file, &mut s.w2);
        for block in x[..rows * self.dim].chunks_mut(GROUP_ROWS * self.dim) {
            let n = block.len() / self.dim;
            s.norm.resize(n * self.dim, 0.0);
            rms_norm_copy(block, n, self.dim, self.dim, &self.norm, EPS, &mut s.norm);
            ops::linear_into(&s.norm, n, self.dim, &s.w1, self.inner, Some(&self.l1.b), &mut s.inner);
            gelu(&mut s.inner);
            linear_add(&s.inner, n, self.inner, &s.w2, self.dim, Some(&self.l2.b), block);
        }
    }
}

/// `x += out(softmax(q·kᵀ/√d_h)·v)` per sequence and head, pre-norm, no mask.
pub struct Attention {
    pub norm: Vec<f32>,
    /// `(3·dim, dim)`: rows are q, then k, then v; head `h` owns columns `h·d_h..`.
    pub qkv: Mat,
    pub out: Linear,
    pub dim: usize,
    pub heads: usize,
}

impl Attention {
    pub fn load(t: &mut WeightFile, p: &str, dim: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            norm: t.vec(&format!("{p}.norm"), &[dim])?,
            qkv: t.mat(&format!("{p}.qkv"), 3 * dim, dim)?,
            out: Linear::load(t, &format!("{p}.out"), dim, dim)?,
            dim,
            heads,
        })
    }

    /// `x` is `(batch, seq, dim)`.
    pub fn apply(&self, file: &WeightFile, x: &mut [f32], batch: usize, seq: usize, s: &mut Scratch) {
        self.qkv.load(file, &mut s.w1);
        self.out.w.load(file, &mut s.w2);
        let group = (GROUP_ROWS / seq).max(1);
        let mut b0 = 0;
        while b0 < batch {
            let n = group.min(batch - b0);
            let rows = &mut x[b0 * seq * self.dim..(b0 + n) * seq * self.dim];
            self.apply_group(rows, n, seq, s);
            b0 += n;
        }
    }

    fn apply_group(&self, x: &mut [f32], batch: usize, seq: usize, s: &mut Scratch) {
        let (dim, heads) = (self.dim, self.heads);
        let hd = dim / heads;
        let rows = batch * seq;
        s.norm.resize(rows * dim, 0.0);
        rms_norm_copy(x, rows, dim, dim, &self.norm, EPS, &mut s.norm);
        s.qkv.resize(rows * 3 * dim, 0.0);
        linear(&s.norm, rows, dim, dim, &s.w1, 3 * dim, None, &mut s.qkv, 3 * dim);
        s.heads.resize(rows * dim, 0.0);
        s.scores.resize(seq * seq, 0.0);
        let scale = 1.0 / (hd as f32).sqrt();
        let stride = 3 * dim;
        for b in 0..batch {
            let base = b * seq;
            for h in 0..heads {
                let q = &s.qkv[base * stride + h * hd..];
                let k = &s.qkv[base * stride + dim + h * hd..];
                // scores (seq, seq) = q (seq, hd) · kᵀ
                gemm(seq, hd, seq, q, stride, 1, k, 1, stride, 0.0, &mut s.scores, seq);
                for v in s.scores.iter_mut() {
                    *v *= scale;
                }
                s.sums.clear();
                s.sums.extend(s.scores.chunks_exact_mut(seq).map(softmax_unnormalised));
                let v = &s.qkv[base * stride + 2 * dim + h * hd..];
                let out = &mut s.heads[base * dim + h * hd..];
                gemm(seq, seq, hd, &s.scores, seq, 1, v, stride, 1, 0.0, out, dim);
                for (i, &sum) in s.sums.iter().enumerate() {
                    let inv = 1.0 / sum;
                    s.heads[(base + i) * dim + h * hd..][..hd].iter_mut().for_each(|v| *v *= inv);
                }
            }
        }
        linear_add(&s.heads, rows, dim, &s.w2, dim, Some(&self.out.b), x);
    }
}

/// `x += l2(gelu(l1(dwconv(norm(x)))))`, the convolution depthwise along time
/// with zero padding. `x` is `(time, lanes, dim)`: every lane (a band for the
/// Stems network, the one sequence for the Ear) is its own sequence.
pub struct TimeBlock {
    pub norm: Vec<f32>,
    /// `(dim, kernel)`
    pub conv_w: Vec<f32>,
    pub conv_b: Vec<f32>,
    pub l1: Linear,
    pub l2: Linear,
    pub dim: usize,
    pub inner: usize,
    pub kernel: usize,
    pub dilation: usize,
}

impl TimeBlock {
    pub fn load(t: &mut WeightFile, p: &str, dim: usize, inner: usize, kernel: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            norm: t.vec(&format!("{p}.norm"), &[dim])?,
            conv_w: t.vec(&format!("{p}.conv.w"), &[dim, kernel])?,
            conv_b: t.vec(&format!("{p}.conv.b"), &[dim])?,
            l1: Linear::load(t, &format!("{p}.l1"), inner, dim)?,
            l2: Linear::load(t, &format!("{p}.l2"), dim, inner)?,
            dim,
            inner,
            kernel,
            dilation,
        })
    }

    /// `h` is a buffer of `x`'s size (the normalised input the convolution reads).
    pub fn apply(&self, file: &WeightFile, x: &mut [f32], h: &mut Vec<f32>, time: usize, lanes: usize, s: &mut Scratch) {
        let d = self.dim;
        let rows = time * lanes;
        h.resize(rows * d, 0.0);
        rms_norm_copy(x, rows, d, d, &self.norm, EPS, h);
        self.l1.w.load(file, &mut s.w1);
        self.l2.w.load(file, &mut s.w2);
        // weights transposed to (kernel, dim) so the inner loop runs over dim
        s.w3.clear();
        for j in 0..self.kernel {
            s.w3.extend((0..d).map(|c| self.conv_w[c * self.kernel + j]));
        }
        let half = (self.kernel / 2) as isize;
        let frames = (GROUP_ROWS / lanes).max(1);
        let mut t0 = 0;
        while t0 < time {
            let g = frames.min(time - t0);
            s.rows.resize(g * lanes * d, 0.0);
            for i in 0..g {
                let t = t0 + i;
                for lane in 0..lanes {
                    let out = &mut s.rows[(i * lanes + lane) * d..][..d];
                    out.copy_from_slice(&self.conv_b);
                    for j in 0..self.kernel {
                        let src = t as isize + (j as isize - half) * self.dilation as isize;
                        if src < 0 || src >= time as isize {
                            continue;
                        }
                        let inp = &h[(src as usize * lanes + lane) * d..][..d];
                        let w = &s.w3[j * d..][..d];
                        for c in 0..d {
                            out[c] += w[c] * inp[c];
                        }
                    }
                }
            }
            let n = g * lanes;
            ops::linear_into(&s.rows, n, d, &s.w1, self.inner, Some(&self.l1.b), &mut s.inner);
            gelu(&mut s.inner);
            linear_add(&s.inner, n, self.inner, &s.w2, d, Some(&self.l2.b), &mut x[t0 * lanes * d..(t0 + g) * lanes * d]);
            t0 += g;
        }
    }
}
