//! Building blocks, each the Burn twin of the loop in `sone_infer::layers`.

use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};
use sone_infer::random::Rng;

/// `x / sqrt(mean(x²) + eps) · gain` over the last dimension.
pub fn rms_norm<B: Backend, const D: usize>(x: Tensor<B, D>, gain: Tensor<B, 1>, eps: f64) -> Tensor<B, D> {
    let ms = x.clone().powi_scalar(2).mean_dim(D - 1);
    x * (ms + eps).sqrt().recip() * gain.unsqueeze::<D>()
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub gain: Param<Tensor<B, 1>>,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self { gain: Param::from_tensor(Tensor::ones([dim], device)) }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        rms_norm(x, self.gain.val(), sone_infer::ops::EPS as f64)
    }
}

pub fn linear<B: Backend>(n_in: usize, n_out: usize, bias: bool, device: &B::Device) -> Linear<B> {
    LinearConfig::new(n_in, n_out).with_bias(bias).init(device)
}

/// A linear layer whose output starts small (`scale` times the default init).
pub fn linear_scaled<B: Backend>(n_in: usize, n_out: usize, scale: f64, device: &B::Device) -> Linear<B> {
    let bound = scale / (n_in as f64).sqrt();
    LinearConfig::new(n_in, n_out).with_initializer(Initializer::Uniform { min: -bound, max: bound }).init(device)
}

/// Uniform in ±`bound`, drawn on the CPU from our own generator so that a
/// model is the same on every backend for the same seed.
pub fn uniform<B: Backend, const D: usize>(rng: &mut Rng, shape: [usize; D], bound: f32, device: &B::Device) -> Tensor<B, D> {
    let n: usize = shape.iter().product();
    let values: Vec<f32> = (0..n).map(|_| rng.sym(bound)).collect();
    Tensor::from_data(TensorData::new(values, shape), device)
}

/// Pre-norm feed-forward, residual: `x + l2(gelu(l1(norm(x))))`.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub norm: RmsNorm<B>,
    pub l1: Linear<B>,
    pub l2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, inner: usize, device: &B::Device) -> Self {
        Self { norm: RmsNorm::new(dim, device), l1: linear(dim, inner, true, device), l2: linear(inner, dim, true, device) }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let h = self.l2.forward(gelu(self.l1.forward(self.norm.forward(x.clone()))));
        x + h
    }
}

/// Pre-norm multi-head self-attention without mask, residual. `x` is `(batch, seq, dim)`.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub norm: RmsNorm<B>,
    /// Output columns: q for all heads, then k, then v.
    pub qkv: Linear<B>,
    pub out: Linear<B>,
    pub heads: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(dim: usize, heads: usize, device: &B::Device) -> Self {
        Self { norm: RmsNorm::new(dim, device), qkv: linear(dim, 3 * dim, false, device), out: linear(dim, dim, true, device), heads }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [n, s, d] = x.dims();
        let (h, hd) = (self.heads, d / self.heads);
        let qkv = self.qkv.forward(self.norm.forward(x.clone()));
        let split = |i: usize| qkv.clone().narrow(2, i * d, d).reshape([n, s, h, hd]).swap_dims(1, 2);
        let (q, k, v) = (split(0), split(1), split(2));
        let scores = q.matmul(k.swap_dims(2, 3)) * (1.0 / (hd as f64).sqrt());
        let heads = softmax(scores, 3).matmul(v);
        let merged = heads.swap_dims(1, 2).reshape([n, s, d]);
        x + self.out.forward(merged)
    }
}

/// Depthwise dilated convolution along time with zero padding, then a
/// feed-forward, pre-norm, residual. `x` is `(batch, time, lanes, dim)`.
#[derive(Module, Debug)]
pub struct TimeBlock<B: Backend> {
    pub norm: RmsNorm<B>,
    /// `(dim, kernel)`
    pub conv_w: Param<Tensor<B, 2>>,
    pub conv_b: Param<Tensor<B, 1>>,
    pub l1: Linear<B>,
    pub l2: Linear<B>,
    pub dilation: usize,
}

impl<B: Backend> TimeBlock<B> {
    pub fn new(dim: usize, inner: usize, kernel: usize, dilation: usize, rng: &mut Rng, device: &B::Device) -> Self {
        Self {
            norm: RmsNorm::new(dim, device),
            conv_w: Param::from_tensor(uniform(rng, [dim, kernel], 1.0 / (kernel as f32).sqrt(), device)),
            conv_b: Param::from_tensor(Tensor::zeros([dim], device)),
            l1: linear(dim, inner, true, device),
            l2: linear(inner, dim, true, device),
            dilation,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, t, _, d] = x.dims();
        let w = self.conv_w.val();
        let kernel = w.dims()[1];
        let pad = (kernel / 2) * self.dilation;
        let h = self.norm.forward(x.clone()).pad([(pad, pad), (0, 0), (0, 0)], burn::tensor::ops::PadMode::Constant(0.0));
        let mut acc = self.conv_b.val().reshape([1, 1, 1, d]);
        for j in 0..kernel {
            let tap = w.clone().narrow(1, j, 1).reshape([1, 1, 1, d]);
            acc = acc + h.clone().narrow(1, j * self.dilation, t) * tap;
        }
        x + self.l2.forward(gelu(self.l1.forward(acc)))
    }
}
