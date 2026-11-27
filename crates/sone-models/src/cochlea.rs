//! The Stems network in Burn (see `sone_infer::cochlea` for the design). Bands
//! of equal width are batched into one matrix product ("groups"); the export
//! splits them back into the per-band tensors `sone-infer` reads.

use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation::gelu;
use sone_infer::CochleaConfig;
use sone_infer::config::STEMS;
use sone_infer::random::Rng;
use sone_infer::st::NamedTensor;

use crate::dsp::Stft;
use crate::export::{row, t1, t2, transposed};
use crate::layers::{Attention, FeedForward, RmsNorm, TimeBlock, rms_norm, uniform};

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub time: TimeBlock<B>,
    pub band: Attention<B>,
    pub ff: FeedForward<B>,
}

#[derive(Module, Debug)]
pub struct CochleaNet<B: Backend> {
    /// Per band group: norm gains `(G, 4w)`, projection `(G, 4w, dim)`, bias `(G, dim)`.
    pub split_norm: Vec<Param<Tensor<B, 2>>>,
    pub split_w: Vec<Param<Tensor<B, 3>>>,
    pub split_b: Vec<Param<Tensor<B, 2>>>,
    pub band_emb: Param<Tensor<B, 2>>,
    pub blocks: Vec<Block<B>>,
    pub norm: RmsNorm<B>,
    /// Per stem: first mask layer of every band `(K, dim, hidden)` and its bias `(K, hidden)`.
    pub mask_w1: Vec<Param<Tensor<B, 3>>>,
    pub mask_b1: Vec<Param<Tensor<B, 2>>>,
    /// Per stem and band group: second layer `(G, hidden, 4w)` and bias `(G, 4w)`.
    pub mask_w2: Vec<Vec<Param<Tensor<B, 3>>>>,
    pub mask_b2: Vec<Vec<Param<Tensor<B, 2>>>>,
    #[module(skip)]
    pub config: CochleaConfig,
}

/// What a forward pass returns for a batch of chunks.
pub struct CochleaOut<B: Backend> {
    /// Mixture STFT `(batch, frames, bins, 4)`.
    pub mix_spec: Tensor<B, 4>,
    /// Estimated STFT of vocals, drums, bass, other: `(batch, 4, frames, bins, 4)`.
    pub spec: Tensor<B, 5>,
    /// Estimated waveforms `(batch, 4, 2, len)`.
    pub wave: Tensor<B, 4>,
}

impl<B: Backend> CochleaNet<B> {
    pub fn new(config: &CochleaConfig, seed: u64, device: &B::Device) -> Self {
        config.validate().expect("valid stems config");
        let mut rng = Rng::new(seed);
        let (d, k, m) = (config.dim, config.bands.len(), config.mask_hidden);
        let groups = config.band_groups();
        let p = |t| Param::from_tensor(t);
        let mut split_norm = Vec::new();
        let mut split_w = Vec::new();
        let mut split_b = Vec::new();
        for &(_, g, w) in &groups {
            split_norm.push(p(Tensor::ones([g, 4 * w], device)));
            split_w.push(Param::from_tensor(uniform(&mut rng, [g, 4 * w, d], 1.0 / ((4 * w) as f32).sqrt(), device)));
            split_b.push(p(Tensor::zeros([g, d], device)));
        }
        let band_emb = Param::from_tensor(uniform(&mut rng, [k, d], 0.02, device));
        let blocks = (0..config.blocks)
            .map(|i| Block {
                time: TimeBlock::new(d, config.ff(), config.kernel, config.dilations[i], &mut rng, device),
                band: Attention::new(d, config.heads, device),
                ff: FeedForward::new(d, config.ff(), device),
            })
            .collect();
        let mut mask_w1 = Vec::new();
        let mut mask_b1 = Vec::new();
        let mut mask_w2 = Vec::new();
        let mut mask_b2 = Vec::new();
        for _ in STEMS {
            mask_w1.push(Param::from_tensor(uniform(&mut rng, [k, d, m], 1.0 / (d as f32).sqrt(), device)));
            mask_b1.push(p(Tensor::zeros([k, m], device)));
            // small masks at the start: "other" (the rest) begins as the whole mixture
            mask_w2.push(groups.iter().map(|&(_, g, w)| Param::from_tensor(uniform(&mut rng, [g, m, 4 * w], 0.1 / (m as f32).sqrt(), device))).collect());
            mask_b2.push(groups.iter().map(|&(_, g, w)| p(Tensor::zeros([g, 4 * w], device))).collect());
        }
        Self { split_norm, split_w, split_b, band_emb, blocks, norm: RmsNorm::new(d, device), mask_w1, mask_b1, mask_w2, mask_b2, config: config.clone() }
    }

    /// Masked estimates of vocals, drums, bass for a mixture STFT
    /// `(batch, frames, bins, 4)`: `(batch, 3, frames, bins, 4)`.
    pub fn forward_spec(&self, spec: Tensor<B, 4>) -> Tensor<B, 5> {
        let c = &self.config;
        let [b, t, bins, _] = spec.dims();
        let (d, k) = (c.dim, c.bands.len());
        let groups = c.band_groups();
        let rows = b * t;

        let mut tokens = Vec::new();
        for (gi, &(first, g, w)) in groups.iter().enumerate() {
            let f0 = c.bands[first][0];
            let x = spec.clone().narrow(2, f0, g * w).reshape([b, t, g, 4 * w]);
            let x = rms_norm_groups(x, self.split_norm[gi].val(), sone_infer::ops::BAND_EPS as f64);
            let x = x.reshape([rows, g, 4 * w]).swap_dims(0, 1);
            let y = x.matmul(self.split_w[gi].val()) + self.split_b[gi].val().reshape([g, 1, d]);
            tokens.push(y.swap_dims(0, 1));
        }
        let mut x = Tensor::cat(tokens, 1).reshape([b, t, k, d]) + self.band_emb.val().reshape([1, 1, k, d]);

        for block in &self.blocks {
            x = block.time.forward(x);
            x = block.band.forward(x.reshape([rows, k, d])).reshape([b, t, k, d]);
            x = block.ff.forward(x);
        }
        let x = self.norm.forward(x).reshape([rows, k, d]).swap_dims(0, 1);

        let mut stems = Vec::new();
        for s in 0..STEMS.len() {
            let h = gelu(x.clone().matmul(self.mask_w1[s].val()) + self.mask_b1[s].val().reshape([k, 1, c.mask_hidden]));
            let mut parts = Vec::new();
            for (gi, &(first, g, w)) in groups.iter().enumerate() {
                let y = h.clone().narrow(0, first, g).matmul(self.mask_w2[s][gi].val()) + self.mask_b2[s][gi].val().reshape([g, 1, 4 * w]);
                parts.push(y.swap_dims(0, 1).reshape([b, t, g * w, 4]));
            }
            let mask = Tensor::cat(parts, 2);
            stems.push(complex_mul(mask, spec.clone(), [b, t, bins]));
        }
        Tensor::stack(stems, 1)
    }

    /// Separate a batch of stereo chunks `(batch, 2, len)`.
    pub fn forward(&self, stft: &Stft<B>, mix: Tensor<B, 3>) -> CochleaOut<B> {
        let [b, _, len] = mix.dims();
        let mix_spec = stft.forward(mix.clone());
        let three = self.forward_spec(mix_spec.clone());
        let [_, _, t, f, _] = three.dims();
        let other_spec = mix_spec.clone() - three.clone().sum_dim(1).reshape([b, t, f, 4]);
        let spec = Tensor::cat(vec![three.clone(), other_spec.reshape([b, 1, t, f, 4])], 1);
        let waves = stft.inverse(three.reshape([b * 3, t, f, 4]), len).reshape([b, 3, 2, len]);
        let other = mix - waves.clone().sum_dim(1).reshape([b, 2, len]);
        let wave = Tensor::cat(vec![waves, other.reshape([b, 1, 2, len])], 1);
        CochleaOut { mix_spec, spec, wave }
    }

    /// The tensors under `sone-infer`'s names (see `CochleaConfig::tensor_shapes`).
    pub fn export(&self) -> Vec<NamedTensor> {
        let c = &self.config;
        let groups = c.band_groups();
        let mut out = Vec::new();
        for (gi, &(first, g, _)) in groups.iter().enumerate() {
            let norm = t2(self.split_norm[gi].val());
            let w = self.split_w[gi].val();
            let bias = t2(self.split_b[gi].val());
            for i in 0..g {
                let band = first + i;
                out.push((format!("split.{band}.norm"), vec![norm.1[1]], row(&norm, i)));
                let wi = w.clone().narrow(0, i, 1).reshape([w.dims()[1], w.dims()[2]]);
                out.push(transposed(&format!("split.{band}.w"), wi));
                out.push((format!("split.{band}.b"), vec![bias.1[1]], row(&bias, i)));
            }
        }
        out.push(("band_emb".into(), self.band_emb.val().dims().to_vec(), t2(self.band_emb.val()).0));
        for (i, block) in self.blocks.iter().enumerate() {
            crate::export::time_block(&mut out, &format!("block{i}.time"), &block.time);
            crate::export::attention(&mut out, &format!("block{i}.band"), &block.band);
            crate::export::feed_forward(&mut out, &format!("block{i}.ff"), &block.ff);
        }
        out.push(("norm".into(), vec![c.dim], t1(self.norm.gain.val())));
        for (s, stem) in STEMS.iter().enumerate() {
            let w1 = self.mask_w1[s].val();
            let b1 = t2(self.mask_b1[s].val());
            for band in 0..c.bands.len() {
                let wi = w1.clone().narrow(0, band, 1).reshape([c.dim, c.mask_hidden]);
                out.push(transposed(&format!("mask.{stem}.{band}.l1.w"), wi));
                out.push((format!("mask.{stem}.{band}.l1.b"), vec![c.mask_hidden], row(&b1, band)));
            }
            for (gi, &(first, g, w)) in groups.iter().enumerate() {
                let w2 = self.mask_w2[s][gi].val();
                let b2 = t2(self.mask_b2[s][gi].val());
                for i in 0..g {
                    let wi = w2.clone().narrow(0, i, 1).reshape([c.mask_hidden, 4 * w]);
                    out.push(transposed(&format!("mask.{stem}.{}.l2.w", first + i), wi));
                    out.push((format!("mask.{stem}.{}.l2.b", first + i), vec![4 * w], row(&b2, i)));
                }
            }
        }
        out
    }
}

/// RMSNorm of `(b, t, g, n)` with one gain row per group member `(g, n)`.
fn rms_norm_groups<B: Backend>(x: Tensor<B, 4>, gain: Tensor<B, 2>, eps: f64) -> Tensor<B, 4> {
    let [g, n] = gain.dims();
    let ones = Tensor::<B, 1>::ones([n], &gain.device());
    rms_norm(x, ones, eps) * gain.reshape([1, 1, g, n])
}

/// `mask ⊗ spec` per bin and channel, both `(b, t, f, 4)` = `[re, im] × 2 channels`.
fn complex_mul<B: Backend>(mask: Tensor<B, 4>, spec: Tensor<B, 4>, [b, t, f]: [usize; 3]) -> Tensor<B, 4> {
    let m = mask.reshape([b, t, f, 2, 2]);
    let x = spec.reshape([b, t, f, 2, 2]);
    let (mr, mi) = (m.clone().narrow(4, 0, 1), m.narrow(4, 1, 1));
    let (xr, xi) = (x.clone().narrow(4, 0, 1), x.narrow(4, 1, 1));
    let re = mr.clone() * xr.clone() - mi.clone() * xi.clone();
    let im = mr * xi + mi * xr;
    Tensor::cat(vec![re, im], 4).reshape([b, t, f, 4])
}
