//! Plan and Critic in Burn (see `sone_infer::beat`).

use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::prelude::*;
use sone_infer::config::{BeatConfig, BeatKind, RECIPES};
use sone_infer::random::Rng;
use sone_infer::st::NamedTensor;

use crate::export::{attention, feed_forward, lin, t1, t2};
use crate::layers::{Attention, FeedForward, RmsNorm, linear, uniform};

#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    pub attn: Attention<B>,
    pub ff: FeedForward<B>,
}

/// Standardisation of one input: `(x - mean) / std`, fitted on the data, not trained.
#[derive(Module, Debug)]
pub struct Standardise<B: Backend> {
    pub mean: Param<Tensor<B, 1>>,
    pub std: Param<Tensor<B, 1>>,
}

impl<B: Backend> Standardise<B> {
    fn new(n: usize, device: &B::Device) -> Self {
        Self {
            mean: Param::from_tensor(Tensor::zeros([n], device)).set_require_grad(false),
            std: Param::from_tensor(Tensor::ones([n], device)).set_require_grad(false),
        }
    }

    /// Replace the statistics (after fitting them on the training set).
    pub fn set(&mut self, mean: &[f32], std: &[f32], device: &B::Device) {
        let n = mean.len();
        self.mean = Param::from_tensor(Tensor::from_data(TensorData::new(mean.to_vec(), [n]), device)).set_require_grad(false);
        self.std = Param::from_tensor(Tensor::from_data(TensorData::new(std.to_vec(), [n]), device)).set_require_grad(false);
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        (x - self.mean.val().unsqueeze::<D>()) / self.std.val().unsqueeze::<D>()
    }
}

#[derive(Module, Debug)]
pub struct BeatNet<B: Backend> {
    pub input_a: Standardise<B>,
    pub input_b: Standardise<B>,
    pub input_g: Standardise<B>,
    pub embed_a: Linear<B>,
    pub embed_b: Linear<B>,
    pub embed_g: Linear<B>,
    pub pos_a: Param<Tensor<B, 2>>,
    pub pos_b: Param<Tensor<B, 2>>,
    pub layers: Vec<EncoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub head_recipe: Option<Linear<B>>,
    pub head_bars: Option<Linear<B>>,
    pub head_lanes: Option<Linear<B>>,
    pub head: Option<Linear<B>>,
    #[module(skip)]
    pub config: BeatConfig,
}

pub enum BeatOut<B: Backend> {
    Cortex {
        /// `(batch, 4)` logits in `RECIPE_ORDER`.
        recipe: Tensor<B, 2>,
        /// `(batch, 2)`: start and swap bars, already `× BARS_SCALE`.
        bars: Tensor<B, 2>,
        /// `(batch, a_len, lanes)` logits (the gains are their sigmoid).
        lanes: Tensor<B, 3>,
    },
    /// `(batch, 1 + issues)`: "good", then every issue.
    Cingulate(Tensor<B, 2>),
}

impl<B: Backend> BeatNet<B> {
    pub fn new(config: &BeatConfig, seed: u64, device: &B::Device) -> Self {
        let mut rng = Rng::new(seed);
        let d = config.dim;
        let plan = config.kind == BeatKind::Cortex;
        Self {
            input_a: Standardise::new(config.a_features(), device),
            input_b: Standardise::new(config.features, device),
            input_g: Standardise::new(config.globals(), device),
            embed_a: linear(config.a_features(), d, true, device),
            embed_b: linear(config.features, d, true, device),
            embed_g: linear(config.globals(), d, true, device),
            pos_a: Param::from_tensor(uniform(&mut rng, [config.a_len, d], 0.02, device)),
            pos_b: Param::from_tensor(uniform(&mut rng, [config.b_len, d], 0.02, device)),
            layers: (0..config.layers)
                .map(|_| EncoderLayer { attn: Attention::new(d, config.heads, device), ff: FeedForward::new(d, config.ff, device) })
                .collect(),
            norm: RmsNorm::new(d, device),
            head_recipe: plan.then(|| linear(d, RECIPES, true, device)),
            head_bars: plan.then(|| linear(d, 2, true, device)),
            head_lanes: plan.then(|| linear(d, config.lanes, true, device)),
            head: (!plan).then(|| linear(d, 1 + config.issues(), true, device)),
            config: config.clone(),
        }
    }

    /// `a (batch, a_len, a_features)`, `b (batch, b_len, features)`, `g (batch, globals)`, all raw.
    pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 3>, g: Tensor<B, 2>) -> BeatOut<B> {
        let c = &self.config;
        let [n, _, _] = a.dims();
        let d = c.dim;
        let tg = self.embed_g.forward(self.input_g.forward(g)).reshape([n, 1, d]);
        let ta = self.embed_a.forward(self.input_a.forward(a)) + self.pos_a.val().unsqueeze::<3>();
        let tb = self.embed_b.forward(self.input_b.forward(b)) + self.pos_b.val().unsqueeze::<3>();
        let mut x = Tensor::cat(vec![tg, ta, tb], 1);
        for layer in &self.layers {
            x = layer.ff.forward(layer.attn.forward(x));
        }
        let x = self.norm.forward(x);
        let global = x.clone().narrow(1, 0, 1).reshape([n, d]);
        match c.kind {
            BeatKind::Cortex => BeatOut::Cortex {
                recipe: self.head_recipe.as_ref().expect("plan").forward(global.clone()),
                bars: self.head_bars.as_ref().expect("plan").forward(global) * sone_infer::config::BARS_SCALE as f64,
                lanes: self.head_lanes.as_ref().expect("plan").forward(x.narrow(1, 1, c.a_len)),
            },
            BeatKind::Cingulate => BeatOut::Cingulate(self.head.as_ref().expect("critic").forward(global)),
        }
    }

    pub fn export(&self) -> Vec<NamedTensor> {
        let mut out = Vec::new();
        for (name, s) in [("a", &self.input_a), ("b", &self.input_b), ("g", &self.input_g)] {
            let (m, sd) = (s.mean.val(), s.std.val());
            out.push((format!("input.{name}_mean"), m.dims().to_vec(), t1(m)));
            out.push((format!("input.{name}_std"), sd.dims().to_vec(), t1(sd)));
        }
        lin(&mut out, "embed_a", &self.embed_a);
        lin(&mut out, "embed_b", &self.embed_b);
        lin(&mut out, "embed_g", &self.embed_g);
        for (name, p) in [("pos_a", &self.pos_a), ("pos_b", &self.pos_b)] {
            let (v, dims) = t2(p.val());
            out.push((name.into(), dims.to_vec(), v));
        }
        for (i, layer) in self.layers.iter().enumerate() {
            attention(&mut out, &format!("layer{i}.attn"), &layer.attn);
            feed_forward(&mut out, &format!("layer{i}.ff"), &layer.ff);
        }
        out.push(("norm".into(), vec![self.config.dim], t1(self.norm.gain.val())));
        for (name, h) in [("head_recipe", &self.head_recipe), ("head_bars", &self.head_bars), ("head_lanes", &self.head_lanes), ("head", &self.head)] {
            if let Some(h) = h {
                lin(&mut out, name, h);
            }
        }
        out
    }
}
