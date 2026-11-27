//! The Ear network in Burn (see `sone_infer::tympanum`).

use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::activation::{gelu, sigmoid};
use sone_infer::TympanumConfig;
use sone_infer::config::TYMPANUM_FRAME_HEADS;
use sone_infer::random::Rng;
use sone_infer::st::NamedTensor;

use crate::export::{lin, t1, time_block, transposed};
use crate::layers::{TimeBlock, linear};

/// One direction of a GRU layer, PyTorch's equations (gates r, z, n).
#[derive(Module, Debug)]
pub struct GruDir<B: Backend> {
    pub ih: Linear<B>,
    pub hh: Linear<B>,
}

#[derive(Module, Debug)]
pub struct GruLayer<B: Backend> {
    pub fwd: GruDir<B>,
    pub bwd: GruDir<B>,
}

#[derive(Module, Debug)]
pub struct TympanumNet<B: Backend> {
    pub in_scale: Param<Tensor<B, 1>>,
    pub in_shift: Param<Tensor<B, 1>>,
    pub proj: Linear<B>,
    pub convs: Vec<TimeBlock<B>>,
    pub grus: Vec<GruLayer<B>>,
    pub frame_head: Linear<B>,
    pub key_head: Linear<B>,
    pub tuning_head: Linear<B>,
    #[module(skip)]
    pub config: TympanumConfig,
}

pub struct TympanumOut<B: Backend> {
    /// `(batch, frames, 5)` logits of `TYMPANUM_FRAME_HEADS`.
    pub frames: Tensor<B, 3>,
    /// `(batch, 24)` key logits.
    pub key: Tensor<B, 2>,
    /// `(batch, 1)` tuning head before `TUNING_RANGE · tanh`.
    pub tuning: Tensor<B, 2>,
}

impl<B: Backend> GruDir<B> {
    fn new(input: usize, hidden: usize, device: &B::Device) -> Self {
        Self { ih: linear(input, 3 * hidden, true, device), hh: linear(hidden, 3 * hidden, true, device) }
    }

    /// `x (batch, time, input)` → `(batch, time, hidden)`, run forwards or backwards in time.
    fn forward(&self, x: Tensor<B, 3>, reverse: bool) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let hidden = self.hh.weight.val().dims()[0];
        let xp = self.ih.forward(x);
        let mut h = Tensor::<B, 2>::zeros([b, hidden], &xp.device());
        let mut outs: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
        for step in 0..t {
            let at = if reverse { t - 1 - step } else { step };
            let xt = xp.clone().narrow(1, at, 1).reshape([b, 3 * hidden]);
            let hp = self.hh.forward(h.clone());
            let r = sigmoid(xt.clone().narrow(1, 0, hidden) + hp.clone().narrow(1, 0, hidden));
            let z = sigmoid(xt.clone().narrow(1, hidden, hidden) + hp.clone().narrow(1, hidden, hidden));
            let n = (xt.narrow(1, 2 * hidden, hidden) + r * hp.narrow(1, 2 * hidden, hidden)).tanh();
            h = (z.clone().neg() + 1.0) * n + z * h;
            outs.push(h.clone().reshape([b, 1, hidden]));
        }
        if reverse {
            outs.reverse();
        }
        Tensor::cat(outs, 1)
    }
}

impl<B: Backend> TympanumNet<B> {
    pub fn new(config: &TympanumConfig, seed: u64, device: &B::Device) -> Self {
        let mut rng = Rng::new(seed);
        let (c, h, mels) = (config.channels, config.gru_hidden, config.n_mels);
        Self {
            in_scale: Param::from_tensor(Tensor::ones([mels], device)),
            in_shift: Param::from_tensor(Tensor::zeros([mels], device)),
            proj: linear(mels, c, true, device),
            convs: config.dilations.iter().map(|&d| TimeBlock::new(c, 2 * c, config.kernel, d, &mut rng, device)).collect(),
            grus: (0..config.gru_layers)
                .map(|l| {
                    let input = if l == 0 { c } else { 2 * h };
                    GruLayer { fwd: GruDir::new(input, h, device), bwd: GruDir::new(input, h, device) }
                })
                .collect(),
            frame_head: linear(2 * h, TYMPANUM_FRAME_HEADS.len(), true, device),
            key_head: linear(2 * h, 24, true, device),
            tuning_head: linear(2 * h, 1, true, device),
            config: config.clone(),
        }
    }

    /// `logmel (batch, frames, n_mels)`.
    pub fn forward(&self, logmel: Tensor<B, 3>) -> TympanumOut<B> {
        let [b, t, mels] = logmel.dims();
        let x = logmel * self.in_scale.val().reshape([1, 1, mels]) + self.in_shift.val().reshape([1, 1, mels]);
        let mut x = gelu(self.proj.forward(x));
        let c = x.dims()[2];
        let mut x4 = x.reshape([b, t, 1, c]);
        for conv in &self.convs {
            x4 = conv.forward(x4);
        }
        x = x4.reshape([b, t, c]);
        for layer in &self.grus {
            let f = layer.fwd.forward(x.clone(), false);
            let r = layer.bwd.forward(x, true);
            x = Tensor::cat(vec![f, r], 2);
        }
        let pooled = x.clone().mean_dim(1).reshape([b, x.dims()[2]]);
        TympanumOut { frames: self.frame_head.forward(x), key: self.key_head.forward(pooled.clone()), tuning: self.tuning_head.forward(pooled) }
    }

    pub fn export(&self) -> Vec<NamedTensor> {
        let mut out = Vec::new();
        let mels = self.config.n_mels;
        out.push(("in_scale".into(), vec![mels], t1(self.in_scale.val())));
        out.push(("in_shift".into(), vec![mels], t1(self.in_shift.val())));
        lin(&mut out, "proj", &self.proj);
        for (i, conv) in self.convs.iter().enumerate() {
            time_block(&mut out, &format!("conv{i}"), conv);
        }
        for (l, layer) in self.grus.iter().enumerate() {
            for (name, dir) in [("fwd", &layer.fwd), ("bwd", &layer.bwd)] {
                let p = format!("gru{l}.{name}");
                out.push(transposed(&format!("{p}.w_ih"), dir.ih.weight.val()));
                out.push(transposed(&format!("{p}.w_hh"), dir.hh.weight.val()));
                let bi = dir.ih.bias.as_ref().expect("bias").val();
                let bh = dir.hh.bias.as_ref().expect("bias").val();
                out.push((format!("{p}.b_ih"), bi.dims().to_vec(), t1(bi)));
                out.push((format!("{p}.b_hh"), bh.dims().to_vec(), t1(bh)));
            }
        }
        lin(&mut out, "frame_head", &self.frame_head);
        lin(&mut out, "key_head", &self.key_head);
        lin(&mut out, "tuning_head", &self.tuning_head);
        out
    }
}
