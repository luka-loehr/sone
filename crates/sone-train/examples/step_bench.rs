//! Where a Stems training step spends its time, per backend:
//! `cargo run -p sone-train --release --example step_bench -- <wgpu|wgpu-ckpt|ndarray> [size] [batch] [seconds]`
//! (`wgpu-ckpt`: activations recomputed in the backward pass where cheap).

use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use sone_infer::CochleaConfig;
use sone_infer::random::Rng;
use sone_models::{Stft, CochleaNet};
use sone_train::losses::{CochleaLossConfig, cochlea_loss};

fn bench<B: AutodiffBackend>(device: B::Device, size: &str, batch: usize, seconds: f64) {
    let c = CochleaConfig::size(size).unwrap();
    let len = sone_train::run::samples(seconds, c.hop);
    let mut model = CochleaNet::<B>::new(&c, 1, &device);
    let mut optim = AdamWConfig::new().init::<B, CochleaNet<B>>();
    let stft = Stft::<B>::new(c.n_fft, c.hop, &device);
    let mr: Vec<Stft<B>> = [512, 1024, 2048].iter().map(|&n| Stft::new(n, n / 4, &device)).collect();
    let lc = CochleaLossConfig { spec_l1: 1.0, wave_l1: 1.0, multires: 1.0, multires_ffts: vec![512, 1024, 2048], leak: 1.0, leak_voice_weight: 4.0, silent_db: -30.0 };
    let mut rng = Rng::new(3);
    let stems: Vec<f32> = (0..batch * 4 * 2 * len).map(|_| rng.sym(0.2)).collect();
    let mut mix = vec![0.0f32; batch * 2 * len];
    for b in 0..batch {
        for s in 0..4 {
            for i in 0..2 * len {
                mix[b * 2 * len + i] += stems[(b * 4 + s) * 2 * len + i];
            }
        }
    }
    let sync = |t: Tensor<B, 1>| t.into_scalar();
    for step in 0..8 {
        let mixt = Tensor::<B, 3>::from_data(TensorData::new(mix.clone(), [batch, 2, len]), &device);
        let tgt = Tensor::<B, 4>::from_data(TensorData::new(stems.clone(), [batch, 4, 2, len]), &device);
        let t0 = Instant::now();
        let out = model.forward(&stft, mixt);
        let _ = sync(out.wave.clone().sum().reshape([1]));
        let t1 = Instant::now();
        let loss = cochlea_loss(&out, tgt, &stft, &mr, &lc);
        let total = loss.total(&lc);
        let _ = sync(total.clone());
        let t2 = Instant::now();
        let grads = GradientsParams::from_grads(total.backward(), &model);
        let t3 = Instant::now();
        model = optim.step(1e-4, model, grads);
        let _ = sync(model.band_emb.val().sum().reshape([1]));
        let t4 = Instant::now();
        println!(
            "step {step}: forward {:.3} s, loss {:.3} s, backward {:.3} s, optimizer {:.3} s, total {:.3} s",
            (t1 - t0).as_secs_f64(),
            (t2 - t1).as_secs_f64(),
            (t3 - t2).as_secs_f64(),
            (t4 - t3).as_secs_f64(),
            (t4 - t0).as_secs_f64()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let size = args.get(1).map(String::as_str).unwrap_or("small");
    let batch = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(4);
    let seconds = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(2.0);
    match args.first().map(String::as_str) {
        Some("wgpu-ckpt") => bench::<Autodiff<Wgpu, BalancedCheckpointing>>(WgpuDevice::default(), size, batch, seconds),
        Some("ndarray") => bench::<Autodiff<NdArray>>(Default::default(), size, batch, seconds),
        _ => bench::<Autodiff<Wgpu>>(WgpuDevice::default(), size, batch, seconds),
    }
}
