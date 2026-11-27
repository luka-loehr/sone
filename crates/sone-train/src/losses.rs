//! Loss terms. Every STFT-domain term is divided by √n_fft so that
//! spectrogram and waveform terms have comparable size (a Hann-windowed
//! STFT scales a signal by about √(3N/8) in energy terms).

use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use sone_models::{Stft, CochleaOut};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CochleaLossConfig {
    /// L1 on the complex spectrogram of all four stems.
    pub spec_l1: f64,
    /// L1 on the four waveforms.
    pub wave_l1: f64,
    /// L1 on STFT magnitudes at several resolutions.
    pub multires: f64,
    pub multires_ffts: Vec<usize>,
    /// Energy a stem puts where its target is silent (see [`leak`]).
    pub leak: f64,
    /// Weight of leak cells in drums, bass and other where the voice is
    /// present: voice bleeding into the instrumentals is the defect heard.
    pub leak_voice_weight: f64,
    /// A stem is "silent" in a cell when it is this far below the mixture there (dB).
    pub silent_db: f64,
}

/// The terms of one batch, as scalar tensors (each already weighted by 1).
pub struct StemsLoss<B: Backend> {
    pub spec_l1: Tensor<B, 1>,
    pub wave_l1: Tensor<B, 1>,
    pub multires: Tensor<B, 1>,
    pub leak: Tensor<B, 1>,
}

impl<B: Backend> StemsLoss<B> {
    pub fn total(&self, c: &CochleaLossConfig) -> Tensor<B, 1> {
        self.spec_l1.clone() * c.spec_l1 + self.wave_l1.clone() * c.wave_l1 + self.multires.clone() * c.multires + self.leak.clone() * c.leak
    }

    pub fn terms(&self) -> [Tensor<B, 1>; 4] {
        [self.spec_l1.clone(), self.wave_l1.clone(), self.multires.clone(), self.leak.clone()]
    }
}

pub const COCHLEA_TERMS: [&str; 4] = ["spec_l1", "wave_l1", "multires", "leak"];

/// `target (batch, 4, 2, len)` are the true stems (vocals, drums, bass, other).
pub fn cochlea_loss<B: Backend>(out: &CochleaOut<B>, target: Tensor<B, 4>, stft: &Stft<B>, multires: &[Stft<B>], c: &CochleaLossConfig) -> StemsLoss<B> {
    let [b, s, ch, len] = target.dims();
    let target_spec = stft.forward(target.clone().reshape([b * s, ch, len]));
    let [_, t, f, _] = target_spec.dims();
    let target_spec = target_spec.reshape([b, s, t, f, 2 * ch]);
    let norm = 1.0 / (stft.n_fft as f64).sqrt();

    // 1. complex spectrogram L1
    let spec_l1 = (out.spec.clone() - target_spec.clone()).abs().mean() * norm;

    // 2. waveform L1
    let wave_l1 = (out.wave.clone() - target.clone()).abs().mean();

    // 3. multi-resolution magnitude L1, averaged over the resolutions
    let est = out.wave.clone().reshape([b * s, ch, len]);
    let tgt = target.reshape([b * s, ch, len]);
    let mut mr = Tensor::<B, 1>::zeros([1], &est.device());
    for m in multires {
        let d = (m.magnitude(est.clone()) - m.magnitude(tgt.clone())).abs().mean();
        mr = mr + d * (1.0 / (m.n_fft as f64).sqrt());
    }
    let multires = mr / multires.len().max(1) as f64;

    // 4. leak
    let leak = leak(&out.spec, &target_spec, &out.mix_spec, c);
    StemsLoss { spec_l1, wave_l1, multires, leak }
}

/// Energy per time-frequency cell, summed over channels: `(…, 2·ch)` → `(…)`.
fn cell_energy<B: Backend, const D: usize, const E: usize>(spec: Tensor<B, D>) -> Tensor<B, E> {
    let dims = spec.dims();
    let mut out = [0usize; E];
    out.copy_from_slice(&dims[..E]);
    spec.powi_scalar(2).sum_dim(D - 1).reshape(out)
}

/// Leak: for every stem, the energy its estimate has in cells where its
/// target is silent (more than `silent_db` below the mixture in that cell,
/// or below a floor of 1e-6 of the chunk's mean mixture energy), as a
/// fraction of the chunk's mixture energy. In drums, bass and other the
/// cells where the voice is present count `leak_voice_weight` times: energy
/// there is most likely voice that should have gone to the vocal stem.
/// The masks come from the targets only, so no gradient flows through them.
pub fn leak<B: Backend>(est: &Tensor<B, 5>, target: &Tensor<B, 5>, mix: &Tensor<B, 4>, c: &CochleaLossConfig) -> Tensor<B, 1> {
    let [b, s, t, f, _] = est.dims();
    let e_est: Tensor<B, 4> = cell_energy::<B, 5, 4>(est.clone());
    let e_tgt: Tensor<B, 4> = cell_energy::<B, 5, 4>(target.clone().detach());
    let e_mix: Tensor<B, 3> = cell_energy::<B, 4, 3>(mix.clone().detach());
    let mean_mix = e_mix.clone().mean_dim(2).mean_dim(1); // (b, 1, 1)
    let tau = 10f64.powf(c.silent_db / 10.0);
    let threshold = (e_mix.clone() * tau + mean_mix.clone() * 1e-6).reshape([b, 1, t, f]);
    let silent = e_tgt.clone().lower(threshold.clone().expand([b, s, t, f])).float();
    let voice = e_tgt.narrow(1, 0, 1).greater_equal(threshold).float(); // (b, 1, t, f)
    // weight 1 everywhere, plus (w - 1) on voiced cells of stems 1.. (not the vocal stem itself)
    let is_instr = Tensor::<B, 1>::from_data(TensorData::new((0..s).map(|i| if i == 0 { 0.0f32 } else { 1.0 }).collect::<Vec<_>>(), [s]), &est.device())
        .reshape([1, s, 1, 1]);
    let weight = voice * is_instr * (c.leak_voice_weight - 1.0) + 1.0;
    let leaked = (e_est * silent * weight).sum_dim(3).sum_dim(2).sum_dim(1).reshape([b]);
    let total = e_mix.sum_dim(2).sum_dim(1).reshape([b]) + 1e-8;
    (leaked / total).mean()
}

/// Binary cross-entropy with logits, soft targets allowed, `pos_weight` on
/// the positive term, per element (reduce yourself).
pub fn bce_with_logits<B: Backend, const D: usize>(x: Tensor<B, D>, y: Tensor<B, D>, pos_weight: f64) -> Tensor<B, D> {
    // log(1 + e^{-|x|}) + max(-x, 0) = -log σ(x)
    let log_sig_neg = (x.clone().abs().neg().exp() + 1.0).log() + x.clone().neg().clamp_min(0.0);
    let one_minus_y = y.clone().neg() + 1.0;
    one_minus_y * x + (y * (pos_weight - 1.0) + 1.0) * log_sig_neg
}

/// Cross-entropy of logits `(n, classes)` against one-hot rows `(n, classes)`
/// (all-zero rows contribute nothing): per row.
pub fn cross_entropy<B: Backend>(logits: Tensor<B, 2>, one_hot: Tensor<B, 2>) -> Tensor<B, 1> {
    let [n, _] = logits.dims();
    (log_softmax(logits, 1) * one_hot).sum_dim(1).neg().reshape([n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn bce_matches_formula() {
        let d = Default::default();
        let x = Tensor::<B, 1>::from_floats([-3.0, 0.0, 2.0], &d);
        let y = Tensor::<B, 1>::from_floats([0.0, 1.0, 0.5], &d);
        let got = bce_with_logits(x, y, 2.0).into_data().to_vec::<f32>().unwrap();
        let s = |v: f32| 1.0 / (1.0 + (-v).exp());
        let want = [-(1.0 - s(-3.0)).ln(), -2.0 * s(0.0).ln(), -(2.0 * 0.5 * s(2.0).ln() + 0.5 * (1.0 - s(2.0)).ln())];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
    }
}
