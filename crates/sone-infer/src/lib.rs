//! The four anatomical ear-and-brain networks of **Sone-6.5M** without any ML
//! runtime: plain loops and `matrixmultiply::sgemm`, weights read in place from
//! a `.safetensors` file (F16 or F32) and widened layer by layer into reused
//! scratch buffers. Builds for `wasm32-unknown-unknown` unchanged.
//!
//! - [`cochlea::Cochlea`]: tonotopic 42-band stem separator (`vocals`, `drums`, `bass`, residual `other`).
//! - [`tympanum::Tympanum`]: 50 Hz rhythmic, vocal-breath, phrase, key and tuning sensor.
//! - [`beat::BeatNet`]: [`BeatKind::Cortex`] (executive transition planner) and
//!   [`BeatKind::Cingulate`] (conflict & 11-defect monitor).
//!
//! All four organs ship inside a single unified `model.safetensors` bundle
//! loaded zero-copy via [`Sone::load`].

pub mod beat;
pub mod chunk;
pub mod cochlea;
pub mod config;
pub mod dsp;
pub mod layers;
pub mod ops;
pub mod random;
pub mod st;
pub mod tympanum;

pub use beat::{BeatNet, BeatWorkspace, CingulateOutput, CortexOutput};
pub use cochlea::{Cochlea, CochleaWorkspace};
pub use config::{BeatConfig, BeatKind, CochleaConfig, TympanumConfig};
pub use st::{WeightFile, merge_bundle};
pub use tympanum::{Tympanum, TympanumOutput, TympanumWorkspace};

/// All four neural organs of **Sone-6.5M** loaded zero-copy from a single unified
/// `.safetensors` file (`cochlea.*`, `tympanum.*`, `cortex.*`, `cingulate.*`).
pub struct Sone {
    pub cochlea: Cochlea,
    pub tympanum: Tympanum,
    pub cortex: BeatNet,
    pub cingulate: BeatNet,
    bytes_len: usize,
}

impl Sone {
    pub const ORGANS: [&'static str; 4] = ["cochlea", "tympanum", "cortex", "cingulate"];

    pub fn load(file: WeightFile) -> anyhow::Result<Self> {
        let bytes_len = file.byte_len();
        let mut parts = file.split_bundle(&Self::ORGANS)?;
        Ok(Self {
            cochlea: Cochlea::load(parts.remove("cochlea").unwrap())?,
            tympanum: Tympanum::load(parts.remove("tympanum").unwrap())?,
            cortex: BeatNet::load(parts.remove("cortex").unwrap())?,
            cingulate: BeatNet::load(parts.remove("cingulate").unwrap())?,
            bytes_len,
        })
    }

    pub fn weights_bytes(&self) -> usize {
        self.bytes_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cochlea_net(half: bool) -> Cochlea {
        let mut c = CochleaConfig::size("small").unwrap();
        c.blocks = 2;
        c.dilations = vec![1, 2];
        let bytes = random::file("cochlea", &c, &c.tensor_shapes(), 3, half).unwrap();
        Cochlea::load(WeightFile::from_vec(bytes).unwrap()).unwrap()
    }

    #[test]
    fn cochlea_stems_sum_to_mixture_and_chunking_agrees() {
        let net = cochlea_net(true);
        let hop = net.config.hop;
        let len = hop * 64;
        let l: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();
        let r: Vec<f32> = (0..len).map(|i| (i as f32 * 0.023).cos() * 0.2).collect();
        let mut ws = CochleaWorkspace::default();
        let mut out = vec![[Vec::new(), Vec::new()]; 4];
        net.separate_chunk(&mut ws, &l, &r, &mut out);
        for i in 0..len {
            let sum: f32 = out.iter().map(|o| o[0][i]).sum();
            assert!((sum - l[i]).abs() < 1e-5);
        }
        assert!(out[0][0].iter().any(|v| v.abs() > 1e-4), "a random network does something");
        let mut whole = vec![[Vec::new(), Vec::new()]; 4];
        net.separate(&mut ws, &l, &r, len, hop * 8, &mut whole);
        for s in 0..4 {
            for i in 0..len {
                assert!((whole[s][1][i] - out[s][1][i]).abs() < 1e-5);
            }
        }
        net.separate(&mut ws, &l, &r, hop * 24, hop * 8, &mut whole);
        for i in 0..len {
            let sum: f32 = whole.iter().map(|o| o[1][i]).sum();
            assert!((sum - r[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn cochlea_v2_studio_time_attn_four_stems_sums_exact() {
        let mut c = CochleaConfig::size("v2-base").unwrap();
        c.blocks = 2;
        c.dilations = vec![1, 2];
        assert_eq!(c.bands.len(), 64);
        assert!(c.time_attn && c.four_stems && c.mask_bound > 0.0);
        let bytes = random::file("cochlea", &c, &c.tensor_shapes(), 7, true).unwrap();
        let net = Cochlea::load(WeightFile::from_vec(bytes).unwrap()).unwrap();
        let hop = net.config.hop;
        let len = hop * 16;
        let l: Vec<f32> = (0..len).map(|i| (i as f32 * 0.013).sin() * 0.4).collect();
        let r: Vec<f32> = (0..len).map(|i| (i as f32 * 0.029).cos() * 0.3).collect();
        let mut ws = CochleaWorkspace::default();
        let mut out = vec![[Vec::new(), Vec::new()]; 4];
        net.separate(&mut ws, &l, &r, hop * 8, hop * 2, &mut out);
        for i in 0..len {
            let sl: f32 = out.iter().map(|o| o[0][i]).sum();
            let sr: f32 = out.iter().map(|o| o[1][i]).sum();
            assert!((sl - l[i]).abs() < 1e-5 && (sr - r[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn tympanum_cortex_and_cingulate_run() {
        let c = TympanumConfig::default();
        let tym = Tympanum::load(WeightFile::from_vec(random::file("tympanum", &c, &c.tensor_shapes(), 1, true).unwrap()).unwrap()).unwrap();
        let mono: Vec<f32> = (0..44_100).map(|i| (i as f32 * 0.05).sin()).collect();
        let out = tym.analyze(&mut TympanumWorkspace::default(), &mono);
        assert_eq!(out.frames.len(), 44_100 / 882 + 1);
        assert!((out.key.iter().sum::<f32>() - 1.0).abs() < 1e-4);

        for cfg in [BeatConfig::cortex(), BeatConfig::cingulate()] {
            let model = if cfg.kind == BeatKind::Cortex { "cortex" } else { "cingulate" };
            let net = BeatNet::load(WeightFile::from_vec(random::file(model, &cfg, &cfg.tensor_shapes(), 2, false).unwrap()).unwrap()).unwrap();
            let a = vec![0.5; cfg.a_len * cfg.a_features()];
            let b = vec![-0.5; cfg.b_len * cfg.features];
            let g: Vec<f32> = (0..cfg.globals()).map(|i| i as f32).collect();
            let raw = net.forward_raw(&mut BeatWorkspace::default(), &a, &b, &g);
            let want = if cfg.kind == BeatKind::Cortex { 4 + 2 + 80 * 8 } else { 1 + sone_core::Issue::ALL.len() };
            assert_eq!(raw.len(), want);
            assert!(raw.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn unified_sone_bundle_loads_all_four_organs_zero_copy() {
        let mut sc = CochleaConfig::size("small").unwrap();
        sc.blocks = 2;
        sc.dilations = vec![1, 2];
        let ec = TympanumConfig::default();
        let pc = BeatConfig::cortex();
        let cc = BeatConfig::cingulate();

        let sb = random::file("cochlea", &sc, &sc.tensor_shapes(), 10, true).unwrap();
        let eb = random::file("tympanum", &ec, &ec.tensor_shapes(), 11, true).unwrap();
        let pb = random::file("cortex", &pc, &pc.tensor_shapes(), 12, true).unwrap();
        let cb = random::file("cingulate", &cc, &cc.tensor_shapes(), 13, true).unwrap();

        let bundle_bytes = merge_bundle(
            &[("cochlea", &sb), ("tympanum", &eb), ("cortex", &pb), ("cingulate", &cb)],
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let sone = Sone::load(WeightFile::from_vec(bundle_bytes).unwrap()).unwrap();

        let solo_cortex = BeatNet::load(WeightFile::from_vec(pb).unwrap()).unwrap();
        let a = vec![0.25; pc.a_len * pc.a_features()];
        let b = vec![-0.25; pc.b_len * pc.features];
        let g = vec![0.1; pc.globals()];
        let r1 = sone.cortex.forward_raw(&mut BeatWorkspace::default(), &a, &b, &g);
        let r2 = solo_cortex.forward_raw(&mut BeatWorkspace::default(), &a, &b, &g);
        assert_eq!(r1, r2);
    }
}
