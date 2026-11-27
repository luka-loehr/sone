//! Parity: every Burn network, exported to safetensors, gives the same
//! output in `sone-infer` (1e-4 relative with F32 weights, 1e-2 with F16),
//! and exports exactly the names and shapes the reader expects.

use burn::backend::NdArray;
use burn::prelude::*;
use sone_infer::config::{BeatConfig, TympanumConfig, CochleaConfig};
use sone_infer::random::Rng;
use sone_infer::{BeatNet as InferBeat, BeatWorkspace, Cochlea, CochleaWorkspace, Tympanum, TympanumWorkspace, WeightFile};
use sone_models::export::{check_names, to_safetensors, values};
use sone_models::{BeatNet, BeatOut, TympanumNet, Stft, CochleaNet};

type B = NdArray<f32>;

fn rel_err(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    want.iter().zip(got).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

fn random(rng: &mut Rng, n: usize, a: f32) -> Vec<f32> {
    (0..n).map(|_| rng.sym(a)).collect()
}

fn check(name: &str, want: &[f32], f32_out: &[f32], f16_out: &[f32]) {
    let (e32, e16) = (rel_err(want, f32_out), rel_err(want, f16_out));
    println!("{name}: relative error f32 {e32:.2e}, f16 {e16:.2e}");
    assert!(e32 < 1e-4, "{name}: f32 parity {e32}");
    assert!(e16 < 1e-2, "{name}: f16 parity {e16}");
}

#[test]
fn cochlea_parity() {
    let device = Default::default();
    for size in ["small", "base"] {
        let mut c = CochleaConfig::size(size).unwrap();
        c.blocks = 3;
        c.dilations = vec![1, 2, 4];
        let net = CochleaNet::<B>::new(&c, 5, &device);
        let tensors = net.export();
        check_names(&tensors, &c.tensor_shapes()).unwrap();
        assert_eq!(c.num_params(), net.num_params(), "parameter count");

        let len = c.hop * 40;
        let mut rng = Rng::new(11);
        let (l, r) = (random(&mut rng, len, 0.5), random(&mut rng, len, 0.5));
        let mut both = l.clone();
        both.extend_from_slice(&r);
        let mix = Tensor::<B, 3>::from_data(TensorData::new(both, [1, 2, len]), &device);
        let stft = Stft::<B>::new(c.n_fft, c.hop, &device);
        let want = values(net.forward(&stft, mix).wave);

        let mut outs = Vec::new();
        for half in [false, true] {
            let bytes = to_safetensors("cochlea", &c, &tensors, half).unwrap();
            let infer = Cochlea::load(WeightFile::from_vec(bytes).unwrap()).unwrap();
            let mut out = vec![[Vec::new(), Vec::new()]; 4];
            infer.separate_chunk(&mut CochleaWorkspace::default(), &l, &r, &mut out);
            outs.push(out.iter().flat_map(|s| s.iter().flatten().copied()).collect::<Vec<f32>>());
        }
        // "other" is mixture minus three: compare the three predicted stems
        let n3 = 3 * 2 * len;
        check(&format!("stems {size}"), &want[..n3], &outs[0][..n3], &outs[1][..n3]);
    }
}

#[test]
fn tympanum_parity() {
    let device = Default::default();
    let c = TympanumConfig::default();
    let net = TympanumNet::<B>::new(&c, 3, &device);
    let tensors = net.export();
    check_names(&tensors, &c.tensor_shapes()).unwrap();
    let t = 120;
    let mut rng = Rng::new(4);
    let mel: Vec<f32> = (0..t * c.n_mels).map(|_| rng.uniform() * 6.0).collect();
    let out = net.forward(Tensor::from_data(TensorData::new(mel.clone(), [1, t, c.n_mels]), &device));
    let mut want = values(out.frames);
    want.extend(values(out.key));
    want.extend(values(out.tuning));
    let mut got = Vec::new();
    for half in [false, true] {
        let tympanum = Tympanum::load(WeightFile::from_vec(to_safetensors("tympanum", &c, &tensors, half).unwrap()).unwrap()).unwrap();
        let (mut f, k, tu) = tympanum.logits(&mut TympanumWorkspace::default(), &mel);
        f.extend(k);
        f.push(tu);
        got.push(f);
    }
    check("ear", &want, &got[0], &got[1]);
}

#[test]
fn cortex_and_cingulate_parity() {
    let device = Default::default();
    for c in [BeatConfig::cortex(), BeatConfig::cingulate()] {
        let mut net = BeatNet::<B>::new(&c, 8, &device);
        let mut rng = Rng::new(21);
        let fa = c.a_features();
        // non-trivial standardisation, as after fitting
        let mean: Vec<f32> = random(&mut rng, fa, 2.0);
        let std: Vec<f32> = (0..fa).map(|_| 0.5 + rng.uniform()).collect();
        net.input_a.set(&mean, &std, &device);
        let gn = c.globals();
        net.input_g.set(&random(&mut rng, gn, 50.0), &vec![7.0; gn], &device);
        let tensors = net.export();
        check_names(&tensors, &c.tensor_shapes()).unwrap();
        assert_eq!(c.num_params(), net.num_params(), "parameter count");

        let a = random(&mut rng, c.a_len * fa, 3.0);
        let b = random(&mut rng, c.b_len * c.features, 3.0);
        let g: Vec<f32> = [128.0f32, 124.0, 1.0, 2.0, 3.5, 1.2, -45.0, 8.0][..gn].to_vec();
        let out = net.forward(
            Tensor::from_data(TensorData::new(a.clone(), [1, c.a_len, fa]), &device),
            Tensor::from_data(TensorData::new(b.clone(), [1, c.b_len, c.features]), &device),
            Tensor::from_data(TensorData::new(g.clone(), [1, gn]), &device),
        );
        let (model, want) = match out {
            BeatOut::Cortex { recipe, bars, lanes } => {
                let mut v = values(recipe);
                v.extend(values(bars));
                v.extend(values(burn::tensor::activation::sigmoid(lanes)));
                ("cortex", v)
            }
            BeatOut::Cingulate(logits) => ("cingulate", values(logits)),
        };
        let mut got = Vec::new();
        for half in [false, true] {
            let infer = InferBeat::load(WeightFile::from_vec(to_safetensors(model, &c, &tensors, half).unwrap()).unwrap()).unwrap();
            got.push(infer.forward_raw(&mut BeatWorkspace::default(), &a, &b, &g));
        }
        check(model, &want, &got[0], &got[1]);
    }
}

#[test]
fn stft_matches_infer() {
    let device = Default::default();
    let (n, hop, len) = (512, 128, 128 * 30);
    let mut rng = Rng::new(2);
    let (l, r) = (random(&mut rng, len, 1.0), random(&mut rng, len, 1.0));
    let mut both = l.clone();
    both.extend_from_slice(&r);
    let stft = Stft::<B>::new(n, hop, &device);
    let spec = stft.forward(Tensor::from_data(TensorData::new(both.clone(), [1, 2, len]), &device));
    let mut want = Vec::new();
    sone_infer::dsp::Stft::new(n, hop).forward_stereo(&l, &r, &mut want);
    let got = values(spec.clone());
    assert!(rel_err(&want, &got) < 1e-5, "{}", rel_err(&want, &got));
    let back = values(stft.inverse(spec, len));
    assert!(rel_err(&both, &back) < 1e-5, "{}", rel_err(&both, &back));
}
