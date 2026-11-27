//! Convolution reverb (wet only) with a decaying stereo noise impulse response.
//!
//! The reference draws its noise from numpy's `default_rng(7)`, which Rust
//! cannot reproduce, so the two responses the specs use (rt60 2.4 s and 2.6 s)
//! are embedded as exported by `reference/python/export_ir.py`; any other rt60
//! gets a response built the same way from a Rust generator.

use std::sync::Arc;

use realfft::RealFftPlanner;

use crate::SR;
use crate::filter::{Band, butter};

static IR_2_4: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/reverb_ir_2.4.f32"));
static IR_2_6: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/reverb_ir_2.6.f32"));

fn decode(bytes: &[u8]) -> Vec<[f32; 2]> {
    bytes
        .as_chunks::<8>().0.iter()
        .map(|c| {
            [f32::from_le_bytes([c[0], c[1], c[2], c[3]]), f32::from_le_bytes([c[4], c[5], c[6], c[7]])]
        })
        .collect()
}

/// The impulse response for `rt60` seconds (stereo, 1.3 × rt60 long).
pub fn impulse_response(rt60: f64) -> Arc<Vec<[f32; 2]>> {
    if rt60 == 2.4 {
        Arc::new(decode(IR_2_4))
    } else if rt60 == 2.6 {
        Arc::new(decode(IR_2_6))
    } else {
        Arc::new(generate_ir(rt60, 7))
    }
}

/// SplitMix64 feeding a Box-Muller transform: any decent white Gaussian noise will do.
struct Noise {
    state: u64,
    spare: Option<f64>,
}

impl Noise {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let (a, b) = (self.uniform(), self.uniform());
        let r = (-2.0 * a.ln()).sqrt();
        let t = std::f64::consts::TAU * b;
        self.spare = Some(r * t.sin());
        r * t.cos()
    }
}

/// Same recipe as the reference: noise under an exponential decay reaching -60 dB at
/// rt60, the first 12 ms silent (pre-delay), band-passed 300-6500 Hz, the louder
/// channel normalised to unit energy.
pub fn generate_ir(rt60: f64, seed: u64) -> Vec<[f32; 2]> {
    let n = (rt60 * 1.3 * SR) as usize;
    let mut rng = Noise { state: seed, spare: None };
    let pre = (0.012 * SR) as usize;
    let mut ch = [vec![0.0f64; n], vec![0.0f64; n]];
    for i in 0..n {
        let env = (-6.91 * (i as f64 / SR) / rt60).exp();
        for c in ch.iter_mut() {
            c[i] = rng.gauss() * env * if i < pre { 0.0 } else { 1.0 };
        }
    }
    let sos = butter(2, Band::Bandpass(300.0, 6500.0), SR);
    let ch = [sos.filter(&ch[0]), sos.filter(&ch[1])];
    let norm = ch.iter().map(|c| c.iter().map(|v| v * v).sum::<f64>().sqrt()).fold(0.0, f64::max);
    (0..n).map(|i| [(ch[0][i] / norm) as f32, (ch[1][i] / norm) as f32]).collect()
}

/// `fftconvolve(x, h)[:len(x)]` by overlap-add, in double precision.
pub fn convolve(x: &[f64], h: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0f64; n];
    if n == 0 || h.is_empty() {
        return out;
    }
    let m = h.len();
    let fft_len = (2 * m).next_power_of_two().max(4096);
    let block = fft_len - m + 1;
    let mut planner = RealFftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(fft_len);
    let inv = planner.plan_fft_inverse(fft_len);
    let mut hbuf = fwd.make_input_vec();
    hbuf[..m].copy_from_slice(h);
    let mut hspec = fwd.make_output_vec();
    fwd.process(&mut hbuf, &mut hspec).expect("fft");
    let mut xbuf = fwd.make_input_vec();
    let mut xspec = fwd.make_output_vec();
    let mut ybuf = inv.make_output_vec();
    let scale = 1.0 / fft_len as f64;
    let mut start = 0;
    while start < n {
        let len = block.min(n - start);
        xbuf.fill(0.0);
        xbuf[..len].copy_from_slice(&x[start..start + len]);
        fwd.process(&mut xbuf, &mut xspec).expect("fft");
        for (a, b) in xspec.iter_mut().zip(&hspec) {
            *a *= *b;
        }
        inv.process(&mut xspec, &mut ybuf).expect("ifft");
        let end = (start + len + m - 1).min(n);
        for (i, o) in out[start..end].iter_mut().enumerate() {
            *o += ybuf[i] * scale;
        }
        start += len;
    }
    out
}

/// Reverb of a stereo send with the response for `rt60`, cut to the input length.
pub fn reverb(x: &[[f32; 2]], rt60: f64) -> Vec<[f32; 2]> {
    let ir = impulse_response(rt60);
    reverb_with(x, &ir)
}

pub fn reverb_with(x: &[[f32; 2]], ir: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let n = x.len();
    let mut out = vec![[0.0f32; 2]; n];
    // Only the span where the send is open needs convolving; the rest of the output is its tail or silence.
    let Some(first) = x.iter().position(|v| v[0] != 0.0 || v[1] != 0.0) else {
        return out;
    };
    for c in 0..2 {
        let xs: Vec<f64> = x[first..].iter().map(|v| f64::from(v[c])).collect();
        let h: Vec<f64> = ir.iter().map(|v| f64::from(v[c])).collect();
        let y = convolve(&xs, &h);
        for (i, v) in y.into_iter().enumerate() {
            out[first + i][c] = v as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolve_matches_direct() {
        let x: Vec<f64> = (0..5000).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect();
        let h: Vec<f64> = (0..300).map(|i| ((i * 31) % 17) as f64 / 8.0 - 1.0).collect();
        let y = convolve(&x, &h);
        for i in [0, 1, 299, 300, 2047, 4999] {
            let d: f64 = (0..=i.min(299)).map(|k| h[k] * x[i - k]).sum();
            assert!((y[i] - d).abs() < 1e-9, "{i}");
        }
    }

    #[test]
    fn embedded_responses_have_the_reference_shape() {
        for rt in [2.4, 2.6] {
            let ir = impulse_response(rt);
            assert_eq!(ir.len(), (rt * 1.3 * SR) as usize);
            assert!(ir[..529].iter().all(|v| v[0] == 0.0 && v[1] == 0.0));
            let e: f64 = ir.iter().map(|v| f64::from(v[0]).powi(2)).sum::<f64>();
            let e1: f64 = ir.iter().map(|v| f64::from(v[1]).powi(2)).sum::<f64>();
            assert!((e.max(e1) - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn generated_response_decays_like_rt60() {
        let ir = generate_ir(2.0, 7);
        let e = |a: usize, b: usize| ir[a..b].iter().map(|v| f64::from(v[0]).powi(2)).sum::<f64>();
        let s = 44100 / 10;
        // 1 s later the energy is 30 dB down (60 dB per rt60 of 2 s).
        let drop = 10.0 * (e(s, 2 * s) / e(11 * s, 12 * s)).log10();
        assert!((drop - 30.0).abs() < 3.0, "{drop}");
    }
}
