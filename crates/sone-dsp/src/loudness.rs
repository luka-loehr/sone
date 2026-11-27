//! Integrated loudness, ITU-R BS.1770-4, computed the way pyloudnorm does it:
//! K-weighting from the RBJ cookbook formulas (high shelf +4 dB at 1500 Hz,
//! high-pass at 38 Hz), 400 ms blocks every 100 ms, an absolute gate at
//! -70 LUFS and a relative gate 10 LU below the absolutely gated mean.

use crate::np::pairwise_sum_f32;

fn rbj(kind: &str, g_db: f64, q: f64, fc: f64, rate: f64) -> ([f64; 3], [f64; 3]) {
    let a = 10f64.powf(g_db / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * (fc / rate);
    let alpha = w0.sin() / (2.0 * q);
    let c = w0.cos();
    let (b, a_) = match kind {
        "high_shelf" => (
            [
                a * ((a + 1.0) + (a - 1.0) * c + 2.0 * a.sqrt() * alpha),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
                a * ((a + 1.0) + (a - 1.0) * c - 2.0 * a.sqrt() * alpha),
            ],
            [
                (a + 1.0) - (a - 1.0) * c + 2.0 * a.sqrt() * alpha,
                2.0 * ((a - 1.0) - (a + 1.0) * c),
                (a + 1.0) - (a - 1.0) * c - 2.0 * a.sqrt() * alpha,
            ],
        ),
        _ => ([(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0], [1.0 + alpha, -2.0 * c, 1.0 - alpha]),
    };
    let a0 = a_[0];
    ([b[0] / a0, b[1] / a0, b[2] / a0], [1.0, a_[1] / a0, a_[2] / a0])
}

/// `scipy.signal.lfilter` of one biquad, stored back as float32 like pyloudnorm's copy of the input.
fn lfilter_f32(b: [f64; 3], a: [f64; 3], x: &mut [f32]) {
    let (mut z0, mut z1) = (0.0f64, 0.0f64);
    for v in x.iter_mut() {
        let xi = f64::from(*v);
        let y = b[0] * xi + z0;
        z0 = b[1] * xi - a[1] * y + z1;
        z1 = b[2] * xi - a[2] * y;
        *v = y as f32;
    }
}

/// Integrated loudness in LUFS of a stereo signal at `rate`. `-inf` for silence.
pub fn integrated_loudness(x: &[[f32; 2]], rate: f64) -> f64 {
    let n = x.len();
    let mut ch: [Vec<f32>; 2] = [x.iter().map(|v| v[0]).collect(), x.iter().map(|v| v[1]).collect()];
    let (bs, as_) = rbj("high_shelf", 4.0, 1.0 / 2f64.sqrt(), 1500.0, rate);
    let (bh, ah) = rbj("high_pass", 0.0, 0.5, 38.0, rate);
    for c in ch.iter_mut() {
        lfilter_f32(bs, as_, c);
    }
    for c in ch.iter_mut() {
        lfilter_f32(bh, ah, c);
    }
    let t_g = 0.4;
    let step = 1.0 - 0.75;
    let t = n as f64 / rate;
    let num_blocks = (((t - t_g) / (t_g * step)).round_ties_even() as i64 + 1).max(0) as usize;
    let mut z = [vec![0.0f64; num_blocks], vec![0.0f64; num_blocks]];
    let mut sq: Vec<f32> = Vec::new();
    for (i, c) in ch.iter().enumerate() {
        for (j, zj) in z[i].iter_mut().enumerate() {
            let l = (t_g * (j as f64 * step) * rate) as usize;
            let u = ((t_g * (j as f64 * step + 1.0) * rate) as usize).min(n);
            sq.clear();
            sq.extend(c[l.min(u)..u].iter().map(|v| v * v));
            // numpy: a python float times a float32 scalar stays float32
            *zj = f64::from((1.0 / (t_g * rate)) as f32 * pairwise_sum_f32(&sq));
        }
    }
    let l: Vec<f64> = (0..num_blocks).map(|j| -0.691 + 10.0 * (z[0][j] + z[1][j]).log10()).collect();
    let gated_mean = |sel: &dyn Fn(f64) -> bool| -> [f64; 2] {
        let js: Vec<usize> = (0..num_blocks).filter(|&j| sel(l[j])).collect();
        let m = |c: &Vec<f64>| crate::np::pairwise_sum_f64(&js.iter().map(|&j| c[j]).collect::<Vec<_>>()) / js.len() as f64;
        [m(&z[0]), m(&z[1])]
    };
    let za = gated_mean(&|v| v >= -70.0);
    let gamma_r = -0.691 + 10.0 * (za[0] + za[1]).log10() - 10.0;
    let zr = gated_mean(&|v| v > gamma_r && v > -70.0);
    let zr = zr.map(|v| if v.is_nan() { 0.0 } else { v });
    -0.691 + 10.0 * (zr[0] + zr[1]).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, amp: f64, secs: f64) -> Vec<[f32; 2]> {
        let n = (secs * 48000.0) as usize;
        (0..n)
            .map(|i| {
                let v = (amp * (std::f64::consts::TAU * freq * i as f64 / 48000.0).sin()) as f32;
                [v, v]
            })
            .collect()
    }

    // Reference values: pyloudnorm 0.1.1 (Meter(44100).integrated_loudness) on the same float32 signals.
    #[test]
    fn matches_pyloudnorm_on_synthetic_signals() {
        let sr = 44100.0;
        let n = 10 * 44100;
        let t = |i: usize| i as f64 / sr;
        let a: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let v = (0.5 * (std::f64::consts::TAU * 997.0 * t(i)).sin()) as f32;
                [v, v]
            })
            .collect();
        let tone = |i: usize, k: f64| {
            [
                (k * 0.3 * (std::f64::consts::TAU * 100.0 * t(i)).sin()) as f32,
                (k * 0.2 * (std::f64::consts::TAU * 5000.0 * t(i)).sin()) as f32,
            ]
        };
        let mut b: Vec<[f32; 2]> = (0..n).map(|i| tone(i, 1.0)).collect();
        b.extend(vec![[0.0f32; 2]; 5 * 44100]);
        // numpy: 0.01 * b[:3 s] on the float32 array
        b.extend((0..3 * 44100).map(|i| {
            let v = tone(i, 1.0);
            [0.01f32 * v[0], 0.01f32 * v[1]]
        }));
        let la = integrated_loudness(&a, sr);
        let lb = integrated_loudness(&b, sr);
        assert!((la - -6.062719366490407).abs() < 1e-3, "{la}");
        assert!((lb - -11.500079532644808).abs() < 1e-3, "{lb}");
    }

    #[test]
    fn level_and_gates() {
        let a = integrated_loudness(&sine(1000.0, 0.1, 10.0), 48000.0);
        let b = integrated_loudness(&sine(1000.0, 0.01, 10.0), 48000.0);
        assert!((a - b - 20.0).abs() < 1e-6);
        // Long silence does not count (absolute gate); only the blocks half over the edge pull it down a little.
        let mut x = sine(1000.0, 0.1, 10.0);
        x.extend(vec![[0.0f32; 2]; 480000]);
        assert!((integrated_loudness(&x, 48000.0) - a).abs() < 0.1);
        // A much quieter part is dropped by the relative gate.
        let mut y = sine(1000.0, 0.1, 10.0);
        y.extend(sine(1000.0, 0.001, 10.0));
        assert!((integrated_loudness(&y, 48000.0) - a).abs() < 0.1);
    }
}
