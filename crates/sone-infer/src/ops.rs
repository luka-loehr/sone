//! The handful of primitives every network here is made of. Each mirrors the
//! Burn op of the same name in `sone-models` exactly (same epsilon, same erf
//! GELU), so the parity tests can hold to 1e-4.

/// Epsilon of every RMSNorm except the band-input one (`BAND_EPS`).
pub const EPS: f32 = 1e-5;
/// The band input spans a large dynamic range (quiet high bands of quiet
/// songs), so its norm gets a far smaller floor.
pub const BAND_EPS: f32 = 1e-10;

/// `C (m×n) = A·B + beta·C` with explicit strides (row stride, column stride),
/// bounds-checked before the unsafe call.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn gemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    rsa: usize,
    csa: usize,
    b: &[f32],
    rsb: usize,
    csb: usize,
    beta: f32,
    c: &mut [f32],
    rsc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(k == 0 || (m - 1) * rsa + (k - 1) * csa < a.len(), "gemm: A out of range");
    assert!(k == 0 || (k - 1) * rsb + (n - 1) * csb < b.len(), "gemm: B out of range");
    assert!((m - 1) * rsc + n <= c.len(), "gemm: C out of range");
    // SAFETY: the asserts above keep every index of A, B and C inside its slice.
    unsafe {
        matrixmultiply::sgemm(
            m, k, n, 1.0,
            a.as_ptr(), rsa as isize, csa as isize,
            b.as_ptr(), rsb as isize, csb as isize,
            beta, c.as_mut_ptr(), rsc as isize, 1,
        );
    }
}

/// `out (rows, n) = x (rows, k; row stride xs) · wᵀ (w: (n, k)) + bias`.
/// `out` has row stride `os` (≥ n) and is written, not accumulated.
#[allow(clippy::too_many_arguments)]
pub fn linear(x: &[f32], rows: usize, k: usize, xs: usize, w: &[f32], n: usize, bias: Option<&[f32]>, out: &mut [f32], os: usize) {
    for r in 0..rows {
        let row = &mut out[r * os..r * os + n];
        match bias {
            Some(b) => row.copy_from_slice(b),
            None => row.fill(0.0),
        }
    }
    gemm(rows, k, n, x, xs, 1, w, 1, k, 1.0, out, os);
}

/// As [`linear`] into a fresh `(rows, n)` vector.
pub fn linear_into(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize, bias: Option<&[f32]>, out: &mut Vec<f32>) {
    out.clear();
    out.resize(rows * n, 0.0);
    linear(x, rows, k, k, w, n, bias, out, n);
}

/// `x += y · wᵀ + bias` for `x (rows, n)`, `y (rows, k)`.
pub fn linear_add(y: &[f32], rows: usize, k: usize, w: &[f32], n: usize, bias: Option<&[f32]>, x: &mut [f32]) {
    if let Some(b) = bias {
        for row in x[..rows * n].chunks_exact_mut(n) {
            for (v, b) in row.iter_mut().zip(b) {
                *v += b;
            }
        }
    }
    gemm(rows, k, n, y, k, 1, w, 1, k, 1.0, x, n);
}

/// RMSNorm per row: `x / sqrt(mean(x²) + eps) · gain`.
pub fn rms_norm_rows(x: &mut [f32], dim: usize, gain: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(dim) {
        let k = 1.0 / (sum_squares(row) / dim as f32 + eps).sqrt();
        for (v, g) in row.iter_mut().zip(gain) {
            *v *= k * g;
        }
    }
}

/// [`rms_norm_rows`] from `src` (row stride `ss`) into `dst` (dense).
pub fn rms_norm_copy(src: &[f32], rows: usize, dim: usize, ss: usize, gain: &[f32], eps: f32, dst: &mut [f32]) {
    for r in 0..rows {
        let row = &src[r * ss..r * ss + dim];
        let k = 1.0 / (sum_squares(row) / dim as f32 + eps).sqrt();
        for ((d, v), g) in dst[r * dim..(r + 1) * dim].iter_mut().zip(row).zip(gain) {
            *d = v * k * g;
        }
    }
}

pub fn sum_squares(x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (chunks, rest) = x.as_chunks::<8>();
    for c in chunks {
        for i in 0..8 {
            acc[i] += c[i] * c[i];
        }
    }
    acc.iter().sum::<f32>() + rest.iter().map(|v| v * v).sum::<f32>()
}

/// GELU, the erf form (Burn's `gelu`): `x/2 · (1 + erf(x/√2))`.
pub fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2));
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// erf, branch-free so it vectorises: Abramowitz & Stegun 7.1.26
/// (|error| < 1.5e-7, in f32 < 5e-7).
#[inline(always)]
pub fn erf(x: f32) -> f32 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let y = 1.0
        - t * (0.254_829_6 + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))))
            * fast_exp(-a * a);
    y.copysign(x)
}

/// `row = exp(row - max(row))`; returns the sum (the caller divides, after
/// the product with V: `head_dim` divisions per row instead of `seq`).
pub fn softmax_unnormalised(row: &mut [f32]) -> f32 {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = fast_exp(*v - max);
        sum += *v;
    }
    sum
}

/// Softmax of a small vector, in place.
pub fn softmax(row: &mut [f32]) {
    let sum = softmax_unnormalised(row);
    row.iter_mut().for_each(|v| *v /= sum);
}

/// `exp` for x ≤ 0 (softmax, erf), branch-free so it vectorises; ~1 ulp.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.max(-87.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    // Cody-Waite: ln 2 in two parts
    let r = x - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_67 + r * (0.041_666_668 + r * (0.008_333_452 + r * (0.001_388_949_4 + r * 0.000_198_527_62))))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// Round to nearest even.
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // subnormal
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = u32::from(rest > mid || (rest == mid && half & 1 == 1));
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    let round = u32::from(rest > 0x1000 || (rest == 0x1000 && half & 1 == 1));
    sign | (half + round) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000,
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips() {
        for h in 0..=u16::MAX {
            let v = f16_to_f32(h);
            if v.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(v), h, "{h:#x}");
        }
        assert_eq!(f32_to_f16(1.0 + 1.0 / 4096.0), 0x3c00);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
    }

    #[test]
    fn exp_and_erf_are_accurate() {
        for i in 0..2000 {
            let x = -(i as f32) * 0.04;
            let (a, b) = (fast_exp(x), x.exp());
            assert!((a - b).abs() <= b * 2e-6 + 1e-38, "{x}: {a} vs {b}");
        }
        // erf(1) and erf(0.5) against their known values
        assert!((erf(1.0) - 0.842_700_8).abs() < 5e-7);
        assert!((erf(-0.5) + 0.520_499_9).abs() < 5e-7);
    }

    #[test]
    fn linear_matches_loops() {
        let (rows, k, n) = (3, 5, 4);
        let x: Vec<f32> = (0..rows * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut out = Vec::new();
        linear_into(&x, rows, k, &w, n, Some(&b), &mut out);
        for r in 0..rows {
            for o in 0..n {
                let want = b[o] + (0..k).map(|i| x[r * k + i] * w[o * k + i]).sum::<f32>();
                assert!((out[r * n + o] - want).abs() < 1e-5);
            }
        }
    }
}
