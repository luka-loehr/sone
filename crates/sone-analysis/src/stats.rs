//! The few numpy routines the references use, with numpy's conventions.

/// `np.median`.
pub fn median(v: &[f64]) -> f64 {
    percentile(v, 50.0)
}

/// `np.percentile` with linear interpolation.
pub fn percentile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = q / 100.0 * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

/// Least-squares line `y = a·x + b` (`np.polyfit(x, y, 1)`); `(a, b)`.
pub fn lstsq_line(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let a = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (a, my - a * mx)
}

/// Python's `round(v, digits)` (ties are too rare in measured values to matter).
pub fn round_to(v: f64, digits: i32) -> f64 {
    let p = 10f64.powi(digits);
    (v * p).round() / p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_conventions() {
        assert_eq!(median(&[3.0, 1.0, 2.0, 10.0]), 2.5);
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 90.0) - 4.6).abs() < 1e-12);
        let (a, b) = lstsq_line(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((a - 2.0).abs() < 1e-12 && (b - 1.0).abs() < 1e-12);
        assert_eq!(round_to(-12.345_6, 1), -12.3);
    }
}
