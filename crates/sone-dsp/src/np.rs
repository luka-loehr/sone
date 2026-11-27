//! The few numpy / scipy.ndimage primitives the renderer leans on, with their
//! exact conventions (window placement, edge modes, rounding, summation order).
//! The reference makes discrete decisions from these values (arg-maxima, bar
//! boundaries, rounding to 0.1 dB), so "close" is not always enough.

/// numpy's pairwise summation (the order `np.sum` adds a contiguous array in).
pub fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0;
        for &v in a {
            r += v;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
    }
}

/// Same for float32 arrays: numpy accumulates them in float32.
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f32;
        for &v in a {
            r += v;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// `np.mean(np.square(x))` of a float32 array, in float32 like numpy.
pub fn mean_square_f32(x: &[f32]) -> f32 {
    let sq: Vec<f32> = x.iter().map(|v| v * v).collect();
    pairwise_sum_f32(&sq) / x.len() as f32
}

/// `np.mean(np.square(x))` of a float64 array.
pub fn mean_square_f64(x: &[f64]) -> f64 {
    let sq: Vec<f64> = x.iter().map(|v| v * v).collect();
    pairwise_sum_f64(&sq) / x.len() as f64
}

/// Python's `round()` and `np.round`: halves go to the even neighbour.
pub fn round_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// `int(round(x))`.
pub fn iround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Index `j` of the last breakpoint with `xp[j] <= x` (`searchsorted(side="right") - 1`).
fn last_le(xp: &[f64], x: f64) -> isize {
    xp.partition_point(|&p| p <= x) as isize - 1
}

/// `np.interp(x, xp, fp)` for one value.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    if x < xp[0] {
        return fp[0];
    }
    let j = last_le(xp, x);
    if j < 0 {
        return fp[0];
    }
    let j = j as usize;
    if j >= n - 1 {
        return fp[n - 1];
    }
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let r = slope * (x - xp[j]) + fp[j];
    if r.is_nan() { slope * (x - xp[j + 1]) + fp[j + 1] } else { r }
}

/// `np.searchsorted(xp, x, side="right") - 1`.
pub fn segment_index(xp: &[f64], x: f64) -> isize {
    last_le(xp, x)
}

/// Edge handling of scipy.ndimage filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// `d c b a | a b c d | d c b a` (scipy's default).
    Reflect,
    /// `a a a a | a b c d | d d d d`.
    Nearest,
}

fn edge_index(i: isize, n: usize, edge: Edge) -> usize {
    let n = n as isize;
    if (0..n).contains(&i) {
        return i as usize;
    }
    match edge {
        Edge::Nearest => i.clamp(0, n - 1) as usize,
        Edge::Reflect => {
            // Half-sample symmetric, periodic with 2n.
            let p = 2 * n;
            let mut k = i.rem_euclid(p);
            if k >= n {
                k = p - 1 - k;
            }
            k as usize
        }
    }
}

/// First input index of the window of output 0: scipy centres a window of
/// `size` on `size / 2`, and `origin` shifts it to the left.
fn window_start(size: usize, origin: isize) -> isize {
    -((size / 2) as isize) - origin
}

/// `scipy.ndimage.uniform_filter1d(x, size, mode, origin=0)`: a running mean that
/// scipy keeps in double precision, updated by one sample per step.
pub fn uniform_filter1d(x: &[f64], size: usize, edge: Edge) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let size = size.max(1);
    let s0 = window_start(size, 0);
    let at = |i: isize| x[edge_index(i, n, edge)];
    let len = size as f64;
    let mut out = Vec::with_capacity(n);
    let mut tmp = 0.0;
    for k in 0..size as isize {
        tmp += at(s0 + k);
    }
    tmp /= len;
    out.push(tmp);
    for i in 1..n as isize {
        tmp += (at(s0 + i + size as isize - 1) - at(s0 + i - 1)) / len;
        out.push(tmp);
    }
    out
}

/// `scipy.ndimage.minimum_filter1d(x, size, mode, origin)` (monotone deque, O(n)).
pub fn minimum_filter1d(x: &[f32], size: usize, edge: Edge, origin: isize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let s0 = window_start(size, origin);
    let at = |i: isize| x[edge_index(i, n, edge)];
    let mut out = Vec::with_capacity(n);
    let mut dq: std::collections::VecDeque<(isize, f32)> = std::collections::VecDeque::new();
    // Window of output i covers [s0 + i, s0 + i + size - 1].
    let mut next = s0;
    for i in 0..n as isize {
        let hi = s0 + i + size as isize - 1;
        while next <= hi {
            let v = at(next);
            while dq.back().is_some_and(|&(_, b)| b >= v) {
                dq.pop_back();
            }
            dq.push_back((next, v));
            next += 1;
        }
        while dq.front().is_some_and(|&(j, _)| j < s0 + i) {
            dq.pop_front();
        }
        out.push(dq.front().unwrap().1);
    }
    out
}

/// `np.median` of a float64 slice.
pub fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// `np.linspace(start, stop, num)`.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let step = (stop - start) / div;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_window_matches_scipy() {
        // scipy: an impulse at 10 reaches outputs 8..=13 for size 6, 8..=12 for size 5.
        let mut x = vec![0.0; 20];
        x[10] = 1.0;
        let nz = |v: Vec<f64>| v.iter().enumerate().filter(|(_, a)| a.abs() > 1e-12).map(|(i, _)| i).collect::<Vec<_>>();
        assert_eq!(nz(uniform_filter1d(&x, 6, Edge::Reflect)), (8..=13).collect::<Vec<_>>());
        assert_eq!(nz(uniform_filter1d(&x, 5, Edge::Reflect)), (8..=12).collect::<Vec<_>>());
        let z: Vec<f64> = (0..6).map(f64::from).collect();
        let r = uniform_filter1d(&z, 3, Edge::Reflect);
        assert!((r[0] - 1.0 / 3.0).abs() < 1e-12 && (r[5] - 14.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn minimum_window_matches_scipy() {
        let mut y = vec![1.0f32; 20];
        y[10] = 0.0;
        let zeros = |v: Vec<f32>| v.iter().enumerate().filter(|(_, a)| **a == 0.0).map(|(i, _)| i).collect::<Vec<_>>();
        assert_eq!(zeros(minimum_filter1d(&y, 6, Edge::Reflect, -2)), (6..=11).collect::<Vec<_>>());
        assert_eq!(zeros(minimum_filter1d(&y, 8, Edge::Reflect, -3)), (4..=11).collect::<Vec<_>>());
        assert_eq!(zeros(minimum_filter1d(&y, 5, Edge::Reflect, 0)), (8..=12).collect::<Vec<_>>());
    }

    #[test]
    fn interp_edges() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [0.0, 10.0, 0.0];
        assert_eq!(interp(-1.0, &xp, &fp), 0.0);
        assert_eq!(interp(0.5, &xp, &fp), 5.0);
        assert_eq!(interp(2.0, &xp, &fp), 0.0);
        assert_eq!(interp(3.0, &xp, &fp), 0.0);
    }

    #[test]
    fn pairwise_sum_is_a_sum() {
        let v: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect();
        assert_eq!(pairwise_sum_f64(&v), 0.5 * 999.0 * 1000.0 / 2.0);
    }
}
