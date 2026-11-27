//! The fast path for DJ transitions: a classic onset envelope over the part
//! of the song the planner needs, a constant-tempo beat grid fitted to it,
//! and Beat This! on one short window only, to decide what a spectral
//! tracker gets wrong — which beat phase is the beat (not the off-beat), the
//! tempo octave and which beat is the "one".
//!
//! - [`OnsetEnvelope`]: spectral flux of log-compressed magnitudes (1024-point
//!   FFT, 20 ms hop, 50 frames per second like the network's) over the span;
//! - [`fit_grid`]: the period and phase that put the grid on the onsets over
//!   the span (a comb search around the network's tempo), with the network's
//!   beat probabilities in its window as the phase vote, and the downbeat
//!   offset by the network's downbeat probabilities.

use super::Fft;

/// Frames per second of the envelope (as the spectrogram's).
pub const FPS: f32 = 50.0;
const N_FFT: usize = 1024;
/// Highest bin used (≈ 8 kHz at 44.1 kHz).
const MAX_HZ: f32 = 8_000.0;

/// Spectral flux, one value per 20 ms frame; frame `t` is centred on `t / 50` s.
pub struct OnsetEnvelope {
    hop: usize,
    bins: usize,
    buf: Vec<f32>,
    /// Absolute index of `buf[0]`.
    base: usize,
    len: usize,
    next: usize,
    window: Vec<f32>,
    fft: Fft,
    frame: Vec<f32>,
    prev: Vec<f32>,
    pub values: Vec<f32>,
}

impl OnsetEnvelope {
    pub fn new(sample_rate: u32) -> Self {
        let hop = (sample_rate as f32 / FPS).round() as usize;
        let bins = ((MAX_HZ / sample_rate as f32 * N_FFT as f32) as usize).min(N_FFT / 2);
        let window = (0..N_FFT)
            .map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        Self {
            hop,
            bins,
            buf: Vec::new(),
            base: 0,
            len: 0,
            next: 0,
            window,
            fft: Fft::new(N_FFT),
            frame: vec![0.0; N_FFT],
            prev: vec![0.0; bins],
            values: Vec::new(),
        }
    }

    pub fn push(&mut self, x: &[f32]) {
        self.buf.extend_from_slice(x);
        self.len += x.len();
        while self.next * self.hop + N_FFT / 2 <= self.len {
            self.compute();
        }
        let keep_from = (self.next * self.hop).saturating_sub(N_FFT / 2);
        if keep_from > self.base + 65_536 {
            self.buf.drain(..keep_from - self.base);
            self.base = keep_from;
        }
    }

    fn compute(&mut self) {
        let first = (self.next * self.hop) as i64 - (N_FFT / 2) as i64;
        let mut frame = std::mem::take(&mut self.frame);
        for (n, v) in frame.iter_mut().enumerate() {
            let i = first + n as i64;
            let s = if i < 0 || i as usize >= self.len { 0.0 } else { self.buf[i as usize - self.base] };
            *v = s * self.window[n];
        }
        let spectrum = self.fft.magnitudes(&frame);
        self.frame = frame;
        let mut flux = 0.0;
        for (k, prev) in self.prev.iter_mut().enumerate().skip(1) {
            let v = (1.0 + 100.0 * spectrum[k]).ln();
            flux += (v - *prev).max(0.0);
            *prev = v;
        }
        self.values.push(if self.next == 0 { 0.0 } else { flux / self.bins as f32 });
        self.next += 1;
    }
}

/// A constant-tempo grid: beat `k` at `phase + k · period` (song seconds);
/// beats with `k ≡ down (mod 4)` are downbeats.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    pub period: f32,
    pub phase: f32,
    pub down: i64,
}

impl Grid {
    /// Beats and downbeats within `[from, to]`.
    pub fn beats(&self, from: f32, to: f32) -> (Vec<f32>, Vec<f32>) {
        let k0 = ((from - self.phase) / self.period).ceil() as i64;
        let k1 = ((to - self.phase) / self.period).floor() as i64;
        let mut beats = Vec::new();
        let mut downs = Vec::new();
        for k in k0..=k1 {
            let t = self.phase + k as f32 * self.period;
            beats.push(t);
            if (k - self.down).rem_euclid(4) == 0 {
                downs.push(t);
            }
        }
        (beats, downs)
    }
}

/// A 50 fps probability series (frame 0 at `t0` s) at time `t`.
pub fn prob_at(series: &[f32], t0: f32, t: f32) -> f32 {
    at(series, t0, t)
}

/// Linear interpolation of a 50 fps series whose frame 0 is at `t0` s.
fn at(series: &[f32], t0: f32, t: f32) -> f32 {
    let x = (t - t0) * FPS;
    if x < 0.0 {
        return 0.0;
    }
    let i = x as usize;
    if i + 1 >= series.len() {
        return 0.0;
    }
    let f = x - i as f32;
    series[i] * (1.0 - f) + series[i + 1] * f
}

/// The network's view of its window: beat and downbeat probabilities per
/// frame, frame 0 at `t0` s.
pub struct Votes<'a> {
    pub t0: f32,
    pub beat: &'a [f32],
    pub down: &'a [f32],
}

/// Weight of the network's beat probabilities against the (normalised)
/// onset envelope when placing the grid.
const NET_WEIGHT: f32 = 2.0;

/// Fit the grid over `span` (song seconds). `env` is the whole envelope
/// (frame 0 at 0 s); `period0` is the starting tempo (from the network's
/// beats), searched within ±4 %. `env`'s frame 0 is at `env_t0` s.
pub fn fit_grid(env: &[f32], env_t0: f32, span: (f32, f32), period0: f32, votes: &Votes, net_beats: &[f32]) -> Grid {
    let (local, t_env) = normalised(env, env_t0, span);
    let net_span = (votes.t0, votes.t0 + votes.beat.len() as f32 / FPS);
    fit(&local, t_env, span, net_span, period0, votes, net_beats)
}

/// The envelope over the span, normalised to mean 1 after removing its
/// local mean (so loud and quiet parts count alike); with its start time.
fn normalised(env: &[f32], env_t0: f32, span: (f32, f32)) -> (Vec<f32>, f32) {
    let f0 = (((span.0 - env_t0) * FPS).max(0.0) as usize).min(env.len());
    let f1 = (((span.1 - env_t0) * FPS).max(0.0) as usize).min(env.len());
    let raw = &env[f0..f1];
    let w = 25; // ±0.5 s
    let mut local: Vec<f32> = (0..raw.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(w), (i + w + 1).min(raw.len()));
            let mean = raw[a..b].iter().sum::<f32>() / (b - a) as f32;
            (raw[i] - mean).max(0.0)
        })
        .collect();
    let mean = local.iter().sum::<f32>() / local.len().max(1) as f32;
    if mean > 0.0 {
        local.iter_mut().for_each(|v| *v /= mean);
    }
    (local, env_t0 + f0 as f32 / FPS)
}

fn fit(local: &[f32], t_env: f32, span: (f32, f32), net_span: (f32, f32), period0: f32, votes: &Votes, net_beats: &[f32]) -> Grid {
    let score = |period: f32, phase: f32| -> f32 {
        let mut s = 0.0;
        let mut n = 0;
        let k0 = ((span.0 - phase) / period).ceil() as i64;
        let k1 = ((span.1 - phase) / period).floor() as i64;
        for k in k0..=k1 {
            let t = phase + k as f32 * period;
            s += at(local, t_env, t);
            n += 1;
        }
        let mut net = 0.0;
        let mut m = 0;
        let k0 = ((net_span.0 - phase) / period).ceil() as i64;
        let k1 = ((net_span.1 - phase) / period).floor() as i64;
        for k in k0..=k1 {
            net += at(votes.beat, votes.t0, phase + k as f32 * period);
            m += 1;
        }
        s / n.max(1) as f32 + NET_WEIGHT * net / m.max(1) as f32
    };

    // The network decides which phase is the beat (a spectral flux often
    // peaks on the off-beat hi-hats): for each period the phase stays within
    // ±40 ms of the network's beats (their circular mean). Coarse: period
    // ±4 % in 0.1 % steps, phase in 5 ms steps.
    let anchor = |period: f32| -> Option<f32> {
        if net_beats.len() < 2 {
            return None;
        }
        let (mut c, mut s) = (0.0f32, 0.0f32);
        for &b in net_beats {
            let a = 2.0 * std::f32::consts::PI * (b / period).fract();
            c += a.cos();
            s += a.sin();
        }
        Some(s.atan2(c).rem_euclid(2.0 * std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * period)
    };
    let mut best = (f32::NEG_INFINITY, period0, 0.0);
    for i in -40..=40 {
        let period = period0 * (1.0 + i as f32 * 0.001);
        let phases: Vec<f32> = match anchor(period) {
            Some(a) => (-8..=8).map(|j| a + j as f32 * 0.005).collect(),
            None => (0..(period / 0.01) as usize).map(|j| j as f32 * 0.01).collect(),
        };
        for phase in phases {
            let s = score(period, phase);
            if s > best.0 {
                best = (s, period, phase);
            }
        }
    }
    // fine: ±0.1 % in 0.01 % steps, ±10 ms in 1 ms steps
    let (_, p_c, ph_c) = best;
    for i in -10..=10 {
        let period = p_c * (1.0 + i as f32 * 0.0001);
        for j in -10..=10 {
            let phase = ph_c + j as f32 * 0.001;
            let s = score(period, phase);
            if s > best.0 {
                best = (s, period, phase);
            }
        }
    }
    let (_, period, phase) = best;
    // the "one": the beat offset whose beats in the window carry the most
    // downbeat probability
    let k0 = ((net_span.0 - phase) / period).ceil() as i64;
    let k1 = ((net_span.1 - phase) / period).floor() as i64;
    let mut votes4 = [0.0f32; 4];
    for k in k0..=k1 {
        votes4[k.rem_euclid(4) as usize] += at(votes.down, votes.t0, phase + k as f32 * period);
    }
    let down = (0..4).max_by(|&a, &b| votes4[a].total_cmp(&votes4[b])).unwrap_or(0) as i64;
    Grid { period, phase, down }
}

/// Median inter-beat interval of beats (s), if there are enough.
pub fn median_period(beats: &[f32]) -> Option<f32> {
    let mut d: Vec<f32> = beats.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 0.2 && d < 1.5).collect();
    if d.len() < 3 {
        return None;
    }
    d.sort_by(f32::total_cmp);
    Some(d[d.len() / 2])
}

/// The tempo from the envelope alone (autocorrelation over the span, with
/// a preference for 120 BPM), when the network found no beats.
pub fn autocorrelation_period(env: &[f32], env_t0: f32, span: (f32, f32)) -> f32 {
    let f0 = (((span.0 - env_t0) * FPS).max(0.0) as usize).min(env.len());
    let f1 = (((span.1 - env_t0) * FPS).max(0.0) as usize).min(env.len());
    let x = &env[f0..f1];
    let mean = x.iter().sum::<f32>() / x.len().max(1) as f32;
    let x: Vec<f32> = x.iter().map(|v| v - mean).collect();
    let mut best = (f32::NEG_INFINITY, 25usize);
    for lag in 15..=60 {
        if lag >= x.len() {
            break;
        }
        let r: f32 = x[lag..].iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() / (x.len() - lag) as f32;
        let bpm = 60.0 * FPS / lag as f32;
        let prior = (-0.5 * ((bpm / 120.0).log2() / 0.9).powi(2)).exp();
        if r * prior > best.0 {
            best = (r * prior, lag);
        }
    }
    best.1 as f32 / FPS
}
