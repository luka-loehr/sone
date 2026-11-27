//! One streaming pass over a song's four stems: everything the analysis needs
//! later, kept small enough for a Worker (about 30 MB for four minutes).
//!
//! The stems arrive in blocks of mono samples (all four the same length). Per
//! hop of 32 samples (0.73 ms) the pass keeps the energy of every band the
//! references look at; any segment's level is then a sum over hops, with the
//! two partial hops at its ends taken pro rata (the error on a 16th note is
//! below 0.01 dB; at cut points given in advance it is exact). The pitched
//! stems go through short-time Fourier transforms folded into a pitch
//! profile at a fifth of a semitone ([`Spectrum`]), so the song's tuning
//! can be applied afterwards.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};
use realfft::num_complex::Complex;

use crate::filter::{Band, Sos};

pub const SR: f64 = 44_100.0;
/// Samples per energy hop.
pub const HOP: usize = 32;
/// Chroma frames: window and hop in samples (186 ms / 23 ms).
pub const N_FFT: usize = 8192;
pub const C_HOP: usize = 1024;
/// The key's frames: below [`F_SPLIT`] a long window resolves the bass
/// notes (1.3 Hz per bin; the short window's 5.4 Hz cannot tell E1 from F1).
pub const N_LONG: usize = 32_768;
pub const K_HOP: usize = 4_096;
pub const F_SPLIT: f64 = 300.0;
/// Pitch bins per octave of the stored profile (a fifth of a semitone).
pub const FINE: usize = 60;
/// Frequencies folded into the pitch profile (librosa's CQT starts at C1).
const F_MIN: f64 = 32.7;
const F_MAX: f64 = 5_000.0;
/// C0 at A = 440 Hz; fine bin 0 is centred on it.
const F_C0: f64 = 16.351_597_831_287_414;

/// Bands whose energy is kept per hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Mix,
    Vocals,
    Drums,
    Bass,
    Other,
    /// Drums below 150 Hz (structure.py's kick).
    Kick,
    /// Drums above 5 kHz.
    Hats,
    /// Drums below 120 Hz (phasecheck.py's kick band).
    Kick120,
    /// Drums 1.5–6 kHz, where claps and snares sit (the backbeat).
    Clap,
    /// Drums above 1.5 kHz: sharp attacks for the grid's phase.
    Attack,
}

pub const LANES: usize = 10;

/// Mono blocks of the four stems, equally long.
pub struct Block<'a> {
    pub vocals: &'a [f32],
    pub drums: &'a [f32],
    pub bass: &'a [f32],
    pub other: &'a [f32],
}

/// What the pass keeps.
#[derive(Debug, Clone)]
pub struct Measurements {
    /// Samples per stem.
    pub len: usize,
    /// Per lane, the energy (sum of squares) of hops `0..h` at `cum[lane][h]`.
    pub cum: Vec<Vec<f64>>,
    /// Per chroma frame (centred on sample `j * C_HOP`), [`FINE`] magnitude
    /// sums of bass + other.
    pub fine_bo: Vec<f32>,
    /// The same for bass + other + vocals.
    pub fine_bov: Vec<f32>,
    /// Per key frame (centred on sample `c * K_HOP`), [`FINE`] values of the
    /// sum of bass, other and vocals: the long window below [`F_SPLIT`], the
    /// short frames around it above.
    pub fine_key: Vec<f32>,
    /// Spectral peaks of bass + other + vocals by their distance from equal
    /// temperament, one bin per cent from −50.
    pub tuning_hist: Vec<f32>,
    /// Sample indices given in advance (sorted) and, per lane, the exact
    /// energy from the start of their hop up to them.
    pub cuts: Vec<usize>,
    pub cut_partial: Vec<[f64; LANES]>,
}

impl Measurements {
    pub fn frames(&self) -> usize {
        self.fine_bo.len() / FINE
    }

    /// Energy of one hop.
    pub fn hop_energy(&self, lane: Lane, h: usize) -> f64 {
        let c = &self.cum[lane as usize];
        if h + 1 < c.len() { c[h + 1] - c[h] } else { 0.0 }
    }

    /// Energy of `lane` in samples `0..x`: exact at the cut points, elsewhere
    /// the partial hop is taken pro rata.
    fn prefix(&self, lane: Lane, x: usize) -> f64 {
        let c = &self.cum[lane as usize];
        let h = x / HOP;
        if h + 1 >= c.len() {
            return c[c.len() - 1];
        }
        if x.is_multiple_of(HOP) {
            return c[h];
        }
        let partial = match self.cuts.binary_search(&x) {
            Ok(i) => self.cut_partial[i][lane as usize],
            Err(_) => {
                let hop_len = (((h + 1) * HOP).min(self.len) - h * HOP) as f64;
                (c[h + 1] - c[h]) * (x - h * HOP) as f64 / hop_len
            }
        };
        c[h] + partial
    }

    /// Mean square of `lane` over samples `a..b` (clipped to the song).
    pub fn mean_square(&self, lane: Lane, a: usize, b: usize) -> Option<f64> {
        let b = b.min(self.len);
        if b <= a {
            return None;
        }
        Some(((self.prefix(lane, b) - self.prefix(lane, a)) / (b - a) as f64).max(0.0))
    }

    /// Level in dB of `lane` over seconds `t0..t1` without the references'
    /// 64-sample floor (−100 when empty).
    pub fn level_db(&self, lane: Lane, t0: f64, t1: f64) -> f64 {
        let (a, b) = seg_bounds(t0, t1, self.len);
        10.0 * (self.mean_square(lane, a, b).unwrap_or(0.0) + 1e-10).log10()
    }

    /// Level in dB of `lane` from `t0` to `t1` seconds, as the references'
    /// `seg_db` (sample bounds `int(t * SR)`, −100 below 64 samples).
    pub fn seg_db(&self, lane: Lane, t0: f64, t1: f64) -> f64 {
        let (a, b) = seg_bounds(t0, t1, self.len);
        if b < a + 64 {
            return -100.0;
        }
        10.0 * (self.mean_square(lane, a, b).unwrap_or(0.0) + 1e-10).log10()
    }

    /// Pitch-class profile (12, C first) of one frame with the song's tuning
    /// applied: fine bins within half a semitone of each (tuned) pitch class.
    pub fn chroma_frame(&self, fine: &[f32], frame: usize, tuning_cents: f64, out: &mut [f64; 12]) {
        *out = [0.0; 12];
        let f = &fine[frame * FINE..(frame + 1) * FINE];
        let per = (FINE / 12) as f64;
        let shift = tuning_cents / 100.0 * per;
        for (i, &v) in f.iter().enumerate() {
            // the nearest tuned pitch class takes the bin
            let pos = ((i as f64 - shift) / per).round().rem_euclid(12.0);
            out[pos as usize % 12] += f64::from(v);
        }
    }

    /// The tuning (cents from A = 440 Hz) that most spectral peaks agree on.
    pub fn tuning_cents(&self) -> f64 {
        let h = &self.tuning_hist;
        // smoothed over ±3 cents: single peaks scatter
        let smooth: Vec<f32> = (0..h.len()).map(|i| (0..7).map(|d| h[(i + h.len() + d - 3) % h.len()]).sum()).collect();
        let best = smooth.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(50, |(i, _)| i);
        best as f64 - 50.0
    }
}

/// Sample bounds of the references' `seg_db(x, t0, t1)`:
/// `int(max(t0, 0) * SR)`, `int(min(t1 * SR, len))`.
pub fn seg_bounds(t0: f64, t1: f64, len: usize) -> (usize, usize) {
    let a = (t0.max(0.0) * SR) as usize;
    let b = (t1 * SR).min(len as f64).max(0.0) as usize;
    (a, b)
}

/// One short-time Fourier transform folded into fine pitch bins: the
/// amplitude in every fifth of a semitone of every octave (the square root
/// of the power there, like a constant-Q filter bank: a partial counts once
/// however many FFT bins its band spans, and noise does not grow with the
/// band's width), each band weighted by 1/sqrt(f) as librosa's CQT does
/// (`scale=True`: by the square root of the filter's length), so the low
/// notes that carry a key's root weigh more.
struct Spectrum {
    n: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    frame_in: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    mags: Vec<f32>,
    /// Per FFT bin: unfolded fine pitch bin and the weight of the next one.
    bin_map: Vec<(usize, usize, f32)>,
    bands: Vec<f32>,
    band_weight: Vec<f32>,
}

impl Spectrum {
    fn new(n: usize, f_lo: f64, f_hi: f64) -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        let window = (0..n).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32).collect();
        let mut bin_map = Vec::new();
        for k in 1..n / 2 {
            let f = k as f64 * SR / n as f64;
            if f < f_lo || f > f_hi {
                continue;
            }
            let pos = FINE as f64 * (f / F_C0).log2();
            let lo = pos.floor();
            bin_map.push((k, lo as usize, (pos - lo) as f32));
        }
        let bands = vec![0.0; bin_map.last().map_or(0, |b| b.1 + 2)];
        let band_weight = (0..bands.len()).map(|i| (F_C0 * 2f64.powf(i as f64 / FINE as f64) / 100.0).powf(-0.5) as f32).collect();
        Self { n, fft, window, frame_in: vec![0.0; n], spectrum, scratch, mags: vec![0.0; n / 2 + 1], bin_map, bands, band_weight }
    }

    /// The frame `samples` (`n` of them) into `out` ([`FINE`] values, added
    /// to), `split`: only bands from that frequency on (0: all).
    fn fold(&mut self, samples: &[f32], out: &mut [f32], split: f64) {
        for ((d, &s), &w) in self.frame_in.iter_mut().zip(samples).zip(&self.window) {
            *d = s * w;
        }
        self.fft.process_with_scratch(&mut self.frame_in, &mut self.spectrum, &mut self.scratch).expect("fft sizes");
        // amplitude units: a Hann window sums to n / 2
        let norm = 2.0 / self.n as f32;
        for (m, c) in self.mags.iter_mut().zip(&self.spectrum) {
            *m = c.norm() * norm;
        }
        self.bands.iter_mut().for_each(|b| *b = 0.0);
        for &(k, lo, frac) in &self.bin_map {
            let p = self.mags[k] * self.mags[k];
            self.bands[lo] += p * (1.0 - frac);
            self.bands[lo + 1] += p * frac;
        }
        let first = if split > 0.0 { (FINE as f64 * (split / F_C0).log2()).ceil() as usize } else { 0 };
        for (i, (&b, &w)) in self.bands.iter().zip(&self.band_weight).enumerate().skip(first) {
            out[i % FINE] += b.sqrt() * w;
        }
    }
}

/// The streaming pass.
pub struct Measurer {
    cuts: Vec<usize>,
    next_cut: usize,
    cut_partial: Vec<[f64; LANES]>,
    filters: [Sos; 5],
    acc: [f64; LANES],
    in_hop: usize,
    cum: Vec<Vec<f64>>,
    len: usize,
    /// Pitched samples (bass + other, and with vocals) from song sample
    /// `buf_start` on (negative: the zeros before the song).
    bo: Vec<f32>,
    bov: Vec<f32>,
    buf_start: i64,
    short: Spectrum,
    long: Spectrum,
    fine_bo: Vec<f32>,
    fine_bov: Vec<f32>,
    /// Key frames: the long window's low part, and the sum and count of the
    /// short frames' high parts that fall into each.
    key_low: Vec<f32>,
    key_high: Vec<f32>,
    key_count: Vec<u32>,
    tuning_hist: Vec<f32>,
    frames: usize,
    long_frames: usize,
    frame_buf: Vec<f32>,
}

impl Default for Measurer {
    fn default() -> Self {
        Self::new()
    }
}

impl Measurer {
    pub fn new() -> Self {
        Self::with_capacity(0.0)
    }

    /// Reserve for a song of about `seconds` (in wasm a grown vector leaves
    /// its old copy behind).
    pub fn with_capacity(seconds: f64) -> Self {
        Self::with_cuts(seconds, Vec::new())
    }

    /// A pass that measures segments bounded by `cuts` (sample indices)
    /// exactly, for a song of about `seconds`.
    pub fn with_cuts(seconds: f64, mut cuts: Vec<usize>) -> Self {
        cuts.sort_unstable();
        cuts.dedup();
        let hops = (seconds * SR / HOP as f64) as usize + 2;
        let frames = (seconds * SR / C_HOP as f64) as usize + 2;
        let key_frames = (seconds * SR / K_HOP as f64) as usize + 2;
        let pad = N_LONG / 2;
        Self {
            cut_partial: Vec::with_capacity(cuts.len()),
            cuts,
            next_cut: 0,
            filters: [
                Sos::butter4(Band::Lowpass(150.0), SR),
                Sos::butter4(Band::Highpass(5_000.0), SR),
                Sos::butter4(Band::Lowpass(120.0), SR),
                Sos::butter4(Band::Bandpass(1_500.0, 6_000.0), SR),
                Sos::butter4(Band::Highpass(1_500.0), SR),
            ],
            acc: [0.0; LANES],
            in_hop: 0,
            cum: (0..LANES)
                .map(|_| {
                    let mut v = Vec::with_capacity(hops + 1);
                    v.push(0.0);
                    v
                })
                .collect(),
            len: 0,
            // centred frames: the first ones reach before the song's start
            bo: vec![0.0; pad],
            bov: vec![0.0; pad],
            buf_start: -(pad as i64),
            short: Spectrum::new(N_FFT, F_MIN, F_MAX),
            long: Spectrum::new(N_LONG, F_MIN, F_SPLIT + 50.0),
            fine_bo: Vec::with_capacity(frames * FINE),
            fine_bov: Vec::with_capacity(frames * FINE),
            key_low: Vec::with_capacity(key_frames * FINE),
            key_high: Vec::with_capacity(key_frames * FINE),
            key_count: Vec::with_capacity(key_frames),
            tuning_hist: vec![0.0; 100],
            frames: 0,
            long_frames: 0,
            frame_buf: vec![0.0; FINE],
        }
    }

    pub fn push(&mut self, block: Block<'_>) {
        let n = block.drums.len();
        assert!(block.vocals.len() == n && block.bass.len() == n && block.other.len() == n, "stem blocks differ in length");
        for i in 0..n {
            let g = self.len + i;
            while self.next_cut < self.cuts.len() && self.cuts[self.next_cut] <= g {
                // (a cut at g: the energy of its hop before sample g)
                self.cut_partial.push(if self.cuts[self.next_cut] == g { self.acc } else { [0.0; LANES] });
                self.next_cut += 1;
            }
            let (v, d, b, o) = (f64::from(block.vocals[i]), f64::from(block.drums[i]), f64::from(block.bass[i]), f64::from(block.other[i]));
            let m = v + d + b + o;
            let kick = self.filters[0].tick(d);
            let hats = self.filters[1].tick(d);
            let k120 = self.filters[2].tick(d);
            let clap = self.filters[3].tick(d);
            let attack = self.filters[4].tick(d);
            for (a, x) in self.acc.iter_mut().zip([m, v, d, b, o, kick, hats, k120, clap, attack]) {
                *a += x * x;
            }
            self.in_hop += 1;
            if self.in_hop == HOP {
                self.flush_hop();
            }
            self.bo.push(block.bass[i] + block.other[i]);
            self.bov.push(block.bass[i] + block.other[i] + block.vocals[i]);
        }
        self.len += n;
        self.frames_until(usize::MAX, usize::MAX);
    }

    fn flush_hop(&mut self) {
        for (c, a) in self.cum.iter_mut().zip(self.acc.iter_mut()) {
            let last = *c.last().expect("starts with 0");
            c.push(last + *a);
            *a = 0.0;
        }
        self.in_hop = 0;
    }

    /// Run every frame whose samples are buffered (at most `short_max` short
    /// and `long_max` long frames in all), then drop what no frame needs.
    fn frames_until(&mut self, short_max: usize, long_max: usize) {
        let end = self.buf_start + self.bo.len() as i64;
        loop {
            let short_end = (self.frames * C_HOP + N_FFT / 2) as i64;
            let long_end = (self.long_frames * K_HOP + N_LONG / 2) as i64;
            let short_ok = self.frames < short_max && short_end <= end;
            let long_ok = self.long_frames < long_max && long_end <= end;
            if short_ok && (!long_ok || short_end <= long_end) {
                self.short_frame();
            } else if long_ok {
                self.long_frame();
            } else {
                break;
            }
        }
        let keep = ((self.frames * C_HOP) as i64 - (N_FFT / 2) as i64).min((self.long_frames * K_HOP) as i64 - (N_LONG / 2) as i64);
        let drop = keep - self.buf_start;
        if drop > 65_536 {
            self.bo.drain(..drop as usize);
            self.bov.drain(..drop as usize);
            self.buf_start += drop;
        }
    }

    fn short_frame(&mut self) {
        let from = ((self.frames * C_HOP) as i64 - (N_FFT / 2) as i64 - self.buf_start) as usize;
        for which in 0..2 {
            let fine = if which == 0 { &mut self.fine_bo } else { &mut self.fine_bov };
            let base = fine.len();
            fine.resize(base + FINE, 0.0);
            let src = if which == 0 { &self.bo } else { &self.bov };
            self.short.fold(&src[from..from + N_FFT], &mut fine[base..], 0.0);
            if which == 1 {
                // the key frame this one falls into takes its part above the split
                self.frame_buf.iter_mut().for_each(|v| *v = 0.0);
                self.short.fold(&self.bov[from..from + N_FFT], &mut self.frame_buf, F_SPLIT);
                let c = (self.frames + 2) / 4;
                while self.key_count.len() <= c {
                    self.key_count.push(0);
                    self.key_high.extend(std::iter::repeat_n(0.0, FINE));
                }
                for (h, &v) in self.key_high[c * FINE..(c + 1) * FINE].iter_mut().zip(&self.frame_buf) {
                    *h += v;
                }
                self.key_count[c] += 1;
                if self.frames.is_multiple_of(2) {
                    collect_peaks(&self.short.mags, &mut self.tuning_hist);
                }
            }
        }
        self.frames += 1;
    }

    fn long_frame(&mut self) {
        let from = ((self.long_frames * K_HOP) as i64 - (N_LONG / 2) as i64 - self.buf_start) as usize;
        let base = self.key_low.len();
        self.key_low.resize(base + FINE, 0.0);
        self.long.fold(&self.bov[from..from + N_LONG], &mut self.key_low[base..], 0.0);
        self.long_frames += 1;
    }

    pub fn finish(mut self) -> Measurements {
        if self.in_hop > 0 {
            self.flush_hop();
        }
        // frames up to the one centred at or before the last sample
        let wanted = self.len.div_ceil(C_HOP).max(1);
        let wanted_long = self.len.div_ceil(K_HOP).max(1);
        self.bo.extend(std::iter::repeat_n(0.0, N_LONG));
        self.bov.extend(std::iter::repeat_n(0.0, N_LONG));
        self.frames_until(wanted, wanted_long);
        let mut fine_key = self.key_low;
        for (c, &count) in self.key_count.iter().enumerate().take(fine_key.len() / FINE) {
            if count > 0 {
                for (k, &h) in fine_key[c * FINE..(c + 1) * FINE].iter_mut().zip(&self.key_high[c * FINE..(c + 1) * FINE]) {
                    *k += h / count as f32;
                }
            }
        }
        let mut cuts = self.cuts;
        cuts.truncate(self.cut_partial.len());
        Measurements {
            len: self.len,
            cum: self.cum,
            fine_bo: self.fine_bo,
            fine_bov: self.fine_bov,
            fine_key,
            tuning_hist: self.tuning_hist,
            cuts,
            cut_partial: self.cut_partial,
        }
    }
}

/// Peaks of the pitched spectrum (short window) between 80 Hz and 2.5 kHz,
/// interpolated, counted by their deviation from equal temperament (as
/// librosa's `estimate_tuning` does with `piptrack`).
fn collect_peaks(mags: &[f32], hist: &mut [f32]) {
    let k0 = (80.0 * N_FFT as f64 / SR) as usize;
    let k1 = (2_500.0 * N_FFT as f64 / SR) as usize;
    let max = mags[k0..k1].iter().copied().fold(0.0f32, f32::max);
    if max <= 1e-9 {
        return;
    }
    for k in k0..k1 {
        let (a, b, c) = (mags[k - 1], mags[k], mags[k + 1]);
        if b < 0.1 * max || b <= a || b < c {
            continue;
        }
        let den = a - 2.0 * b + c;
        let off = if den.abs() > 1e-12 { 0.5 * (a - c) / den } else { 0.0 };
        let f = (k as f64 + f64::from(off)) * SR / N_FFT as f64;
        let semis = 12.0 * (f / 440.0).log2();
        let dev = semis - semis.round();
        let bin = ((dev * 100.0 + 50.0).floor() as isize).clamp(0, 99) as usize;
        hist[bin] += 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(f: f64, secs: f64, amp: f32) -> Vec<f32> {
        (0..(secs * SR) as usize).map(|i| amp * (2.0 * std::f64::consts::PI * f * i as f64 / SR).sin() as f32).collect()
    }

    #[test]
    fn levels_and_pitch_of_a_sine() {
        let v = sine(440.0 * 2f64.powf(0.2 / 12.0), 3.0, 0.5); // A, 20 cents sharp
        let z = vec![0.0; v.len()];
        let mut m = Measurer::new();
        for i in (0..v.len()).step_by(5_000) {
            let j = (i + 5_000).min(v.len());
            m.push(Block { vocals: &v[i..j], drums: &z[i..j], bass: &z[i..j], other: &z[i..j] });
        }
        let m = m.finish();
        // a sine of amplitude 0.5 has a mean square of 1/8
        let db = m.seg_db(Lane::Vocals, 0.5, 2.5);
        assert!((db - 10.0 * (0.125f64).log10()).abs() < 0.01, "{db}");
        assert!((m.seg_db(Lane::Drums, 0.5, 2.5) + 100.0).abs() < 1e-9);
        let t = m.tuning_cents();
        assert!((t - 20.0).abs() <= 3.0, "{t}");
        let mut c = [0.0; 12];
        m.chroma_frame(&m.fine_bov, m.frames() / 2, t, &mut c);
        let best = (0..12).max_by(|&a, &b| c[a].total_cmp(&c[b])).unwrap();
        assert_eq!(best, 9, "{c:?}");
    }
}
