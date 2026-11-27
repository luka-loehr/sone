//! Phase-vocoder engine: every frequency keeps its own phase continuity while
//! all events stay on the ideal time line.
//!
//! * STFT with 4096-sample Hann frames every 512 input samples; each frame is
//!   placed on the output at the ideal time of its centre, so any event lands
//!   within a fraction of a millisecond of `t × ratio`.
//! * Identity phase locking (Laroche & Dolson): spectral peaks advance their
//!   phase by their measured frequency times the output hop; every other bin
//!   keeps its phase relation to its peak, so a note's partials stay together.
//!   The phase decision is made on the mid signal and applied to both channels
//!   as a rotation, which keeps the stereo image.
//! * Strong attacks: the frame at the attack is reset to its own phases, frames
//!   before it do not see the attack (their input is faded out 3 ms before it,
//!   so there is no pre-echo), and the attack itself (3 ms before to 40 ms after)
//!   is copied from the input verbatim with short crossfades.

use std::collections::VecDeque;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use super::detect::{Detector, Onset};

const N: usize = 4096;
const HA: i64 = 512;
const HALF: i64 = (N / 2) as i64;
/// Verbatim region around a strong attack, and its crossfades.
const V_PRE: i64 = 132;
const V_POST: i64 = 1764;
const V_XF: i64 = 64;
/// Fade of the input before an attack in the frames that must not see it.
const MASK_FADE: i64 = 64;

pub(super) struct PvStretcher {
    ratio: f64,
    detector: Detector,
    onsets: VecDeque<Onset>,
    buf: VecDeque<[f32; 2]>,
    base: i64,
    input_len: i64,
    finished: bool,
    fwd: Arc<dyn RealToComplex<f64>>,
    inv: Arc<dyn ComplexToReal<f64>>,
    window: Vec<f64>,
    time_buf: Vec<f64>,
    scratch: Vec<Complex<f64>>,
    /// Next frame to analyse (centre at k * HA) and the output centre of the last one.
    k: i64,
    o_prev: f64,
    c_prev: i64,
    /// Phases of the previous frame: analysis (mid) and synthesis, per bin.
    phi_a: Vec<f64>,
    phi_s: Vec<f64>,
    have_prev: bool,
    /// Overlap-add accumulators from output position `out_base` on: signal and window power.
    acc: VecDeque<[f64; 2]>,
    wsum: VecDeque<f64>,
    out_base: i64,
    /// Output samples handed out so far.
    emitted: i64,
    /// Verbatim regions: (output start, output end, input offset = input - output).
    verbatim: VecDeque<(i64, i64, i64)>,
    pub(super) resets: usize,
    pub(super) n_onsets: usize,
    pub(super) n_strong: usize,
}

impl PvStretcher {
    pub(super) fn new(ratio: f64) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(N);
        let inv = planner.plan_fft_inverse(N);
        let scratch = vec![Complex::default(); fwd.get_scratch_len().max(inv.get_scratch_len())];
        let time_buf = inv.make_output_vec();
        let window = (0..N).map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N as f64).cos()).collect();
        let first = -(HALF / HA) - 1;
        PvStretcher {
            ratio,
            detector: Detector::new(),
            onsets: VecDeque::new(),
            buf: VecDeque::new(),
            base: 0,
            input_len: 0,
            finished: false,
            fwd,
            inv,
            window,
            time_buf,
            scratch,
            k: first,
            o_prev: 0.0,
            c_prev: 0,
            phi_a: vec![0.0; N / 2 + 1],
            phi_s: vec![0.0; N / 2 + 1],
            have_prev: false,
            acc: VecDeque::new(),
            wsum: VecDeque::new(),
            out_base: (first * HA) - HALF,
            emitted: 0,
            verbatim: VecDeque::new(),
            resets: 0,
            n_onsets: 0,
            n_strong: 0,
        }
    }

    pub(super) fn set_ratio(&mut self, r: f64) {
        self.ratio = r;
    }

    /// Output position of the end of the input, on the time line so far.
    pub(super) fn end_of_input(&self) -> i64 {
        if !self.have_prev {
            return (self.input_len as f64 * self.ratio).round() as i64;
        }
        let a_prev = (self.k - 1) * HA;
        (self.o_prev + (self.input_len - a_prev) as f64 * self.ratio).round() as i64
    }

    pub(super) fn latency(&self) -> usize {
        (HALF + HA) as usize + super::detect::HOP * (2 * super::detect::PEAK_HOPS + 2)
    }

    pub(super) fn push(&mut self, x: &[[f32; 2]]) {
        let mut found = Vec::new();
        self.detector.push(x, &mut found);
        self.add_onsets(found);
        self.buf.extend(x.iter().copied());
        self.input_len += x.len() as i64;
    }

    pub(super) fn finish(&mut self) {
        if !self.finished {
            let mut found = Vec::new();
            self.detector.finish(&mut found);
            self.add_onsets(found);
            self.finished = true;
        }
    }

    fn add_onsets(&mut self, found: Vec<Onset>) {
        for o in found {
            self.n_onsets += 1;
            self.n_strong += usize::from(o.strong);
            self.onsets.push_back(o);
        }
    }

    fn known(&self) -> i64 {
        if self.finished { i64::MAX / 4 } else { self.detector.done_until().min(self.input_len) }
    }

    #[inline]
    fn at(&self, i: i64) -> [f32; 2] {
        if i < self.base || i >= self.base + self.buf.len() as i64 {
            [0.0; 2]
        } else {
            self.buf[(i - self.base) as usize]
        }
    }

    /// Output samples up to (not including) `limit`, as far as they are final.
    pub(super) fn pull(&mut self, limit: Option<i64>, out: &mut Vec<[f32; 2]>) {
        loop {
            // A frame needs its whole window and the attacks up to its end.
            let a = self.k * HA;
            let can_frame = self.known() >= a + HALF + HA;
            // Output before the next frame's first sample is final.
            let final_until =
                if self.have_prev { (self.o_prev + HA as f64 * self.ratio).round() as i64 - HALF } else { i64::MIN };
            while self.emitted < final_until && limit.is_none_or(|l| self.emitted < l) {
                out.push(self.emit());
            }
            if limit.is_some_and(|l| self.emitted >= l) {
                break;
            }
            if !can_frame {
                break;
            }
            if self.finished && a - HALF > self.input_len && limit.is_none() {
                break;
            }
            self.frame();
        }
        self.trim();
    }

    fn emit(&mut self) -> [f32; 2] {
        let o = self.emitted;
        let i = (o - self.out_base) as usize;
        let (v, w) = (self.acc.get(i).copied().unwrap_or([0.0; 2]), self.wsum.get(i).copied().unwrap_or(0.0));
        let mut y = if w > 1e-9 { [v[0] / w, v[1] / w] } else { [0.0, 0.0] };
        while self.verbatim.front().is_some_and(|r| r.1 + V_XF <= o) {
            self.verbatim.pop_front();
        }
        if let Some(&(s, e, off)) = self.verbatim.iter().find(|r| r.0 - V_XF <= o && o < r.1 + V_XF) {
            let g = if o < s {
                0.5 - 0.5 * (std::f64::consts::PI * (o - (s - V_XF)) as f64 / V_XF as f64).cos()
            } else if o >= e {
                0.5 + 0.5 * (std::f64::consts::PI * (o - e) as f64 / V_XF as f64).cos()
            } else {
                1.0
            };
            let x = self.at(o + off);
            y = [(1.0 - g) * y[0] + g * f64::from(x[0]), (1.0 - g) * y[1] + g * f64::from(x[1])];
        }
        self.emitted += 1;
        [y[0] as f32, y[1] as f32]
    }

    fn frame(&mut self) {
        let a = self.k * HA;
        let o = if self.have_prev { self.o_prev + HA as f64 * self.ratio } else { a as f64 * self.ratio };
        let c = o.round() as i64;
        let hs = (c - self.c_prev) as f64;
        // Strong attacks this frame's window reaches: the first one at or after the centre
        // cuts the input off (no pre-echo); one within half a hop of the centre resets the phases.
        let mut cut: Option<i64> = None;
        let mut reset = false;
        for on in self.onsets.iter().filter(|on| on.strong) {
            let t = on.pos;
            if (t - a).abs() * 2 <= HA {
                reset = true;
                // Its verbatim region, placed at the attack's ideal output time.
                let ot = c + ((t - a) as f64 * self.ratio).round() as i64;
                if self.verbatim.back().is_none_or(|r| r.0 != ot - V_PRE) {
                    self.verbatim.push_back((ot - V_PRE, ot + V_POST, t - ot));
                }
            }
            if t > a + HA / 2 && t - V_PRE < a + HALF && cut.is_none_or(|x| t < x) {
                cut = Some(t);
            }
        }
        // Windowed input, both channels.
        let mut xs = [self.fwd.make_input_vec(), self.fwd.make_input_vec()];
        for (i, w) in self.window.iter().enumerate() {
            let pos = a - HALF + i as i64;
            let mut g = *w;
            if let Some(t) = cut {
                let edge = t - V_PRE;
                if pos >= edge {
                    g = 0.0;
                } else if pos > edge - MASK_FADE {
                    g *= (edge - pos) as f64 / MASK_FADE as f64;
                }
            }
            let v = self.at(pos);
            xs[0][i] = f64::from(v[0]) * g;
            xs[1][i] = f64::from(v[1]) * g;
        }
        let mut sp = [self.fwd.make_output_vec(), self.fwd.make_output_vec()];
        for c_ in 0..2 {
            self.fwd.process_with_scratch(&mut xs[c_], &mut sp[c_], &mut self.scratch).expect("fft");
        }
        let nb = N / 2 + 1;
        let mid: Vec<Complex<f64>> = (0..nb).map(|k| (sp[0][k] + sp[1][k]) * 0.5).collect();
        let mag: Vec<f64> = mid.iter().map(|z| z.norm()).collect();
        let phi: Vec<f64> = mid.iter().map(|z| z.arg()).collect();
        let mut rot = vec![0.0f64; nb];
        if self.have_prev && !reset {
            let top = mag.iter().copied().fold(0.0, f64::max);
            let floor = top * 1e-4;
            let peaks: Vec<usize> = (0..nb)
                .filter(|&k| {
                    mag[k] > floor
                        && (k < 1 || mag[k] > mag[k - 1])
                        && (k < 2 || mag[k] > mag[k - 2])
                        && (k + 1 >= nb || mag[k] >= mag[k + 1])
                        && (k + 2 >= nb || mag[k] >= mag[k + 2])
                })
                .collect();
            if peaks.is_empty() {
                rot.fill(0.0);
            } else {
                // Rotation of every peak, then of its region (bounded by the magnitude minimum between peaks).
                let mut prot = Vec::with_capacity(peaks.len());
                for &p in &peaks {
                    let omega = std::f64::consts::TAU * p as f64 / N as f64;
                    let dphi = princarg(phi[p] - self.phi_a[p] - omega * HA as f64);
                    let inst = omega + dphi / HA as f64;
                    let ps = self.phi_s[p] + inst * hs;
                    prot.push(ps - phi[p]);
                }
                let mut start = 0;
                for (i, &p) in peaks.iter().enumerate() {
                    let end = if i + 1 < peaks.len() {
                        let q = peaks[i + 1];
                        (p..q).min_by(|&x, &y| mag[x].total_cmp(&mag[y])).unwrap() + 1
                    } else {
                        nb
                    };
                    for r in rot.iter_mut().take(end).skip(start) {
                        *r = prot[i];
                    }
                    start = end;
                }
            }
        } else if reset {
            self.resets += 1;
        }
        for k in 0..nb {
            self.phi_s[k] = phi[k] + rot[k];
            self.phi_a[k] = phi[k];
        }
        self.have_prev = true;
        // Rotate, back to time, window again and add.
        let start = c - HALF;
        if self.acc.is_empty() && start < self.out_base {
            self.out_base = start;
        }
        while self.out_base + (self.acc.len() as i64) < start + N as i64 {
            self.acc.push_back([0.0; 2]);
            self.wsum.push_back(0.0);
        }
        let scale = 1.0 / N as f64;
        let turn: Vec<Complex<f64>> = rot
            .iter()
            .map(|&r| {
                let (s, co) = r.sin_cos();
                Complex::new(co, s)
            })
            .collect();
        #[allow(clippy::needless_range_loop)]
        for c_ in 0..2 {
            for k in 0..nb {
                sp[c_][k] *= turn[k];
            }
            sp[c_][0].im = 0.0;
            sp[c_][nb - 1].im = 0.0;
            self.inv.process_with_scratch(&mut sp[c_], &mut self.time_buf, &mut self.scratch).expect("ifft");
            // Positions before `out_base` were handed out already (or lie before the output's start).
            for (i, w) in self.window.iter().enumerate() {
                let idx = start + i as i64 - self.out_base;
                if idx >= 0 {
                    self.acc[idx as usize][c_] += self.time_buf[i] * scale * w;
                }
            }
        }
        for (i, w) in self.window.iter().enumerate() {
            let idx = start + i as i64 - self.out_base;
            if idx >= 0 {
                self.wsum[idx as usize] += w * w;
            }
        }
        self.o_prev = o;
        self.c_prev = c;
        self.k += 1;
    }

    fn trim(&mut self) {
        while self.out_base < self.emitted && !self.acc.is_empty() {
            self.acc.pop_front();
            self.wsum.pop_front();
            self.out_base += 1;
        }
        // Input: frames still to come start at k*HA - HALF; verbatim copies read around the output.
        let oldest_verbatim = self.verbatim.front().map_or(i64::MAX, |r| r.0 - V_XF + r.2);
        let keep = (self.k * HA - HALF).min(oldest_verbatim);
        while self.base < keep && !self.buf.is_empty() {
            self.buf.pop_front();
            self.base += 1;
        }
        while self.onsets.front().is_some_and(|o| o.pos + V_POST < keep) {
            self.onsets.pop_front();
        }
    }
}

fn princarg(x: f64) -> f64 {
    x - std::f64::consts::TAU * (x / std::f64::consts::TAU).round()
}
