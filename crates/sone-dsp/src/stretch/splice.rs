//! Splicing engine: plays the input verbatim and repeats or skips a little where
//! the waveform matches itself (see the module documentation).

use std::collections::VecDeque;

use super::detect::{self, Onset};
use super::lane::{Ctx, Lane, LaneParams};
use super::{LOOKAHEAD, POST, StretchStats};

static FULL: LaneParams = LaneParams {
    xf: 1024,
    jmax: 1100,
    span: 441,
    all_targets: false,
    protect_weak: true,
    tol_strong: 8,
    tol_weak: 8,
    fine: true,
    soft: 441.0,
    ahead: 8192,
    dec: 4,
};

/// The splicing engine.
pub(super) struct SpliceStretcher {
    ratio: f64,
    detector: detect::Detector,
    /// Input from absolute position `base`.
    buf: VecDeque<[f32; 2]>,
    base: i64,
    input_len: i64,
    finished: bool,
    onsets: VecDeque<Onset>,
    n_onsets: usize,
    n_strong: usize,
    /// Ideal input position of the next output sample.
    p: f64,
    lane: Lane,
}

impl SpliceStretcher {
    pub(super) fn new(ratio: f64) -> Self {
        SpliceStretcher {
            ratio,
            detector: detect::Detector::new(),
            buf: VecDeque::new(),
            base: 0,
            input_len: 0,
            finished: false,
            onsets: VecDeque::new(),
            n_onsets: 0,
            n_strong: 0,
            p: 0.0,
            lane: Lane::new(&FULL),
        }
    }

    /// Applies to output produced from now on. A splice already planned is kept;
    /// the next plan is measured against the new ratio.
    pub(super) fn set_ratio(&mut self, ratio: f64) {
        self.ratio = ratio;
    }

    /// How many input samples the output runs behind the input when streaming:
    /// the output sample that leaves now belongs to the input this many samples
    /// before the newest one.
    pub(super) fn latency(&self) -> usize {
        LOOKAHEAD + 1100 + detect::HOP * (2 * detect::PEAK_HOPS + 1)
    }

    pub(super) fn stats(&self) -> StretchStats {
        let l = &self.lane.stats;
        let n = l.coarse + l.fine;
        StretchStats {
            onsets: self.n_onsets,
            strong_onsets: self.n_strong,
            coarse: l.coarse,
            fine: l.fine,
            missed: l.missed,
            relaxed: l.relaxed,
            mean_rho: if n > 0 { l.rho_sum / n as f64 } else { 1.0 },
            min_rho: l.min_rho,
            resets: 0,
        }
    }

    /// Feeds interleaved stereo input, returns the interleaved output that is ready.
    pub(super) fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let frames: Vec<[f32; 2]> = input.as_chunks::<2>().0.iter().map(|c| [c[0], c[1]]).collect();
        self.push_frames(&frames);
        let mut out = Vec::new();
        while self.ready() {
            out.push(self.next_sample());
        }
        self.trim();
        out.into_iter().flatten().collect()
    }

    /// End of input: returns the rest of the output, up to the ideal end of the input.
    pub(super) fn flush(&mut self) -> Vec<f32> {
        self.finish();
        let mut out = Vec::new();
        while self.lane.in_crossfade() || self.p < self.input_len as f64 {
            out.push(self.next_sample());
        }
        out.into_iter().flatten().collect()
    }

    /// Offline use: exactly `round(n × ratio)` samples for the whole input.
    pub(super) fn render_all(&mut self, x: &[[f32; 2]]) -> Vec<[f32; 2]> {
        let want = (x.len() as f64 * self.ratio).round() as usize;
        self.push_frames(x);
        self.finish();
        let mut out = Vec::with_capacity(want);
        while out.len() < want {
            out.push(self.next_sample());
        }
        out
    }

    pub(super) fn finish(&mut self) {
        if self.finished {
            return;
        }
        let mut found = Vec::new();
        self.detector.finish(&mut found);
        self.add_onsets(found);
        self.finished = true;
    }

    pub(super) fn push_frames(&mut self, frames: &[[f32; 2]]) {
        let mut found = Vec::new();
        self.detector.push(frames, &mut found);
        self.add_onsets(found);
        self.buf.extend(frames.iter().copied());
        self.input_len += frames.len() as i64;
    }

    fn add_onsets(&mut self, found: Vec<Onset>) {
        for o in found {
            self.n_onsets += 1;
            self.n_strong += usize::from(o.strong);
            self.onsets.push_back(o);
        }
    }

    /// How far the input is known to the planner (samples and attacks).
    fn known_until(&self) -> i64 {
        if self.finished { i64::MAX / 4 } else { self.detector.done_until().min(self.input_len) }
    }

    pub(super) fn ready(&self) -> bool {
        self.known_until() >= self.lane.q + LOOKAHEAD as i64 + 1100
    }

    pub(super) fn next_sample(&mut self) -> [f32; 2] {
        let ctx = Ctx {
            buf: &self.buf,
            base: self.base,
            onsets: &self.onsets,
            p: self.p,
            ratio: self.ratio,
            known: self.known_until(),
        };
        let y = self.lane.next(&ctx);
        self.p += 1.0 / self.ratio;
        y
    }

    pub(super) fn trim(&mut self) {
        let keep_from = self.lane.oldest_needed();
        while self.base < keep_from && !self.buf.is_empty() {
            self.buf.pop_front();
            self.base += 1;
        }
        while self.onsets.front().is_some_and(|o| o.pos + POST < keep_from) {
            self.onsets.pop_front();
        }
    }
}

