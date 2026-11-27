//! Key-locked time stretch for small ratios (0.96 to 1.04), offline and streaming.
//!
//! # Choice of algorithm
//!
//! Two engines were built and measured against Rubber Band on real music
//! (`reference/python/stretch_eval.py`: six songs, 30 s each, ratios 0.965,
//! 0.985, 1.016, 1.035; `nudge_probe.py` / `compare_nudge.py` on the specs).
//!
//! * [`Method::PhaseVocoder`] (default, [`pv`](self)): 4096-sample STFT every 512
//!   input samples, each frame placed at the ideal output time of its centre;
//!   identity phase locking (Laroche & Dolson) decided on the mid signal and
//!   applied to both channels as a rotation; at every strong attack the frame is
//!   reset to its own phases, earlier frames do not see the attack (no pre-echo)
//!   and the attack itself (3 ms before to 40 ms after) is copied verbatim.
//!   Every frequency keeps its phase continuity, so held notes and bass stay
//!   pure, and *every* event, strong or weak, stays on the ideal time line.
//! * [`Method::Splice`] ([`splice`](self)): time-domain splicing (WSOLA family).
//!   The input plays verbatim; now and then a few ms are repeated or skipped
//!   where the waveform matches itself, never across an attack, steered so that
//!   strong attacks land within 0.2 ms. Attacks are bit-exact and levels exact,
//!   and it is three times faster. Its limit is physical: a clean splice of a
//!   held bass note needs a jump of whole periods (up to 25 ms), so between two
//!   strong attacks weaker events (hi-hats) drift by up to a few ms.
//!
//! Measured: both engines keep length exact and stay within 0.16 ms of the
//! ideal on average; per-5 s-window timing of the onset envelope was mostly
//! under 1 ms for the phase vocoder but up to 3.5 ms for the splicer, and the
//! renderer's automatic nudge agreed with the reference's (which uses Rubber
//! Band R3) within 0.8 ms on 35 of 36 specs with the phase vocoder, 30 of 36
//! with the splicer. The log-spectral distance to Rubber Band R3 is at the level
//! of Rubber Band R2's for both. The phase vocoder streams with about 0.1 s of
//! latency, the splicer needs 0.8 s of lookahead. Hence the phase vocoder is the
//! default.
//!
//! Streaming: [`Stretcher::process`] takes interleaved stereo and returns what
//! is ready; the output runs [`Stretcher::latency`] input samples behind. The
//! ratio may change at any time (`set_ratio`); the ideal input position advances
//! by `1 / ratio` per output sample.

mod detect;
mod lane;
mod pv;
mod splice;

pub use detect::{Onset, detect_onsets};

/// Input the splicing engine's planner looks ahead of the read position (samples).
pub const LOOKAHEAD: usize = 32768;
/// Protected zone of an attack: before and after its detected position.
pub(crate) const PRE: i64 = 220;
pub(crate) const POST: i64 = 1323;

/// Which engine a [`Stretcher`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    /// Time-domain splicing: verbatim audio, exact attacks and levels; weak events may drift a few ms.
    Splice,
    /// Phase vocoder with phase locking and verbatim strong attacks: every event on time.
    #[default]
    PhaseVocoder,
}

/// Counters of what the engine did, for tests and diagnostics.
#[derive(Debug, Clone, Copy, Default)]
pub struct StretchStats {
    pub onsets: usize,
    pub strong_onsets: usize,
    /// Splicing: ordinary splices and fine ones (right before an attack).
    pub coarse: usize,
    pub fine: usize,
    /// Splicing: strong attacks the planner could not land within tolerance.
    pub missed: usize,
    /// Splicing: splices that had to cross a weak attack's protected zone.
    pub relaxed: usize,
    /// Splicing: mean waveform correlation at the splices (1 = seamless).
    pub mean_rho: f64,
    /// Splicing: worst correlation of an ordinary splice (fine ones sit under an attack).
    pub min_rho: f64,
    /// Phase vocoder: frames reset to their own phases at an attack.
    pub resets: usize,
}

enum Engine {
    Splice(Box<splice::SpliceStretcher>),
    Pv(Box<pv::PvStretcher>),
}

/// Streaming key-locked time stretcher for stereo at 44.1 kHz.
pub struct Stretcher {
    ratio: f64,
    engine: Engine,
}

impl Stretcher {
    /// `ratio` = output duration / input duration.
    pub fn new(ratio: f64) -> Self {
        Self::with_method(ratio, Method::default())
    }

    pub fn with_method(ratio: f64, method: Method) -> Self {
        let engine = match method {
            Method::Splice => Engine::Splice(Box::new(splice::SpliceStretcher::new(ratio))),
            Method::PhaseVocoder => Engine::Pv(Box::new(pv::PvStretcher::new(ratio))),
        };
        Stretcher { ratio, engine }
    }

    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Applies to output produced from now on.
    pub fn set_ratio(&mut self, ratio: f64) {
        self.ratio = ratio;
        match &mut self.engine {
            Engine::Splice(e) => e.set_ratio(ratio),
            Engine::Pv(e) => e.set_ratio(ratio),
        }
    }

    /// How many input samples the output runs behind the input when streaming:
    /// the output sample that leaves now belongs to the input this many samples
    /// before the newest one.
    pub fn latency(&self) -> usize {
        match &self.engine {
            Engine::Splice(e) => e.latency(),
            Engine::Pv(e) => e.latency(),
        }
    }

    pub fn stats(&self) -> StretchStats {
        match &self.engine {
            Engine::Splice(e) => e.stats(),
            Engine::Pv(e) => StretchStats {
                onsets: e.n_onsets,
                strong_onsets: e.n_strong,
                resets: e.resets,
                ..Default::default()
            },
        }
    }

    /// Feeds interleaved stereo input, returns the interleaved output that is ready.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        match &mut self.engine {
            Engine::Splice(e) => e.process(input),
            Engine::Pv(e) => {
                let frames: Vec<[f32; 2]> = input.as_chunks::<2>().0.iter().map(|c| [c[0], c[1]]).collect();
                e.push(&frames);
                let mut out = Vec::new();
                e.pull(None, &mut out);
                out.into_iter().flatten().collect()
            }
        }
    }

    /// End of input: returns the rest of the output, up to the ideal end of the input.
    pub fn flush(&mut self) -> Vec<f32> {
        match &mut self.engine {
            Engine::Splice(e) => e.flush(),
            Engine::Pv(e) => {
                e.finish();
                let end = e.end_of_input();
                let mut out = Vec::new();
                e.pull(Some(end), &mut out);
                out.into_iter().flatten().collect()
            }
        }
    }

    /// Offline use: exactly `round(n × ratio)` samples for the whole input.
    pub fn render_all(&mut self, x: &[[f32; 2]]) -> Vec<[f32; 2]> {
        let want = (x.len() as f64 * self.ratio).round() as usize;
        match &mut self.engine {
            Engine::Splice(e) => e.render_all(x),
            Engine::Pv(e) => {
                e.push(x);
                e.finish();
                let mut out = Vec::with_capacity(want);
                e.pull(Some(want as i64), &mut out);
                out.resize(want, [0.0; 2]);
                out
            }
        }
    }
}

/// Offline stretch: output length is exactly `round(len × ratio)`.
pub fn stretch(x: &[[f32; 2]], ratio: f64) -> Vec<[f32; 2]> {
    if (ratio - 1.0).abs() < 1e-5 {
        return x.to_vec();
    }
    Stretcher::new(ratio).render_all(x)
}

/// Offline stretch with a chosen engine, also returning what it did.
pub fn stretch_with(x: &[[f32; 2]], ratio: f64, method: Method) -> (Vec<[f32; 2]>, StretchStats) {
    let mut s = Stretcher::with_method(ratio, method);
    let y = s.render_all(x);
    (y, s.stats())
}

/// Offline stretch that also returns what the engine did.
pub fn stretch_with_stats(x: &[[f32; 2]], ratio: f64) -> (Vec<[f32; 2]>, StretchStats) {
    stretch_with(x, ratio, Method::default())
}

#[cfg(test)]
mod tests;
