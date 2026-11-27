//! Beat and downbeat tracking: the neural beat tracker Beat This! ([`BeatThis`];
//! Foscarin, Schlüter, Widmer, ISMIR 2024) in plain Rust, and where the song
//! plays at full loudness.
//!
//! Standalone neural beat and downbeat tracker in plain Rust
//! (`src/analysis.rs`, `src/analysis/beat_this.rs`, `src/analysis/grid.rs`),
//! without its Spotify/CDN streaming, so this repository stands alone. Keep the
//! two in step when the network code changes.
//!
//! The pipeline is the Python reference's (`beat_this.inference.Audio2Beats`,
//! `dbn=False`): mono → 22050 Hz (soxr's 2:1 filter, measured and embedded
//! here) → log-mel spectrogram (torchaudio `MelSpectrogram` + `log1p(1000·x)`)
//! → the network in 30 s chunks → the "minimal" postprocessor (peaks of the
//! logits above 0 within ±70 ms, adjacent peaks merged, downbeats snapped to
//! the nearest beat). Everything streams: [`Analyzer`] takes the audio in
//! pieces and keeps only the spectrogram (128 floats per 20 ms), so a whole
//! song never has to sit in memory as PCM.
//!
//! Two models: `small0` (2.1 M parameters, F16, 4.2 MB, embedded with the
//! feature `embed-small`) and `final0` (20 M parameters, F16, 41 MB, loaded at
//! run time with [`load_model`]; too big to embed in a Worker bundle). The
//! checkpoints and the code are MIT-licensed (© 2024 Institute of
//! Computational Perception, JKU Linz; github.com/CPJKU/beat_this); the weights
//! files are the checkpoints with their BatchNorms folded.
//!
//! Two paths: the whole song ([`track`], [`Analyzer`]) and one transition
//! point ([`track_focus`], [`FocusAnalyzer`]), which runs the network only on
//! the chunk(s) of the song's own chunk grid around the point, so its beats
//! there equal the whole-song run's.

mod beat_this;
pub mod grid;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

pub use beat_this::{BeatThis, Workspace};

use beat_this::N_MELS;

/// Sample rate the public entry points take.
pub const SAMPLE_RATE: u32 = 44_100;

/// Beats of one song (or of a window of it).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatAnalysis {
    /// Seconds from the start of the audio.
    pub beats: Vec<f32>,
    /// The beats that start a bar (a subset of `beats`).
    pub downbeats: Vec<f32>,
    /// Where the song first plays at (nearly) full loudness: the first 4 s
    /// window whose mean level is within 3 dB of the loud average.
    pub full_start: f32,
    /// The end of the last 1 s window within 9 dB of the loud average.
    pub loud_end: f32,
    pub duration: f32,
}

impl BeatAnalysis {
    /// `(seconds, is_downbeat)` per beat, the form the Python reference's
    /// `.beats` files have.
    pub fn flagged(&self) -> Vec<(f64, bool)> {
        self.beats.iter().map(|&b| (f64::from(b), self.downbeats.contains(&b))).collect()
    }
}

#[cfg(feature = "embed-small")]
static SMALL0: &[u8] = include_bytes!("../../../models/beat_this_small0_f16.safetensors");

/// The embedded `small0` network (parsed once, ~8.4 MB as f32 for its
/// convolutions; its F16 matrices are read in place).
#[cfg(feature = "embed-small")]
pub fn small() -> &'static BeatThis {
    static MODEL: std::sync::OnceLock<BeatThis> = std::sync::OnceLock::new();
    MODEL.get_or_init(|| BeatThis::from_safetensors(SMALL0).expect("the embedded Beat This weights parse"))
}

/// A network from a weights file's bytes (e.g. `final0` F16). The bytes are
/// kept for the life of the process, because the F16 matrices are read in
/// place: load a model once and keep it.
pub fn load_model(bytes: Vec<u8>) -> Result<BeatThis> {
    BeatThis::from_vec(bytes)
}

/// [`load_model`] from a file.
#[cfg(feature = "files")]
pub fn load_model_file(path: &std::path::Path) -> Result<BeatThis> {
    use anyhow::Context;
    load_model(std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
}

/// Stereo (interleaved) or mono 44.1 kHz audio as mono, the way the Python
/// reference mixes down (the mean of the channels).
pub fn to_mono(audio: &[f32], channels: usize) -> Vec<f32> {
    match channels {
        0 | 1 => audio.to_vec(),
        c => audio.chunks_exact(c).map(|f| f.iter().sum::<f32>() / c as f32).collect(),
    }
}

/// Beats and downbeats of a whole song: `audio` is 44.1 kHz, `channels`
/// interleaved channels (1 or 2). Returns `(seconds, is_downbeat)` per beat.
pub fn track(model: &BeatThis, audio: &[f32], channels: usize) -> Result<Vec<(f64, bool)>> {
    Ok(analyze_with(model, audio, channels)?.flagged())
}

/// [`track`] with the loudness summary.
pub fn analyze_with(model: &BeatThis, audio: &[f32], channels: usize) -> Result<BeatAnalysis> {
    let mut analyzer = Analyzer::new(SAMPLE_RATE)?;
    let channels = channels.max(1);
    analyzer.reserve_seconds((audio.len() / channels) as f32 / SAMPLE_RATE as f32);
    // in pieces, so neither the mono nor the resampled copy exists whole
    let mut mono = Vec::with_capacity(SAMPLE_RATE as usize);
    for piece in audio.chunks(SAMPLE_RATE as usize * channels) {
        mono.clear();
        mono.extend(piece.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32));
        analyzer.push(&mono);
    }
    Ok(analyzer.finish(model))
}

/// Beats around one transition point ([`Focus`]): the network on the
/// chunk(s) of the song's chunk grid that cover `options.net` around the
/// point (`max_chunks` ≥ 1), else the fast grid fit ([`grid`]).
pub fn track_focus(model: &BeatThis, audio: &[f32], channels: usize, focus: Focus, options: FocusOptions) -> Result<Vec<(f64, bool)>> {
    let mono = to_mono(audio, channels);
    Ok(analyze_focus(&mono, SAMPLE_RATE, focus, options, model)?.flagged())
}

/// Analyse a window of a song: `mono` is the song's audio from `offset_s`
/// on. Every time in the result is from the start of the song. The loudness
/// is the window's own.
pub fn analyze_window(model: &BeatThis, mono: &[f32], sample_rate: u32, offset_s: f32) -> Result<BeatAnalysis> {
    let mut analyzer = Analyzer::new(sample_rate)?;
    analyzer.reserve_seconds(mono.len() as f32 / sample_rate as f32);
    for piece in mono.chunks(sample_rate as usize) {
        analyzer.push(piece);
    }
    let mut analysis = analyzer.finish(model);
    for t in analysis.beats.iter_mut().chain(analysis.downbeats.iter_mut()) {
        *t += offset_s;
    }
    analysis.full_start += offset_s;
    analysis.loud_end += offset_s;
    analysis.duration += offset_s;
    Ok(analysis)
}

/// The whole song's loudness, the network on `window` only.
pub fn analyze_net_window(model: &BeatThis, mono: &[f32], sample_rate: u32, window: NetWindow) -> Result<BeatAnalysis> {
    let mut analyzer = Analyzer::new(sample_rate)?;
    analyzer.reserve_seconds(mono.len() as f32 / sample_rate as f32);
    for piece in mono.chunks(sample_rate as usize) {
        analyzer.push(piece);
    }
    let (prediction, loudness) = analyzer.into_prediction_window(model, window);
    Ok(loudness.with_beats(prediction.finish()))
}

/// Where the network runs (song seconds). The loudness always covers
/// everything decoded; the adaptive windows are placed from it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NetWindow {
    Fixed { from_s: f32, to_s: f32 },
    /// `loud_end − before_s ..= loud_end + after_s` (the song's end: decode
    /// to the end for an exact `loud_end`).
    AroundLoudEnd { before_s: f32, after_s: f32 },
    /// `full_start − before_s ..= full_start + after_s` (the song's start).
    AroundFullStart { before_s: f32, after_s: f32 },
}

impl NetWindow {
    /// The span in song seconds, given the loudness of the decoded audio
    /// (clamped to it).
    pub fn span(&self, loudness: &LoudnessSummary) -> (f32, f32) {
        let (from, to) = match *self {
            Self::Fixed { from_s, to_s } => (from_s, to_s),
            Self::AroundLoudEnd { before_s, after_s } => (loudness.loud_end - before_s, loudness.loud_end + after_s),
            Self::AroundFullStart { before_s, after_s } => (loudness.full_start - before_s, loudness.full_start + after_s),
        };
        (from.max(0.0), to.min(loudness.duration).max(from.max(0.0)))
    }
}

/// Which transition point a fast analysis is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Focus {
    /// The song on air: where it is left (around `loud_end`, whole file
    /// decoded).
    MixOut,
    /// The next song: where it comes in (around `full_start`, only its head
    /// decoded).
    Entry,
}

/// Where the fast analysis looks, relative to the focus point (`loud_end`
/// for [`Focus::MixOut`], `full_start` for [`Focus::Entry`]), in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FocusOptions {
    /// The grid's span (the beats in the result).
    pub span: (f32, f32),
    /// The region the network should cover (the chunk(s) of the song's own
    /// chunk grid that overlap it most are run; outside them there are no
    /// beats in the result).
    pub net: (f32, f32),
    /// Decode no further than this (song seconds; ∞: the whole file).
    pub decode_to_s: f32,
    /// Mix-out: fetch and decode only the last this many seconds of the
    /// file (∞: all of it). The loud average (and so `loud_end`) then comes
    /// from that tail; `full_start` is meaningless.
    #[serde(default = "infinite")]
    pub tail_s: f32,
    /// 0 (the default): the fast path — the network on the short `net`
    /// window, a constant-tempo grid over the span. 1 or 2: the network on
    /// at most this many chunks of the song's own chunk grid covering `net`
    /// (a second one when one covers less than 90 % of it) — the whole-song
    /// result there, at 29.76 s of network per chunk.
    #[serde(default)]
    pub max_chunks: usize,
}

fn infinite() -> f32 {
    f32::INFINITY
}

impl FocusOptions {
    /// The audio the chunk-aligned path needs (a chunk around the point and
    /// a second of context): at least the last 100 s / the first 92 s.
    fn with_room(mut self) -> Self {
        if self.max_chunks > 0 {
            self.tail_s = self.tail_s.max(100.0);
            self.decode_to_s = self.decode_to_s.max(92.0);
        }
        self
    }

    pub fn default_for(focus: Focus) -> Self {
        match focus {
            Focus::MixOut => Self { span: (-45.0, 3.0), net: (-30.0, -18.0), decode_to_s: f32::INFINITY, tail_s: 75.0, max_chunks: 0 },
            Focus::Entry => Self { span: (-4.0, 32.0), net: (-2.0, 10.0), decode_to_s: 60.0, tail_s: f32::INFINITY, max_chunks: 0 },
        }
    }
}

/// The analysis of one transition point: loudness over what is decoded, then
/// Beat This! (the given model) on the chunk(s) of the song's own chunk grid
/// around the point — the result there is exactly the whole-song result.
/// When no such chunk's audio is at hand, the fallback is a constant-tempo
/// grid on an onset envelope ([`grid`]) with the network on a short window.
/// Keeps the audio it may need: the last 100 s (mix-out) or everything
/// decoded (entry), as 16-bit.
pub struct FocusAnalyzer {
    rate: u32,
    focus: Focus,
    options: FocusOptions,
    loudness: Loudness,
    kept: Vec<i16>,
    /// Absolute sample index of `kept[0]`.
    kept_base: u64,
    pushed: u64,
}

/// Audio a mix-out analysis keeps (the end of the song).
const MIX_OUT_KEEP_S: u64 = 100;
/// Head the entry's loudness is measured on.
const ENTRY_LOUDNESS_S: u64 = 60;

impl FocusAnalyzer {
    pub fn new(sample_rate: u32, focus: Focus, options: FocusOptions) -> Result<Self> {
        Self::new_at(sample_rate, focus, options, 0)
    }

    /// For audio that starts at song sample `start` (a decoded tail).
    pub fn new_at(sample_rate: u32, focus: Focus, options: FocusOptions, start: u64) -> Result<Self> {
        if sample_rate != 44_100 && sample_rate != 22_050 {
            bail!("track analysis supports 44.1 and 22.05 kHz, not {sample_rate} Hz");
        }
        Ok(Self {
            rate: sample_rate,
            focus,
            options,
            loudness: Loudness::new_at(sample_rate, start),
            kept: Vec::new(),
            kept_base: start,
            pushed: start,
        })
    }

    pub fn push(&mut self, mono: &[f32]) {
        // the entry's loudness rule is tuned on a 60 s head
        let loud = if self.focus == Focus::Entry && self.kept_base == 0 {
            let limit = ENTRY_LOUDNESS_S * u64::from(self.rate);
            &mono[..(limit.saturating_sub(self.pushed) as usize).min(mono.len())]
        } else {
            mono
        };
        self.loudness.push(loud);
        self.pushed += mono.len() as u64;
        self.kept.extend(mono.iter().map(|&v| (v.clamp(-1.0, 1.0) * 32_767.0).round() as i16));
        if self.focus == Focus::MixOut {
            let keep = (MIX_OUT_KEEP_S * u64::from(self.rate)) as usize;
            if self.kept.len() > keep + self.rate as usize * 5 {
                let drop = self.kept.len() - keep;
                self.kept.drain(..drop);
                self.kept_base += drop as u64;
            }
        }
    }

    /// The analysis: the network on the one chunk of the song's own chunk
    /// grid (the one a whole-song run uses: 1500 frames every 1488 from
    /// frame −6, the last one ending at the song's end) that covers most of
    /// the target region, so the beats and downbeats there are exactly what
    /// Beat This finds on the whole song, for the cost of one chunk.
    pub fn finish(self, model: &BeatThis) -> BeatAnalysis {
        match (self.options.max_chunks > 0).then(|| self.aligned_chunks()).flatten() {
            Some(chunks) => self.finish_aligned(model, &chunks),
            None => self.finish_grid(model),
        }
    }

    fn loudness_summary(&self) -> LoudnessSummary {
        let loudness = self.loudness.clone();
        match self.focus {
            Focus::Entry if self.kept_base == 0 => loudness.summary_head(),
            _ => loudness.summary(),
        }
    }

    /// Samples per spectrogram frame at the input rate.
    fn hop(&self) -> u64 {
        u64::from(self.rate / 50)
    }

    /// Frames of the whole song's spectrogram (known once decoded to its
    /// end: the mix-out), else `None` (the head of a longer song).
    fn song_frames(&self) -> Option<i64> {
        let n22 = if self.rate == 44_100 { self.pushed.div_ceil(2) } else { self.pushed };
        (self.focus == Focus::MixOut).then(|| 1 + (n22 / 441) as i64)
    }

    /// The chunks to run (start frame, kept range of frames; one, or two
    /// neighbours), if their audio (with a second of context either side) is
    /// at hand.
    fn aligned_chunks(&self) -> Option<Vec<(i64, (i64, i64))>> {
        let loudness = self.loudness_summary();
        let key = match self.focus {
            Focus::MixOut => loudness.loud_end,
            Focus::Entry => loudness.full_start,
        };
        let target = (((key + self.options.net.0) * 50.0) as i64, ((key + self.options.net.1) * 50.0) as i64);
        let len = self.song_frames();
        let starts: Vec<i64> = match len {
            Some(len) => beat_this::chunk_starts(len as usize),
            None => (0..).map(|k| -6 + 1488 * k).take_while(|&s| s < target.1 + 2 * 1488).collect(),
        };
        let end = len.unwrap_or(i64::MAX);
        let hop = self.hop() as i64;
        let decoded_frames = self.pushed as i64 / hop;
        let chunks: Vec<Option<(i64, (i64, i64))>> = starts
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let mut kept = ((s + 6).max(0), (s + 1494).min(end));
                if i > 0 && i + 1 == starts.len() && len.is_some() {
                    // the last chunk (shifted to the end) keeps only what the
                    // one before it does not cover (keep_first)
                    kept.0 = kept.0.max(starts[i - 1] + 1494);
                }
                // its input and a second either side must be decoded and kept
                let from_ok = ((s.max(0) - 50).max(0) * hop) >= self.kept_base as i64;
                let to_ok = match len {
                    Some(_) => true,
                    None => s + 1500 + 50 <= decoded_frames,
                };
                (kept.1 > kept.0 && from_ok && to_ok).then_some((s, kept))
            })
            .collect();
        let overlap = |k: (i64, i64)| (k.1.min(target.1) - k.0.max(target.0)).max(0);
        let (best, _) = chunks
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (i, overlap(c.1))))
            .max_by_key(|&(_, o)| o)
            .filter(|&(_, o)| o > 0)?;
        let mut picked = vec![chunks[best].expect("available")];
        let covered = overlap(picked[0].1);
        if self.options.max_chunks >= 2 && (covered as f32) < 0.9 * (target.1 - target.0) as f32 {
            let neighbours = [best.checked_sub(1), Some(best + 1)];
            if let Some(extra) = neighbours
                .into_iter()
                .flatten()
                .filter_map(|i| chunks.get(i).copied().flatten())
                .max_by_key(|c| overlap(c.1))
                .filter(|c| overlap(c.1) > 0)
            {
                picked.push(extra);
                picked.sort_by_key(|c| c.0);
            }
        }
        Some(picked)
    }

    fn finish_aligned(self, model: &BeatThis, chunks: &[(i64, (i64, i64))]) -> BeatAnalysis {
        let loudness = self.loudness_summary();
        let hop = self.hop() as i64;
        let end = self.song_frames().unwrap_or(i64::MAX);
        let first = chunks[0].0;
        let last = chunks[chunks.len() - 1].0;
        let m0 = first.max(0);
        let m1 = (last + 1500).min(end);
        // the audio from a second before the chunks (on the frame grid) to a
        // second after them, or the song's start / end
        let a = ((m0 - 50).max(0) * hop).max(self.kept_base as i64);
        let a = a.div_euclid(hop) * hop + if a % hop == 0 { 0 } else { hop };
        let b = ((m1 + 50) * hop).min(self.pushed as i64);
        let slice_frame0 = a / hop;
        let audio: Vec<f32> = self.kept[(a as u64 - self.kept_base) as usize..(b as u64 - self.kept_base) as usize]
            .iter()
            .map(|&v| f32::from(v) / 32_767.0)
            .collect();
        let mut analyzer = Analyzer::new(self.rate).expect("a supported rate");
        analyzer.push(&audio);
        drop(audio);
        let mel = analyzer.spectrogram();
        let from = ((m0 - slice_frame0) as usize * N_MELS).min(mel.len());
        let to = ((m1 - slice_frame0) as usize * N_MELS).min(mel.len());
        let part = &mel[from..to];
        let frames = part.len() / N_MELS;
        let mut beat = vec![-1000.0; frames];
        let mut down = vec![-1000.0; frames];
        let mut workspace = Workspace::default();
        // keep_first: the later chunk first, the earlier one overwrites it
        for &(start, _) in chunks.iter().rev() {
            model.predict_chunk(&mut workspace, part, start - m0, &mut beat, &mut down);
        }
        let kept = (chunks[0].1.0, chunks[chunks.len() - 1].1.1);
        let (ka, kb) = (((kept.0 - m0).max(0) as usize).min(frames), ((kept.1 - m0).max(0) as usize).min(frames));
        let (beats, downs) = postprocess(&beat[ka..kb], &down[ka..kb]);
        let t0 = (m0 + ka as i64) as f32 / 50.0;
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var_os("ANALYSIS_DEBUG").is_some() {
            eprintln!("aligned chunks {chunks:?} slice from frame {slice_frame0} part {m0}..{m1}");
        }
        let shift = |v: Vec<f32>| v.into_iter().map(|t| t + t0).collect();
        loudness.with_beats((shift(beats), shift(downs)))
    }

    fn finish_grid(self, model: &BeatThis) -> BeatAnalysis {
        let rate = self.rate as f32;
        let loudness = match self.focus {
            Focus::Entry if self.kept_base == 0 => self.loudness.summary_head(),
            _ => self.loudness.summary(),
        };
        let key = match self.focus {
            Focus::MixOut => loudness.loud_end,
            Focus::Entry => loudness.full_start,
        };
        let kept_from = self.kept_base as f32 / rate;
        let kept_to = self.pushed as f32 / rate;
        let span = ((key + self.options.span.0).max(kept_from), (key + self.options.span.1).min(kept_to));
        // the onset envelope over the span only (from the kept audio), its
        // frames on the song's 20 ms grid
        let hop = u64::from(self.rate / 50);
        let env_start = ((span.0 * rate) as u64 / hop * hop).max(self.kept_base.div_ceil(hop) * hop);
        let env_end = ((span.1 * rate) as u64).min(self.pushed);
        let mut onset = grid::OnsetEnvelope::new(self.rate);
        let piece: Vec<f32> = self.kept
            [(env_start - self.kept_base) as usize..(env_end.max(env_start) - self.kept_base) as usize]
            .iter()
            .map(|&v| f32::from(v) / 32_767.0)
            .collect();
        onset.push(&piece);
        drop(piece);
        let env_t0 = env_start as f32 / rate;
        // the network's window, inside the kept audio
        let mut net = ((key + self.options.net.0).max(kept_from), (key + self.options.net.1).min(kept_to));
        if net.1 - net.0 < 4.0 {
            net = (kept_from.max(kept_to - (self.options.net.1 - self.options.net.0)), kept_to);
        }
        let first = ((net.0 * rate) as u64 - self.kept_base) as usize;
        let last = (((net.1 * rate) as u64).saturating_sub(self.kept_base) as usize).min(self.kept.len());
        let net_t0 = (self.kept_base + first as u64) as f32 / rate;
        let mut analyzer = Analyzer::new(self.rate).expect("a supported rate");
        let audio: Vec<f32> = self.kept[first..last.max(first)].iter().map(|&v| f32::from(v) / 32_767.0).collect();
        drop(self.kept);
        analyzer.push(&audio);
        drop(audio);
        let (prediction, _) = analyzer.into_prediction(model);
        let (beat, down) = prediction.logits();
        let (net_beats, _) = postprocess(&beat, &down);
        let net_beats: Vec<f32> = net_beats.iter().map(|b| b + net_t0).collect();
        let sigmoid = |v: &f32| 1.0 / (1.0 + (-v).exp());
        let (beat_p, down_p): (Vec<f32>, Vec<f32>) = (beat.iter().map(sigmoid).collect(), down.iter().map(sigmoid).collect());
        let env = &onset.values;
        let mut period0 = grid::median_period(&net_beats).unwrap_or_else(|| grid::autocorrelation_period(env, env_t0, span));
        // A short window sometimes hears double time (200+ BPM); no song of
        // the lab set is above 160 BPM: fold into 70–180 BPM, keeping the half
        // of the network's beats that carries the downbeats.
        let mut net_beats = net_beats;
        while period0 < 60.0 / 180.0 {
            period0 *= 2.0;
            let votes_at = |parity: usize| -> f32 {
                net_beats.iter().skip(parity).step_by(2).map(|&b| grid::prob_at(&down_p, net_t0, b)).sum()
            };
            let parity = usize::from(votes_at(1) > votes_at(0));
            net_beats = net_beats.iter().skip(parity).step_by(2).copied().collect();
        }
        let votes = grid::Votes { t0: net_t0, beat: &beat_p, down: &down_p };
        let fitted = grid::fit_grid(env, env_t0, span, period0, &votes, &net_beats);
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var_os("ANALYSIS_DEBUG").is_some() {
            eprintln!("net window {net:?} t0 {net_t0} beats {net_beats:?} period0 {period0} grid {fitted:?} span {span:?}");
        }
        loudness.with_beats(fitted.beats(span.0, span.1))
    }
}

/// [`FocusAnalyzer`] over a whole mono signal (offline).
pub fn analyze_focus(mono: &[f32], sample_rate: u32, focus: Focus, options: FocusOptions, model: &BeatThis) -> Result<BeatAnalysis> {
    let options = options.with_room();
    let limit = if options.decode_to_s.is_finite() { (options.decode_to_s * sample_rate as f32) as usize } else { usize::MAX };
    let mono = &mono[..mono.len().min(limit)];
    let start = if focus == Focus::MixOut && options.tail_s.is_finite() {
        mono.len().saturating_sub((options.tail_s * sample_rate as f32) as usize)
    } else {
        0
    };
    let mut analyzer = FocusAnalyzer::new_at(sample_rate, focus, options, start as u64)?;
    for piece in mono[start..].chunks(sample_rate as usize) {
        analyzer.push(piece);
    }
    Ok(analyzer.finish(model))
}

/// Streaming analysis: push mono audio in pieces, then [`Analyzer::finish`].
pub struct Analyzer {
    loudness: Loudness,
    decimator: Option<Decimator>,
    mel: MelFrontend,
    scratch: Vec<f32>,
    /// Samples pushed so far, and the window (in samples) the network sees.
    pushed: u64,
    from: u64,
    to: u64,
    /// The window's start in seconds (added to the beat times).
    offset_s: f32,
}

impl Analyzer {
    pub fn new(sample_rate: u32) -> Result<Self> {
        let decimator = match sample_rate {
            44_100 => Some(Decimator::new()),
            22_050 => None,
            other => bail!("track analysis supports 44.1 and 22.05 kHz, not {other} Hz"),
        };
        Ok(Self {
            loudness: Loudness::new(sample_rate),
            decimator,
            mel: MelFrontend::new(),
            scratch: Vec::new(),
            pushed: 0,
            from: 0,
            to: u64::MAX,
            offset_s: 0.0,
        })
    }

    /// An analyzer whose network sees only `from_s..to_s` of what is pushed
    /// (the loudness sees everything); beat times stay song times.
    pub fn with_window(sample_rate: u32, from_s: f32, to_s: f32) -> Result<Self> {
        let mut analyzer = Self::new(sample_rate)?;
        let rate = f64::from(sample_rate);
        analyzer.from = (f64::from(from_s.max(0.0)) * rate) as u64;
        analyzer.to = if to_s.is_finite() { (f64::from(to_s.max(0.0)) * rate) as u64 } else { u64::MAX };
        analyzer.offset_s = (analyzer.from as f64 / rate) as f32;
        Ok(analyzer)
    }

    /// Size the spectrogram for a track of about `seconds` (it then never
    /// reallocates, which in wasm would leave the old copy's memory behind).
    pub fn reserve_seconds(&mut self, seconds: f32) {
        let frames = (seconds.max(0.0) * FPS as f32) as usize + 2;
        self.mel.out.reserve(frames.saturating_sub(self.mel.out.len() / N_MELS) * N_MELS);
    }

    pub fn push(&mut self, mono: &[f32]) {
        self.loudness.push(mono);
        let start = self.pushed;
        self.pushed += mono.len() as u64;
        let lo = self.from.clamp(start, self.pushed) - start;
        let hi = self.to.clamp(start, self.pushed) - start;
        let mono = &mono[lo as usize..hi as usize];
        if mono.is_empty() {
            return;
        }
        match &mut self.decimator {
            Some(decimator) => {
                self.scratch.clear();
                decimator.push(mono, &mut self.scratch);
                self.mel.push(&self.scratch);
            }
            None => self.mel.push(mono),
        }
    }

    /// The analysis, with the network run in one go.
    pub fn finish(self, model: &BeatThis) -> BeatAnalysis {
        let (mut prediction, loudness) = self.into_prediction(model);
        while prediction.step() {}
        loudness.with_beats(prediction.finish())
    }

    /// The whole spectrogram and the loudness summary; the network then runs
    /// chunk by chunk ([`Prediction::step`]).
    /// Like [`Analyzer::into_prediction`], the network on `window` only
    /// (which must lie inside the analyzer's own window, if it has one).
    pub fn into_prediction_window(self, model: &BeatThis, window: NetWindow) -> (Prediction<'_>, LoudnessSummary) {
        let offset_s = self.offset_s;
        let (mut prediction, loudness) = self.into_prediction(model);
        let (from_s, to_s) = window.span(&loudness);
        let frames = prediction.mel.len() / N_MELS;
        let first = (((from_s - offset_s) * FPS as f32).round().max(0.0) as usize).min(frames);
        let end = (((to_s - offset_s) * FPS as f32).round().max(0.0) as usize).clamp(first, frames);
        if first > 0 || end < frames {
            let mut mel = std::mem::take(&mut prediction.mel);
            mel.truncate(end * N_MELS);
            mel.drain(..first * N_MELS);
            prediction = Prediction::new(model, mel);
            prediction.offset_s = offset_s + first as f32 / FPS as f32;
        }
        (prediction, loudness)
    }

    pub fn into_prediction(mut self, model: &BeatThis) -> (Prediction<'_>, LoudnessSummary) {
        if let Some(decimator) = &mut self.decimator {
            self.scratch.clear();
            decimator.finish(&mut self.scratch);
            self.mel.push(&self.scratch);
        }
        let mel = self.mel.finish();
        let mut prediction = Prediction::new(model, mel);
        prediction.offset_s = self.offset_s;
        (prediction, self.loudness.summary())
    }

    /// The log-mel spectrogram so far (`(frames, 128)`); for tests.
    pub fn spectrogram(mut self) -> Vec<f32> {
        if let Some(decimator) = &mut self.decimator {
            self.scratch.clear();
            decimator.finish(&mut self.scratch);
            self.mel.push(&self.scratch);
        }
        self.mel.finish()
    }
}

/// The network over a whole spectrogram, a slice of a chunk (a layer) per
/// [`Prediction::step`].
pub struct Prediction<'a> {
    model: &'a BeatThis,
    mel: Vec<f32>,
    /// Chunk starts still to run, the last one first.
    starts: Vec<i64>,
    /// Next step of the chunk `starts.last()`.
    step: usize,
    workspace: Workspace,
    beat: Vec<f32>,
    down: Vec<f32>,
    /// Added to the times (a window's start in the song).
    offset_s: f32,
}

impl<'a> Prediction<'a> {
    fn new(model: &'a BeatThis, mel: Vec<f32>) -> Self {
        let len = mel.len() / N_MELS;
        // keep_first: the last chunk first, so earlier ones overwrite it
        let starts = beat_this::chunk_starts(len);
        Self { model, mel, starts, step: 0, workspace: Workspace::default(), beat: vec![-1000.0; len], down: vec![-1000.0; len], offset_s: 0.0 }
    }

    /// Run the next step (in wasm about 0.1–0.4 s of CPU); false when all
    /// are done.
    pub fn step(&mut self) -> bool {
        let Some(&start) = self.starts.last() else { return false };
        self.model.chunk_step(&mut self.workspace, &self.mel, start, self.step, &mut self.beat, &mut self.down);
        self.step += 1;
        if self.step == self.model.chunk_steps() {
            self.step = 0;
            self.starts.pop();
        }
        !self.starts.is_empty()
    }

    /// Frame logits → beat and downbeat times.
    pub fn finish(mut self) -> (Vec<f32>, Vec<f32>) {
        while self.step() {}
        let (mut beats, mut downbeats) = postprocess(&self.beat, &self.down);
        for t in beats.iter_mut().chain(downbeats.iter_mut()) {
            *t += self.offset_s;
        }
        (beats, downbeats)
    }

    /// Song time of the spectrogram's first frame (added to the times).
    pub fn offset_s(&self) -> f32 {
        self.offset_s
    }

    /// The spectrogram being analysed, `(frames, 128)`.
    pub fn spectrogram(&self) -> &[f32] {
        &self.mel
    }

    /// The frame logits (beat, downbeat); [`postprocess`] turns them into times.
    pub fn logits(mut self) -> (Vec<f32>, Vec<f32>) {
        while self.step() {}
        (self.beat, self.down)
    }
}

/// Frames per second of the spectrogram and the logits.
const FPS: f64 = 50.0;

/// Beat This's minimal postprocessor (`postp_minimal`).
pub fn postprocess(beat: &[f32], down: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let beats = peaks(beat);
    let mut downs: Vec<f64> = peaks(down)
        .into_iter()
        .map(|d| {
            // snap to the nearest beat (the first on a tie, like argmin)
            let mut best = d;
            let mut dist = f64::INFINITY;
            for &b in &beats {
                if (b - d).abs() < dist {
                    dist = (b - d).abs();
                    best = b;
                }
            }
            best
        })
        .collect();
    if beats.is_empty() {
        downs.clear();
    }
    downs.sort_by(f64::total_cmp);
    downs.dedup();
    let seconds = |v: Vec<f64>| v.into_iter().map(|f| (f / FPS) as f32).collect();
    (seconds(beats), seconds(downs))
}

/// Frames (possibly fractional) of the logit peaks: local maxima within ±3
/// frames that are above 0, runs of adjacent peaks merged into their mean.
fn peaks(logits: &[f32]) -> Vec<f64> {
    let n = logits.len();
    let mut out = Vec::new();
    let mut run: Option<(f64, f64)> = None; // (mean, count)
    for i in 0..n {
        let v = logits[i];
        if v <= 0.0 {
            continue;
        }
        let window = &logits[i.saturating_sub(3)..(i + 4).min(n)];
        if window.iter().any(|&w| w > v) {
            continue;
        }
        run = match run {
            // (Python compares with the running mean, not the last peak)
            Some((mean, count)) if i as f64 - mean <= 1.0 => {
                let count = count + 1.0;
                Some((mean + (i as f64 - mean) / count, count))
            }
            Some((mean, _)) => {
                out.push(mean);
                Some((i as f64, 1.0))
            }
            None => Some((i as f64, 1.0)),
        };
    }
    if let Some((mean, _)) = run {
        out.push(mean);
    }
    out
}

/// Levels of 100 ms windows (mono, dB), as the DJ lab's `loudness`.
#[derive(Clone)]
struct Loudness {
    window: usize,
    energy: f64,
    count: usize,
    db: Vec<f32>,
    samples: u64,
    sample_rate: u32,
    /// Song sample of the first pushed sample, and samples still to skip so
    /// the windows sit on the song's 100 ms grid.
    start: u64,
    skip: usize,
}

impl Loudness {
    fn new(sample_rate: u32) -> Self {
        Self::new_at(sample_rate, 0)
    }

    /// For audio that starts at song sample `start` (a decoded tail).
    fn new_at(sample_rate: u32, start: u64) -> Self {
        let window = sample_rate as usize / 10;
        let skip = (window - (start % window as u64) as usize) % window;
        Self { window, energy: 0.0, count: 0, db: Vec::new(), samples: 0, sample_rate, start, skip }
    }

    fn push(&mut self, mono: &[f32]) {
        self.samples += mono.len() as u64;
        let skip = self.skip.min(mono.len());
        self.skip -= skip;
        for &s in &mono[skip..] {
            self.energy += f64::from(s * s);
            self.count += 1;
            if self.count == self.window {
                self.db.push((10.0 * (self.energy / self.window as f64 + 1e-12).log10()) as f32);
                self.energy = 0.0;
                self.count = 0;
            }
        }
    }

    /// The song's head only (the entry analysis): the loud average of a head
    /// is often too low (a quiet intro weighs more than in the whole song),
    /// so `full_start` is the later of the lab's rule and the first 4 s window
    /// within 5 dB of the head's loudest 4 s window — on the 40 lab songs
    /// (60 s heads) never more than 0.7 s before the full-song value.
    fn summary_head(self) -> LoudnessSummary {
        const W4: usize = 40;
        let db: Vec<f64> = self.db.iter().map(|&d| f64::from(d)).collect();
        let means: Vec<f64> = db.windows(W4).map(|w| w.iter().sum::<f64>() / W4 as f64).collect();
        let loudest = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let by_peak = means.iter().take(900).position(|&m| m > loudest - 5.0);
        let mut summary = self.summary();
        if let Some(i) = by_peak {
            // (a head starts at song time 0, so its windows are song windows)
            summary.full_start = summary.full_start.max(i as f32 / 10.0);
        }
        summary
    }

    fn summary(self) -> LoudnessSummary {
        let rate = f64::from(self.sample_rate);
        // times below are relative to the first whole window, then shifted
        let first = ((self.start + (self.window as u64 - self.start % self.window as u64) % self.window as u64) as f64 / rate) as f32;
        let end = ((self.start + self.samples) as f64 / rate) as f32;
        let duration = end - first;
        let db = &self.db;
        let loud: Vec<f64> = db.iter().filter(|&&d| d > -40.0).map(|&d| f64::from(d)).collect();
        let avg = if loud.is_empty() { -20.0 } else { loud.iter().sum::<f64>() / loud.len() as f64 };
        let mean = |i: usize, w: usize| db[i..i + w].iter().map(|&d| f64::from(d)).sum::<f64>() / w as f64;
        const W1: usize = 10;
        const W4: usize = 40;
        let mut loud_end = duration;
        if db.len() >= W1
            && let Some(i) = (0..=db.len() - W1).rev().find(|&i| mean(i, W1) > avg - 9.0)
        {
            loud_end = (i + W1) as f32 / 10.0;
        }
        let loud_start = if db.len() >= W1 {
            (0..=db.len() - W1).find(|&i| mean(i, W1) > avg - 9.0).map_or(0.0, |i| i as f32 / 10.0)
        } else {
            0.0
        };
        let full_start = if db.len() >= W4 {
            (0..=db.len() - W4).take(900).find(|&i| mean(i, W4) > avg - 3.0).map_or(loud_start, |i| i as f32 / 10.0)
        } else {
            loud_start
        };
        LoudnessSummary { full_start: first + full_start, loud_end: first + loud_end, duration: end }
    }
}

/// The loudness part of a [`BeatAnalysis`].
pub struct LoudnessSummary {
    pub full_start: f32,
    pub loud_end: f32,
    pub duration: f32,
}

impl LoudnessSummary {
    pub fn with_beats(self, (beats, downbeats): (Vec<f32>, Vec<f32>)) -> BeatAnalysis {
        BeatAnalysis { beats, downbeats, full_start: self.full_start, loud_end: self.loud_end, duration: self.duration }
    }
}

/// 44.1 → 22.05 kHz with soxr's default (HQ) filter: `y[n] = Σ h[k]·x[2n−k]`
/// over |k| ≤ 200, the signal zero outside, `ceil(len / 2)` outputs — soxr
/// itself within 5e-7.
struct Decimator {
    /// Input from index `base` on (negative: the leading zeros).
    buf: Vec<f32>,
    base: i64,
    /// Next output index.
    next: i64,
    /// Inputs pushed.
    len: i64,
    /// `HALFBAND[1..]` reversed, so both halves of the symmetric filter are
    /// plain dot products.
    rev: [f32; TAPS],
}

const TAPS: usize = 200;

impl Decimator {
    fn new() -> Self {
        let mut rev = [0.0; TAPS];
        for (i, r) in rev.iter_mut().enumerate() {
            *r = HALFBAND[TAPS - i];
        }
        Self { buf: vec![0.0; TAPS], base: -(TAPS as i64), next: 0, len: 0, rev }
    }

    fn push(&mut self, x: &[f32], out: &mut Vec<f32>) {
        self.buf.extend_from_slice(x);
        self.len += x.len() as i64;
        self.run(out, self.len);
    }

    fn finish(&mut self, out: &mut Vec<f32>) {
        let outputs = (self.len + 1) / 2;
        self.buf.extend(std::iter::repeat_n(0.0, 2 * TAPS + 2));
        while self.next < outputs {
            self.one(out);
        }
    }

    /// Every output whose inputs up to `2n + 200` are in (`available` real
    /// inputs pushed).
    fn run(&mut self, out: &mut Vec<f32>, available: i64) {
        while 2 * self.next + (TAPS as i64) < available {
            self.one(out);
        }
        let keep_from = 2 * self.next - TAPS as i64;
        let drop = (keep_from - self.base).max(0) as usize;
        if drop > 65_536 {
            self.buf.drain(..drop);
            self.base += drop as i64;
        }
    }

    fn one(&mut self, out: &mut Vec<f32>) {
        let c = (2 * self.next - self.base) as usize;
        let lo = &self.buf[c - TAPS..c];
        let hi = &self.buf[c + 1..c + 1 + TAPS];
        let y = HALFBAND[0] * self.buf[c] + dot(&self.rev, lo) + dot(&HALFBAND[1..], hi);
        out.push(y);
        self.next += 1;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

const N_FFT: usize = 1024;
const HOP: usize = 441;
const BINS: usize = N_FFT / 2 + 1;

/// torchaudio `MelSpectrogram(22050, n_fft=1024, hop_length=441, f_min=30,
/// f_max=11000, n_mels=128, mel_scale="slaney", normalized="frame_length",
/// power=1)` with centred, reflect-padded frames, then `log1p(1000·x)`.
struct MelFrontend {
    /// 22.05 kHz input from index `base` on.
    buf: Vec<f32>,
    base: usize,
    len: usize,
    /// Next frame.
    next: usize,
    /// Frame 0 and 1 reach back into the reflection of the first samples.
    head: Vec<f32>,
    window: Vec<f32>,
    fft: Fft,
    /// Per mel band: first bin and weights.
    filters: Vec<(usize, Vec<f32>)>,
    frame: Vec<f32>,
    out: Vec<f32>,
}

impl MelFrontend {
    fn new() -> Self {
        let window = (0..N_FFT)
            .map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        Self {
            buf: Vec::new(),
            base: 0,
            len: 0,
            next: 0,
            head: Vec::new(),
            window,
            fft: Fft::new(N_FFT),
            filters: mel_filters(),
            frame: vec![0.0; N_FFT],
            out: Vec::new(),
        }
    }

    fn push(&mut self, x: &[f32]) {
        if self.head.len() < N_FFT / 2 + 1 {
            let take = (N_FFT / 2 + 1 - self.head.len()).min(x.len());
            self.head.extend_from_slice(&x[..take]);
        }
        self.buf.extend_from_slice(x);
        self.len += x.len();
        // frame t covers samples t·441 − 512 ..= t·441 + 511
        while self.next * HOP + N_FFT / 2 < self.len && self.head.len() > N_FFT / 2 {
            self.compute(None);
        }
        let keep_from = (self.next * HOP).saturating_sub(N_FFT / 2);
        if keep_from > self.base + 65_536 {
            self.buf.drain(..keep_from - self.base);
            self.base = keep_from;
        }
    }

    fn finish(mut self) -> Vec<f32> {
        // reflect padding needs more than 512 samples (as torch)
        if self.len > N_FFT / 2 {
            let frames = 1 + self.len / HOP;
            while self.next < frames {
                self.compute(Some(self.len));
            }
        }
        self.out
    }

    fn sample(&self, i: i64, end: Option<usize>) -> f32 {
        let mut i = i;
        if i < 0 {
            return self.head[(-i) as usize];
        }
        if let Some(len) = end
            && i >= len as i64
        {
            i = 2 * (len as i64 - 1) - i;
        }
        self.buf[i as usize - self.base]
    }

    fn compute(&mut self, end: Option<usize>) {
        let first = (self.next * HOP) as i64 - (N_FFT / 2) as i64;
        let mut frame = std::mem::take(&mut self.frame);
        for (n, v) in frame.iter_mut().enumerate() {
            *v = self.sample(first + n as i64, end) * self.window[n];
        }
        let spectrum = self.fft.magnitudes(&frame);
        self.frame = frame;
        let norm = 1.0 / (N_FFT as f32).sqrt();
        for (lo, weights) in &self.filters {
            let e: f32 = weights.iter().zip(&spectrum[*lo..]).map(|(w, m)| w * m).sum();
            self.out.push((1000.0 * e * norm).ln_1p());
        }
        self.next += 1;
    }
}

/// torchaudio `melscale_fbanks(513, 30, 11000, 128, 22050, norm=None,
/// mel_scale="slaney")`, as sparse rows.
fn mel_filters() -> Vec<(usize, Vec<f32>)> {
    fn hz_to_mel(f: f64) -> f64 {
        let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4f64.ln() / 27.0;
        if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp }
    }
    fn mel_to_hz(m: f64) -> f64 {
        let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4f64.ln() / 27.0;
        if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m }
    }
    let (m_min, m_max) = (hz_to_mel(30.0), hz_to_mel(11_000.0));
    let f_pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let freqs: Vec<f64> = (0..BINS).map(|k| 11_025.0 * k as f64 / (BINS - 1) as f64).collect();
    (0..N_MELS)
        .map(|m| {
            let weights: Vec<f32> = freqs
                .iter()
                .map(|&f| {
                    let down = (f - f_pts[m]) / (f_pts[m + 1] - f_pts[m]);
                    let up = (f_pts[m + 2] - f) / (f_pts[m + 2] - f_pts[m + 1]);
                    down.min(up).max(0.0) as f32
                })
                .collect();
            let lo = weights.iter().position(|&w| w > 0.0).unwrap_or(0);
            let hi = weights.iter().rposition(|&w| w > 0.0).map_or(lo, |i| i + 1);
            (lo, weights[lo..hi].to_vec())
        })
        .collect()
}

/// Radix-2 complex FFT of a real frame; returns `|X[k]|` for `k ≤ n/2`.
struct Fft {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    mag: Vec<f32>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let (cos, sin) = (0..n / 2)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .unzip();
        Self { n, cos, sin, re: vec![0.0; n], im: vec![0.0; n], mag: vec![0.0; n / 2 + 1] }
    }

    fn magnitudes(&mut self, x: &[f32]) -> &[f32] {
        let n = self.n;
        let bits = n.trailing_zeros();
        for (i, &v) in x.iter().enumerate() {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            self.re[j] = v;
            self.im[j] = 0.0;
        }
        let mut size = 2;
        while size <= n {
            let half = size / 2;
            let step = n / size;
            for start in (0..n).step_by(size) {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = self.re[b] * wr - self.im[b] * wi;
                    let ti = self.re[b] * wi + self.im[b] * wr;
                    self.re[b] = self.re[a] - tr;
                    self.im[b] = self.im[a] - ti;
                    self.re[a] += tr;
                    self.im[a] += ti;
                }
            }
            size *= 2;
        }
        for k in 0..=n / 2 {
            self.mag[k] = self.re[k].hypot(self.im[k]);
        }
        &self.mag
    }
}

/// soxr's 44.1 → 22.05 kHz filter (`soxr.resample`, default quality),
/// `h[0..=200]` of the symmetric impulse response, measured with unit
/// impulses at even and odd input positions.
#[rustfmt::skip]
const HALFBAND: [f32; TAPS + 1] = [
    0.47843587, 0.31752712, 0.021483934, -0.103764735,
    -0.02124477, 0.059796467, 0.020850463, -0.040127836,
    -0.02030735, 0.028608462, 0.019624168, -0.020845246,
    -0.018811718, 0.01515715, 0.017882764, -0.010765795,
    -0.016851714, 0.007263194, 0.015734274, -0.00441558,
    -0.014547175, 0.00207924, 0.013307779, -0.0001603062,
    -0.012033744, -0.0014061689, 0.010742681, 0.002667794,
    -0.0094517805, -0.0036615066, 0.008177526, 0.0044176322,
    -0.006935385, -0.0049623493, 0.0057395287, 0.0053191953,
    -0.004602653, -0.0055100136, 0.0035357517, 0.0055554938,
    -0.0025480026, -0.00547543, 0.0016466845, 0.005288887,
    -0.0008371187, -0.005014159, 0.00012269332, 0.0046687466,
    0.0004951003, -0.004269246, -0.0010165772, 0.0038312017,
    0.0014437139, -0.0033690212, -0.0017800243, 0.00289584,
    0.0020303505, -0.0024234145, -0.0022006594, 0.0019620634,
    0.002297861, -0.0015206054, -0.0023295502, 0.001106342,
    0.0023038003, -0.0007250959, -0.002228962, 0.00038118288,
    0.0021134275, -7.755514e-05, -0.001965466, -0.00018414907,
    0.0017930517, 0.000403509, -0.0016036923, -0.0005811346,
    0.0014043411, 0.0007185978, -0.0012012676, -0.0008182359,
    0.001000001, 0.0008830024, -0.00080527714, -0.0009163539,
    0.00062103174, 0.00092205487, -0.00045037435, -0.00090406457,
    0.00029563764, 0.0008663978, -0.0001584118, -0.00081304065,
    3.9581646e-05, 0.00074779877, 6.0581417e-05, -0.0006742566,
    -0.0001423465, 0.0005956767, 0.00020646097, -0.0005150084,
    -0.00025403523, 0.0004347806, 0.00028647826, -0.00035712367,
    -0.0003054085, 0.00028377865, 0.0003125816, -0.00021606582,
    -0.00030980445, 0.00015494926, 0.0002988901, -0.00010102913,
    -0.00028159725, 5.458918e-05, 0.00025956382, -1.5657986e-05,
    -0.00023432204, -1.5989692e-05, 0.00020722132, 4.0756637e-05,
    -0.0001794354, -5.9183305e-05, 0.00015195446, 7.192575e-05,
    -0.00012558363, -7.9696256e-05, 0.0001009382, 8.323237e-05,
    -7.8477104e-05, -8.331168e-05, 5.845757e-05, 8.0606755e-05,
    -4.106604e-05, -7.581021e-05, 2.6326083e-05, 6.9538146e-05,
    -1.4168585e-05, -6.232578e-05, 4.450075e-06, 5.4640444e-05,
    3.0274118e-06, -4.686228e-05, -8.504128e-06, 3.930671e-05,
    1.2247707e-05, -3.219711e-05, -1.452303e-05, 2.5704232e-05,
    1.559721e-05, -1.9933927e-05, -1.573228e-05, 1.4930669e-05,
    1.5155167e-05, -1.070899e-05, -1.4079573e-05, 7.237053e-06,
    1.2675282e-05, -4.4754947e-06, -1.1100877e-05, 2.3479543e-06,
    9.472439e-06, -7.702149e-07, -7.879189e-06, -3.3696497e-07,
    6.3875277e-06, 1.0489279e-06, -5.0462413e-06, -1.459398e-06,
    3.880249e-06, 1.6374057e-06, -2.8879008e-06, -1.642066e-06,
    2.078759e-06, 1.539325e-06, -1.4364665e-06, -1.3629045e-06,
    9.472677e-07, 1.1509658e-06, -5.807799e-07, -9.340398e-07,
    3.2396824e-07, 7.2955993e-07, -1.5215227e-07, -5.4558603e-07,
    4.2712692e-08, 3.863458e-07, 1.2606167e-08, -2.6012322e-07,
    -3.4801783e-08, 1.6708401e-07, 4.1547008e-08, -9.879096e-08,
    -3.9708727e-08, 4.873243e-08, 2.6130728e-08, -9.485167e-09,
    4.0048337e-10, 7.306777e-09, -5.4896345e-09, -4.4313647e-10,
    8.350828e-09,
];

/// F-measure of `estimate` against `reference` (times in seconds) with a
/// tolerance of ±`window` seconds, each reference beat matched at most once
/// (mir_eval's `beat.f_measure` without its 5 s trim).
pub fn f_measure(reference: &[f64], estimate: &[f64], window: f64) -> f64 {
    if reference.is_empty() && estimate.is_empty() {
        return 1.0;
    }
    if reference.is_empty() || estimate.is_empty() {
        return 0.0;
    }
    // both sorted; beats are further apart than twice the window, so the
    // greedy walk finds the maximum matching
    let (mut i, mut j, mut hits) = (0, 0, 0usize);
    while i < reference.len() && j < estimate.len() {
        let d = estimate[j] - reference[i];
        if d.abs() <= window {
            hits += 1;
            i += 1;
            j += 1;
        } else if d < 0.0 {
            j += 1;
        } else {
            i += 1;
        }
    }
    let p = hits as f64 / estimate.len() as f64;
    let r = hits as f64 / reference.len() as f64;
    if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

#[cfg(all(test, feature = "embed-small"))]
mod tests {
    use super::*;

    /// A click every `period` seconds from 0.5 s on (a short decaying burst
    /// of noise, accented every fourth), 44.1 kHz.
    fn click_track(bpm: f64, seconds: f64) -> (Vec<f32>, Vec<f64>) {
        let sr = 44_100.0;
        let period = 60.0 / bpm;
        let mut x = vec![0.0f32; (seconds * sr) as usize];
        let mut beats = Vec::new();
        let mut seed = 12_345u32;
        let mut t = 0.5;
        let mut n = 0;
        while t + 0.1 < seconds {
            beats.push(t);
            let start = (t * sr) as usize;
            let gain = if n % 4 == 0 { 1.0 } else { 0.5 };
            for i in 0..(0.06 * sr) as usize {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
                let env = (-(i as f32) / (0.01 * sr as f32)).exp();
                let tone = (2.0 * std::f32::consts::PI * 60.0 * i as f32 / sr as f32).sin();
                x[start + i] += gain * env * (0.5 * noise + 0.8 * tone);
            }
            t += period;
            n += 1;
        }
        (x, beats)
    }

    #[test]
    fn click_track_beats_are_found() {
        if SMALL0.starts_with(b"version https://git-lfs") {
            return;
        }
        // 16 s: one short chunk, so the test is quick in debug builds
        let (x, truth) = click_track(120.0, 16.0);
        let analysis = analyze_window(small(), &x, 44_100, 0.0).expect("analysis");
        assert!((analysis.duration - 16.0).abs() < 1e-3);
        // every true beat (after the first second) has a detected beat within
        // 20 ms (one frame; the network puts these clicks one frame late)
        let inner: Vec<f64> = truth.iter().copied().filter(|&t| t > 1.0 && t < 15.0).collect();
        let hits = inner
            .iter()
            .filter(|&&t| analysis.beats.iter().any(|&b| (f64::from(b) - t).abs() <= 0.0201))
            .count();
        assert!(hits as f64 >= 0.95 * inner.len() as f64, "{hits} of {} beats found: {:?}", inner.len(), analysis.beats);
        assert!(analysis.beats.len() <= truth.len() + 2, "extra beats: {:?}", analysis.beats);
        assert!(analysis.downbeats.iter().all(|d| analysis.beats.contains(d)));
    }

    #[test]
    fn window_times_are_song_times() {
        if SMALL0.starts_with(b"version https://git-lfs") {
            return;
        }
        let (x, truth) = click_track(120.0, 24.0);
        let from = 8 * 44_100;
        let window = analyze_window(small(), &x[from..], 44_100, 8.0).expect("analysis");
        assert!((window.duration - 24.0).abs() < 1e-3);
        let inner: Vec<f64> = truth.iter().copied().filter(|&t| t > 9.0 && t < 23.0).collect();
        let hits = inner
            .iter()
            .filter(|&&t| window.beats.iter().any(|&b| (f64::from(b) - t).abs() <= 0.0201))
            .count();
        assert!(hits as f64 >= 0.95 * inner.len() as f64, "{hits} of {}: {:?}", inner.len(), window.beats);
        assert!(window.beats.iter().all(|&b| b >= 8.0));
        // the same through a windowed Analyzer (the loudness sees it all)
        let mut analyzer = Analyzer::with_window(44_100, 8.0, 24.0).unwrap();
        for piece in x.chunks(10_000) {
            analyzer.push(piece);
        }
        let windowed = analyzer.finish(small());
        assert_eq!(windowed.beats, window.beats);
        assert!((windowed.duration - 24.0).abs() < 1e-3);
    }

    #[test]
    fn focus_beats_sit_on_the_clicks() {
        if SMALL0.starts_with(b"version https://git-lfs") {
            return;
        }
        let (x, truth) = click_track(126.0, 50.0);
        let a = analyze_focus(&x, 44_100, Focus::MixOut, FocusOptions::default_for(Focus::MixOut), small()).expect("analysis");
        assert!(a.beats.len() > 40, "{:?}", a.beats);
        // (the network may put a beat before the first click or after the last)
        let last = truth[truth.len() - 1] as f32 + 0.1;
        for &b in a.beats.iter().filter(|&&b| b > 0.4 && b < last) {
            let near = truth.iter().map(|&t| (f64::from(b) - t).abs()).fold(f64::INFINITY, f64::min);
            assert!(near <= 0.0201, "beat {b} is {near} s off");
        }
        let period = (a.beats[a.beats.len() - 1] - a.beats[0]) / (a.beats.len() - 1) as f32;
        assert!((period - 60.0 / 126.0).abs() < 0.002, "{period}");
    }

    #[test]
    fn streaming_matches_one_piece() {
        let (x, _) = click_track(100.0, 7.0);
        let mut whole = Analyzer::new(44_100).unwrap();
        whole.push(&x);
        let whole = whole.spectrogram();
        let mut pieces = Analyzer::new(44_100).unwrap();
        for chunk in x.chunks(1_000) {
            pieces.push(chunk);
        }
        let pieces = pieces.spectrogram();
        assert_eq!(whole.len(), (1 + x.len() / 2 / HOP) * N_MELS);
        assert_eq!(whole, pieces);
    }

    #[test]
    fn loudness_finds_the_full_part() {
        // 10 s quiet intro (−33 dB), 60 s loud (−3 dB), 10 s fade to silence:
        // the loud average is about −8 dB, so the first 4 s window within
        // 3 dB of it starts at 9 s (one quiet second in it)
        let sr = 44_100usize;
        let mut x = Vec::new();
        for i in 0..80 * sr {
            let t = i as f32 / sr as f32;
            let amp = if t < 10.0 { 0.03 } else if t < 70.0 { 1.0 } else { (1.0 - (t - 70.0) / 10.0).max(0.0).powi(4) };
            x.push(amp * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        let mut l = Loudness::new(44_100);
        l.push(&x);
        let s = l.summary();
        assert!((s.full_start - 9.0).abs() < 0.05, "{}", s.full_start);
        assert!(s.loud_end > 71.0 && s.loud_end < 75.0, "{}", s.loud_end);
        assert!((s.duration - 80.0).abs() < 1e-3);
    }

    #[test]
    fn f_measure_counts_each_beat_once() {
        let r = [1.0, 1.5, 2.0, 2.5];
        assert_eq!(f_measure(&r, &r, 0.07), 1.0);
        let e = [1.05, 1.06, 2.0, 3.0];
        // 1.05 and 2.0 match: p = 2/4, r = 2/4
        assert!((f_measure(&r, &e, 0.07) - 0.5).abs() < 1e-12);
        assert_eq!(f_measure(&r, &[], 0.07), 0.0);
    }

    #[test]
    fn peaks_merge_adjacent_maxima() {
        let logits = [-1.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5, -3.0];
        assert_eq!(peaks(&logits), vec![1.5, 8.0]);
    }
}
