//! One training example for the planner and the critic, as written by `sone-sim`
//! (JSON lines). Everything is at beat resolution; no audio is needed to train on it.

use serde::{Deserialize, Serialize};

use crate::{issue::Issue, spec::Recipe};

/// Per beat of a song, what the planner may know (see `FEATURES`).
pub type BeatFeatures = Vec<f32>;

/// Names of the per-beat features, in order.
pub const FEATURES: [&str; 24] = [
    "vocals_db", "drums_db", "kick_db", "hats_db", "bass_db", "other_db", "mix_db", "sung",
    "line_end", "beat_in_bar", "bar_in_phrase8", "novelty",
    "chroma_c", "chroma_cs", "chroma_d", "chroma_ds", "chroma_e", "chroma_f", "chroma_fs", "chroma_g", "chroma_gs", "chroma_a", "chroma_as", "chroma_b",
];

/// Per beat of the ruler, the gain of each stem of both decks after its fader
/// (A: vocals, drums, bass, other; B: the same), 0..1.
pub type BeatLanes = [f32; 8];

/// What was measured on the audio and cannot be seen at beat resolution.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Measures {
    /// Offset between the two songs' drum attacks, milliseconds.
    #[serde(default)]
    pub beat_offset_ms: f32,
    /// How much song B is stretched, percent.
    #[serde(default)]
    pub stretch_pct: f32,
    /// Level of the voice that leaks into the muted stems, dB relative to the voice (-60: clean).
    #[serde(default)]
    pub vocal_leak_db: f32,
    /// Difference of the two songs' tuning, cents.
    #[serde(default)]
    pub tuning_diff_cents: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Label {
    /// 1: a transition the rules allow (or the owner accepted), 0: broken.
    pub good: u8,
    /// What is wrong with it.
    #[serde(default)]
    pub issues: Vec<Issue>,
    /// "rules", "perturbed", "owner".
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Example {
    pub id: String,
    pub a: String,
    pub b: String,
    /// Song A, the 64 beats before the one and the 16 after it.
    pub a_beats: Vec<BeatFeatures>,
    /// Song B from 64 beats before its bar on the one to 64 after it (padded with silence before its start).
    pub b_beats: Vec<BeatFeatures>,
    pub tempo_a: f32,
    pub tempo_b: f32,
    /// Beats of B per beat of A on the ruler (1, 0.5, 2, 1.5).
    pub beat_ratio: f32,
    /// Distance on the Camelot wheel (0 same, 1 neighbour or related, more: clash).
    pub key_distance: u8,
    pub recipe: Recipe,
    /// Bars before the one at which the transition starts.
    pub start_bars: f32,
    /// Bars before the one at which B's kick and bass come in (0: on the one).
    pub swap_bars: f32,
    /// Gains over the ruler, four values per bar (per beat), from 16 bars before the one to 4 after.
    pub lanes: Vec<BeatLanes>,
    #[serde(default)]
    pub measures: Measures,
    pub label: Label,
}
