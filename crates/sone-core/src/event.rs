//! What the running engine reports, as JSON lines next to the audio.
//! Times are seconds on the output stream's clock.

use serde::{Deserialize, Serialize};

use crate::spec::Recipe;

/// Where the DJ's voice fits: a stretch before the one in which nobody sings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsSlot {
    pub slot_from: f64,
    /// The voice has to be done here (0.3 s before the one).
    pub must_end_by: f64,
    pub max_seconds: f64,
    /// What plays under the voice.
    pub over: String,
    /// How far the music steps back under the voice.
    pub duck_db: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tempo {
    pub a: f64,
    pub b: f64,
    /// "1:1", "1:2", "2:1" or "2:3".
    pub ratio: String,
    pub stretch_pct: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Keys {
    pub a: String,
    pub b: String,
    /// "same", "related", "neighbour" or "clash".
    pub fit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub beat_offset_ms: f32,
    pub voices_at_once: u8,
    pub melodies_at_once: u8,
    pub basses_at_once: u8,
    /// Largest drop of the mix below its level before the transition, dB.
    pub level_dip_db: f32,
    pub lookahead_s: f32,
    /// CPU time spent per second of audio.
    pub cpu_core_fraction: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Sent as soon as the transition is decided, before it starts.
    Plan {
        t: f64,
        a: String,
        b: String,
        recipe: Recipe,
        confidence: f32,
        starts_at: f64,
        one_at: f64,
        ends_at: f64,
        tempo: Tempo,
        key: Keys,
        tts: Option<TtsSlot>,
    },
    /// A plan that was announced will not happen (its audio was not there in time, or another
    /// song was asked for). Another plan may follow.
    PlanDropped { t: f64, reason: String },
    TransitionStarted { t: f64 },
    TheOne { t: f64 },
    TransitionEnded { t: f64 },
    Stats {
        t: f64,
        #[serde(flatten)]
        stats: Stats,
    },
}
