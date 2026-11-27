//! The transition spec: what every stem of both songs does over the bar ruler.
//! The JSON form is the one the lab wrote by hand (`data/versions/*.json`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Breakpoints `[u, value]`, ascending in `u`.
pub type Points = Vec<[f64; 2]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stem {
    Vocals,
    Drums,
    Bass,
    /// Everything else: the melody and chords ("Melodie" on the site).
    Other,
}

impl Stem {
    pub const ALL: [Stem; 4] = [Stem::Vocals, Stem::Drums, Stem::Bass, Stem::Other];

    pub fn name(self) -> &'static str {
        match self {
            Stem::Vocals => "vocals",
            Stem::Drums => "drums",
            Stem::Bass => "bass",
            Stem::Other => "other",
        }
    }
}

/// How a gain lane moves between two breakpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Shape {
    /// Straight line in amplitude. What the owner asked for.
    Lin,
    /// Smoothstep in amplitude (the renderer's default when a deck names no shape).
    #[default]
    S,
    /// Straight line in dB, silence is -70 dB. Rejected by the owner: its start is audible.
    Db,
}

/// A piece of the song's own bars placed on the ruler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub u0: f64,
    /// `None`: to the end of the deck.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u1: Option<f64>,
    /// The song's position (bars relative to ruler zero) that plays at `u0`.
    pub src: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SendKind {
    Echo,
    Reverb,
}

/// A stem sent to the echo or the reverb.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Send {
    #[serde(rename = "type")]
    pub kind: SendKind,
    pub stem: Stem,
    /// Send level over the ruler.
    pub send: Points,
    /// Echo time in beats of the deck.
    #[serde(default = "Send::default_beats")]
    pub beats: f64,
    #[serde(default = "Send::default_feedback")]
    pub feedback: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<f64>,
    /// Echo band, Hz.
    #[serde(default = "Send::default_hp")]
    pub hp: f64,
    #[serde(default = "Send::default_lp")]
    pub lp: f64,
    /// Reverb decay, seconds.
    #[serde(default = "Send::default_rt60")]
    pub rt60: f64,
    /// Feed the send after the stem's fader.
    #[serde(default)]
    pub post: bool,
}

impl Send {
    fn default_beats() -> f64 {
        0.75
    }
    fn default_feedback() -> f64 {
        0.55
    }
    fn default_hp() -> f64 {
        350.0
    }
    fn default_lp() -> f64 {
        4500.0
    }
    fn default_rt60() -> f64 {
        2.4
    }
    /// Output level of the wet signal (echo 0.7, reverb 0.5 unless given).
    pub fn wet_level(&self) -> f64 {
        self.level.unwrap_or(match self.kind {
            SendKind::Echo => 0.7,
            SendKind::Reverb => 0.5,
        })
    }
}

/// One song in a transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deck {
    /// Unique track identifier.
    pub id: String,
    /// Bar of the song's analysis that sits at `anchor_u`.
    pub anchor_bar: usize,
    #[serde(default)]
    pub anchor_u: f64,
    /// Moves the bar line by whole beats when the analysis has the one on the wrong beat.
    #[serde(default)]
    pub beat_shift: i32,
    /// Bars the tempo grid is fitted on (the part with a steady beat).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_bars: Option<[usize; 2]>,
    /// Beats of this song per ruler bar (4; 2 for a song counted in half time, 6 for a 2:3 lock).
    #[serde(default = "Deck::default_beats_per_bar")]
    pub beats_per_bar: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo_bpm: Option<f64>,
    /// Song A: silent from here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub until: Option<f64>,
    /// Song B: its first sound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<f64>,
    /// Song B without tempo sync: where on A's ruler it starts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_u: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub segments: Vec<Segment>,
    /// Gain per stem over the ruler. A stem without a lane plays at 1.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub lanes: BTreeMap<Stem, Points>,
    #[serde(default)]
    pub shape: Shape,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub shapes: BTreeMap<Stem, Shape>,
    /// High-pass cutoff (Hz) per stem over the ruler.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub hpf: BTreeMap<Stem, Points>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub lpf: BTreeMap<Stem, Points>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sends: Vec<Send>,
    /// Trim of the whole deck in dB over the ruler.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gain_db: Points,
    #[serde(default)]
    pub trim_db: f64,
    #[serde(default)]
    pub nudge_ms: f64,
    #[serde(default = "Deck::default_true")]
    pub auto_nudge: bool,
    /// Display only: the song is counted at half its felt tempo.
    #[serde(default)]
    pub double: bool,
}

impl Deck {
    fn default_beats_per_bar() -> u32 {
        4
    }
    fn default_true() -> bool {
        true
    }
    pub fn shape_of(&self, stem: Stem) -> Shape {
        self.shapes.get(&stem).copied().unwrap_or(self.shape)
    }
}

/// Ruler windows of both songs' drums that are compared to align the beats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignPairs {
    pub a: [f64; 2],
    pub b: [f64; 2],
}

/// A line of the explanation shown on the site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub u: f64,
    pub text: String,
}

/// The four ways a transition is built (docs/RULES.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recipe {
    /// B's voice rides on A's beat, B takes over on the one.
    VoiceOverBeat,
    /// B's beat sneaks under A's last chorus, B's voice starts on the one.
    Sneak,
    /// A's outro beat under B's beatless intro.
    OutroUnderIntro,
    /// Tempo gap or clashing keys: A's hook alone in the room, B hard on the one.
    Gap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spec {
    pub id: String,
    /// The rendered clip on the ruler.
    pub clip: [f64; 2],
    pub first_event: f64,
    pub last_event: f64,
    /// Where kick and bass change hands when that is not the one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swap_u: Option<f64>,
    /// B plays at its own tempo, not locked to A's bars.
    #[serde(default)]
    pub no_sync: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align_pairs: Option<AlignPairs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<Recipe>,
    pub a: Deck,
    pub b: Deck,
    // The rest is for people.
    #[serde(default)]
    pub technique: String,
    #[serde(default)]
    pub why: String,
    #[serde(default)]
    pub answer: String,
    #[serde(default)]
    pub key_txt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<Step>,
    #[serde(default)]
    pub fx: serde_json::Value,
}

impl Spec {
    /// Whether B's drums play under A's before the one (a sneak). When they only start on the one (B's voice
    /// over A's beat, A's outro under B's intro), the two beats never sound together and cannot drift apart.
    pub fn beats_overlap(&self) -> bool {
        self.b.lanes.get(&Stem::Drums).is_none_or(|l| l.iter().any(|p| p[0] < 0.0 && p[1] > 0.0))
    }

    /// Every spec of a lab version file (`data/versions/S7.json`).
    pub fn load_all(path: &std::path::Path) -> anyhow::Result<Vec<Spec>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_lab_version_parses() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/versions");
        let mut n = 0;
        for v in ["S1", "S3", "S4", "S5", "S6", "S7"] {
            let specs = Spec::load_all(&root.join(format!("{v}.json"))).unwrap_or_else(|e| panic!("{v}: {e}"));
            assert!(specs.len() >= 10, "{v}");
            for s in &specs {
                assert!(s.clip[0] < s.clip[1]);
                assert!(!s.a.id.is_empty() && !s.b.id.is_empty());
            }
            n += specs.len();
        }
        assert_eq!(n, 75);
    }

    #[test]
    fn round_trip() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/versions/S7.json");
        let specs = Spec::load_all(&root).unwrap();
        let again: Vec<Spec> = serde_json::from_str(&serde_json::to_string(&specs).unwrap()).unwrap();
        assert_eq!(specs, again);
    }
}
