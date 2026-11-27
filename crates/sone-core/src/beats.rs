//! A song at beat resolution: what the planner, the simulator and the models
//! read (`data/analysis/beats/<id>.json`, written by `sone-analysis`).

use serde::{Deserialize, Serialize};

use crate::example::BeatFeatures;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatTable {
    pub id: String,
    pub duration: f64,
    /// Tempo of the steady part, as the song is counted by the analysis.
    pub bpm: f64,
    pub beat_s: f64,
    /// Beat times on the fitted grid, seconds. Beat `i` of the table.
    pub times: Vec<f64>,
    /// Position of the beat in its bar, 0..3, after the bar-phase check.
    pub beat_in_bar: Vec<u8>,
    /// Beats the analysis' bar line was moved by to put the backbeat on 2 and 4 (0, or -1, 1, 2).
    pub beat_shift: i8,
    /// How sure the bar-phase check is, 0..1.
    pub phase_confidence: f32,
    /// Index into `times` of the first beat of every bar.
    pub bars: Vec<usize>,
    /// Bars with `bar % 8 == phrase8` start an eight-bar phrase.
    pub phrase8: u8,
    pub key: String,
    pub camelot: String,
    /// The second-best key, when it is nearly as likely (major and minor of one tonic are easily confused).
    #[serde(default)]
    pub camelot_alt: String,
    pub tuning_cents: f32,
    /// Level of the song's loud part, dB full scale (the reference of every dB below).
    pub loud_db: f32,
    /// `sone_core::example::FEATURES` per beat.
    pub features: Vec<BeatFeatures>,
    /// Vocal level per sixteenth note of every beat, dB relative to the song's sung level.
    pub v16: Vec<[f32; 4]>,
    /// Sixteenth notes (index = beat * 4 + sixteenth) on which a sung line ends.
    pub line_ends: Vec<usize>,
    /// Sixteenth notes on which a sung line begins.
    pub line_starts: Vec<usize>,
}

impl BeatTable {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Bar index of a beat.
    pub fn bar_of(&self, beat: usize) -> usize {
        match self.bars.binary_search(&beat) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
    }
}
