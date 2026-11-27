//! What is known about a song before a transition is planned
//! (`data/analysis/struct/<id>.json`, written by `sone-analysis`).

use serde::{Deserialize, Serialize};

/// One bar of the song. Levels are dB relative to the song's loud level
/// (the 90th percentile of its bars), so 0 is "as loud as the song gets".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BarRow {
    pub bar: usize,
    /// Time of the bar's downbeat, seconds.
    pub t: f64,
    pub len: f64,
    pub mix: f32,
    pub vocals: f32,
    pub drums: f32,
    /// Drum stem below 150 Hz.
    pub kick: f32,
    /// Drum stem above 5 kHz.
    pub hats: f32,
    pub bass: f32,
    pub other: f32,
    /// Vocal level per beat of the bar.
    pub vbeats: [f32; 4],
    /// Is somebody singing on that beat.
    pub sung: [u8; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GridResidual {
    pub rms: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackAnalysis {
    pub id: String,
    pub duration: f64,
    pub bpm: f64,
    /// Beat period of the steady part, seconds.
    pub beat_s: f64,
    pub t0: f64,
    pub grid_resid_ms: GridResidual,
    #[serde(default)]
    pub steady: Vec<f64>,
    #[serde(default)]
    pub local_bpm: Vec<f32>,
    pub drum_delta_ms: f32,
    /// Which beat (mod 4) the analysis takes for the one.
    pub bar_phase: u8,
    pub downbeat_votes: Vec<u32>,
    /// Bars with `bar % 8 == phrase8` start an eight-bar phrase.
    pub phrase8: u8,
    #[serde(default)]
    pub phrase8_scores: Vec<f32>,
    pub phrase4: u8,
    pub loud_db: f32,
    pub bars: Vec<BarRow>,
    pub novelty: Vec<f32>,
    /// Beat times, seconds.
    pub beats: Vec<f64>,
    /// Beat numbers (a missed beat leaves a gap).
    pub beat_idx: Vec<i64>,
}

impl TrackAnalysis {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Indices into `beats` of the downbeats, in bar order.
    pub fn downbeats(&self) -> Vec<usize> {
        (0..self.beats.len()).filter(|&i| self.beat_idx[i].rem_euclid(4) == i64::from(self.bar_phase)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_song_parses() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/analysis/struct");
        let mut n = 0;
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "json") {
                let a = TrackAnalysis::load(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
                assert_eq!(a.beats.len(), a.beat_idx.len());
                assert_eq!(a.downbeats().len(), a.bars.len() + 1, "{}", a.id);
                n += 1;
            }
        }
        assert_eq!(n, 40);
    }
}
