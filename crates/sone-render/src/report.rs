//! What a render reports (the same JSON as the reference's `<id>.report.json`)
//! and what the website draws (`<id>.json`).

use std::collections::BTreeMap;

use sone_core::Stem;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub nudge_ms: f64,
    pub clean_vocals: [bool; 2],
    pub leak_free: [bool; 2],
    pub tempo_a: f64,
    pub tempo_b: f64,
    /// Stretch of B, percent of its tempo (positive: sped up).
    pub stretch_pct: f64,
    pub lufs_a: f64,
    pub lufs_b: f64,
    /// Phase correction of each grid from the drum stem, ms.
    pub phase_a_ms: f64,
    pub phase_b_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<Align>,
    pub shortterm_db: ShortTerm,
    pub bars: Vec<BarRow>,
    pub limiter_min_gain_db: f64,
}

/// More than this between the drums' lag in the first and in the second half of the blend, and the
/// songs drift apart audibly: the owner's accepted transitions stay within ±2 ms, the ones he heard as
/// "not synced" drift 14 to 36 ms.
pub const MAX_DRIFT_MS: f64 = 12.0;

/// More than this between the two songs' drums over the whole align window and they are not on top of
/// each other (the alignment did not find them); every accepted transition sits within 1 ms.
pub const MAX_LAG_MS: f64 = 3.0;

impl Report {
    /// How far B's drums lie from A's over the whole align window, ms.
    pub fn lag_ms(&self) -> f64 {
        self.align.as_ref().map_or(0.0, |a| a.all.0)
    }

    /// How far B's drums move against A's from the first to the second half of the blend, ms.
    pub fn drift_ms(&self) -> f64 {
        self.align.as_ref().map_or(0.0, |a| a.second_half.0 - a.first_half.0)
    }
}

/// Beat lag between both decks' drums: (lag ms, sharpness); positive: B early.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Align {
    pub all: (f64, f64),
    pub first_half: (f64, f64),
    pub second_half: (f64, f64),
    pub drums_a_db: f64,
    pub drums_b_db: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ShortTerm {
    #[serde(rename = "ref")]
    pub reference: f64,
    pub min: f64,
    pub max: f64,
}

/// One bar of the meter bridge, dB: the decks, the sum and its bands, every stem
/// of both decks after its fader.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BarRow {
    pub u: i64,
    #[serde(rename = "A")]
    pub a: f64,
    #[serde(rename = "B")]
    pub b: f64,
    pub mix: f64,
    pub low: f64,
    pub high: f64,
    pub a_v: f64,
    pub b_v: f64,
    pub a_d: f64,
    pub b_d: f64,
    pub a_b: f64,
    pub b_b: f64,
    pub a_o: f64,
    pub b_o: f64,
}

impl BarRow {
    pub fn set(&mut self, stem: Stem, a: f64, b: f64) {
        let (x, y) = match stem {
            Stem::Vocals => (&mut self.a_v, &mut self.b_v),
            Stem::Drums => (&mut self.a_d, &mut self.b_d),
            Stem::Bass => (&mut self.a_b, &mut self.b_b),
            Stem::Other => (&mut self.a_o, &mut self.b_o),
        };
        *x = a;
        *y = b;
    }

    pub fn get(&self, stem: Stem) -> (f64, f64) {
        match stem {
            Stem::Vocals => (self.a_v, self.b_v),
            Stem::Drums => (self.a_d, self.b_d),
            Stem::Bass => (self.a_b, self.b_b),
            Stem::Other => (self.a_o, self.b_o),
        }
    }
}

/// Fader lanes of one deck, sampled 16 times per bar of its own ruler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeckLanes {
    /// Ruler position of the first point.
    pub u0: f64,
    /// Ruler bars (of A) between two points.
    pub step_bars: f64,
    pub gain: BTreeMap<Stem, Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanesForSite {
    pub id: String,
    pub clip: [f64; 2],
    pub bar_s: f64,
    pub seconds: f64,
    pub report: Report,
    pub a: DeckLanes,
    pub b: DeckLanes,
}
