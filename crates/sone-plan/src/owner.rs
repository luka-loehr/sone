//! What the owner decided for single pairs of songs and for single songs (`data/owner/pair_rules.json`).

use serde::Deserialize;

use crate::PlanOptions;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PairRules {
    #[serde(default)]
    pub pairs: Vec<PairRule>,
    #[serde(default)]
    pub songs: Vec<SongRule>,
}

/// What the owner decided for a song whatever it is paired with.
#[derive(Debug, Clone, Deserialize)]
pub struct SongRule {
    pub id: String,
    /// The song's intro voice never rides on another song's beat: as song B its voice comes on the one.
    #[serde(default)]
    pub no_voice_over_beat_as_b: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PairRule {
    pub a: String,
    pub b: String,
    #[serde(default)]
    pub no_voice_over_beat: bool,
}

impl PairRules {
    /// The rules file, or no rules when there is none.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(serde_json::from_str(&text)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// The options with what the owner decided for the pair `a` → `b`.
    pub fn apply(&self, a: &str, b: &str, opt: PlanOptions) -> PlanOptions {
        let no_voice_over_beat = opt.no_voice_over_beat
            || self.pairs.iter().any(|r| r.a == a && r.b == b && r.no_voice_over_beat)
            || self.songs.iter().any(|r| r.id == b && r.no_voice_over_beat_as_b);
        PlanOptions { no_voice_over_beat, ..opt }
    }
}
