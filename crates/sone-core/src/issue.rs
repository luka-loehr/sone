//! What can be wrong with a transition. Every entry is something the owner
//! heard and named (`data/labels.json`), or the cause found behind it.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Issue {
    RollOut,
    PitchRiding,
    BMelodyEntersAtOnce,
    AbruptStemExit,
    TwoMelodies,
    CutNotTransition,
    StemsLeaveSeparately,
    AMelodyLeftEarly,
    WrongRecipe,
    DownbeatOffByOne,
    FadeAtRandomPoint,
    FadeTooShort,
    BMelodyBeforeOne,
    BDrumsBeforeOne,
    FadeCurve,
    ABeatGoneBeforeOne,
    AudibleFadeOnset,
    VocalLeak,
    BeatHoleBeforeOne,
    TailAfterOne,
    BVoiceBeforeOne,
    KeyClashLayering,
    PolyrhythmLayering,
    /// Not named by the owner, but ruled out from the start.
    TwoVoices,
    TwoBasses,
    BeatsOffset,
    VoiceCutInWord,
    /// Named by the owner on 2026-10-01: seconds before the one in which nothing but some drums and
    /// bass play ("3 seconds of kinda nothing happening", "1 sec of pure silence").
    SilenceBeforeOne,
    /// B's voice comes in well after the one, or in the middle of a word ("no reason not to start B's
    /// voice at the 1", "it started mid word").
    BVoiceLate,
    /// The two songs do not go together, however the transition is built ("it's just a bad pair").
    BadPair,
}

impl Issue {
    pub const ALL: [Issue; 30] = [
        Issue::RollOut,
        Issue::PitchRiding,
        Issue::BMelodyEntersAtOnce,
        Issue::AbruptStemExit,
        Issue::TwoMelodies,
        Issue::CutNotTransition,
        Issue::StemsLeaveSeparately,
        Issue::AMelodyLeftEarly,
        Issue::WrongRecipe,
        Issue::DownbeatOffByOne,
        Issue::FadeAtRandomPoint,
        Issue::FadeTooShort,
        Issue::BMelodyBeforeOne,
        Issue::BDrumsBeforeOne,
        Issue::FadeCurve,
        Issue::ABeatGoneBeforeOne,
        Issue::AudibleFadeOnset,
        Issue::VocalLeak,
        Issue::BeatHoleBeforeOne,
        Issue::TailAfterOne,
        Issue::BVoiceBeforeOne,
        Issue::KeyClashLayering,
        Issue::PolyrhythmLayering,
        Issue::TwoVoices,
        Issue::TwoBasses,
        Issue::BeatsOffset,
        Issue::VoiceCutInWord,
        Issue::SilenceBeforeOne,
        Issue::BVoiceLate,
        Issue::BadPair,
    ];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&i| i == self).expect("listed")
    }
}
