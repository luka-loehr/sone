//! Types every crate shares.
//!
//! A transition is described on one bar ruler `u` (in bars). `u = 0` is "the
//! one": the bar line on which song B takes over. Song A (the one that
//! leaves) plays untouched at its own tempo; song B is cut together from its
//! own bars, mixed from its stems in its own time and then brought to A's
//! tempo with its pitch unchanged.

pub mod analysis;
pub mod beats;
pub mod event;
pub mod example;
pub mod issue;
pub mod spec;

pub use analysis::{BarRow, TrackAnalysis};
pub use beats::BeatTable;
pub use event::{Event, Stats, TtsSlot};
pub use example::{Example, Label, Measures};
pub use issue::Issue;
pub use spec::{AlignPairs, Deck, Points, Recipe, Segment, Send, SendKind, Shape, Spec, Stem, Step};

/// Sample rate of songs, stems and renders.
pub const SAMPLE_RATE: u32 = 44_100;
