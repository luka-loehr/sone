//! Training data without a studio.
//!
//! The owner accepted a dozen transitions, which is far too few to learn
//! from. But the rules behind them are known, and so is every way in which
//! a transition was wrong. So: plan transitions by the rules for many pairs
//! of songs (good examples), then break each one in a named way (bad
//! examples, the fault is known because it was put there). The owner's own
//! verdicts are written in the same form; they calibrate, the generated ones
//! teach.
//!
//! Everything is at beat resolution and needs no audio.

pub mod build;
pub mod generate;
pub mod owner;
pub mod perturb;
pub mod rng;

pub use build::{A_AFTER, A_BEFORE, B_AFTER, B_BEFORE, LANE_TICKS, example_from_spec};
