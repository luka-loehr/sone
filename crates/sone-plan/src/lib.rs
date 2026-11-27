//! The planner: the rules of `docs/RULES.md` as code.
//!
//! Given two songs at beat resolution it decides how song B takes over from
//! song A: which of the four recipes, where "the one" is, which of B's bars
//! sits on it, where A's voice leaves and B's voice enters, and what every
//! stem does until then. The answer is a `sone_core::Spec`, the same form the
//! lab wrote by hand, so the renderer plays it unchanged.
//!
//! Nothing here is learned. It is the baseline the trained planner has to
//! beat and the fallback when that one is not sure.

pub mod compat;
pub mod keys;
pub mod lanes;
pub mod owner;
pub mod plan;
pub mod tempo;
pub mod tts;
pub mod view;

pub use owner::PairRules;
pub use plan::{Plan, PlanOptions, plan};
pub use tempo::{Lock, TempoFit};
