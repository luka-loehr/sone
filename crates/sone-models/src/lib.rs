//! The four networks in Burn, generic over the backend: Stems, Ear, Plan and
//! Critic. Their designs are described next to their loop twins in
//! `sone-infer`; each network exports its tensors under the names
//! `sone_infer::config` lists, so a trained model runs without Burn.

pub mod beat;
pub mod dsp;
pub mod tympanum;
pub mod export;
pub mod layers;
pub mod cochlea;

pub use beat::{BeatNet, BeatOut};
pub use dsp::Stft;
pub use tympanum::{TympanumNet, TympanumOut};
pub use cochlea::{CochleaNet, CochleaOut};
