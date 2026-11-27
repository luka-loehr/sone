//! The signal processing of the transition renderer (`sone-render`), ported
//! from `reference/python/handmix.py` and its numpy/scipy building blocks.
//! Everything here is pure computation on in-memory buffers, so it builds for
//! `wasm32-unknown-unknown`; WAV files are only touched by the `stretch` tool.
//!
//! Stereo signals are `[f32; 2]` frames at [`SR`] (44.1 kHz).

pub mod echo;
pub mod filter;
pub mod lane;
pub mod limiter;
pub mod loudness;
pub mod np;
pub mod onset;
pub mod reverb;
pub mod stretch;
pub mod sweep;

pub use echo::echo;
pub use filter::{Band, Biquad, Sos, SosState, butter};
pub use lane::lane;
pub use limiter::limiter;
pub use loudness::integrated_loudness;
pub use onset::{onset_env, xcorr_lag};
pub use reverb::{impulse_response, reverb};
pub use stretch::{Stretcher, stretch};
pub use sweep::{SweepKind, sweep};

/// Sample rate everything runs at.
pub const SR: f64 = sone_core::SAMPLE_RATE as f64;

/// Mean power in dB, as the reference's `db()` (float32 mean, +1e-10 floor).
pub fn db_f32(x: &[f32]) -> f32 {
    10.0 * (np::mean_square_f32(x) + 1e-10).log10()
}

pub fn db_f64(x: &[f64]) -> f64 {
    10.0 * (np::mean_square_f64(x) + 1e-10).log10()
}
