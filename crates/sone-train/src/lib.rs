//! Training programs of the four networks, their data loading and export.
//! The binary `train` (src/main.rs) runs them from `cookbooks/<model>.toml`.

pub mod audio;
pub mod config;
pub mod data_tympanum;
pub mod data_examples;
pub mod data_cochlea;
pub mod losses;
pub mod run;
pub mod synth;
pub mod train_beat;
pub mod train_tympanum;
pub mod train_cochlea;
