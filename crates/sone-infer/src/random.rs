//! Random weights of the right names and shapes, for speed and memory
//! measurements before a trained file exists (and for tests).

use std::collections::BTreeMap;

use crate::config::Shapes;
use crate::st::{self, NamedTensor};

/// xorshift64*: deterministic across platforms, no dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [-a, a).
    pub fn sym(&mut self, a: f32) -> f32 {
        (self.uniform() * 2.0 - 1.0) * a
    }
}

/// Values like a freshly initialised network: gains near 1, deviations
/// positive, matrices uniform in ±1/√fan_in, biases and embeddings small.
pub fn tensors(shapes: &Shapes, seed: u64) -> Vec<NamedTensor> {
    let mut rng = Rng::new(seed);
    shapes
        .iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let values: Vec<f32> = if name.ends_with("norm") || name.ends_with("in_scale") || name.ends_with("_std") {
                (0..n).map(|_| 1.0 + rng.sym(0.1)).collect()
            } else if shape.len() == 2 && !name.starts_with("pos_") && name != "band_emb" {
                let a = 1.0 / (shape[1] as f32).sqrt();
                (0..n).map(|_| rng.sym(a)).collect()
            } else {
                (0..n).map(|_| rng.sym(0.05)).collect()
            };
            (name.clone(), shape.clone(), values)
        })
        .collect()
}

/// A complete weights file for `model` ("stems", "ear", "plan", "critic").
pub fn file(model: &str, config: &impl serde::Serialize, shapes: &Shapes, seed: u64, half: bool) -> anyhow::Result<Vec<u8>> {
    let meta: BTreeMap<String, String> = st::metadata(model, config)?;
    st::write(&tensors(shapes, seed), half, &meta)
}
