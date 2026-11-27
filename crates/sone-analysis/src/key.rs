//! Key and tuning of a song (`reference/python/keys.py`): the mean chroma of
//! bass + other + vocals (every frame scaled to its strongest class, as
//! librosa's `chroma_cqt`) against the Krumhansl–Kessler profiles of the 24
//! keys; the best correlation wins. The chroma here comes from a short-time
//! Fourier transform folded onto the tuned pitch classes, not librosa's
//! constant-Q transform, so close calls can differ from the reference.

use serde::{Deserialize, Serialize};

use crate::chroma::Chroma;

const MAJ: [f64; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MIN: [f64; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
const NOTE: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
/// Camelot number of the major key on each tonic (C = 8B) and of the minor
/// key on each tonic (A minor = 8A).
const CAM_MAJ: [u8; 12] = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1];
const CAM_MIN: [u8; 12] = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Key {
    /// "C", "F#m", …
    pub key: String,
    pub camelot: String,
    /// Correlation of the best key.
    pub score: f32,
    /// The runner-up: key, Camelot code, correlation.
    pub second: (String, String, f32),
    pub tuning_cents: f32,
    pub chroma: [f32; 12],
}

/// The runner-up counts as nearly as likely within this much correlation.
pub const ALT_MARGIN: f32 = 0.05;

impl Key {
    /// The second key's Camelot code when it is nearly as likely.
    pub fn alternative(&self) -> Option<String> {
        (self.score - self.second.2 < ALT_MARGIN).then(|| self.second.1.clone())
    }
}

fn corr(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let ma = a.iter().sum::<f64>() / 12.0;
    let mb = b.iter().sum::<f64>() / 12.0;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..12 {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    sab / (saa * sbb).sqrt().max(1e-12)
}

/// The key whose profile correlates best with `chroma` (C first).
pub fn estimate(chroma: &[f64; 12], tuning_cents: f64) -> Key {
    let mut all: Vec<(f64, String, String)> = Vec::with_capacity(24);
    for k in 0..12 {
        let rot = |p: &[f64; 12]| -> [f64; 12] { std::array::from_fn(|i| p[(i + 12 - k) % 12]) };
        all.push((corr(&rot(&MAJ), chroma), NOTE[k].to_string(), format!("{}B", CAM_MAJ[k])));
        all.push((corr(&rot(&MIN), chroma), format!("{}m", NOTE[k]), format!("{}A", CAM_MIN[k])));
    }
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    Key {
        key: all[0].1.clone(),
        camelot: all[0].2.clone(),
        score: all[0].0 as f32,
        second: (all[1].1.clone(), all[1].2.clone(), all[1].0 as f32),
        tuning_cents: tuning_cents as f32,
        chroma: chroma.map(|v| v as f32),
    }
}

/// Mean scaled chroma of the song (frames with sound only).
pub fn song_chroma(c: &Chroma) -> [f64; 12] {
    let mut sum = [0.0; 12];
    let mut n = 0usize;
    for (f, &w) in c.frames.iter().zip(&c.weight) {
        if w <= 1e-6 {
            continue;
        }
        for (s, &v) in sum.iter_mut().zip(f) {
            *s += f64::from(v);
        }
        n += 1;
    }
    sum.map(|v| v / n.max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_find_their_own_keys() {
        // the profile itself, rotated to A minor and to E-flat major
        let a_minor: [f64; 12] = std::array::from_fn(|i| MIN[(i + 12 - 9) % 12]);
        let k = estimate(&a_minor, 0.0);
        assert_eq!((k.key.as_str(), k.camelot.as_str()), ("Am", "8A"));
        let es: [f64; 12] = std::array::from_fn(|i| MAJ[(i + 12 - 3) % 12]);
        let k = estimate(&es, 0.0);
        assert_eq!((k.key.as_str(), k.camelot.as_str()), ("D#", "5B"));
    }
}
