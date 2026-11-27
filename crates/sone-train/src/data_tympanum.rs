//! Ear labels: one JSON file per song next to (or pointing at) its audio.
//!
//! ```json
//! {
//!   "audio": "song.flac",             // relative to the JSON file, 44.1 kHz
//!   "beats": [0.52, 1.01],            // seconds
//!   "downbeats": [0.52, 2.48],
//!   "vocals": [[12.3, 18.9]],         // sung segments [start, end]
//!   "line_ends": [18.9],              // a sung line ends here
//!   "phrase_starts": [0.52, 16.2],    // an eight-bar phrase starts here
//!   "key": "A minor",                 // or "F# major", "Bbm", "C:min"; null if unknown
//!   "tuning_cents": -8.0              // deviation from A = 440 Hz; null if unknown
//! }
//! ```
//!
//! A field that is missing (or null) is unknown: its head is masked out of
//! the loss for that song. An empty list means "none in this song".

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sone_infer::dsp::LogMel;
use sone_infer::random::Rng;
use sone_infer::{TympanumConfig, config::TYMPANUM_FRAME_HEADS};
use serde::{Deserialize, Serialize};

use crate::audio;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EarLabel {
    pub audio: PathBuf,
    #[serde(default)]
    pub beats: Option<Vec<f64>>,
    #[serde(default)]
    pub downbeats: Option<Vec<f64>>,
    #[serde(default)]
    pub vocals: Option<Vec<[f64; 2]>>,
    #[serde(default)]
    pub line_ends: Option<Vec<f64>>,
    #[serde(default)]
    pub phrase_starts: Option<Vec<f64>>,
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub tuning_cents: Option<f32>,
}

/// Key name → class (pitch class of the tonic, + 12 for minor).
pub fn parse_key(s: &str) -> Option<usize> {
    let t = s.trim();
    let mut chars = t.chars();
    let root = match chars.next()?.to_ascii_uppercase() {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let rest: String = chars.collect();
    let (shift, rest) = if let Some(r) = rest.strip_prefix('#').or_else(|| rest.strip_prefix('♯')) {
        (1, r)
    } else if let Some(r) = rest.strip_prefix('b').or_else(|| rest.strip_prefix('♭')) {
        (11, r)
    } else {
        (0, rest.as_str())
    };
    let mode = rest.trim_start_matches([':', ' ', '-', '_']).to_ascii_lowercase();
    let minor = mode.starts_with("min") || mode == "m";
    if !(minor || mode.is_empty() || mode.starts_with("maj")) {
        return None;
    }
    Some((root + shift) % 12 + if minor { 12 } else { 0 })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TympanumData {
    /// Directories of label files (`*.json`).
    pub train: Vec<PathBuf>,
    pub valid: Vec<PathBuf>,
    pub chunk_seconds: f64,
    /// Seconds of each validation song (in chunks from the start).
    #[serde(default = "default_valid_seconds")]
    pub valid_seconds: f64,
}

fn default_valid_seconds() -> f64 {
    60.0
}

#[derive(Debug, Clone)]
pub struct TympanumSong {
    pub label: EarLabel,
    pub audio: PathBuf,
    pub len: u64,
    pub key: Option<usize>,
}

pub fn index(roots: &[PathBuf]) -> Result<Vec<TympanumSong>> {
    let mut out = Vec::new();
    for root in roots {
        let root = crate::config::expand(root);
        let mut files: Vec<PathBuf> = std::fs::read_dir(&root)
            .with_context(|| format!("reading {}", root.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        files.sort();
        for f in files {
            let label: EarLabel = serde_json::from_str(&std::fs::read_to_string(&f)?).with_context(|| format!("parsing {}", f.display()))?;
            let audio = if label.audio.is_absolute() { label.audio.clone() } else { f.parent().unwrap_or(Path::new(".")).join(&label.audio) };
            let key = match &label.key {
                Some(k) => Some(parse_key(k).with_context(|| format!("{}: unknown key {k}", f.display()))?),
                None => None,
            };
            let len = audio::length(&audio)?;
            out.push(TympanumSong { label, audio, len, key });
        }
    }
    Ok(out)
}

/// One batch: log-mel `(batch, frames, mels)`, frame targets `(batch, frames, 5)`,
/// head masks `(batch, 5)`, key one-hot `(batch, 24)` (zero row: unknown),
/// tuning `(batch)` in cents with its mask.
pub struct TympanumBatch {
    pub mel: Vec<f32>,
    pub targets: Vec<f32>,
    pub head_mask: Vec<f32>,
    pub key: Vec<f32>,
    pub tuning: Vec<f32>,
    pub tuning_mask: Vec<f32>,
    pub batch: usize,
    pub frames: usize,
}

/// The chunk of `song` from `start` (samples) as one batch row.
fn fill(song: &TympanumSong, start: u64, len: usize, c: &TympanumConfig, mel: &mut LogMel, b: &mut TympanumBatch) -> Result<()> {
    let [l, r] = audio::read(&song.audio, start, len)?;
    let mono: Vec<f32> = l.iter().zip(&r).map(|(a, b)| 0.5 * (a + b)).collect();
    let mut m = Vec::new();
    let t = mel.compute(&mono, &mut m);
    b.mel.extend_from_slice(&m);
    let fps = c.fps();
    let t0 = start as f64 / c.sample_rate as f64;
    let mut tg = vec![0.0f32; t * 5];
    let lab = &song.label;
    let events = [&lab.beats, &lab.downbeats, &None, &lab.line_ends, &lab.phrase_starts];
    for (h, ev) in events.iter().enumerate() {
        if let Some(ev) = ev {
            for &e in ev.iter() {
                let i = ((e - t0) * fps).round() as i64;
                // the frame on the event is 1, its neighbours 0.5 (timing tolerance)
                for (d, v) in [(-1i64, 0.5f32), (0, 1.0), (1, 0.5)] {
                    let j = i + d;
                    if j >= 0 && (j as usize) < t {
                        let at = j as usize * 5 + h;
                        tg[at] = tg[at].max(v);
                    }
                }
            }
        }
    }
    if let Some(segs) = &lab.vocals {
        for i in 0..t {
            let at = t0 + i as f64 / fps;
            if segs.iter().any(|[s, e]| at >= *s && at <= *e) {
                tg[i * 5 + 2] = 1.0;
            }
        }
    }
    b.targets.extend_from_slice(&tg);
    let known = [lab.beats.is_some(), lab.downbeats.is_some(), lab.vocals.is_some(), lab.line_ends.is_some(), lab.phrase_starts.is_some()];
    b.head_mask.extend(known.iter().map(|&k| if k { 1.0 } else { 0.0 }));
    let mut key = [0.0f32; 24];
    if let Some(k) = song.key {
        key[k] = 1.0;
    }
    b.key.extend_from_slice(&key);
    b.tuning.push(lab.tuning_cents.unwrap_or(0.0));
    b.tuning_mask.push(if lab.tuning_cents.is_some() { 1.0 } else { 0.0 });
    b.batch += 1;
    b.frames = t;
    Ok(())
}

impl TympanumBatch {
    fn new() -> Self {
        Self { mel: vec![], targets: vec![], head_mask: vec![], key: vec![], tuning: vec![], tuning_mask: vec![], batch: 0, frames: 0 }
    }
}

pub fn sample(songs: &[TympanumSong], n: usize, len: usize, c: &TympanumConfig, rng: &mut Rng) -> Result<TympanumBatch> {
    if songs.is_empty() {
        bail!("no Ear training songs");
    }
    let mut mel = LogMel::for_tympanum(c);
    let mut b = TympanumBatch::new();
    for _ in 0..n {
        let s = &songs[(rng.next_u64() % songs.len() as u64) as usize];
        let start = if s.len as usize > len { rng.next_u64() % (s.len - len as u64) } else { 0 };
        fill(s, start, len, c, &mut mel, &mut b)?;
    }
    Ok(b)
}

/// Consecutive chunks of a song from its start, up to `seconds`, one batch each.
pub fn song_chunks(song: &TympanumSong, len: usize, seconds: f64, c: &TympanumConfig) -> Result<Vec<TympanumBatch>> {
    let mut mel = LogMel::for_tympanum(c);
    let limit = ((seconds * c.sample_rate as f64) as u64).min(song.len);
    let mut out = Vec::new();
    let mut start = 0u64;
    while start + (len as u64) / 2 <= limit.max(1) {
        let mut b = TympanumBatch::new();
        fill(song, start, len, c, &mut mel, &mut b)?;
        out.push(b);
        start += len as u64;
    }
    Ok(out)
}

pub const HEADS: [&str; 5] = TYMPANUM_FRAME_HEADS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_parse() {
        assert_eq!(parse_key("A minor"), Some(21));
        assert_eq!(parse_key("Am"), Some(21));
        assert_eq!(parse_key("C:min"), Some(12));
        assert_eq!(parse_key("F# major"), Some(6));
        assert_eq!(parse_key("Bb"), Some(10));
        assert_eq!(parse_key("Bbm"), Some(22));
        assert_eq!(parse_key("H"), None);
        assert_eq!(parse_key(sone_infer::config::KEYS[15]), Some(15));
    }
}
