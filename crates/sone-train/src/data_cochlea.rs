//! Songs with stems for the Stems network.
//!
//! A song is a directory holding its stems as FLAC or WAV (44.1 kHz, stereo
//! or mono): `vocals` (or `vocals_r`, the BS-RoFormer teacher's), `drums`
//! (or `drums_i`), `bass` (or `bass_i`), optionally `other`, and the mixture
//! as `mixture` or `mix`, or as `<mixture_dir>/<song>.f32` (raw interleaved
//! f32, the `~/djlab/tracks` layout the teachers read). Without an `other`
//! file, other = mixture − vocals − drums − bass (what the teachers define).
//!
//! Training examples are random chunks; with probability `remix_prob` every
//! stem comes from a different random song and position (remixing, standard
//! in source separation), each stem gets a random gain, channels may be
//! swapped, and the mixture is the sum of the four, so targets and input
//! always agree.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sone_infer::random::Rng;
use serde::{Deserialize, Serialize};

use crate::audio;

pub const STEM_NAMES: [&str; 4] = ["vocals", "drums", "bass", "other"];
const ALIASES: [&[&str]; 4] = [&["vocals", "vocals_r"], &["drums", "drums_i"], &["bass", "bass_i"], &["other"]];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CochleaData {
    /// Directories whose subdirectories are songs (or that are songs themselves).
    pub train: Vec<PathBuf>,
    pub valid: Vec<PathBuf>,
    /// Where `<song>.f32` mixtures live, for song directories without a mixture file.
    #[serde(default)]
    pub mixture_dir: Option<PathBuf>,
    pub chunk_seconds: f64,
    #[serde(default = "default_remix")]
    pub remix_prob: f64,
    /// Random gain per stem, dB, uniform in `[min, max]`.
    #[serde(default = "default_gain")]
    pub gain_db: [f64; 2],
    #[serde(default = "default_swap")]
    pub swap_prob: f64,
    /// Songs used for validation (the first N, sorted by name).
    #[serde(default = "default_valid_songs")]
    pub valid_songs: usize,
    /// Seconds of each validation song, from `valid_offset_seconds`.
    #[serde(default = "default_valid_seconds")]
    pub valid_seconds: f64,
    #[serde(default)]
    pub valid_offset_seconds: f64,
}

fn default_remix() -> f64 {
    0.5
}
fn default_gain() -> [f64; 2] {
    [-6.0, 3.0]
}
fn default_swap() -> f64 {
    0.5
}
fn default_valid_songs() -> usize {
    10
}
fn default_valid_seconds() -> f64 {
    60.0
}

#[derive(Debug, Clone)]
pub struct Song {
    pub name: String,
    /// vocals, drums, bass, other (None: derived from the mixture).
    pub stems: [Option<PathBuf>; 4],
    pub mixture: Option<PathBuf>,
    pub len: u64,
}

fn find(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    for n in names {
        for ext in ["flac", "wav"] {
            let p = dir.join(format!("{n}.{ext}"));
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

impl Song {
    fn open(dir: &Path, mixture_dir: Option<&Path>) -> Result<Option<Self>> {
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
        let stems: [Option<PathBuf>; 4] = std::array::from_fn(|i| find(dir, ALIASES[i]));
        let mut mixture = find(dir, &["mixture", "mix"]);
        if mixture.is_none()
            && let Some(md) = mixture_dir
        {
            let p = md.join(format!("{name}.f32"));
            mixture = p.exists().then_some(p);
        }
        if stems[..3].iter().any(Option::is_none) || (stems[3].is_none() && mixture.is_none()) {
            return Ok(None);
        }
        let mut len = u64::MAX;
        for p in stems.iter().flatten().chain(mixture.iter()) {
            len = len.min(audio::length(p)?);
        }
        Ok(Some(Self { name, stems, mixture, len }))
    }

    /// Stem `s` from `start`, `len` samples.
    pub fn read_stem(&self, s: usize, start: u64, len: usize) -> Result<[Vec<f32>; 2]> {
        if let Some(p) = &self.stems[s] {
            return audio::read(p, start, len);
        }
        let mut other = audio::read(self.mixture.as_ref().context("no mixture")?, start, len)?;
        for i in 0..3 {
            let s = self.read_stem(i, start, len)?;
            for c in 0..2 {
                for (o, v) in other[c].iter_mut().zip(&s[c]) {
                    *o -= v;
                }
            }
        }
        Ok(other)
    }
}

/// All songs under the given roots, sorted by name.
pub fn index(roots: &[PathBuf], mixture_dir: Option<&Path>) -> Result<Vec<Song>> {
    let mut songs = Vec::new();
    for root in roots {
        let root = crate::config::expand(root);
        if let Some(s) = Song::open(&root, mixture_dir)? {
            songs.push(s);
            continue;
        }
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
            .with_context(|| format!("reading {}", root.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for d in dirs {
            if let Some(s) = Song::open(&d, mixture_dir)? {
                songs.push(s);
            }
        }
    }
    songs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(songs)
}

/// One batch: stems `(batch, 4, 2, len)` and mixture `(batch, 2, len)`, flat.
pub struct StemsBatch {
    pub stems: Vec<f32>,
    pub mix: Vec<f32>,
    pub batch: usize,
    pub len: usize,
}

pub struct Sampler {
    pub songs: Vec<Song>,
    pub cfg: CochleaData,
    pub len: usize,
}

impl Sampler {
    pub fn new(songs: Vec<Song>, cfg: CochleaData, len: usize) -> Result<Self> {
        let usable = songs.iter().filter(|s| s.len as usize > len).count();
        if usable == 0 {
            bail!("no training song is longer than one chunk ({len} samples)");
        }
        Ok(Self { songs: songs.into_iter().filter(|s| s.len as usize > len).collect(), cfg, len })
    }

    pub fn batch(&self, n: usize, rng: &mut Rng) -> Result<StemsBatch> {
        let len = self.len;
        let mut stems = vec![0.0f32; n * 4 * 2 * len];
        let mut mix = vec![0.0f32; n * 2 * len];
        let pick = |rng: &mut Rng| -> (usize, u64) {
            let s = (rng.next_u64() % self.songs.len() as u64) as usize;
            let off = rng.next_u64() % (self.songs[s].len - len as u64);
            (s, off)
        };
        for b in 0..n {
            let (song, off) = pick(rng);
            let remix = (rng.uniform() as f64) < self.cfg.remix_prob;
            for s in 0..4 {
                let (song, off) = if remix { pick(rng) } else { (song, off) };
                let [mut l, mut r] = self.songs[song].read_stem(s, off, len)?;
                if (rng.uniform() as f64) < self.cfg.swap_prob {
                    std::mem::swap(&mut l, &mut r);
                }
                let [lo, hi] = self.cfg.gain_db;
                let gain = 10f64.powf((lo + (hi - lo) * rng.uniform() as f64) / 20.0) as f32;
                for (c, ch) in [l, r].iter().enumerate() {
                    let dst = &mut stems[((b * 4 + s) * 2 + c) * len..][..len];
                    let m = &mut mix[(b * 2 + c) * len..][..len];
                    for i in 0..len {
                        dst[i] = ch[i] * gain;
                        m[i] += dst[i];
                    }
                }
            }
        }
        Ok(StemsBatch { stems, mix, batch: n, len })
    }
}
