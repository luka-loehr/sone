//! `Example`s for the planner and the critic, from the JSON-lines files the
//! simulator writes (one `sone_core::Example` per line). Parsed once and
//! packed: token inputs as f16 (about 10 KB per example), everything else
//! f32. The planner keeps only `label.good == 1`.

use std::io::BufRead;
use std::path::PathBuf;

use anyhow::{Context, Result};
use sone_core::Example;
use sone_infer::beat::{pack, recipe_index};
use sone_infer::config::{BeatConfig, BeatKind};
use sone_infer::ops::{f16_to_f32, f32_to_f16};
use sone_infer::random::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamplesData {
    /// `.jsonl` files, or directories of them.
    pub train: Vec<PathBuf>,
    /// Held-out files; when empty, `valid_fraction` of the training examples
    /// (chosen by a hash of the example id, so stable across runs).
    #[serde(default)]
    pub valid: Vec<PathBuf>,
    #[serde(default = "default_fraction")]
    pub valid_fraction: f64,
    /// Read at most this many examples (memory: about 10 KB each).
    #[serde(default)]
    pub max_examples: Option<usize>,
}

fn default_fraction() -> f64 {
    0.05
}

fn files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in paths {
        let p = crate::config::expand(p);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(&p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.extension().is_some_and(|e| e == "jsonl"))
                .collect();
            v.sort();
            out.extend(v);
        } else {
            out.push(p);
        }
    }
    Ok(out)
}

/// Every example of the files, in order; `keep` filters while reading.
pub fn read(paths: &[PathBuf], max: Option<usize>, mut keep: impl FnMut(&Example) -> bool, mut each: impl FnMut(Example)) -> Result<usize> {
    let mut n = 0;
    for f in files(paths)? {
        let reader = std::io::BufReader::new(std::fs::File::open(&f).with_context(|| format!("opening {}", f.display()))?);
        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let ex: Example = serde_json::from_str(&line).with_context(|| format!("{}:{}", f.display(), i + 1))?;
            if keep(&ex) {
                each(ex);
                n += 1;
                if max.is_some_and(|m| n >= m) {
                    return Ok(n);
                }
            }
        }
    }
    Ok(n)
}

/// Inputs and targets of many examples, flat.
#[derive(Default)]
pub struct Packed {
    pub a: Vec<u16>,
    pub b: Vec<u16>,
    pub g: Vec<f32>,
    /// Plan: recipe index; critic: unused.
    pub recipe: Vec<u8>,
    /// Plan: start and swap bars.
    pub bars: Vec<f32>,
    /// Plan: lane gains `(a_len, lanes)` per example.
    pub lanes: Vec<u16>,
    /// Critic: good, then one per issue.
    pub labels: Vec<f32>,
    pub n: usize,
}

impl Packed {
    pub fn push(&mut self, c: &BeatConfig, ex: &Example) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let g = pack(c, ex, &mut a, &mut b);
        self.a.extend(a.iter().map(|&v| f32_to_f16(v)));
        self.b.extend(b.iter().map(|&v| f32_to_f16(v)));
        self.g.extend(g);
        match c.kind {
            BeatKind::Cortex => {
                self.recipe.push(recipe_index(ex.recipe) as u8);
                self.bars.extend([ex.start_bars, ex.swap_bars]);
                for i in 0..c.a_len {
                    let lane = ex.lanes.get(i).copied().unwrap_or([0.0; 8]);
                    self.lanes.extend(lane[..c.lanes].iter().map(|&v| f32_to_f16(v)));
                }
            }
            BeatKind::Cingulate => {
                self.labels.push(f32::from(ex.label.good));
                let mut issues = vec![0.0f32; c.issues()];
                for i in &ex.label.issues {
                    issues[i.index()] = 1.0;
                }
                self.labels.extend(issues);
            }
        }
        self.n += 1;
    }

    /// Mean and standard deviation (floored at 1e-3) of every input column:
    /// A tokens, B tokens, globals.
    pub fn stats(&self, c: &BeatConfig) -> [(Vec<f32>, Vec<f32>); 3] {
        let col = |data: &mut dyn Iterator<Item = f32>, width: usize| -> (Vec<f32>, Vec<f32>) {
            let (mut s, mut q, mut n) = (vec![0.0f64; width], vec![0.0f64; width], 0usize);
            for (i, v) in data.enumerate() {
                s[i % width] += v as f64;
                q[i % width] += (v as f64) * (v as f64);
                n += usize::from(i % width == 0);
            }
            let n = n.max(1) as f64;
            let mean: Vec<f32> = s.iter().map(|v| (v / n) as f32).collect();
            let std: Vec<f32> = q.iter().zip(&s).map(|(q, s)| ((q / n - (s / n).powi(2)).max(0.0).sqrt().max(1e-3)) as f32).collect();
            (mean, std)
        };
        [
            col(&mut self.a.iter().map(|&h| f16_to_f32(h)), c.a_features()),
            col(&mut self.b.iter().map(|&h| f16_to_f32(h)), c.features),
            col(&mut self.g.iter().copied(), c.globals()),
        ]
    }
}

/// One batch, flat, in the shapes of `BeatNet::forward` and its targets.
pub struct BeatBatch {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub g: Vec<f32>,
    pub recipe: Vec<f32>,
    pub bars: Vec<f32>,
    pub lanes: Vec<f32>,
    pub labels: Vec<f32>,
    pub n: usize,
}

impl Packed {
    pub fn batch(&self, c: &BeatConfig, idx: &[usize]) -> BeatBatch {
        let (fa, fb, gn) = (c.a_len * c.a_features(), c.b_len * c.features, c.globals());
        let mut out = BeatBatch { a: vec![], b: vec![], g: vec![], recipe: vec![], bars: vec![], lanes: vec![], labels: vec![], n: idx.len() };
        let lab = 1 + c.issues();
        for &i in idx {
            out.a.extend(self.a[i * fa..(i + 1) * fa].iter().map(|&h| f16_to_f32(h)));
            out.b.extend(self.b[i * fb..(i + 1) * fb].iter().map(|&h| f16_to_f32(h)));
            out.g.extend_from_slice(&self.g[i * gn..(i + 1) * gn]);
            match c.kind {
                BeatKind::Cortex => {
                    let mut one = [0.0f32; 4];
                    one[self.recipe[i] as usize] = 1.0;
                    out.recipe.extend(one);
                    out.bars.extend_from_slice(&self.bars[i * 2..i * 2 + 2]);
                    let fl = c.a_len * c.lanes;
                    out.lanes.extend(self.lanes[i * fl..(i + 1) * fl].iter().map(|&h| f16_to_f32(h)));
                }
                BeatKind::Cingulate => out.labels.extend_from_slice(&self.labels[i * lab..(i + 1) * lab]),
            }
        }
        out
    }

    pub fn random_batch(&self, c: &BeatConfig, n: usize, rng: &mut Rng) -> BeatBatch {
        let idx: Vec<usize> = (0..n).map(|_| (rng.next_u64() % self.n as u64) as usize).collect();
        self.batch(c, &idx)
    }
}

/// FNV-1a of the id: which examples are held out when no validation files are given.
pub fn held_out(id: &str, fraction: f64) -> bool {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in id.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    (h % 10_000) as f64 / 10_000.0 < fraction
}
