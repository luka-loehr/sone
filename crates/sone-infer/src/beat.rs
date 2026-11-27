//! Cortex and Cingulate: a transformer encoder over the beat tokens of both songs.
//!
//! - Cortex (executive transition planner): recipe logits and `(start_bars, swap_bars)`
//!   from the global token; 8 stem fader trajectories from A's 80 beat tokens.
//! - Cingulate (conflict & defect monitor): from the global token, `p(good)` and 11
//!   defect probabilities (`DoubleVocal`, `VoiceCutInWord`, `AMelodyLeftEarly`, etc.).

use anyhow::{Result, ensure};
use sone_core::{Example, Issue, spec::Recipe};

use crate::config::{BARS_SCALE, BeatConfig, BeatKind, RECIPES};
use crate::layers::{Attention, FeedForward, Linear, Scratch};
use crate::ops::{EPS, linear_into, rms_norm_rows, sigmoid, softmax};
use crate::st::WeightFile;

pub const RECIPE_ORDER: [Recipe; RECIPES] = [Recipe::VoiceOverBeat, Recipe::Sneak, Recipe::OutroUnderIntro, Recipe::Gap];

pub fn recipe_index(r: Recipe) -> usize {
    RECIPE_ORDER.iter().position(|&x| x == r).expect("listed")
}

struct Input {
    mean: Vec<f32>,
    std: Vec<f32>,
    embed: Linear,
}

enum Heads {
    Cortex { recipe: Linear, bars: Linear, lanes: Linear },
    Cingulate { head: Linear },
}

pub struct BeatNet {
    pub config: BeatConfig,
    file: WeightFile,
    a: Input,
    b: Input,
    g: Input,
    pos_a: Vec<f32>,
    pos_b: Vec<f32>,
    layers: Vec<(Attention, FeedForward)>,
    norm: Vec<f32>,
    heads: Heads,
}

#[derive(Default)]
pub struct BeatWorkspace {
    x: Vec<f32>,
    s: Scratch,
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CortexOutput {
    /// Probabilities in [`RECIPE_ORDER`].
    pub recipe_probs: [f32; RECIPES],
    pub recipe: Recipe,
    pub start_bars: f32,
    pub swap_bars: f32,
    /// Per beat of the ruler (16 bars before the one to 4 after), the gains 0..1.
    pub lanes: Vec<[f32; 8]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CingulateOutput {
    pub good: f32,
    /// Probability per issue, in `Issue::ALL` order.
    pub issues: Vec<(Issue, f32)>,
}

/// The raw, unstandardised inputs of an example for this network: A tokens
/// `(a_len, a_features)` (the critic's with the lanes appended), B tokens
/// `(b_len, features)`, the globals (the critic's with the measures).
/// Short sequences are zero-padded at the end, long ones cut.
pub fn pack(c: &BeatConfig, ex: &Example, a: &mut Vec<f32>, b: &mut Vec<f32>) -> Vec<f32> {
    let with_lanes = c.kind == BeatKind::Cingulate;
    let fa = c.a_features();
    a.clear();
    a.resize(c.a_len * fa, 0.0);
    for (i, beat) in ex.a_beats.iter().take(c.a_len).enumerate() {
        for (j, &v) in beat.iter().take(c.features).enumerate() {
            a[i * fa + j] = v;
        }
    }
    if with_lanes {
        for (i, lane) in ex.lanes.iter().take(c.a_len).enumerate() {
            a[i * fa + c.features..(i + 1) * fa].copy_from_slice(&lane[..c.lanes]);
        }
    }
    b.clear();
    b.resize(c.b_len * c.features, 0.0);
    for (i, beat) in ex.b_beats.iter().take(c.b_len).enumerate() {
        for (j, &v) in beat.iter().take(c.features).enumerate() {
            b[i * c.features + j] = v;
        }
    }
    let mut g = vec![ex.tempo_a, ex.tempo_b, ex.beat_ratio, f32::from(ex.key_distance)];
    if with_lanes {
        let m = &ex.measures;
        g.extend([m.beat_offset_ms, m.stretch_pct, m.vocal_leak_db, m.tuning_diff_cents]);
    }
    g
}

impl BeatNet {
    pub fn load(mut file: WeightFile) -> Result<Self> {
        let kind = file.metadata.get("model").cloned().unwrap_or_default();
        let config: BeatConfig = file.config(&kind)?;
        ensure!(
            ((kind == "cortex" || kind == "plan") && config.kind == BeatKind::Cortex)
                || ((kind == "cingulate" || kind == "critic") && config.kind == BeatKind::Cingulate),
            "weights: model {kind} does not match its config"
        );
        let d = config.dim;
        let t = &mut file;
        let mut input = |name: &str, n: usize, embed: &str| -> Result<Input> {
            Ok(Input {
                mean: t.vec(&format!("input.{name}_mean"), &[n])?,
                std: t.vec(&format!("input.{name}_std"), &[n])?,
                embed: Linear::load(t, embed, d, n)?,
            })
        };
        let a = input("a", config.a_features(), "embed_a")?;
        let b = input("b", config.features, "embed_b")?;
        let g = input("g", config.globals(), "embed_g")?;
        let pos_a = t.vec("pos_a", &[config.a_len, d])?;
        let pos_b = t.vec("pos_b", &[config.b_len, d])?;
        let layers = (0..config.layers)
            .map(|i| {
                Ok((
                    Attention::load(t, &format!("layer{i}.attn"), d, config.heads)?,
                    FeedForward::load(t, &format!("layer{i}.ff"), d, config.ff)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = t.vec("norm", &[d])?;
        let heads = match config.kind {
            BeatKind::Cortex => Heads::Cortex {
                recipe: Linear::load(t, "head_recipe", RECIPES, d)?,
                bars: Linear::load(t, "head_bars", 2, d)?,
                lanes: Linear::load(t, "head_lanes", config.lanes, d)?,
            },
            BeatKind::Cingulate => Heads::Cingulate { head: Linear::load(t, "head", 1 + config.issues(), d)? },
        };
        file.finish()?;
        Ok(Self { config, file, a, b, g, pos_a, pos_b, layers, norm, heads })
    }

    /// Final token states `(tokens, dim)` for raw inputs (see [`pack`]).
    fn encode<'a>(&self, ws: &'a mut BeatWorkspace, a: &[f32], b: &[f32], g: &[f32]) -> &'a [f32] {
        let c = &self.config;
        let d = c.dim;
        let n = c.tokens();
        let BeatWorkspace { x, s, .. } = ws;
        x.clear();
        x.resize(n * d, 0.0);
        let mut embed = |inp: &Input, raw: &[f32], rows: usize, at: usize, pos: Option<&[f32]>, s: &mut Scratch| {
            let f = inp.mean.len();
            s.rows.clear();
            s.rows.extend(raw.iter().enumerate().map(|(i, &v)| (v - inp.mean[i % f]) / inp.std[i % f]));
            let w = inp.embed.w.load(&self.file, &mut s.w1).to_vec();
            linear_into(&s.rows, rows, f, &w, d, Some(&inp.embed.b), &mut s.inner);
            let dst = &mut x[at * d..(at + rows) * d];
            dst.copy_from_slice(&s.inner);
            if let Some(pos) = pos {
                for (v, p) in dst.iter_mut().zip(pos) {
                    *v += p;
                }
            }
        };
        embed(&self.g, g, 1, 0, None, s);
        embed(&self.a, a, c.a_len, 1, Some(&self.pos_a), s);
        embed(&self.b, b, c.b_len, 1 + c.a_len, Some(&self.pos_b), s);
        for (attn, ff) in &self.layers {
            attn.apply(&self.file, x, 1, n, s);
            ff.apply(&self.file, x, n, s);
        }
        rms_norm_rows(x, d, &self.norm, EPS);
        x
    }

    /// Raw head outputs. Plan: `[4 recipe logits, 2 bars (already × BARS_SCALE),
    /// a_len × lanes gains (after sigmoid)]`; Critic: `1 + issues` logits.
    pub fn forward_raw(&self, ws: &mut BeatWorkspace, a: &[f32], b: &[f32], g: &[f32]) -> Vec<f32> {
        let c = self.config.clone();
        let d = c.dim;
        let x = self.encode(ws, a, b, g).to_vec();
        let s = &mut ws.s;
        let mut out = Vec::new();
        let mut tmp = Vec::new();
        let head = |l: &Linear, rows: &[f32], n_rows: usize, s: &mut Scratch, tmp: &mut Vec<f32>| {
            let w = l.w.load(&self.file, &mut s.w1).to_vec();
            linear_into(rows, n_rows, d, &w, l.n_out(), Some(&l.b), tmp);
        };
        match &self.heads {
            Heads::Cortex { recipe, bars, lanes } => {
                head(recipe, &x[..d], 1, s, &mut tmp);
                out.extend_from_slice(&tmp);
                head(bars, &x[..d], 1, s, &mut tmp);
                out.extend(tmp.iter().map(|v| v * BARS_SCALE));
                head(lanes, &x[d..(1 + c.a_len) * d], c.a_len, s, &mut tmp);
                out.extend(tmp.iter().map(|&v| sigmoid(v)));
            }
            Heads::Cingulate { head: h } => {
                head(h, &x[..d], 1, s, &mut tmp);
                out.extend_from_slice(&tmp);
            }
        }
        out
    }

    pub fn cortex(&self, ws: &mut BeatWorkspace, ex: &Example) -> CortexOutput {
        assert_eq!(self.config.kind, BeatKind::Cortex, "not a cortex planner");
        let (mut a, mut b) = (std::mem::take(&mut ws.a), std::mem::take(&mut ws.b));
        let g = pack(&self.config, ex, &mut a, &mut b);
        let raw = self.forward_raw(ws, &a, &b, &g);
        ws.a = a;
        ws.b = b;
        let mut probs = [raw[0], raw[1], raw[2], raw[3]];
        softmax(&mut probs);
        let best = (0..RECIPES).max_by(|&i, &j| probs[i].total_cmp(&probs[j])).unwrap_or(0);
        let lanes = raw[6..].as_chunks::<8>().0.to_vec();
        CortexOutput { recipe_probs: probs, recipe: RECIPE_ORDER[best], start_bars: raw[4], swap_bars: raw[5], lanes }
    }

    pub fn cingulate(&self, ws: &mut BeatWorkspace, ex: &Example) -> CingulateOutput {
        assert_eq!(self.config.kind, BeatKind::Cingulate, "not a cingulate monitor");
        let (mut a, mut b) = (std::mem::take(&mut ws.a), std::mem::take(&mut ws.b));
        let g = pack(&self.config, ex, &mut a, &mut b);
        let raw = self.forward_raw(ws, &a, &b, &g);
        ws.a = a;
        ws.b = b;
        CingulateOutput { good: sigmoid(raw[0]), issues: Issue::ALL.iter().zip(&raw[1..]).map(|(&i, &v)| (i, sigmoid(v))).collect() }
    }
}
