//! One deck: a song's stems cut together from its own bars, each through its
//! fader lane, filters and sends (`assemble` and `build_deck` of the reference).

use std::collections::BTreeMap;

use sone_core::{Deck, Segment, SendKind, Shape, Stem};
use sone_dsp::np::{iround, linspace};
use sone_dsp::{SR, SweepKind, echo, lane, reverb, sweep};

use crate::source::StemSource;

/// A deck's signal on [u_from, u_to) of the ruler, in the deck's own time.
pub struct DeckOut {
    pub total: Vec<[f32; 2]>,
    /// Ruler position of every sample.
    pub u: Vec<f64>,
    /// Fader gain of every stem per sample.
    pub lanes: BTreeMap<Stem, Vec<f32>>,
    /// Every stem after its fader, mono, per sample (for the meter bridge).
    pub parts: BTreeMap<Stem, Vec<f32>>,
}

/// Samples between two ruler positions (`int(round((u1 - u0) * bar))`).
fn span(u0: f64, u1: f64, bar: f64) -> usize {
    iround((u1 - u0) * bar).max(0) as usize
}

/// One stem on the ruler [u_from, u_to) in the deck's own time, cut from its bars.
/// Every splice after the first gets a 6 ms equal-power crossfade into it.
#[allow(clippy::too_many_arguments)]
pub fn assemble(
    src: &dyn StemSource,
    stem: Stem,
    t_anchor: f64,
    beat: f64,
    segments: &[Segment],
    u_from: f64,
    u_to: f64,
    bpb: f64,
) -> Vec<[f32; 2]> {
    let bar = bpb * beat * SR;
    let n = span(u_from, u_to, bar);
    let mut out = vec![[0.0f32; 2]; n];
    const XF: usize = 264; // int(0.006 * SR)
    for (si, s) in segments.iter().enumerate() {
        let u0 = s.u0.max(u_from);
        let u1 = s.u1.unwrap_or(u_to).min(u_to);
        if u1 <= u0 {
            continue;
        }
        let d0 = iround((u0 - u_from) * bar) as usize;
        let d1 = iround((u1 - u_from) * bar) as usize;
        let s0 = iround((t_anchor + (s.src + (u0 - s.u0)) * bpb * beat) * SR);
        let pre = if si > 0 && d0 >= XF { XF } else { 0 };
        let a = s0 - pre as i64;
        let mut seg = vec![[0.0f32; 2]; pre + (d1 - d0)];
        src.read(stem, a, &mut seg);
        if pre > 0 {
            let w = linspace(0.0, std::f64::consts::FRAC_PI_2, pre);
            for (i, &wi) in w.iter().enumerate() {
                let sin = wi.sin() as f32;
                let cos = wi.cos();
                let o = &mut out[d0 - pre + i];
                for c in 0..2 {
                    o[c] = (f64::from(o[c]) * cos + f64::from(seg[i][c] * sin)) as f32;
                }
            }
        }
        out[d0..d1].copy_from_slice(&seg[pre..]);
    }
    out
}

/// Which parts of the deck to build.
#[derive(Clone, Copy)]
pub struct Build<'a> {
    /// Only these stems (the beat-alignment checks use the drums alone).
    pub only: Option<&'a [Stem]>,
    /// Stems as they are: no faders, filters, sends or trim.
    pub raw: bool,
    /// Use the deck's segments (the alignment by `align_pairs` reads the song straight).
    pub segments: bool,
}

impl Default for Build<'_> {
    fn default() -> Self {
        Build { only: None, raw: false, segments: true }
    }
}

/// A deck on [u_from, u_to) in its own time; `t_anchor` is the song time of ruler 0.
pub fn build_deck(
    src: &dyn StemSource,
    t_anchor: f64,
    beat: f64,
    deck: &Deck,
    u_from: f64,
    u_to: f64,
    how: Build,
) -> DeckOut {
    let bpb = f64::from(deck.beats_per_bar);
    let bar = bpb * beat * SR;
    let n = span(u_from, u_to, bar);
    let u: Vec<f64> = (0..n).map(|i| u_from + i as f64 / bar).collect();
    let default_seg = [Segment { u0: u_from, u1: None, src: u_from }];
    let segments: &[Segment] = if how.segments && !deck.segments.is_empty() { &deck.segments } else { &default_seg };
    let mut total = vec![[0.0f32; 2]; n];
    let mut lanes = BTreeMap::new();
    let mut parts = BTreeMap::new();
    for stem in Stem::ALL {
        if how.only.is_some_and(|o| !o.contains(&stem)) {
            continue;
        }
        let mut x = assemble(src, stem, t_anchor, beat, segments, u_from, u_to, bpb);
        if how.raw {
            for (t, v) in total.iter_mut().zip(&x) {
                t[0] += v[0];
                t[1] += v[1];
            }
            continue;
        }
        let default_lane = vec![[u_from, 1.0]];
        let pts = deck.lanes.get(&stem).unwrap_or(&default_lane);
        let g = lane(pts, &u, 6.0, deck.shape_of(stem)).expect("a stem's lane has no points");
        for (kind, map) in [(SweepKind::Highpass, &deck.hpf), (SweepKind::Lowpass, &deck.lpf)] {
            let Some(fp) = map.get(&stem) else { continue };
            if fp.is_empty() {
                continue;
            }
            // Filter only where the cutoff moves; the rest stays bit-exact.
            let clip = |v: f64| v.clamp(0.0, n as f64) as usize;
            let mut a = clip((fp[0][0] - u_from) * bar);
            let mut b = clip((fp[fp.len() - 1][0] - u_from) * bar);
            let flat = if kind == SweepKind::Highpass { 20.0 } else { 20000.0 };
            if (fp[fp.len() - 1][1] - flat).abs() > 1.0 {
                b = n;
            }
            if (fp[0][1] - flat).abs() > 1.0 {
                a = 0;
            }
            if b > a {
                let y = sweep(&x[a..b], fp, &u[a..b], kind);
                x[a..b].copy_from_slice(&y);
            }
        }
        let dry: Vec<[f32; 2]> = x.iter().zip(&g).map(|(v, &gi)| [v[0] * gi, v[1] * gi]).collect();
        for (t, v) in total.iter_mut().zip(&dry) {
            t[0] += v[0];
            t[1] += v[1];
        }
        parts.insert(stem, dry.iter().map(|v| (v[0] + v[1]) / 2.0).collect::<Vec<f32>>());
        for fx in deck.sends.iter().filter(|fx| fx.stem == stem) {
            let s = lane(&fx.send, &u, 6.0, Shape::Lin).expect("a send has no points");
            // numpy: (x * send) * fader, all float32
            let feed: Vec<[f32; 2]> = x
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let f = [v[0] * s[i], v[1] * s[i]];
                    if fx.post { [f[0] * g[i], f[1] * g[i]] } else { f }
                })
                .collect();
            let level = fx.wet_level() as f32;
            let wet = match fx.kind {
                SendKind::Echo => echo(&feed, fx.beats * beat, fx.feedback, 10, fx.hp, fx.lp),
                SendKind::Reverb => reverb(&feed, fx.rt60),
            };
            for (t, w) in total.iter_mut().zip(&wet) {
                t[0] += w[0] * level;
                t[1] += w[1] * level;
            }
        }
        lanes.insert(stem, g);
    }
    if !deck.gain_db.is_empty() && !how.raw {
        let gd = lane(&deck.gain_db, &u, 50.0, Shape::Lin).expect("gain_db");
        let gl: Vec<f32> = gd.iter().map(|&v| 10f32.powf(v / 20.0)).collect();
        for (t, &g) in total.iter_mut().zip(&gl) {
            t[0] *= g;
            t[1] *= g;
        }
        for p in parts.values_mut() {
            for (v, &g) in p.iter_mut().zip(&gl) {
                *v *= g;
            }
        }
    }
    DeckOut { total, u, lanes, parts }
}
