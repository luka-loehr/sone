//! Renders one transition between song A and song B from their stems according
//! to a spec (the port of `reference/python/handmix.py`).
//!
//! Song A plays at its own tempo. Song B is cut together from its own bars,
//! mixed from its stems in its own time, then time-stretched as one signal to
//! A's tempo with its pitch kept, so its stems stay phase-coherent. Both decks
//! are gain-matched by loudness; the sum runs through a look-ahead limiter.
//!
//! The core has no file access and builds for `wasm32-unknown-unknown`; stems
//! come through a [`StemSource`]. The `files` feature adds [`files::FileStems`]
//! (FLAC stems and JSON beat tables) and WAV writing.

pub mod deck;
#[cfg(feature = "files")]
pub mod files;
pub mod grid;
pub mod report;
pub mod source;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use anyhow::{Context, bail};
use sone_core::{Spec, Stem};
use sone_dsp::filter::{Band, butter};
use sone_dsp::np::{Edge, iround, median, uniform_filter1d};
use sone_dsp::{SR, db_f32, db_f64, limiter, xcorr_lag};

use deck::{Build, build_deck};
pub use grid::{Grid, grid};
pub use report::{Align, BarRow, DeckLanes, LanesForSite, MAX_DRIFT_MS, MAX_LAG_MS, Report, ShortTerm};
pub use source::{BeatGrid, MemoryStems, StemSource};

/// The rendered clip and what goes with it.
pub struct Rendered {
    /// The clip, 44.1 kHz.
    pub audio: Vec<[f32; 2]>,
    pub report: Report,
    /// What the website draws.
    pub lanes: LanesForSite,
    /// Seconds per ruler bar.
    pub bar_s: f64,
    /// Time in song A of ruler position 0.
    pub a_zero_s: f64,
    /// Time in song B (its own time) of ruler position 0.
    pub b_zero_s: f64,
    /// Output seconds per second of B.
    pub ratio: f64,
    /// Intermediate signals, when asked for.
    pub signals: Option<Signals>,
}

/// The stages of a render, for comparing against the reference.
pub struct Signals {
    /// Deck A after its faders and effects (before the loudness match).
    pub deck_a: Vec<[f32; 2]>,
    /// Deck B after its faders and effects, in its own time, and after the stretch.
    pub deck_b: Vec<[f32; 2]>,
    pub deck_b_stretched: Vec<[f32; 2]>,
    /// The sum before the limiter.
    pub mix_unlimited: Vec<[f32; 2]>,
    /// Both grids.
    pub grid_a: Grid,
    pub grid_b: Grid,
    /// The nudge used, seconds (exact, not rounded like the report's).
    pub nudge_s: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    /// Use this nudge of B (seconds) instead of the spec's and the automatic one.
    pub nudge_s: Option<f64>,
    /// Keep the intermediate signals in [`Rendered::signals`].
    pub keep_signals: bool,
    /// The stretcher's engine.
    pub method: sone_dsp::stretch::Method,
    /// Use this as deck B after the stretch (a reference render's, to check the
    /// stages after the stretch on their own).
    pub stretched_b: Option<Vec<[f32; 2]>>,
}

/// Renders the transition `spec` between the songs `a` and `b`.
pub fn render(spec: &Spec, a: &dyn StemSource, b: &dyn StemSource) -> anyhow::Result<Rendered> {
    render_with(spec, a, b, &RenderOptions::default())
}

fn mono(x: &[[f32; 2]]) -> Vec<f32> {
    x.iter().map(|v| (v[0] + v[1]) / 2.0).collect()
}

fn flat(x: &[[f32; 2]]) -> &[f32] {
    x.as_flattened()
}

fn round_to(x: f64, digits: i32) -> f64 {
    let s = 10f64.powi(digits);
    (x * s).round_ties_even() / s
}

/// numpy's rounding of a float32 value, returned as the float64 json.dump writes.
fn round1_f32(x: f32) -> f64 {
    f64::from((x * 10.0).round_ties_even() / 10.0)
}

pub fn render_with(spec: &Spec, a: &dyn StemSource, b: &dyn StemSource, opt: &RenderOptions) -> anyhow::Result<Rendered> {
    let stretch = |x: &[[f32; 2]], ratio: f64| -> Vec<[f32; 2]> {
        if (ratio - 1.0).abs() < 1e-5 { x.to_vec() } else { sone_dsp::stretch::Stretcher::with_method(ratio, opt.method).render_all(x) }
    };
    let ga = grid(a, spec.a.anchor_bar, 16, spec.a.beat_shift, spec.a.fit_bars)?;
    let gb = grid(b, spec.b.anchor_bar, 16, spec.b.beat_shift, spec.b.fit_bars)?;
    let (mut t_a, mut beat_a) = (ga.t, ga.beat);
    let (mut t_b, mut beat_b) = (gb.t, gb.beat);
    if let Some(bpm) = spec.a.tempo_bpm.filter(|&v| v != 0.0) {
        beat_a = 60.0 / bpm;
    }
    if let Some(bpm) = spec.b.tempo_bpm.filter(|&v| v != 0.0) {
        beat_b = 60.0 / bpm;
    }
    let bpb_a = f64::from(spec.a.beats_per_bar);
    let bpb_b = f64::from(spec.b.beats_per_bar);
    // The anchor bar sits at ruler position anchor_u; t_* is the song time of ruler position 0.
    t_a -= spec.a.anchor_u * bpb_a * beat_a;
    t_b -= spec.b.anchor_u * bpb_b * beat_b;
    // Output length per input length of B: one ruler bar of B lasts one ruler bar of A.
    let ratio = if spec.no_sync { 1.0 } else { (bpb_a * beat_a) / (bpb_b * beat_b) };
    if !(ratio.is_finite() && ratio > 0.0 && t_a.is_finite() && t_b.is_finite()) {
        bail!("no usable grid (ratio {ratio})");
    }
    let [u0, u1] = spec.clip;
    let a_end = spec.a.until.context("deck a has no `until`")?;
    let b_from = spec.b.from.context("deck b has no `from`")?;
    let bar = bpb_a * beat_a * SR;
    let n = iround((u1 - u0) * bar).max(0) as usize;
    // Loudness match: the louder deck comes down to the quieter one.
    let target = a.lufs().min(b.lufs());
    let g_a = 10f64.powf((target - a.lufs()) / 20.0) * 10f64.powf(spec.a.trim_db / 20.0);
    let g_b = 10f64.powf((target - b.lufs()) / 20.0) * 10f64.powf(spec.b.trim_db / 20.0);
    // Deck A in its own time = output time, with 4 bars for the echo and reverb tails.
    let deck_a = build_deck(a, t_a, beat_a, &spec.a, u0, a_end + 4.0, Build::default());
    let drums_only: &[Stem] = &[Stem::Drums];
    let raw_drums = |segments: bool| Build { only: Some(drums_only), raw: true, segments };
    let mut nudge = spec.b.nudge_ms / 1000.0;
    // Both songs' drums over the align windows (A's, and B's stretched to A's tempo), as mono.
    let drums_pair = |pairs: &sone_core::AlignPairs, t_b: f64, beat_b: f64, ratio: f64, nudge: f64| -> (Vec<f32>, Vec<f32>) {
        let ad = build_deck(a, t_a, beat_a, &spec.a, pairs.a[0], pairs.a[1], raw_drums(false)).total;
        let bd = build_deck(b, t_b + nudge, beat_b, &spec.b, pairs.b[0], pairs.b[1], raw_drums(false)).total;
        let bd = stretch(&bd, ratio);
        let m = ad.len().min(bd.len());
        (mono(&ad[..m]), mono(&bd[..m]))
    };
    // Up to three rounds within ±45 ms; a peak at the edge of the search is no answer (hi-hats on
    // the off-sixteenths make one there). (The lab's renderer searched ±40 ms and moved at most
    // 30 ms; for the transitions it built the result is the same.)
    let align = |pairs: &sone_core::AlignPairs, t_b: f64, beat_b: f64, ratio: f64, mut nudge: f64| -> f64 {
        for _ in 0..3 {
            let (am, bm) = drums_pair(pairs, t_b, beat_b, ratio, nudge);
            let (lag, sharp) = xcorr_lag(&am, &bm, 45.0, 32);
            if lag.abs() < 0.4 || lag.abs() > 43.0 || sharp < 2.0 {
                break;
            }
            // B early by `lag`: read B that much earlier in its own time.
            nudge -= lag / 1000.0 / ratio;
        }
        nudge
    };
    if let Some(v) = opt.nudge_s {
        nudge = v;
    } else if let Some(pairs) = spec.align_pairs.as_ref().filter(|_| spec.b.auto_nudge && !spec.no_sync) {
        nudge = align(pairs, t_b, beat_b, ratio, nudge);
    }
    // B's deck ruler: in sync it is the clip's ruler; without sync B counts its own bars from start_u on.
    let start_u = if spec.no_sync { spec.b.start_u.unwrap_or(b_from) } else { b_from };
    let native_per_bar = if spec.no_sync { bpb_b * beat_b * SR } else { bar / ratio };
    let b_to = b_from + (u1 - start_u) * bar / (native_per_bar * ratio);
    let deck_b = build_deck(b, t_b + nudge, beat_b, &spec.b, b_from, b_to, Build::default());
    let b_out = match &opt.stretched_b {
        Some(v) => v.clone(),
        None => stretch(&deck_b.total, ratio),
    };

    let mut mix = vec![[0.0f32; 2]; n];
    let na = deck_a.total.len().min(n);
    for (m, v) in mix[..na].iter_mut().zip(&deck_a.total) {
        for c in 0..2 {
            m[c] = (f64::from(m[c]) + f64::from(v[c]) * g_a) as f32;
        }
    }
    let ob = iround((start_u - u0) * bar);
    if ob < 0 {
        bail!("B starts before the clip (start_u {start_u} < clip start {u0})");
    }
    let ob = ob as usize;
    let nb = b_out.len().min(n.saturating_sub(ob));
    for (m, v) in mix[ob..ob + nb].iter_mut().zip(&b_out) {
        for c in 0..2 {
            m[c] = (f64::from(m[c]) + f64::from(v[c]) * g_b) as f32;
        }
    }

    let mut report = Report {
        nudge_ms: round_to(nudge * 1000.0, 2),
        clean_vocals: [a.clean_vocals(), b.clean_vocals()],
        leak_free: [a.leak_free(), b.leak_free()],
        tempo_a: round_to(60.0 / beat_a, 3),
        tempo_b: round_to(60.0 / beat_b, 3),
        stretch_pct: round_to((1.0 / ratio - 1.0) * 100.0, 2),
        lufs_a: round_to(a.lufs(), 2),
        lufs_b: round_to(b.lufs(), 2),
        phase_a_ms: round_to(ga.shift * 1000.0, 1),
        phase_b_ms: round_to(gb.shift * 1000.0, 1),
        align: None,
        shortterm_db: ShortTerm::default(),
        bars: Vec::new(),
        limiter_min_gain_db: 0.0,
    };

    // Beat alignment on the raw drum stems where both decks' sources have drums.
    let (am, bm) = if let Some(pairs) = &spec.align_pairs {
        // Bars of A against other bars of B (same length on the ruler), e.g. where both have their full beat.
        let ad = build_deck(a, t_a, beat_a, &spec.a, pairs.a[0], pairs.a[1], raw_drums(false)).total;
        let bd = stretch(&build_deck(b, t_b + nudge, beat_b, &spec.b, pairs.b[0], pairs.b[1], raw_drums(false)).total, ratio);
        let m = ad.len().min(bd.len());
        (mono(&ad[..m]), mono(&bd[..m]))
    } else {
        let (lo_u, hi_u) = if spec.no_sync { (b_from, b_from + 2.0) } else { (b_from.max(-8.0), a_end.min(8.0)) };
        let ad = build_deck(a, t_a, beat_a, &spec.a, lo_u, hi_u, raw_drums(true)).total;
        let bd = stretch(&build_deck(b, t_b + nudge, beat_b, &spec.b, lo_u, hi_u, raw_drums(true)).total, ratio);
        let m = ad.len().min(bd.len());
        (mono(&ad[..m]), mono(&bd[..m]))
    };
    let h = am.len() / 2;
    report.align = Some(Align {
        all: xcorr_lag(&am, &bm, 45.0, 32),
        first_half: xcorr_lag(&am[..h], &bm[..h], 45.0, 32),
        second_half: xcorr_lag(&am[h..], &bm[h..], 45.0, 32),
        drums_a_db: round1_f32(db_f32(&am)),
        drums_b_db: round1_f32(db_f32(&bm)),
    });

    // Bus: short-term loudness (3 s, 100 Hz high-pass), measured.
    let win = (3.0 * SR) as usize;
    let kw = butter(2, Band::Highpass(100.0), SR).filter_f32(&mono(&mix));
    let p = uniform_filter1d(&kw.iter().map(|v| v * v).collect::<Vec<_>>(), win, Edge::Nearest);
    let st: Vec<f64> = p.iter().map(|v| 10.0 * (v + 1e-10).log10()).collect();
    let first = ((spec.first_event - u0) * bar) as usize;
    let ref_a = median(&st[..first.min(st.len())]);
    let inner = &st[(win / 2).min(st.len())..st.len().saturating_sub(win / 2).max(win / 2)];
    report.shortterm_db = ShortTerm {
        reference: round_to(ref_a, 2),
        min: round_to(inner.iter().copied().fold(f64::INFINITY, f64::min), 2),
        max: round_to(inner.iter().copied().fold(f64::NEG_INFINITY, f64::max), 2),
    };
    let mix_unlimited = if opt.keep_signals { Some(mix.clone()) } else { None };
    let (mix, gr) = limiter(&mix, 0.89);

    // Meter bridge per bar: every stem after its fader, the decks, the sum and its bands.
    let mut a_full = vec![[0.0f32; 2]; n];
    for (o, v) in a_full[..na].iter_mut().zip(&deck_a.total) {
        *o = [(f64::from(v[0]) * g_a) as f32, (f64::from(v[1]) * g_a) as f32];
    }
    let mut b_full = vec![[0.0f32; 2]; n];
    for (o, v) in b_full[ob..ob + nb].iter_mut().zip(&b_out) {
        *o = [(f64::from(v[0]) * g_b) as f32, (f64::from(v[1]) * g_b) as f32];
    }
    let mono_mix = mono(&mix);
    let lowb = butter(4, Band::Lowpass(150.0), SR).filter_f32(&mono_mix);
    let highb = butter(4, Band::Highpass(4000.0), SR).filter_f32(&mono_mix);
    let stem_db = |part: &[f32], lo: usize, hi: usize, g: f64| -> f64 {
        let v: Vec<f64> = part[lo.min(part.len())..hi.min(part.len())].iter().map(|&x| f64::from(x) * g).collect();
        if v.is_empty() { -100.0 } else { round_to(db_f64(&v), 1) }
    };
    for k in 0..(u1 - u0) as i64 {
        let i0 = (k as f64 * bar) as usize;
        let i1 = ((k + 1) as f64 * bar) as usize;
        let uu = u0 + k as f64;
        let (i0c, i1c) = (i0.min(n), i1.min(n));
        let mut row = BarRow {
            u: uu as i64,
            a: round1_f32(db_f32(flat(&a_full[i0c..i1c]))),
            b: round1_f32(db_f32(flat(&b_full[i0c..i1c]))),
            mix: round1_f32(db_f32(flat(&mix[i0c..i1c]))),
            low: round_to(db_f64(&lowb[i0c..i1c]), 1),
            high: round_to(db_f64(&highb[i0c..i1c]), 1),
            ..Default::default()
        };
        let j0 = ((uu - start_u) * bar / ratio) as i64;
        let j1 = ((uu + 1.0 - start_u) * bar / ratio) as i64;
        for stem in Stem::ALL {
            let pa = &deck_a.parts[&stem];
            let va = if i0 < pa.len() { stem_db(pa, i0, i1, g_a) } else { -100.0 };
            let pb = &deck_b.parts[&stem];
            let vb = if j1 > 0 { stem_db(pb, j0.max(0) as usize, j1.max(0) as usize, g_b) } else { -100.0 };
            row.set(stem, va, vb);
        }
        report.bars.push(row);
    }
    report.limiter_min_gain_db = round_to(20.0 * gr.max(1e-9).log10(), 2);

    // Lanes for the site, 16 points per bar.
    let step_a = ((bar / 16.0) as usize).max(1);
    let step_b = ((bpb_b * beat_b * SR / 16.0) as usize).max(1);
    let sample = |lanes: &BTreeMap<Stem, Vec<f32>>, step: usize| -> BTreeMap<Stem, Vec<f64>> {
        lanes.iter().map(|(k, g)| (*k, g.iter().step_by(step).map(|&v| round_to(f64::from(v), 3)).collect())).collect()
    };
    let lanes = LanesForSite {
        id: spec.id.clone(),
        clip: spec.clip,
        bar_s: bpb_a * beat_a,
        seconds: n as f64 / SR,
        report: report.clone(),
        a: DeckLanes { u0: deck_a.u.first().copied().unwrap_or(u0), step_bars: step_a as f64 / bar, gain: sample(&deck_a.lanes, step_a) },
        b: DeckLanes { u0: start_u, step_bars: step_b as f64 * ratio / bar, gain: sample(&deck_b.lanes, step_b) },
    };
    let signals = opt.keep_signals.then(|| Signals {
        deck_a: deck_a.total.clone(),
        deck_b: deck_b.total.clone(),
        deck_b_stretched: b_out.clone(),
        mix_unlimited: mix_unlimited.unwrap_or_default(),
        grid_a: Grid { t: t_a, beat: beat_a, shift: ga.shift },
        grid_b: Grid { t: t_b, beat: beat_b, shift: gb.shift },
        nudge_s: nudge,
    });
    Ok(Rendered { audio: mix, report, lanes, bar_s: bpb_a * beat_a, a_zero_s: t_a, b_zero_s: t_b + nudge, ratio, signals })
}

/// The report's per-bar table as the reference tool prints it.
pub fn meter_table(report: &Report) -> String {
    let f = |v: f64| if v < -70.0 { "  -- ".to_string() } else { format!("{:5.0}", v) };
    let mut s = String::from("   u |  A: voc  drm  bas  oth  sum |  B: voc  drm  bas  oth  sum |  mix   low  high\n");
    for b in &report.bars {
        s += &format!(
            "{:4} |    {}{}{}{}{} |    {}{}{}{}{} |{} {} {}\n",
            b.u,
            f(b.a_v),
            f(b.a_d),
            f(b.a_b),
            f(b.a_o),
            f(b.a),
            f(b.b_v),
            f(b.b_d),
            f(b.b_b),
            f(b.b_o),
            f(b.b),
            f(b.mix),
            f(b.low),
            f(b.high)
        );
    }
    s
}
