//! Low-Memory Boundary-Streaming Studio DJ Mix Renderer (`Sone-Pro`).
//!
//! Key studio guarantees:
//! 1. Structural indexing uses `Cochlea-v2-Base` (6.73 dB vocal SDR, 4096-FFT, 64 bands) gated by
//!    `Tympanum-Pro`'s 50 Hz neural `singing` + `breath` heads so `SongAnalyzer` recovers studio-clean
//!    sixteenth-note vocal lines, breaths, and true vocal entry bars with < 400 MB RSS.
//! 2. Candidate search runs `sone_plan::plan` across forced recipes (`VoiceOverBeat`, `Sneak`,
//!    `OutroUnderIntro`) so vocal-overlap transitions anchor Song B right on its real vocal entry
//!    (skipping empty beat intros) and never cut Song A mid-buildup.
//! 3. Zero separated stems outside the blend: `spec.clip` ends at `u = +1.5` bars (all 4 Deck B stems
//!    locked to `1.0` for `u >= 0.0` so `ai_lanes` never dips vocals post-handoff), crossfading with
//!    pre-rolled sample-exact phase alignment into 100% unseparated master stereo Song B.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sone_analysis::{SongAnalyzer, measure::Block, structure::Beats};
use sone_beats::grid::{FPS, Grid, OnsetEnvelope, Votes, autocorrelation_period, fit_grid, median_period, prob_at};
use sone_core::{BeatTable, Label, Points, Recipe, Shape, Spec, Stem};
use sone_dsp::Stretcher;
use sone_infer::{
    BeatWorkspace, Cochlea, CochleaWorkspace, Sone, Tympanum, TympanumWorkspace, WeightFile,
};
use sone_plan::PlanOptions;
use sone_plan::keys::song_distance;
use sone_plan::lanes::at as lane_at;
use sone_plan::view::Song as PlanSong;
use sone_render::{BeatGrid, RenderOptions, StemSource, files};
use sone_sim::build::{A_BEFORE, example_from_spec};

const STATION_LUFS: f64 = -14.0;
const SEAM: usize = 2048;

#[derive(Serialize, Deserialize)]
struct CachedSongMeta {
    total_len: usize,
    duration_s: f64,
    lufs: f64,
    ear_s: f64,
    analyse_s: f64,
    table: BeatTable,
}

struct WindowStems {
    id: String,
    total_len: usize,
    offset: usize,
    stems: [Vec<[f32; 2]>; 4],
    grid: BeatGrid,
    lufs: f64,
}

impl StemSource for WindowStems {
    fn id(&self) -> &str {
        &self.id
    }
    fn len(&self) -> usize {
        self.total_len
    }
    fn read(&self, stem: Stem, start: i64, out: &mut [[f32; 2]]) {
        let idx = match stem {
            Stem::Vocals => 0,
            Stem::Drums => 1,
            Stem::Bass => 2,
            Stem::Other => 3,
        };
        let src = &self.stems[idx];
        let off = self.offset as i64;
        for (k, o) in out.iter_mut().enumerate() {
            let j = start + k as i64 - off;
            *o = if j >= 0 && (j as usize) < src.len() { src[j as usize] } else { [0.0; 2] };
        }
    }
    fn beats(&self) -> &BeatGrid {
        &self.grid
    }
    fn lufs(&self) -> f64 {
        self.lufs
    }
}

fn peaks(p: &[f32], thr: f32, min_dist: usize) -> Vec<usize> {
    let n = p.len();
    let mut cands = Vec::new();
    for i in 0..n {
        if p[i] < thr {
            continue;
        }
        let lo = i.saturating_sub(2);
        let hi = (i + 3).min(n);
        if (lo..hi).all(|j| p[j] < p[i] || (j >= i && p[j] <= p[i])) {
            cands.push(i);
        }
    }
    let mut out: Vec<usize> = Vec::new();
    for idx in cands {
        if let Some(&last) = out.last() {
            if idx - last < min_dist {
                if p[idx] > p[last] {
                    *out.last_mut().unwrap() = idx;
                }
                continue;
            }
        }
        out.push(idx);
    }
    out
}

fn fit_neuro_symbolic_beats(mono: &[f32], beat_probs: &[f32], down_probs: &[f32], duration_s: f64) -> Beats {
    let mut env = OnsetEnvelope::new(44_100);
    env.push(mono);
    let dur_f = duration_s as f32;
    let span = (dur_f.min(4.0), (dur_f - 2.0).max(dur_f.min(8.0)));
    let votes = Votes { t0: 0.0, beat: beat_probs, down: down_probs };

    let raw_peaks = peaks(beat_probs, 0.32, 10);
    let net_beats: Vec<f32> = raw_peaks.iter().map(|&i| i as f32 / FPS).collect();
    let strong_peaks: Vec<(f32, f32)> = peaks(beat_probs, 0.45, 10)
        .into_iter()
        .map(|i| (i as f32 / FPS, beat_probs[i]))
        .filter(|&(t, _)| t >= span.0 && t <= span.1)
        .collect();
    let eval_peaks: Vec<(f32, f32)> = if strong_peaks.len() >= 8 {
        strong_peaks
    } else {
        raw_peaks
            .iter()
            .map(|&i| (i as f32 / FPS, beat_probs[i]))
            .filter(|&(t, _)| t >= span.0 && t <= span.1)
            .collect()
    };
    let total_peak_w: f32 = eval_peaks.iter().map(|&(_, w)| w).sum::<f32>().max(1e-6);

    let mut cands: Vec<f32> = Vec::new();
    let ac_p = autocorrelation_period(&env.values, 0.0, span);
    for m in [0.5f32, 0.6667, 0.75, 1.0, 1.3333, 1.5, 2.0] {
        let p = ac_p * m;
        if (0.32..=0.96).contains(&p) {
            cands.push(p);
        }
    }
    if let Some(mp) = median_period(&net_beats) {
        for m in [0.5f32, 0.6667, 0.75, 1.0, 1.3333, 1.5, 2.0] {
            let p = mp * m;
            if (0.32..=0.96).contains(&p) {
                cands.push(p);
            }
        }
    }
    let mut bpm = 64.0f32;
    while bpm <= 185.0 {
        cands.push(60.0 / bpm);
        bpm *= 1.035;
    }

    let w = 25usize;
    let env_norm: Vec<f32> = {
        let mut loc: Vec<f32> = (0..env.values.len())
            .map(|i| {
                let (a, b) = (i.saturating_sub(w), (i + w + 1).min(env.values.len()));
                let m = env.values[a..b].iter().sum::<f32>() / (b - a).max(1) as f32;
                (env.values[i] - m).max(0.0)
            })
            .collect();
        let mean = loc.iter().sum::<f32>() / loc.len().max(1) as f32;
        if mean > 0.0 {
            loc.iter_mut().for_each(|v| *v /= mean);
        }
        loc
    };

    let mut best: Option<(f32, Grid)> = None;
    for p0 in cands {
        let g = fit_grid(&env.values, 0.0, span, p0, &votes, &net_beats);
        if !(0.31..=0.98).contains(&g.period) {
            continue;
        }
        let (bs, _) = g.beats(span.0, span.1);
        if bs.len() < 16 {
            continue;
        }
        let mut s_env = 0.0f32;
        let mut s_prec = 0.0f32;
        for &t in &bs {
            s_env += prob_at(&env_norm, 0.0, t);
            s_prec += prob_at(beat_probs, 0.0, t);
        }
        let n = bs.len() as f32;
        let prec = s_prec / n;
        let mut covered_w = 0.0f32;
        for &(pt, pw) in &eval_peaks {
            let k = ((pt - g.phase) / g.period).round();
            let nearest = g.phase + k * g.period;
            if (pt - nearest).abs() <= 0.045 {
                covered_w += pw;
            }
        }
        let rec = if eval_peaks.is_empty() { prec } else { covered_w / total_peak_w };
        let f1_net = 2.0 * prec * rec / (prec + rec + 1e-6);
        let bpm_g = 60.0 / g.period;
        let tempo_prior = 0.08 * (-0.5 * ((bpm_g / 115.0).log2() / 0.75).powi(2)).exp();
        let score = 0.35 * (s_env / n) + 3.0 * f1_net + tempo_prior;
        if best.is_none_or(|(bs_best, _)| score > bs_best) {
            best = Some((score, g));
        }
    }

    let grid = best.map(|(_, g)| g).unwrap_or(Grid { period: 0.5, phase: 0.0, down: 0 });
    let (bt, dt) = grid.beats(0.0, dur_f);
    let flagged: Vec<(f64, bool)> = bt
        .into_iter()
        .map(|t| {
            let is_down = dt.iter().any(|&d| (d - t).abs() < 1e-3);
            (f64::from(t), is_down)
        })
        .collect();
    Beats::from_flagged(&flagged)
}

fn raw_stereo_from_song_dir(dir: &Path) -> Result<Vec<[f32; 2]>> {
    if dir.join("mix.flac").exists() {
        return files::read_flac(&dir.join("mix.flac"));
    }
    let (v, d, b, o) = (
        files::read_flac(&dir.join("vocals.flac"))?,
        files::read_flac(&dir.join("drums.flac"))?,
        files::read_flac(&dir.join("bass.flac"))?,
        files::read_flac(&dir.join("other.flac"))?,
    );
    let n = v.len().min(d.len()).min(b.len()).min(o.len());
    Ok((0..n)
        .map(|i| std::array::from_fn(|c| ((v[i][c] + d[i][c]) + b[i][c]) + o[i][c]))
        .collect())
}

fn clean_gain(v: f32) -> f64 {
    if v < 0.03 {
        0.0
    } else if v > 0.97 {
        1.0
    } else {
        f64::from(v)
    }
}

/// Refines a planned `Spec` with `Cortex`'s continuous fader shaping inside the blend while strictly
/// preserving:
/// 1) `sone_plan`'s exact sixteenth-note vocal exit (`a.vocals`) & vocal entry (`b.vocals`) — never
///    cutting mid-word and never playing `a.vocals` across a `segments` backward loop jump.
/// 2) Full `1.0` unity lock on ALL 4 Deck B stems for `u >= +0.5` (and `vocals/drums/bass = 1.0` for
///    `u >= 0.0`), so `ai_lanes` can NEVER dip Song B's vocals or stems after the handoff!
/// 3) Vocal-pocket ducking (`Other = 0.52`, `LPF = 3800 Hz` on Deck A; `HPF = 135 Hz` on Deck B vocals)
///    during `VoiceOverBeat` so Song B's singer sits crystal-clear over Song A's groove.
fn apply_studio_lanes(
    base: &Spec,
    ai_lanes: &[[f32; 8]],
    recipe: Recipe,
    clash: bool,
) -> Spec {
    let mut spec = base.clone();
    let b_from = spec.b.from.unwrap_or(-8.0);
    let first_ev = spec.first_event.min(b_from);
    // Keep separated stems ONLY inside the active blend + 2.25-bar settle window (matching `at_one(0, 2.0)`)!
    let clip_end = if recipe == Recipe::VoiceOverBeat { 2.25 } else { 1.5 };
    spec.clip = [(first_ev - 2.0).min(-6.0), clip_end];
    spec.a.until = Some(1.0);
    spec.a.shape = Shape::Lin;
    spec.b.shape = Shape::Lin;
    spec.a.shapes.clear();
    spec.b.shapes.clear();

    // Never allow Deck A vocals to continue across a backward `segments` loop jump!
    let earliest_loop_u: Option<f64> = spec
        .a
        .segments
        .iter()
        .filter(|g| (g.src - g.u0).abs() > 1e-6 && g.u0 < 0.0)
        .map(|g| g.u0)
        .fold(None, |acc, u| Some(acc.map_or(u, |m: f64| m.min(u))));

    let base_a_voc = spec.a.lanes.get(&Stem::Vocals).cloned().unwrap_or_default();
    let base_b_voc = spec.b.lanes.get(&Stem::Vocals).cloned().unwrap_or_default();

    // Find exact ruler position where Deck A vocals finish in `sone_plan` (clamped before any loop!).
    let mut a_voc_exit = base_a_voc
        .iter()
        .find(|pt| pt[1] <= 0.02)
        .map(|pt| pt[0])
        .unwrap_or(0.0);
    if let Some(lu) = earliest_loop_u {
        if a_voc_exit > lu {
            a_voc_exit = lu;
            spec.a.lanes.insert(
                Stem::Vocals,
                vec![[spec.clip[0], 1.0], [lu - 0.12, 1.0], [lu, 0.0]],
            );
            for s in &mut spec.a.sends {
                if s.stem == Stem::Vocals {
                    s.send = vec![[lu - 0.55, 0.0], [lu - 0.45, 1.0], [lu - 0.04, 1.0], [lu, 0.0]];
                }
            }
        }
    }

    // Find exact ruler position where Deck B vocals open in `sone_plan`.
    let b_voc_entry = base_b_voc
        .iter()
        .find(|pt| pt[1] >= 0.5)
        .map(|pt| pt[0])
        .unwrap_or(if recipe == Recipe::VoiceOverBeat { b_from } else { 0.0 });

    if recipe == Recipe::VoiceOverBeat {
        // Keep `sone_plan`'s exact sixteenth-note `a.vocals` & `b.vocals` and harmonic-clash handling
        // (when `clash == true`, `sone_plan` fades `a.Bass` and `a.Other` out cleanly before `b_from` so
        // `a.Drums` at full club punch carries Song B's vocals with zero key clash and Cingulate ~ 1.0!).
        let pocket_start = b_voc_entry.min(first_ev);
        spec.a.lanes.insert(
            Stem::Drums,
            vec![[spec.clip[0], 1.0], [-0.02, 1.0], [0.0, 0.0], [1.5, 0.0]],
        );
        if clash {
            let fade_start = (pocket_start - 1.5).max(spec.clip[0] + 0.25);
            spec.a.lanes.insert(
                Stem::Bass,
                vec![[spec.clip[0], 1.0], [pocket_start - 0.25, 1.0], [-0.02, 0.85], [0.0, 0.0], [1.5, 0.0]],
            );
            spec.a.lanes.insert(
                Stem::Other,
                vec![[spec.clip[0], 1.0], [fade_start, 1.0], [pocket_start + 0.5, 0.12], [-0.02, 0.0], [1.5, 0.0]],
            );
            spec.a.lpf.insert(
                Stem::Bass,
                vec![[pocket_start - 0.25, 18_000.0], [pocket_start + 0.25, 145.0], [0.0, 140.0]],
            );
            spec.a.lpf.insert(
                Stem::Other,
                vec![[fade_start, 18_000.0], [pocket_start, 1_400.0], [0.0, 1_200.0]],
            );
        } else {
            let mut pa_bas: Points = vec![[spec.clip[0], 1.0], [pocket_start - 0.25, 1.0]];
            let mut pa_oth: Points = vec![[spec.clip[0], 1.0], [pocket_start - 0.25, 1.0]];
            for (idx, row) in ai_lanes.iter().enumerate() {
                let j = -A_BEFORE + idx as i64;
                let u = (j as f64 + 0.5) / 4.0;
                if u < pocket_start || u >= -0.05 {
                    continue;
                }
                let gb = clean_gain(row[2]).clamp(0.86, 0.98);
                let go = clean_gain(row[3]).clamp(0.48, 0.62);
                pa_bas.push([u, gb]);
                pa_oth.push([u, go]);
            }
            pa_bas.extend_from_slice(&[[-0.02, 0.88], [0.0, 0.0], [1.5, 0.0]]);
            pa_oth.extend_from_slice(&[[-0.02, 0.45], [0.0, 0.0], [1.5, 0.0]]);
            spec.a.lanes.insert(Stem::Bass, pa_bas);
            spec.a.lanes.insert(Stem::Other, pa_oth);
            spec.a.lpf.insert(
                Stem::Other,
                vec![[pocket_start - 0.25, 18_000.0], [pocket_start + 0.5, 3_600.0], [0.0, 3_200.0]],
            );
        }

        // Ensure Song B's isolated vocal has zero low-end rumble over Song A's kick/bass, then full range on the 1!
        spec.b.hpf.insert(
            Stem::Vocals,
            vec![[b_from, 135.0], [-0.05, 135.0], [0.05, 20.0]],
        );
        // Ensure ALL 4 Deck B stems reach exact `1.0` cleanly on/right after `u = 0.0` (mix_007: zero 2-bar melody hole after A cuts!)
        spec.b.lanes.insert(Stem::Drums, vec![[b_from, 0.0], [-0.02, 0.0], [0.0, 1.0], [clip_end, 1.0]]);
        spec.b.lanes.insert(Stem::Bass, vec![[b_from, 0.0], [-0.02, 0.0], [0.0, 1.0], [clip_end, 1.0]]);
        spec.b.lanes.insert(Stem::Other, vec![[b_from, 0.0], [-0.04, 0.0], [0.0, 0.85], [0.5, 1.0], [clip_end, 1.0]]);
        let mut bv = base_b_voc;
        bv.retain(|pt| pt[0] < -0.02);
        if bv.is_empty() {
            bv.push([b_from, 1.0]);
        }
        bv.push([0.0, 1.0]);
        bv.push([clip_end, 1.0]);
        spec.b.lanes.insert(Stem::Vocals, bv);

        let mut av = spec.a.lanes.get(&Stem::Vocals).cloned().unwrap_or_default();
        av.retain(|pt| pt[0] <= a_voc_exit + 0.15);
        av.push([(a_voc_exit + 0.16).max(0.05), 0.0]);
        av.push([clip_end, 0.0]);
        spec.a.lanes.insert(Stem::Vocals, av);
    } else {
        // `Sneak` or `OutroUnderIntro`: keep `sone_plan`'s clean vocal timing + drum/bass handoff,
        // subtly shaped by Cortex BEFORE `u = 0.0`, and strictly locked to `ga = 0.0, gb = 1.0` after `u = 0.0`!
        let swap = spec.swap_u.unwrap_or(-4.0);
        for &stem in &[Stem::Drums, Stem::Bass, Stem::Other] {
            let k = match stem {
                Stem::Vocals => 0,
                Stem::Drums => 1,
                Stem::Bass => 2,
                Stem::Other => 3,
            };
            let base_pa = spec.a.lanes.get(&stem).cloned().unwrap_or_default();
            let base_pb = spec.b.lanes.get(&stem).cloned().unwrap_or_default();
            let floor_a = if stem == Stem::Other && b_voc_entry < -0.5 { 0.22 } else { 0.45 };
            let mut pa: Points = vec![[spec.clip[0], 1.0], [first_ev, 1.0]];
            let mut pb: Points = vec![[b_from, lane_at(&base_pb, b_from)]];
            for (idx, row) in ai_lanes.iter().enumerate() {
                let j = -A_BEFORE + idx as i64;
                let u = (j as f64 + 0.5) / 4.0;
                if u <= first_ev || u >= -0.05 {
                    continue;
                }
                let ref_a = lane_at(&base_pa, u);
                let ref_b = lane_at(&base_pb, u);
                let ga = if stem == Stem::Other && b_voc_entry < -0.5 && u >= b_voc_entry - 0.25 {
                    ref_a.clamp(0.22, 0.35)
                } else {
                    (0.65 * ref_a + 0.35 * clean_gain(row[k])).clamp(floor_a, 1.0)
                };
                let gb = if stem == Stem::Bass && u < 0.0 && recipe == Recipe::Sneak {
                    0.0
                } else if u < swap && stem == Stem::Other && recipe == Recipe::Sneak {
                    0.0
                } else {
                    (0.70 * ref_b + 0.30 * clean_gain(row[4 + k])).clamp(0.0, 1.0)
                };
                pa.push([u, ga]);
                pb.push([u, gb]);
            }
            pa.extend_from_slice(&[[-0.015, if stem == Stem::Other && b_voc_entry < -0.5 { 0.22 } else { 0.50 }], [0.0, 0.0], [1.5, 0.0]]);
            if stem == Stem::Other {
                pb.extend_from_slice(&[[0.0, 0.85], [0.5, 1.0], [1.5, 1.0]]);
            } else {
                pb.extend_from_slice(&[[0.0, 1.0], [1.5, 1.0]]);
            }
            spec.a.lanes.insert(stem, pa);
            spec.b.lanes.insert(stem, pb);
        }
        // Lock vocals cleanly: preserve early vocal overlap (`b_voc_entry < 0` in mix_002!), never chop Song A mid-line,
        // and NEVER mute Song B after `u = 0` (`mix_006`)!
        let mut av = spec.a.lanes.get(&Stem::Vocals).cloned().unwrap_or_default();
        av.retain(|pt| pt[0] <= a_voc_exit + 0.15);
        av.push([(a_voc_exit + 0.16).max(0.02), 0.0]);
        av.push([1.5, 0.0]);
        spec.a.lanes.insert(Stem::Vocals, av);

        let bv = if b_voc_entry < -0.05 {
            vec![
                [b_from, 0.0],
                [(b_voc_entry - 0.04).max(b_from), 0.0],
                [b_voc_entry, 1.0],
                [0.0, 1.0],
                [1.5, 1.0],
            ]
        } else if b_voc_entry <= 0.05 {
            vec![[b_from, 0.0], [-0.03, 0.0], [0.0, 1.0], [0.5, 1.0], [1.5, 1.0]]
        } else {
            vec![
                [b_from, 0.0],
                [b_voc_entry - 0.03, 0.0],
                [b_voc_entry, 1.0],
                [(b_voc_entry + 0.25).max(0.5), 1.0],
                [1.5, 1.0],
            ]
        };
        spec.b.lanes.insert(Stem::Vocals, bv);
    }

    // Lift curves must settle to 0.0 dB before the seam at u = 1.5 so master splice gain matches 100%!
    if !spec.b.gain_db.is_empty() {
        spec.b.gain_db = vec![[b_from, spec.b.gain_db[0][1]], [0.5, spec.b.gain_db[0][1]], [1.25, 0.0]];
    }
    spec
}

/// Splices untouched raw stereo Song A before `spec.clip[0]` and untouched raw stereo Song B right
/// at `spec.clip[1]` (`u = +1.5` bars, once Deck A is 100% silent), pre-rolling `PhaseVocoder` on
/// unseparated `raw_b` with fine sample cross-correlation so the seam is imperceptible and ZERO
/// separated stems ever play after the transition!
fn splice_master_showcase(
    raw_a: &[[f32; 2]],
    raw_b: &[[f32; 2]],
    a_gain_db: f64,
    b_gain_db: f64,
    rendered: &sone_render::Rendered,
    spec: &Spec,
    lead_in_s: f64,
    follow_out_s: f64,
) -> Vec<[f32; 2]> {
    let sr = 44_100.0;
    let ga = 10.0f64.powf(a_gain_db / 20.0) as f32;
    let gb = 10.0f64.powf(b_gain_db / 20.0) as f32;
    let a_from = ((rendered.a_zero_s + spec.clip[0] * rendered.bar_s) * sr).round().max(0.0) as usize;
    let seam = SEAM.min(rendered.audio.len() / 6);

    let a_lead_start = a_from.saturating_sub((lead_in_s * sr) as usize);
    let a_end = a_from.min(raw_a.len());
    let mut out: Vec<[f32; 2]> = Vec::new();
    if a_lead_start < a_end {
        for &f in &raw_a[a_lead_start..a_end] {
            out.push([f[0] * ga, f[1] * ga]);
        }
    }

    let clip = &rendered.audio;
    for (i, &cf) in clip.iter().enumerate() {
        if i < seam && a_end + i < raw_a.len() {
            let w = 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / seam as f32).cos());
            let af = raw_a[a_end + i];
            out.push([af[0] * ga * (1.0 - w) + cf[0] * w, af[1] * ga * (1.0 - w) + cf[1] * w]);
        } else {
            out.push(cf);
        }
    }

    let bar_b = rendered.bar_s / rendered.ratio;
    let u_seam = spec.clip[1] - seam as f64 / sr / rendered.bar_s;
    let seg = spec.b.segments.iter().rev().find(|s| s.u0 <= u_seam && s.u1.is_none_or(|u1| u_seam < u1));
    let bars_b = seg.map_or(u_seam, |s| s.src + (u_seam - s.u0));
    let b_resume = ((rendered.b_zero_s + bars_b * bar_b) * sr).round().max(0.0) as usize;

    let want_out = (follow_out_s * sr) as usize;
    if b_resume + 4096 < raw_b.len() && want_out > seam && out.len() > seam {
        let preroll_in = ((2.0 * bar_b * sr) as usize).min(b_resume);
        let in_start = b_resume - preroll_in;
        let in_need = ((want_out as f64 / rendered.ratio) as usize) + preroll_in + 8192;
        let in_end = (in_start + in_need).min(raw_b.len());
        let scaled_b: Vec<[f32; 2]> = raw_b[in_start..in_end]
            .iter()
            .map(|&f| [f[0] * gb, f[1] * gb])
            .collect();
        let stretched_b = if (rendered.ratio - 1.0).abs() < 1e-4 {
            scaled_b
        } else {
            Stretcher::new(rendered.ratio).render_all(&scaled_b)
        };
        let nom_off = (preroll_in as f64 * rendered.ratio).round() as usize;
        let clip_tail_start = out.len() - seam;

        // Fine sample-exact alignment (±96 samples) across the 2048-sample overlap so phase correlation ~ 1.0:
        let mut best_off = nom_off;
        let mut best_dot = f64::NEG_INFINITY;
        for delta in -96i64..=96 {
            let cand = (nom_off as i64 + delta).max(0) as usize;
            if cand + seam > stretched_b.len() {
                continue;
            }
            let mut dot = 0.0f64;
            for i in (0..seam).step_by(2) {
                let a0 = out[clip_tail_start + i];
                let b0 = stretched_b[cand + i];
                dot += f64::from(a0[0] * b0[0] + a0[1] * b0[1]);
            }
            if dot > best_dot {
                best_dot = dot;
                best_off = cand;
            }
        }
        if best_off + seam < stretched_b.len() {
            for i in 0..seam {
                let w = 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / seam as f32).cos());
                let cf = out[clip_tail_start + i];
                let bf = stretched_b[best_off + i];
                out[clip_tail_start + i] = [cf[0] * (1.0 - w) + bf[0] * w, cf[1] * (1.0 - w) + bf[1] * w];
            }
            let end_s = (best_off + want_out).min(stretched_b.len());
            out.extend_from_slice(&stretched_b[best_off + seam..end_s]);
        }
    }
    out
}

fn sync_sung_flag(meta: &mut CachedSongMeta) {
    for (b, row) in meta.table.v16.iter().enumerate() {
        let voiced = row.iter().filter(|&&x| x > -20.0).count() >= 2;
        if let Some(feat) = meta.table.features.get_mut(b) {
            if feat.len() > 7 {
                feat[7] = if voiced { 1.0 } else { 0.0 };
            }
        }
    }
}

fn index_song_low_mem(
    id: &str,
    song_dir: &Path,
    cache_dir: &Path,
    tympanum: &Tympanum,
    ear_ws: &mut TympanumWorkspace,
    scan_cochlea: &Cochlea,
    scan_ws: &mut CochleaWorkspace,
) -> Result<CachedSongMeta> {
    let cp = cache_dir.join(format!("{id}.json"));
    if cp.exists() {
        if let Ok(mut m) = serde_json::from_str::<CachedSongMeta>(&std::fs::read_to_string(&cp)?) {
            sync_sung_flag(&mut m);
            return Ok(m);
        }
    }
    let stereo = raw_stereo_from_song_dir(song_dir)?;
    let total_len = stereo.len();
    let duration_s = total_len as f64 / 44_100.0;
    let lufs = sone_dsp::integrated_loudness(&stereo, sone_dsp::SR);

    let mono: Vec<f32> = stereo.iter().map(|s| 0.5 * (s[0] + s[1])).collect();
    let t_ear = Instant::now();
    let ear_out = tympanum.analyze(ear_ws, &mono);
    let bp: Vec<f32> = ear_out.frames.iter().map(|f| f[0]).collect();
    let dp: Vec<f32> = ear_out.frames.iter().map(|f| f[1]).collect();
    let beats = fit_neuro_symbolic_beats(&mono, &bp, &dp, duration_s);
    let ear_s = t_ear.elapsed().as_secs_f64();
    drop(mono);

    let left: Vec<f32> = stereo.iter().map(|s| s[0]).collect();
    let right: Vec<f32> = stereo.iter().map(|s| s[1]).collect();
    drop(stereo);

    let hop = scan_cochlea.config.hop;
    let chunk = (8 * 44_100 / hop) * hop;
    let overlap = (44_100 / hop) * hop;
    let mut sep: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
    scan_cochlea.separate(scan_ws, &left, &right, chunk, overlap, &mut sep);
    drop(left);
    drop(right);

    let t_an = Instant::now();
    let mut analyzer = SongAnalyzer::new(id, beats, duration_s);
    let n_frames = ear_out.frames.len();
    for i in (0..total_len).step_by(16_384) {
        let j = (i + 16_384).min(total_len);
        // Gate measurement vocal energy by Tympanum-Pro's 50 Hz `singing` + `breath` probabilities
        // so residual -15 dB accompaniment bleed during breaths/instrumental bars drops cleanly
        // below `VOICED_DB` (-20 dB) & `SILENT_DB` (-30 dB), giving studio-accurate line starts/ends!
        let v: Vec<f32> = (i..j)
            .map(|k| {
                let raw_v = 0.5 * (sep[0][0][k] + sep[0][1][k]);
                let fi = ((k as u64 * 50) / 44_100) as usize;
                let (p_sing, p_breath) = if fi < n_frames {
                    (ear_out.frames[fi][2], ear_out.frames[fi][3])
                } else {
                    (0.5, 0.0)
                };
                let sing_w = ((p_sing - 0.22) / 0.36).clamp(0.0, 1.0);
                let breath_w = 1.0 - 0.85 * ((p_breath - 0.18) / 0.45).clamp(0.0, 1.0);
                let gate = 0.06 + 0.94 * sing_w * breath_w;
                raw_v * gate
            })
            .collect();
        let d: Vec<f32> = (i..j).map(|k| 0.5 * (sep[1][0][k] + sep[1][1][k])).collect();
        let b: Vec<f32> = (i..j).map(|k| 0.5 * (sep[2][0][k] + sep[2][1][k])).collect();
        let o: Vec<f32> = (i..j).map(|k| 0.5 * (sep[3][0][k] + sep[3][1][k])).collect();
        analyzer.push(Block { vocals: &v, drums: &d, bass: &b, other: &o });
    }
    drop(sep);
    let song = analyzer.finish().with_context(|| format!("SongAnalyzer failed on {id}"))?;
    let analyse_s = t_an.elapsed().as_secs_f64();

    let meta = CachedSongMeta {
        total_len,
        duration_s,
        lufs,
        ear_s,
        analyse_s,
        table: song.table,
    };
    std::fs::write(&cp, serde_json::to_string(&meta)?)?;
    Ok(meta)
}

/// Zero-phase Mid-channel vocal-formant dropout bridge (`250 Hz .. 6200 Hz`):
/// When a song contains telephone/radio-filtered call-and-response vocal lines (high-passed above
/// `350 Hz` with no chest fundamental), neural band-split separators can route those phrases into
/// `Other`, creating sudden `-15..-20 dB` vocal dropouts mid-verse when `Other` is muted during
/// `VoiceOverBeat`. Detecting frames inside an active singing phrase (`+-1.8s` local vocal peak
/// `> -32.5 dBFS`) where `Vocals` suddenly drops below `-34.5 dBFS` and transferring only the
/// zero-phase bandpass center (`Mid = 0.5*(L+R)`) component from `Other` back to `Vocals` restores
/// continuous vocal intelligibility while preserving exact sample-level mixture conservation
/// (`Vocals + Other` invariant to machine precision) and leaving clean vocals & instrumental
/// sections 100% untouched.
fn bridge_filtered_vocal_dropouts(stems: &mut [Vec<[f32; 2]>; 4]) {
    let n = stems[0].len();
    if n < 44_100 {
        return;
    }
    let sr = 44_100.0f64;
    // 20 ms frame RMS envelope (`hop = 882` = 50 Hz)
    let hop = 882usize;
    let nf = n / hop;
    if nf < 10 {
        return;
    }
    let mut voc_db = vec![-120.0f32; nf];
    let mut mid_ratio = vec![0.0f32; nf];
    for fi in 0..nf {
        let s0 = fi * hop;
        let s1 = (s0 + hop).min(n);
        let mut ev = 0.0f64;
        let mut em = 0.0f64;
        let mut es = 0.0f64;
        for i in s0..s1 {
            let (vl, vr) = (f64::from(stems[0][i][0]), f64::from(stems[0][i][1]));
            ev += 0.5 * (vl * vl + vr * vr);
            let (ol, or) = (f64::from(stems[3][i][0]), f64::from(stems[3][i][1]));
            let m = 0.5 * (ol + or);
            let s = 0.5 * (ol - or);
            em += m * m;
            es += s * s;
        }
        let inv = 1.0 / (s1 - s0) as f64;
        voc_db[fi] = (10.0 * (ev * inv + 1e-12).log10()) as f32;
        mid_ratio[fi] = (em / (em + es + 1e-12)) as f32;
    }

    // Active singing context: rolling max over +-1.5s (+-75 frames)
    let mut raw_gate = vec![0.0f32; nf];
    let mut any_active = false;
    for fi in 0..nf {
        let lo = fi.saturating_sub(75);
        let hi = (fi + 76).min(nf);
        let mut near_pk = -120.0f32;
        for &d in &voc_db[lo..hi] {
            if d > near_pk {
                near_pk = d;
            }
        }
        if near_pk > -32.5 && voc_db[fi] < -34.5 {
            let hole = ((-34.5 - voc_db[fi]) / 5.0).clamp(0.0, 1.0);
            let center = ((mid_ratio[fi] - 0.35) / 0.40).clamp(0.0, 1.0);
            let g = hole * center;
            if g > 0.02 {
                raw_gate[fi] = g;
                any_active = true;
            }
        }
    }
    if !any_active {
        return;
    }

    // Smooth frame gate over 9 frames (180 ms) for click-free transitions
    let mut gate_f = vec![0.0f32; nf];
    for fi in 0..nf {
        let lo = fi.saturating_sub(4);
        let hi = (fi + 5).min(nf);
        gate_f[fi] = raw_gate[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
    }

    // Zero-phase forward-backward 2nd-order Butterworth HPF (260 Hz) + LPF (6200 Hz) on Other Mid
    fn biquad_zero_phase(sig: &mut [f32], b: [f64; 3], a: [f64; 2]) {
        let (mut z1, mut z2) = (0.0f64, 0.0f64);
        for x in sig.iter_mut() {
            let inp = f64::from(*x);
            let y = b[0] * inp + z1;
            z1 = b[1] * inp - a[0] * y + z2;
            z2 = b[2] * inp - a[1] * y;
            *x = y as f32;
        }
        let (mut z1, mut z2) = (0.0f64, 0.0f64);
        for x in sig.iter_mut().rev() {
            let inp = f64::from(*x);
            let y = b[0] * inp + z1;
            z1 = b[1] * inp - a[0] * y + z2;
            z2 = b[2] * inp - a[1] * y;
            *x = y as f32;
        }
    }
    let butter_hpf = |fc: f64| -> ([f64; 3], [f64; 2]) {
        let w0 = 2.0 * std::f64::consts::PI * fc / sr;
        let (cw, sw) = (w0.cos(), w0.sin());
        let alpha = sw / std::f64::consts::SQRT_2;
        let a0 = 1.0 + alpha;
        (
            [(1.0 + cw) * 0.5 / a0, -(1.0 + cw) / a0, (1.0 + cw) * 0.5 / a0],
            [-2.0 * cw / a0, (1.0 - alpha) / a0],
        )
    };
    let butter_lpf = |fc: f64| -> ([f64; 3], [f64; 2]) {
        let w0 = 2.0 * std::f64::consts::PI * fc / sr;
        let (cw, sw) = (w0.cos(), w0.sin());
        let alpha = sw / std::f64::consts::SQRT_2;
        let a0 = 1.0 + alpha;
        (
            [(1.0 - cw) * 0.5 / a0, (1.0 - cw) / a0, (1.0 - cw) * 0.5 / a0],
            [-2.0 * cw / a0, (1.0 - alpha) / a0],
        )
    };

    let mut mid_band: Vec<f32> = stems[3].iter().map(|s| 0.5 * (s[0] + s[1])).collect();
    let (bh, ah) = butter_hpf(260.0);
    let (bl, al) = butter_lpf(6200.0);
    biquad_zero_phase(&mut mid_band, bh, ah);
    biquad_zero_phase(&mut mid_band, bl, al);

    for i in 0..n {
        let pf = (i as f32) / (hop as f32);
        let f0 = (pf.floor() as usize).min(nf - 1);
        let f1 = (f0 + 1).min(nf - 1);
        let frac = pf - (f0 as f32);
        let g = gate_f[f0] * (1.0 - frac) + gate_f[f1] * frac;
        if g > 1e-4 {
            let delta = g * mid_band[i];
            stems[0][i][0] += delta;
            stems[0][i][1] += delta;
            stems[3][i][0] -= delta;
            stems[3][i][1] -= delta;
        }
    }
}

fn separate_window_cached(
    id: &str,
    raw: &[[f32; 2]],
    anchor_bar: usize,
    meta: &CachedSongMeta,
    before_s: f64,
    after_s: f64,
    win_cache: &Path,
    cochlea_pro: &Cochlea,
    ws: &mut CochleaWorkspace,
) -> Result<(usize, [Vec<[f32; 2]>; 4], f64)> {
    let hop = cochlea_pro.config.hop;
    let n = raw.len();
    let t_anc = meta.table.bars.get(anchor_bar).and_then(|&bi| meta.table.times.get(bi)).copied().unwrap_or(0.5 * meta.duration_s);
    let q_step = 8 * 44_100;
    let s0_raw = ((t_anc - before_s).max(0.0) * 44_100.0) as usize;
    let s0 = ((s0_raw / q_step) * q_step / hop) * hop;
    let s1_want = (((t_anc + after_s) * 44_100.0) as usize).min(n);
    let s1_q = (((s1_want + q_step - 1) / q_step) * q_step).min(n);
    let s1 = ((s1_q / hop) * hop).max((s0 + hop).min(n)).min(n);
    let wlen = s1 - s0;

    let cache_file = win_cache.join(format!("{id}_{s0}_{s1}.f32bin"));
    if cache_file.exists() {
        let bytes = std::fs::read(&cache_file)?;
        if bytes.len() == wlen * 4 * 2 * 4 {
            let mut stems: [Vec<[f32; 2]>; 4] = std::array::from_fn(|_| vec![[0.0; 2]; wlen]);
            let mut cur = 0usize;
            for st in 0..4 {
                for i in 0..wlen {
                    let l = f32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
                    let r = f32::from_le_bytes(bytes[cur + 4..cur + 8].try_into().unwrap());
                    stems[st][i] = [l, r];
                    cur += 8;
                }
            }
            bridge_filtered_vocal_dropouts(&mut stems);
            return Ok((s0, stems, 0.0));
        }
    }

    let l: Vec<f32> = raw[s0..s1].iter().map(|f| f[0]).collect();
    let r: Vec<f32> = raw[s0..s1].iter().map(|f| f[1]).collect();
    let chunk = (8 * 44_100 / hop) * hop;
    let overlap = (44_100 / hop) * hop;
    let mut sep: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
    let t0 = Instant::now();
    cochlea_pro.separate(ws, &l, &r, chunk, overlap, &mut sep);
    let el = t0.elapsed().as_secs_f64();

    let mlen = sep[0][0].len().min(wlen);
    let mut stems: [Vec<[f32; 2]>; 4] = std::array::from_fn(|st| (0..mlen).map(|i| [sep[st][0][i], sep[st][1][i]]).collect());
    drop(sep);

    let mut bytes = Vec::with_capacity(mlen * 4 * 2 * 4);
    for st in 0..4 {
        for &[lv, rv] in &stems[st] {
            bytes.extend_from_slice(&lv.to_le_bytes());
            bytes.extend_from_slice(&rv.to_le_bytes());
        }
    }
    let _ = std::fs::write(&cache_file, bytes);
    bridge_filtered_vocal_dropouts(&mut stems);
    Ok((s0, stems, el))
}

fn required_window_secs(spec: &Spec, is_a: bool, bar_s: f64) -> (f64, f64) {
    let mut min_u = spec.clip[0] - 2.0;
    let mut max_u = spec.clip[1] + 5.0;
    if let Some(ap) = &spec.align_pairs {
        let pair_u = if is_a { ap.a } else { ap.b };
        min_u = min_u.min(pair_u[0] - 2.0);
        max_u = max_u.max(pair_u[1] + 2.0);
    }
    let segs = if is_a { &spec.a.segments } else { &spec.b.segments };
    for g in segs {
        min_u = min_u.min(g.src - 2.0);
    }
    let anchor_u = if is_a { spec.a.anchor_u } else { spec.b.anchor_u };
    let before_s = ((anchor_u - min_u).max(4.0) * bar_s) + 8.0;
    let after_s = ((max_u - anchor_u).max(4.0) * bar_s) + 8.0;
    (before_s, after_s)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exports = Path::new(args.first().map(|s| s.as_str()).unwrap_or("/home/user/bot/exports_pro"));
    let stems_dir = Path::new(args.get(1).map(|s| s.as_str()).unwrap_or("/home/user/djlab/stems"));
    let out_dir = PathBuf::from(args.get(2).map(|s| s.as_str()).unwrap_or("/home/user/bot/e2e_mixes_pro"));
    let max_pairs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    std::fs::create_dir_all(&out_dir)?;

    let cache_dir_env = std::env::var("SONE_STRUCT_CACHE").unwrap_or_else(|_| "/home/user/bot/.sone_struct_cache_v2".into());
    let cache_dir = Path::new(&cache_dir_env);
    let win_cache_env = std::env::var("SONE_WIN_CACHE").unwrap_or_else(|_| "/home/user/bot/.sone_win_cache".into());
    let win_cache = Path::new(&win_cache_env);
    std::fs::create_dir_all(cache_dir)?;
    std::fs::create_dir_all(win_cache)?;

    let bp = exports.join("sone-pro-f16.safetensors");
    println!("Loading unified Sone bundle from {} (win_cache={})...", bp.display(), win_cache.display());
    let bundle = Sone::load(WeightFile::from_vec(std::fs::read(&bp)?)?)?;
    let (tympanum, mut cochlea_pro, mut plan_net, mut crit_net) = (bundle.tympanum, bundle.cochlea, bundle.cortex, bundle.cingulate);
    if let Ok(cp) = std::env::var("SONE_COCHLEA_PATH") {
        if Path::new(&cp).exists() {
            println!("  -> Overriding Cochlea with {cp}");
            cochlea_pro = Cochlea::load(WeightFile::from_vec(std::fs::read(&cp)?)?)?;
        }
    }
    if let Ok(xp) = std::env::var("SONE_CORTEX_PATH") {
        if Path::new(&xp).exists() {
            println!("  -> Overriding Cortex with {xp}");
            plan_net = sone_infer::BeatNet::load(WeightFile::from_vec(std::fs::read(&xp)?)?)?;
        }
    }
    if let Ok(gp) = std::env::var("SONE_CINGULATE_PATH") {
        if Path::new(&gp).exists() {
            println!("  -> Overriding Cingulate with {gp}");
            crit_net = sone_infer::BeatNet::load(WeightFile::from_vec(std::fs::read(&gp)?)?)?;
        }
    }
    let base_bundle = Sone::load(WeightFile::from_vec(std::fs::read("/home/user/bot/exports_v2_base/sone-v2-base-f16.safetensors")?)?)?;
    let scan_cochlea = base_bundle.cochlea;

    let mut song_ids: Vec<String> = std::fs::read_dir(stems_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && (e.path().join("vocals.flac").exists() || e.path().join("mix.flac").exists()))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    song_ids.sort();
    println!("Indexing {} songs with Cochlea-v2-Base (6.73 dB) + Tympanum-Pro vocal gate...", song_ids.len());

    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let results: std::sync::Mutex<Vec<(usize, String, CachedSongMeta)>> = std::sync::Mutex::new(Vec::with_capacity(song_ids.len()));
    std::thread::scope(|sc| -> Result<()> {
        let mut handles = Vec::new();
        for _ in 0..2 {
            handles.push(sc.spawn(|| -> Result<()> {
                let mut loc_ear = TympanumWorkspace::default();
                let mut loc_scan = CochleaWorkspace::default();
                loop {
                    let k = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if k >= song_ids.len() {
                        break;
                    }
                    let id = &song_ids[k];
                    let meta = index_song_low_mem(id, &stems_dir.join(id), cache_dir, &tympanum, &mut loc_ear, &scan_cochlea, &mut loc_scan)?;
                    println!(
                        "  [INDEX {}/{}] {id} ({:.1}s) | BPM={:.2} Key={} lines={}/{} LUFS={:.1}",
                        k + 1,
                        song_ids.len(),
                        meta.duration_s,
                        meta.table.bpm,
                        meta.table.camelot,
                        meta.table.line_starts.len(),
                        meta.table.line_ends.len(),
                        meta.lufs
                    );
                    results.lock().unwrap().push((k, id.clone(), meta));
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().expect("worker panicked")?;
        }
        Ok(())
    })?;
    drop(scan_cochlea);

    let mut songs = results.into_inner().unwrap();
    songs.sort_by_key(|(k, _, _)| *k);

    struct PairCand {
        a_idx: usize,
        b_idx: usize,
        spec: Spec,
        recipe: Recipe,
        recipe_prob: f32,
        critic_good: f32,
        rank_score: f32,
        stretch_pct: f32,
        bar_s: f64,
        plan_ms: f64,
        critic_ms: f64,
    }

    let mut beat_ws = BeatWorkspace::default();
    let mut cands: Vec<PairCand> = Vec::new();
    for a_idx in 0..songs.len() {
        for b_idx in 0..songs.len() {
            if a_idx == b_idx {
                continue;
            }
            let (_, id_a, sa) = &songs[a_idx];
            let (_, id_b, sb) = &songs[b_idx];
            let kd = song_distance(&sa.table.camelot, &sa.table.camelot_alt, &sb.table.camelot, &sb.table.camelot_alt);
            let clash = kd >= 2 || (sa.table.tuning_cents - sb.table.tuning_cents).abs() > 25.0;
            let sa_view = PlanSong::new(&sa.table);
            let sb_view = PlanSong::new(&sb.table);
            let b_early_vocal = sb.table.line_ends.len() >= 3 && sb_view.voice_bar().is_some_and(|vb| vb <= 12);

            let mut default_plan = None;
            for stretch_lim in [4.0, 5.3, 6.8] {
                let opt = PlanOptions {
                    id: format!("pair_{}_{}", &id_a[..6], &id_b[..6]),
                    max_stretch_pct: stretch_lim,
                    clip_before: 2.0,
                    clip_after: 1.5,
                    ..PlanOptions::default()
                };
                if let Ok(p) = sone_plan::plan(&sa.table, &sb.table, &opt) {
                    if !p.spec.no_sync {
                        default_plan = Some((p, stretch_lim));
                        break;
                    }
                }
            }
            let Some((def_plan, used_lim)) = default_plan else { continue };
            let stretch_pct = def_plan.fit.stretch_pct as f32;

            let t_p = Instant::now();
            let mut plan_ms = 0.0f64;
            let mut best_cand: Option<(f32, Spec, Recipe, f32, f32, f64)> = None;
            for cand_r in [Recipe::VoiceOverBeat, Recipe::Sneak, Recipe::OutroUnderIntro] {
                for &lead_bars in &[8.0, 6.0] {
                    let f_opt = PlanOptions {
                        id: def_plan.spec.id.clone(),
                        max_stretch_pct: used_lim,
                        lead_bars,
                        clip_before: 2.0,
                        clip_after: 1.5,
                        force: Some(cand_r),
                        ..PlanOptions::default()
                    };
                    let Ok(rp) = sone_plan::plan(&sa.table, &sb.table, &f_opt) else { continue };
                    if rp.spec.no_sync || rp.recipe == Recipe::Gap {
                        continue;
                    }
                    // For VoiceOverBeat, verify Song B actually sings during the overlap!
                    if cand_r == Recipe::VoiceOverBeat && !b_early_vocal {
                        continue;
                    }
                    // When `VoiceOverBeat` runs on `clash == true`, `apply_studio_lanes` low-passes Deck A's
                    // sub-bass (`145 Hz`) and fades Deck A's `Other` out (`eval_kd = 1` adjacent-key manifold).
                    let eval_kd = if cand_r == Recipe::VoiceOverBeat && clash { 1 } else { kd };
                    let raw_ex = example_from_spec(
                        &rp.spec.id,
                        &rp.spec,
                        &sa.table,
                        &sb.table,
                        cand_r,
                        stretch_pct,
                        eval_kd,
                        Label { good: 1, issues: vec![], source: "e2e".into() },
                    );
                    let t_cp = Instant::now();
                    let ai_p = plan_net.cortex(&mut beat_ws, &raw_ex);
                    plan_ms = t_cp.elapsed().as_secs_f64() * 1000.0;

                    let cand_spec = apply_studio_lanes(&rp.spec, &ai_p.lanes, cand_r, clash);
                    let cand_ex = example_from_spec(
                        &cand_spec.id,
                        &cand_spec,
                        &sa.table,
                        &sb.table,
                        cand_r,
                        stretch_pct,
                        eval_kd,
                        Label { good: 1, issues: vec![], source: "e2e".into() },
                    );
                    let crit = crit_net.cingulate(&mut beat_ws, &cand_ex);
                    let r_prob = match cand_r {
                        Recipe::VoiceOverBeat => ai_p.recipe_probs[0],
                        Recipe::Sneak => ai_p.recipe_probs[1],
                        Recipe::OutroUnderIntro => ai_p.recipe_probs[2],
                        Recipe::Gap => ai_p.recipe_probs[3],
                    };

                    // Check if Song A's blend bed is already instrumental (zero vocal conflict -> ideal vocal-over-beat!):
                    let a_anc = cand_spec.a.anchor_bar.min(sa_view.bars());
                    let a_bed_lo = a_anc.saturating_sub(6);
                    let a_bed_sung = if a_anc > a_bed_lo {
                        (a_bed_lo..a_anc).map(|k| sa_view.sung_share(k)).sum::<f32>() / (a_anc - a_bed_lo) as f32
                    } else {
                        1.0
                    };
                    let a_outro_instrumental = a_bed_sung <= 0.20;

                    let worst_defect = crit.issues.iter().map(|&(_, p)| p).fold(0.0f32, f32::max);
                    let stretch_pen = 0.018 * (stretch_pct.abs() - 2.0).max(0.0);
                    let key_pen = if kd >= 2 { 0.04 } else { 0.0 };
                    let vob_bonus = if cand_r == Recipe::VoiceOverBeat && b_early_vocal && crit.good >= 0.72 {
                        if a_outro_instrumental { 0.24 } else { 0.14 }
                    } else {
                        0.0
                    };
                    let loop_pen = if cand_spec.a.segments.len() > 1 { 0.12 } else { 0.0 };
                    let rank_score = 0.70 * crit.good + 0.18 * r_prob + 0.15 * rp.confidence + vob_bonus
                        - 0.25 * worst_defect - stretch_pen - key_pen - loop_pen;
                    if best_cand.as_ref().is_none_or(|(bs, _, _, _, _, _)| rank_score > *bs) {
                        best_cand = Some((rank_score, cand_spec, cand_r, r_prob, crit.good, rp.bar_s));
                    }
                }
            }
            let critic_ms = t_p.elapsed().as_secs_f64() * 1000.0;
            let Some((rank_score, spec, chosen_recipe, recipe_prob, crit_good, bar_s)) = best_cand else { continue };
            cands.push(PairCand {
                a_idx,
                b_idx,
                spec,
                recipe: chosen_recipe,
                recipe_prob,
                critic_good: crit_good,
                rank_score,
                stretch_pct,
                bar_s,
                plan_ms,
                critic_ms,
            });
        }
    }
    cands.sort_by(|a, b| b.rank_score.total_cmp(&a.rank_score));
    println!("Evaluated {} clean syncable pairs (<= 6.8% stretch); rendering top {max_pairs} studio transitions...", cands.len());

    // Pin the exact 3 pairs Luka critiqued first so Mix 1, Mix 2, Mix 3 can be directly A/B verified!
    let pinned_prefixes: [(&str, &str); 3] = [
        ("6N185A", "7yoqSl"),
        ("1fVQmM", "5ihq4q"),
        ("6wPupZ", "0hkR93"),
    ];
    let mut ordered_indices: Vec<usize> = Vec::new();
    for (pa, pb) in pinned_prefixes {
        if let Some(pos) = cands.iter().position(|c| songs[c.a_idx].1.starts_with(pa) && songs[c.b_idx].1.starts_with(pb)) {
            ordered_indices.push(pos);
        }
    }

    let mut count_a = vec![0usize; songs.len()];
    let mut count_b = vec![0usize; songs.len()];
    let mut chosen_pairs: Vec<(usize, usize)> = Vec::new();
    let mut bench_rows = Vec::new();
    let mut pro_ws = CochleaWorkspace::default();
    let mut made = 0usize;

    let mut queue: Vec<usize> = ordered_indices;
    for min_crit in [0.85f32, 0.65, 0.45] {
        for (i, c) in cands.iter().enumerate() {
            if !queue.contains(&i) && c.critic_good >= min_crit {
                queue.push(i);
            }
        }
    }

    for pass_cap in [1usize, 2, 3] {
        for &ci in &queue {
            if made >= max_pairs {
                break;
            }
            let cand = &cands[ci];
            let is_pinned = made < 3 && pinned_prefixes.iter().any(|(pa, pb)| songs[cand.a_idx].1.starts_with(pa) && songs[cand.b_idx].1.starts_with(pb));
            if chosen_pairs.contains(&(cand.a_idx, cand.b_idx)) {
                continue;
            }
            if !is_pinned {
                if count_a[cand.a_idx] >= pass_cap || count_b[cand.b_idx] >= pass_cap {
                    continue;
                }
                if pass_cap < 3 && chosen_pairs.contains(&(cand.b_idx, cand.a_idx)) {
                    continue;
                }
            }

            let (_, id_a, sa) = &songs[cand.a_idx];
            let (_, id_b, sb) = &songs[cand.b_idx];
            let raw_a = raw_stereo_from_song_dir(&stems_dir.join(id_a))?;
            let raw_b = raw_stereo_from_song_dir(&stems_dir.join(id_b))?;

            let (bef_a, aft_a) = required_window_secs(&cand.spec, true, cand.bar_s);
            let bar_b_s = 60.0 / sb.table.bpm.max(40.0) * f64::from(cand.spec.b.beats_per_bar);
            let (bef_b, aft_b) = required_window_secs(&cand.spec, false, bar_b_s);

            let (off_a, stems_a, sep_a_s) = separate_window_cached(id_a, &raw_a, cand.spec.a.anchor_bar, sa, bef_a, aft_a, win_cache, &cochlea_pro, &mut pro_ws)?;
            let (off_b, stems_b, sep_b_s) = separate_window_cached(id_b, &raw_b, cand.spec.b.anchor_bar, sb, bef_b, aft_b, win_cache, &cochlea_pro, &mut pro_ws)?;

            let win_a = WindowStems { id: id_a.clone(), total_len: sa.total_len, offset: off_a, stems: stems_a, grid: BeatGrid::from_table(&sa.table), lufs: sa.lufs };
            let win_b = WindowStems { id: id_b.clone(), total_len: sb.total_len, offset: off_b, stems: stems_b, grid: BeatGrid::from_table(&sb.table), lufs: sb.lufs };

            let mut spec = cand.spec.clone();
            spec.id = format!("sone_mix{:02}_{}_{}", made + 1, &id_a[..6], &id_b[..6]);
            let a_gain_db = (STATION_LUFS - sa.lufs).min(0.0);
            let b_gain_db = (STATION_LUFS - sb.lufs).min(0.0);
            let pair_lufs = sa.lufs.min(sb.lufs);
            spec.a.trim_db += a_gain_db - (pair_lufs - sa.lufs);
            spec.b.trim_db += b_gain_db - (pair_lufs - sb.lufs);

            let t_r = Instant::now();
            let mut rendered = match sone_render::render_with(&spec, &win_a, &win_b, &RenderOptions::default()) {
                Ok(r) => r,
                Err(e) => {
                    println!("  [skip] {} -> {}: render error: {e:#}", id_a, id_b);
                    continue;
                }
            };
            // Closed-form joint Newton drum-slope + phase-nudge lock (fixes live-drumming tempo wander in mix_005
            // and re-centers `nudge_ms` even when `spec.align_pairs` is `None` on Sneak blends like mix_002!):
            if !spec.no_sync && rendered.report.lag_ms().abs() > 0.4 && rendered.report.lag_ms().abs() < 42.0 {
                let mut re_nudge = spec.clone();
                re_nudge.b.nudge_ms -= rendered.report.lag_ms() / rendered.ratio;
                if let Ok(rn) = sone_render::render_with(&re_nudge, &win_a, &win_b, &RenderOptions::default()) {
                    if rn.report.lag_ms().abs() < rendered.report.lag_ms().abs() {
                        spec = re_nudge;
                        rendered = rn;
                    }
                }
            }
            for _ in 0..4 {
                let d_ms = rendered.report.drift_ms();
                if d_ms.abs() <= 1.0 || spec.no_sync {
                    break;
                }
                let span_bars = spec.align_pairs.as_ref().map_or_else(
                    || {
                        let lo = spec.b.from.unwrap_or(-8.0).max(-8.0);
                        let hi = spec.a.until.unwrap_or(1.0).min(8.0);
                        (hi - lo).abs().max(2.0)
                    },
                    |ap| (ap.a[1] - ap.a[0]).abs().max(2.0),
                );
                let half_span_ms = (500.0 * span_bars * rendered.bar_s).max(1500.0);
                let cur_bpm_b = spec.b.tempo_bpm.unwrap_or(rendered.report.tempo_b);
                let rel = d_ms / half_span_ms;
                let mut best_improve: Option<(Spec, sone_render::Rendered)> = None;
                for mult in [1.0, -1.0, 0.65, -0.65, 1.35, -1.35] {
                    let try_bpm = cur_bpm_b * (1.0 + mult * rel);
                    let mut try_spec = spec.clone();
                    try_spec.b.tempo_bpm = Some(try_bpm);
                    if let Ok(mut r_try) = sone_render::render_with(&try_spec, &win_a, &win_b, &RenderOptions::default()) {
                        if r_try.report.lag_ms().abs() > 0.4 && r_try.report.lag_ms().abs() < 42.0 {
                            try_spec.b.nudge_ms -= r_try.report.lag_ms() / r_try.ratio;
                            if let Ok(r_re) = sone_render::render_with(&try_spec, &win_a, &win_b, &RenderOptions::default()) {
                                r_try = r_re;
                            }
                        }
                        let cur_best_d = best_improve.as_ref().map_or(rendered.report.drift_ms().abs(), |(_, rb)| rb.report.drift_ms().abs());
                        if r_try.report.drift_ms().abs() + 0.10 < cur_best_d && r_try.report.lag_ms().abs() <= 2.5 {
                            let done_early = r_try.report.drift_ms().abs() <= 0.8 && r_try.report.lag_ms().abs() <= 1.0;
                            best_improve = Some((try_spec, r_try));
                            if done_early {
                                break;
                            }
                        }
                    }
                }
                if let Some((bs, br)) = best_improve {
                    spec = bs;
                    rendered = br;
                } else {
                    break;
                }
            }
            let drift = rendered.report.drift_ms();
            let lag = rendered.report.lag_ms();
            // Strict ear-calibrated studio sync gate (Luka's ear: <= 2.2 ms drift, <= 2.5 ms lag):
            if drift.abs() > 2.2 || lag.abs() > 2.5 {
                println!("  [skip] {} -> {}: drift={:+.1}ms lag={:+.1}ms (exceeds 2.2ms studio sync gate)", id_a, id_b, drift, lag);
                continue;
            }
            let showcase_audio = splice_master_showcase(&raw_a, &raw_b, a_gain_db, b_gain_db, &rendered, &spec, 10.0, 12.0);
            let render_s = t_r.elapsed().as_secs_f64();

            let wav_path = out_dir.join(format!("{}.wav", spec.id));
            files::write_wav16(&wav_path, &showcase_audio)?;
            std::fs::write(out_dir.join(format!("{}.spec.json", spec.id)), serde_json::to_string_pretty(&[&spec])?)?;

            chosen_pairs.push((cand.a_idx, cand.b_idx));
            count_a[cand.a_idx] += 1;
            count_b[cand.b_idx] += 1;
            made += 1;

            println!(
                "RENDERED [{made}/{max_pairs}] {} | {:?} (p={:.2}) | Cingulate={:.3} | A_bar={} B_bar={} (from={:+.1}) | BPM {:.1}->{:.1} ({:+.1}%) | Key {}->{} | drift={:+.1}ms lag={:+.1}ms | SepA={:.1}s SepB={:.1}s Render={:.2}s",
                spec.id, cand.recipe, cand.recipe_prob, cand.critic_good, spec.a.anchor_bar, spec.b.anchor_bar, spec.b.from.unwrap_or(0.0), sa.table.bpm, sb.table.bpm, cand.stretch_pct, sa.table.camelot, sb.table.camelot, drift, lag, sep_a_s, sep_b_s, render_s
            );

            bench_rows.push(serde_json::json!({
                "id": spec.id,
                "a": id_a,
                "b": id_b,
                "a_anchor_bar": spec.a.anchor_bar,
                "b_anchor_bar": spec.b.anchor_bar,
                "b_from_u": spec.b.from,
                "bpm_a": sa.table.bpm,
                "bpm_b": sb.table.bpm,
                "stretch_pct": cand.stretch_pct,
                "key_a": sa.table.camelot,
                "key_b": sb.table.camelot,
                "recipe": format!("{:?}", cand.recipe),
                "recipe_prob": cand.recipe_prob,
                "critic_good": cand.critic_good,
                "drift_ms": drift,
                "lag_ms": lag,
                "plan_ms": cand.plan_ms,
                "critic_ms": cand.critic_ms,
                "render_s": render_s,
            }));
        }
        if made >= max_pairs {
            break;
        }
    }
    std::fs::write(out_dir.join("e2e_benchmark.json"), serde_json::to_string_pretty(&bench_rows)?)?;
    println!("Completed {made} studio mixes -> {}", out_dir.display());
    Ok(())
}
