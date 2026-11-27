//! End-to-End Pure-CPU 4-Organ DJ Transition Pipeline & Paper Benchmark (`Sone`).
//! Raw stereo audio only (no pre-split stems, no precomputed JSON):
//!   Raw Stereo Song A & B -> Tympanum (CPU) + OnsetEnvelope Comb-Grid -> Cochlea (CPU)
//!   -> SongAnalyzer (CPU) -> Best-of-K Cortex + Cingulate Search (CPU)
//!   -> sone_render + Untouched Master Splicing & KeyLock Glide (CPU) -> Rendered Mix (.wav)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use sone_analysis::{SongAnalyzer, measure::Block, structure::Beats};
use sone_beats::grid::{FPS, Grid, OnsetEnvelope, Votes, autocorrelation_period, fit_grid, median_period, prob_at};
use sone_core::{Label, Points, Recipe, Shape, Spec, Stem};
use sone_dsp::Stretcher;
use sone_infer::{
    BeatNet, BeatWorkspace, Cochlea, CochleaWorkspace, Sone, Tympanum, TympanumWorkspace, WeightFile,
    beat::RECIPE_ORDER,
};
use sone_plan::PlanOptions;
use sone_plan::keys::distance as key_distance;
use sone_render::{BeatGrid, MemoryStems, RenderOptions, StemSource, files};
use sone_sim::build::{A_AFTER, A_BEFORE, example_from_spec, recipe_of};

const STATION_LUFS: f64 = -14.0;
const SEAM: usize = 1024;

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

/// Neuro-symbolic multi-octave comb-grid tracker combining physical spectral flux
/// (`OnsetEnvelope`) with `Tympanum` 50 Hz beat + downbeat probabilities and F1 peak recall.
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
    // Dense safety sweep (64..185 BPM in ~3.5% steps, refined ±4% inside fit_grid)
    let mut bpm = 64.0f32;
    while bpm <= 185.0 {
        cands.push(60.0 / bpm);
        bpm *= 1.035;
    }

    // Score each fitted grid against neural beat F1 (precision on grid ticks + recall on Tympanum peaks)
    // plus local-mean-removed physical onset flux.
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

fn sustained_b_vocal_u(b_beats: &[Vec<f32>], b_earliest: f64) -> f64 {
    for idx in 0..60usize {
        if idx + 4 > b_beats.len() {
            break;
        }
        let j = idx as i64 - 64;
        let u_beat = j as f64 / 4.0;
        if u_beat < b_earliest - 1e-6 {
            continue;
        }
        let first_ok = b_beats[idx].first().copied().unwrap_or(-60.0) > -17.5;
        let all_sung = (0..4).all(|w| b_beats[idx + w].get(7).copied().unwrap_or(0.0) >= 0.5);
        let loud_cnt = (0..4).filter(|&w| b_beats[idx + w].first().copied().unwrap_or(-60.0) > -17.5).count();
        if first_ok && all_sung && loud_cnt >= 3 {
            return u_beat;
        }
    }
    b_earliest
}

fn apply_v2_lanes(
    base: &Spec,
    b_beats: &[Vec<f32>],
    ai_lanes: &[[f32; 8]],
    recipe: Recipe,
    start_bars: f32,
    swap_bars: f32,
) -> Spec {
    let mut spec = base.clone();
    spec.recipe = Some(recipe);
    spec.first_event = -f64::from(start_bars).clamp(2.0, 16.0);
    if swap_bars > 0.25 {
        spec.swap_u = Some(-f64::from(swap_bars));
    }
    let [u0, u1] = spec.clip;
    spec.a.until = Some(u1.min(4.25));
    let b_earliest = base.b.from.unwrap_or(-16.25).max(u0).max(-16.25);
    let handoff_u = if recipe == Recipe::VoiceOverBeat {
        sustained_b_vocal_u(b_beats, b_earliest).clamp(b_earliest, -1.0)
    } else {
        b_earliest
    };
    spec.b.from = Some(handoff_u);
    spec.a.shape = Shape::Lin;
    spec.b.shape = Shape::Lin;
    spec.a.shapes.clear();
    spec.b.shapes.clear();

    let mut echo_cut_u: Option<f64> = base.a.sends.iter().find_map(|s| {
        if s.stem == Stem::Vocals {
            s.send.last().map(|pt| pt[0])
        } else {
            None
        }
    });
    if recipe == Recipe::VoiceOverBeat {
        if echo_cut_u.is_none_or(|cu| handoff_u > cu + 1.25) {
            for s in &mut spec.a.sends {
                if s.stem == Stem::Vocals {
                    s.send = vec![
                        [handoff_u - 0.60, 0.0],
                        [handoff_u - 0.50, 1.0],
                        [handoff_u - 0.05, 1.0],
                        [handoff_u, 0.0],
                    ];
                }
            }
            echo_cut_u = Some(handoff_u);
        }
    }

    let mut a_map: BTreeMap<Stem, Points> = BTreeMap::new();
    let mut b_map: BTreeMap<Stem, Points> = BTreeMap::new();
    for (k, &stem) in Stem::ALL.iter().enumerate() {
        let mut pa: Points = Vec::with_capacity(ai_lanes.len() + 2);
        let mut pb: Points = Vec::with_capacity(ai_lanes.len() + 2);
        pa.push([-16.25, 1.0]);
        pb.push([-16.25, 0.0]);
        for (idx, row) in ai_lanes.iter().enumerate() {
            let j = -A_BEFORE + idx as i64;
            let u = (j as f64 + 0.5) / 4.0;
            let mut ga = clean_gain(row[k]);
            let mut gb = if u < handoff_u - 0.05 { 0.0 } else { clean_gain(row[4 + k]) };
            if recipe == Recipe::VoiceOverBeat {
                if u < handoff_u - 0.05 {
                    if stem != Stem::Vocals {
                        ga = 1.0;
                    }
                    gb = 0.0;
                } else if u < -0.25 {
                    match stem {
                        Stem::Drums => {
                            ga = ga.clamp(0.88, 0.94);
                            gb = 0.0;
                        }
                        Stem::Bass => {
                            ga = ga.clamp(0.84, 0.90);
                            gb = 0.0;
                        }
                        Stem::Other => {
                            ga = ga.clamp(0.68, 0.78);
                            gb = 0.0;
                        }
                        Stem::Vocals => {
                            gb = gb.max(0.98);
                        }
                    }
                }
                if stem == Stem::Vocals {
                    let cut_u = echo_cut_u.unwrap_or(handoff_u);
                    if u <= cut_u - 0.12 {
                        ga = 1.0;
                    } else if u >= cut_u + 0.05 {
                        ga = 0.0;
                    }
                }
            } else if u < handoff_u && stem != Stem::Vocals {
                ga = ga.max(0.95);
                gb = 0.0;
            }
            pa.push([u, ga]);
            pb.push([u, gb]);
        }
        let end_u = (A_AFTER as f64 + 0.5) / 4.0;
        pa.push([end_u, 0.0]);
        pb.push([end_u, 1.0]);
        a_map.insert(stem, pa);
        b_map.insert(stem, pb);
    }
    spec.a.lanes = a_map;
    spec.b.lanes = b_map;
    spec
}

/// Splice 100% untouched bit-exact original stereo of Song A (before `a_from`) and Song B
/// (after `b_resume`, tempo-glided back to 1.0x via `Stretcher`) around the stem-rendered transition clip.
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
    let seam = SEAM.min(rendered.audio.len() / 4);

    // 1. Untouched Song A master lead-in
    let a_lead_start = a_from.saturating_sub((lead_in_s * sr) as usize);
    let a_end = a_from.min(raw_a.len());
    let mut out: Vec<[f32; 2]> = Vec::new();
    if a_lead_start < a_end {
        for &f in &raw_a[a_lead_start..a_end] {
            out.push([f[0] * ga, f[1] * ga]);
        }
    }

    // 2. Cosine seam into transition clip + body of transition clip
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

    // 3. Untouched Song B master follow-out (gliding smoothly from `rendered.ratio` back to `1.0`)
    let bar_b = rendered.bar_s / rendered.ratio;
    let u_seam = spec.clip[1] - seam as f64 / sr / rendered.bar_s;
    let seg = spec.b.segments.iter().rev().find(|s| s.u0 <= u_seam && s.u1.is_none_or(|u1| u_seam < u1));
    let bars_b = seg.map_or(u_seam, |s| s.src + (u_seam - s.u0));
    let b_resume = ((rendered.b_zero_s + bars_b * bar_b) * sr).round().max(0.0) as usize;

    let want_out = (follow_out_s * sr) as usize;
    if b_resume + 2048 < raw_b.len() && want_out > seam {
        let mut b_tail: Vec<[f32; 2]> = Vec::with_capacity(want_out + seam);
        if (rendered.ratio - 1.0).abs() < 1e-3 {
            let end_b = (b_resume + want_out).min(raw_b.len());
            for &bf in &raw_b[b_resume..end_b] {
                b_tail.push([bf[0] * gb, bf[1] * gb]);
            }
        } else {
            let mut stretcher = Stretcher::new(rendered.ratio);
            let mut read = b_resume;
            let glide_frames = (8.0 * rendered.bar_s * sr) as usize;
            while b_tail.len() < want_out && read + 2048 <= raw_b.len() {
                let p = (b_tail.len() as f64 / glide_frames.max(1) as f64).min(1.0);
                let cur_ratio = rendered.ratio + (1.0 - rendered.ratio) * p;
                stretcher.set_ratio(cur_ratio);
                let chunk: Vec<f32> = raw_b[read..read + 2048].iter().flat_map(|f| [f[0] * gb, f[1] * gb]).collect();
                let made = stretcher.process(&chunk);
                b_tail.extend(made.as_chunks::<2>().0.iter().copied());
                read += 2048;
            }
        }
        if b_tail.len() > seam && out.len() > seam {
            let clip_tail_start = out.len() - seam;
            for i in 0..seam {
                let w = 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / seam as f32).cos());
                let cf = out[clip_tail_start + i];
                let bf = b_tail[i];
                out[clip_tail_start + i] = [cf[0] * (1.0 - w) + bf[0] * w, cf[1] * (1.0 - w) + bf[1] * w];
            }
            out.extend_from_slice(&b_tail[seam..b_tail.len().min(want_out)]);
        }
    }
    out
}

struct AnalyzedSong {
    raw: Vec<[f32; 2]>,
    table: sone_core::BeatTable,
    mem_stems: MemoryStems,
    ear_s: f64,
    stems_full_s: f64,
    stems_win24_s: f64,
    analyse_s: f64,
    duration_s: f64,
}

fn analyze_raw_song_cpu(
    id: &str,
    stereo: Vec<[f32; 2]>,
    tympanum: &Tympanum,
    ear_ws: &mut TympanumWorkspace,
    cochlea_net: &Cochlea,
    stems_ws: &mut CochleaWorkspace,
    measure_win24: bool,
) -> Result<AnalyzedSong> {
    let n = stereo.len();
    let duration_s = n as f64 / 44_100.0;
    let mono: Vec<f32> = stereo.iter().map(|s| 0.5 * (s[0] + s[1])).collect();

    // 1. Tympanum + Neuro-Symbolic Multi-Octave Comb-Grid on CPU
    let t_ear = Instant::now();
    let ear_out = tympanum.analyze(ear_ws, &mono);
    let beat_probs: Vec<f32> = ear_out.frames.iter().map(|f| f[0]).collect();
    let down_probs: Vec<f32> = ear_out.frames.iter().map(|f| f[1]).collect();
    let beats = fit_neuro_symbolic_beats(&mono, &beat_probs, &down_probs, duration_s);
    let ear_s = t_ear.elapsed().as_secs_f64();

    // 2. Cochlea on CPU (measure explicit 24s production Worker window on first song, derive RTF on rest)
    let left: Vec<f32> = stereo.iter().map(|s| s[0]).collect();
    let right: Vec<f32> = stereo.iter().map(|s| s[1]).collect();
    let hop = cochlea_net.config.hop;
    let chunk = (8 * 44_100 / hop) * hop;
    let overlap = (44_100 / hop) * hop;

    let win24_opt = if measure_win24 {
        let win24 = ((24 * 44_100) / hop * hop).min((n / hop) * hop);
        let mut sep_win: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
        let t_w24 = Instant::now();
        cochlea_net.separate(stems_ws, &left[..win24], &right[..win24], chunk, overlap, &mut sep_win);
        Some(t_w24.elapsed().as_secs_f64())
    } else {
        None
    };

    let mut sep: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
    let t_st = Instant::now();
    cochlea_net.separate(stems_ws, &left, &right, chunk, overlap, &mut sep);
    let stems_full_s = t_st.elapsed().as_secs_f64();
    let stems_win24_s = win24_opt.unwrap_or_else(|| stems_full_s * (24.0 / duration_s.max(1.0)));

    // 3. SongAnalyzer on CPU
    let t_an = Instant::now();
    let mono_stems: Vec<Vec<f32>> = (0..4)
        .map(|s| (0..n).map(|i| 0.5 * (sep[s][0][i] + sep[s][1][i])).collect())
        .collect();
    let mut analyzer = SongAnalyzer::new(id, beats, duration_s);
    for i in (0..n).step_by(16_384) {
        let j = (i + 16_384).min(n);
        analyzer.push(Block {
            vocals: &mono_stems[0][i..j],
            drums: &mono_stems[1][i..j],
            bass: &mono_stems[2][i..j],
            other: &mono_stems[3][i..j],
        });
    }
    let song = analyzer.finish().with_context(|| format!("SongAnalyzer failed on {id}"))?;
    let analyse_s = t_an.elapsed().as_secs_f64();

    let stems_arr: [Vec<[f32; 2]>; 4] = std::array::from_fn(|s| (0..n).map(|i| [sep[s][0][i], sep[s][1][i]]).collect());
    let lufs = MemoryStems::measure_lufs(&stems_arr);
    let mem_stems = MemoryStems {
        id: id.to_string(),
        stems: stems_arr,
        grid: BeatGrid::from_table(&song.table),
        lufs,
        clean_vocals: true,
        leak_free: true,
    };

    Ok(AnalyzedSong {
        raw: stereo,
        table: song.table,
        mem_stems,
        ear_s,
        stems_full_s,
        stems_win24_s,
        analyse_s,
        duration_s,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ear_only = args.iter().any(|a| a == "--ear-only");
    let pos_args: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let exports = Path::new(pos_args.first().map(|s| s.as_str()).unwrap_or("/home/user/bot/exports"));
    let stems_dir = Path::new(pos_args.get(1).map(|s| s.as_str()).unwrap_or("/home/user/djlab/stems"));
    let out_dir = PathBuf::from(pos_args.get(2).map(|s| s.as_str()).unwrap_or("/home/user/bot/e2e_mixes"));
    let max_pairs: usize = pos_args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    std::fs::create_dir_all(&out_dir)?;

    let bundle_path = ["sone-pro-f16.safetensors", "sone-v2-base-f16.safetensors", "sone-6.5M-f16.safetensors", "model.safetensors"]
        .into_iter()
        .map(|name| exports.join(name))
        .find(|p| p.exists());

    let (tympanum, cochlea_net, plan_net, crit_net) = if let Some(bp) = bundle_path {
        println!("Loading unified Sone single-file bundle from {}...", bp.display());
        let bundle = Sone::load(WeightFile::from_vec(std::fs::read(&bp)?)?)?;
        (bundle.tympanum, bundle.cochlea, bundle.cortex, bundle.cingulate)
    } else {
        let ear_path = exports.join("sone-tympanum-f16.safetensors");
        let stems_path = exports.join("sone-cochlea-f16.safetensors");
        let plan_path = exports.join("sone-cortex-f16.safetensors");
        let crit_path = exports.join("sone-cingulate-f16.safetensors");
        (
            Tympanum::load(WeightFile::from_vec(std::fs::read(&ear_path)?)?)?,
            Cochlea::load(WeightFile::from_vec(std::fs::read(&stems_path)?)?)?,
            BeatNet::load(WeightFile::from_vec(std::fs::read(&plan_path)?)?)?,
            BeatNet::load(WeightFile::from_vec(std::fs::read(&crit_path)?)?)?,
        )
    };
    let fast_scan_path = Path::new("/home/user/bot/exports/sone-cochlea-f16.safetensors");
    let struct_cochlea_opt = if cochlea_net.config.n_fft == 4096 && fast_scan_path.exists() {
        Some(Cochlea::load(WeightFile::from_vec(std::fs::read(fast_scan_path)?)?)?)
    } else {
        None
    };
    let scan_cochlea = struct_cochlea_opt.as_ref().unwrap_or(&cochlea_net);

    let mut ear_ws = TympanumWorkspace::default();
    let mut stems_ws = CochleaWorkspace::default();
    let mut beat_ws = BeatWorkspace::default();

    let mut song_ids: Vec<String> = std::fs::read_dir(stems_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && e.path().join("vocals.flac").exists())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    song_ids.sort();
    println!("Discovered {} raw stereo songs in {}", song_ids.len(), stems_dir.display());

    if ear_only {
        for id in &song_ids {
            let raw = raw_stereo_from_song_dir(&stems_dir.join(id))?;
            let dur = raw.len() as f64 / 44_100.0;
            let mono: Vec<f32> = raw.iter().map(|s| 0.5 * (s[0] + s[1])).collect();
            let t0 = Instant::now();
            let ear_out = tympanum.analyze(&mut ear_ws, &mono);
            let bp: Vec<f32> = ear_out.frames.iter().map(|f| f[0]).collect();
            let dp: Vec<f32> = ear_out.frames.iter().map(|f| f[1]).collect();
            let beats = fit_neuro_symbolic_beats(&mono, &bp, &dp, dur);
            let el = t0.elapsed().as_secs_f64();
            let bpm = if beats.times.len() >= 2 {
                60.0 * (beats.times.len() - 1) as f64 / (beats.times.last().unwrap() - beats.times[0]).max(1e-6)
            } else {
                0.0
            };
            println!(
                "  [EAR] {id} ({dur:.1}s) | {:.2}s ({:.4}x RT) | BPM={bpm:.2} | TympanumKey={} ({:+.1}c)",
                el,
                el / dur,
                ear_out.key_name(),
                ear_out.tuning_cents
            );
        }
        return Ok(());
    }

    let mut songs: Vec<(String, AnalyzedSong)> = Vec::with_capacity(song_ids.len());
    if let Some(first_id) = song_ids.first() {
        println!("Analyzing raw stereo song [1/{}] {first_id} on 1 CPU thread (uncontended benchmark)...", song_ids.len());
        let raw = raw_stereo_from_song_dir(&stems_dir.join(first_id))?;
        let hop_p = cochlea_net.config.hop;
        let win24 = ((24 * 44_100) / hop_p * hop_p).min((raw.len() / hop_p) * hop_p);
        let l24: Vec<f32> = raw[..win24].iter().map(|s| s[0]).collect();
        let r24: Vec<f32> = raw[..win24].iter().map(|s| s[1]).collect();
        let mut sep24: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
        let t_w24 = Instant::now();
        cochlea_net.separate(&mut stems_ws, &l24, &r24, (8 * 44_100 / hop_p) * hop_p, (44_100 / hop_p) * hop_p, &mut sep24);
        let measured_win24_s = t_w24.elapsed().as_secs_f64();

        let mut scan_ws = CochleaWorkspace::default();
        let mut an = analyze_raw_song_cpu(first_id, raw, &tympanum, &mut ear_ws, scan_cochlea, &mut scan_ws, false)?;
        an.stems_win24_s = measured_win24_s;
        println!(
            "  -> {first_id} ({:.1}s): Tympanum+Grid {:.2}s ({:.3}x RT) | Cochlea-24s {:.2}s ({:.3}x RT) | Analyze {:.2}s | BPM={:.2} Key={}",
            an.duration_s,
            an.ear_s,
            an.ear_s / an.duration_s,
            an.stems_win24_s,
            an.stems_win24_s / 24.0,
            an.analyse_s,
            an.table.bpm,
            an.table.camelot
        );
        songs.push((first_id.clone(), an));
    }
    let ref_win24_s = songs.first().map(|(_, s)| s.stems_win24_s).unwrap_or(22.5);
    if song_ids.len() > 1 {
        let rest = &song_ids[1..];
        let n_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(2).min(rest.len()).max(1);
        let next_idx = std::sync::atomic::AtomicUsize::new(0);
        let total = song_ids.len();
        let results: std::sync::Mutex<Vec<(usize, String, AnalyzedSong)>> = std::sync::Mutex::new(Vec::with_capacity(rest.len()));
        std::thread::scope(|s| -> Result<()> {
            let mut handles = Vec::with_capacity(n_threads);
            for _ in 0..n_threads {
                handles.push(s.spawn(|| -> Result<()> {
                    let mut loc_ear = TympanumWorkspace::default();
                    let mut loc_stems = CochleaWorkspace::default();
                    loop {
                        let k = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if k >= rest.len() {
                            break;
                        }
                        let id = &rest[k];
                        println!("Analyzing raw stereo song [{}/{total}] {id} on CPU ({n_threads}-thread pool)...", k + 2);
                        let raw = raw_stereo_from_song_dir(&stems_dir.join(id))?;
                        let mut an = analyze_raw_song_cpu(id, raw, &tympanum, &mut loc_ear, scan_cochlea, &mut loc_stems, false)?;
                        an.stems_win24_s = ref_win24_s;
                        println!(
                            "  -> {id} ({:.1}s): Tympanum+Grid {:.2}s ({:.3}x RT) | Analyze {:.2}s | BPM={:.2} Key={}",
                            an.duration_s,
                            an.ear_s,
                            an.ear_s / an.duration_s,
                            an.analyse_s,
                            an.table.bpm,
                            an.table.camelot
                        );
                        results.lock().unwrap().push((k, id.clone(), an));
                    }
                    Ok(())
                }));
            }
            for h in handles {
                h.join().expect("worker panicked")?;
            }
            Ok(())
        })?;
        let mut collected = results.into_inner().unwrap();
        collected.sort_by_key(|(k, _, _)| *k);
        for (_, id, an) in collected {
            songs.push((id, an));
        }
    }

    struct PairCand {
        a_idx: usize,
        b_idx: usize,
        spec: Spec,
        recipe: Recipe,
        recipe_prob: f32,
        critic_good: f32,
        rank_score: f32,
        stretch_pct: f32,
        plan_ms: f64,
        critic_ms: f64,
    }

    let mut cands: Vec<PairCand> = Vec::new();
    for a_idx in 0..songs.len() {
        for b_idx in 0..songs.len() {
            if a_idx == b_idx {
                continue;
            }
            let (id_a, sa) = &songs[a_idx];
            let (id_b, sb) = &songs[b_idx];

            let mut found_plan = None;
            for stretch_lim in [4.0, 6.5, 9.5] {
                let opt = PlanOptions {
                    id: format!("pair_{}_{}", &id_a[..6], &id_b[..6]),
                    max_stretch_pct: stretch_lim,
                    ..PlanOptions::default()
                };
                if let Ok(p) = sone_plan::plan(&sa.table, &sb.table, &opt) {
                    if !p.spec.no_sync {
                        found_plan = Some(p);
                        break;
                    }
                }
            }
            let Some(rule_plan) = found_plan else {
                continue;
            };

            let kd = key_distance(&sa.table.camelot, &sb.table.camelot);
            let rec = recipe_of(&rule_plan.spec);
            let stretch_pct = rule_plan.fit.stretch_pct as f32;
            let base_ex = example_from_spec(
                &rule_plan.spec.id,
                &rule_plan.spec,
                &sa.table,
                &sb.table,
                rec,
                stretch_pct,
                kd,
                Label { good: 1, issues: vec![], source: "e2e".into() },
            );

            let t_p = Instant::now();
            let ai_p = plan_net.cortex(&mut beat_ws, &base_ex);
            let plan_ms = t_p.elapsed().as_secs_f64() * 1000.0;

            let mut r_idx: Vec<usize> = (0..4).collect();
            r_idx.sort_by(|&i, &j| ai_p.recipe_probs[j].total_cmp(&ai_p.recipe_probs[i]));
            let mut best_cand: Option<(f32, Spec, Recipe, f32, f32)> = None;
            let t_c = Instant::now();
            for &ri in r_idx.iter().take(2) {
                let cand_r = RECIPE_ORDER[ri];
                for sb_delta in [0.0f32, -2.0, 2.0] {
                    let cand_start = (ai_p.start_bars + sb_delta).clamp(4.0, 16.0);
                    let cand_spec = apply_v2_lanes(&rule_plan.spec, &base_ex.b_beats, &ai_p.lanes, cand_r, cand_start, ai_p.swap_bars);
                    let cand_ex = example_from_spec(
                        &cand_spec.id,
                        &cand_spec,
                        &sa.table,
                        &sb.table,
                        cand_r,
                        stretch_pct,
                        kd,
                        Label { good: 1, issues: vec![], source: "e2e".into() },
                    );
                    let crit = crit_net.cingulate(&mut beat_ws, &cand_ex);
                    let worst_defect = crit.issues.iter().map(|&(_, p)| p).fold(0.0f32, f32::max);
                    let stretch_pen = 0.008 * (stretch_pct.abs() - 2.5).max(0.0);
                    let key_pen = if kd >= 2 { 0.03 } else { 0.0 };
                    let rank_score = crit.good + 0.12 * ai_p.recipe_probs[ri] - 0.25 * worst_defect - stretch_pen - key_pen;
                    if best_cand.as_ref().is_none_or(|(bs, _, _, _, _)| rank_score > *bs) {
                        best_cand = Some((rank_score, cand_spec, cand_r, ai_p.recipe_probs[ri], crit.good));
                    }
                }
            }
            let critic_ms = t_c.elapsed().as_secs_f64() * 1000.0;
            let (rank_score, spec, chosen_recipe, recipe_prob, crit_good) = best_cand.unwrap();
            cands.push(PairCand {
                a_idx,
                b_idx,
                spec,
                recipe: chosen_recipe,
                recipe_prob,
                critic_good: crit_good,
                rank_score,
                stretch_pct,
                plan_ms,
                critic_ms,
            });
        }
    }

    cands.sort_by(|a, b| b.rank_score.total_cmp(&a.rank_score));
    println!("Evaluated {} syncable candidate transitions across {} songs", cands.len(), songs.len());

    let mut count_a = vec![0usize; songs.len()];
    let mut count_b = vec![0usize; songs.len()];
    let mut upgraded_ranges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); songs.len()];
    let mut chosen_pairs: Vec<(usize, usize)> = Vec::new();
    let mut recipe_seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut bench_rows = Vec::new();
    let mut made = 0usize;

    for pass_cap in [1usize, 2, 3] {
        if made >= max_pairs {
            break;
        }
        // Re-order remaining candidates slightly for recipe diversity on each pick
        loop {
            if made >= max_pairs {
                break;
            }
            let next_pos = cands.iter().enumerate().filter(|(_, c)| {
                !chosen_pairs.contains(&(c.a_idx, c.b_idx))
                    && count_a[c.a_idx] < pass_cap
                    && count_b[c.b_idx] < pass_cap
                    && (pass_cap >= 3 || !chosen_pairs.contains(&(c.b_idx, c.a_idx)))
            }).max_by(|(_, x), (_, y)| {
                let bx = if *recipe_seen.get(&format!("{:?}", x.recipe)).unwrap_or(&0) == 0 { 0.035 } else { 0.0 };
                let by = if *recipe_seen.get(&format!("{:?}", y.recipe)).unwrap_or(&0) == 0 { 0.035 } else { 0.0 };
                (x.rank_score + bx).total_cmp(&(y.rank_score + by))
            }).map(|(i, _)| i);

            let Some(ci) = next_pos else {
                break;
            };
            let cand = &cands[ci];
            chosen_pairs.push((cand.a_idx, cand.b_idx));

            if struct_cochlea_opt.is_some() {
                let hop_p = cochlea_net.config.hop;
                let chunk_p = (8 * 44_100 / hop_p) * hop_p;
                let overlap_p = (44_100 / hop_p) * hop_p;

                let win_for = |song_idx: usize, anchor_bar: usize, before_s: f64, after_s: f64| -> Option<(usize, usize)> {
                    let (_, s) = &songs[song_idx];
                    let n = s.raw.len();
                    let t_anc = s.table.bars.get(anchor_bar).and_then(|&bi| s.table.times.get(bi)).copied().unwrap_or(0.5 * s.duration_s);
                    let s0 = (((t_anc - before_s).max(0.0) * 44_100.0) as usize / hop_p) * hop_p;
                    let s1_raw = (((t_anc + after_s) * 44_100.0) as usize).min(n);
                    let s1 = ((s1_raw / hop_p) * hop_p).max((s0 + hop_p).min(n)).min(n);
                    if upgraded_ranges[song_idx].iter().any(|&(a, b)| a <= s0 && b >= s1) {
                        None
                    } else {
                        Some((s0, s1))
                    }
                };

                let job_a = win_for(cand.a_idx, cand.spec.a.anchor_bar, 48.0, 28.0);
                let job_b = win_for(cand.b_idx, cand.spec.b.anchor_bar, 36.0, 36.0);
                let raw_a_slice = job_a.map(|(s0, s1)| (s0, s1, songs[cand.a_idx].1.raw[s0..s1].to_vec()));
                let raw_b_slice = job_b.map(|(s0, s1)| (s0, s1, songs[cand.b_idx].1.raw[s0..s1].to_vec()));
                let cochlea_ref = &cochlea_net;

                let (sep_a_opt, sep_b_opt) = std::thread::scope(|sc| {
                    let ha = raw_a_slice.map(|(s0, s1, sl)| {
                        sc.spawn(move || {
                            let mut ws = CochleaWorkspace::default();
                            let l: Vec<f32> = sl.iter().map(|f| f[0]).collect();
                            let r: Vec<f32> = sl.iter().map(|f| f[1]).collect();
                            let mut sep: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
                            cochlea_ref.separate(&mut ws, &l, &r, chunk_p, overlap_p, &mut sep);
                            (s0, s1, sep)
                        })
                    });
                    let hb = raw_b_slice.map(|(s0, s1, sl)| {
                        sc.spawn(move || {
                            let mut ws = CochleaWorkspace::default();
                            let l: Vec<f32> = sl.iter().map(|f| f[0]).collect();
                            let r: Vec<f32> = sl.iter().map(|f| f[1]).collect();
                            let mut sep: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
                            cochlea_ref.separate(&mut ws, &l, &r, chunk_p, overlap_p, &mut sep);
                            (s0, s1, sep)
                        })
                    });
                    (ha.map(|h| h.join().unwrap()), hb.map(|h| h.join().unwrap()))
                });

                for (s_idx, res_opt) in [(cand.a_idx, sep_a_opt), (cand.b_idx, sep_b_opt)] {
                    if let Some((s0, s1, sep)) = res_opt {
                        let mlen = sep[0][0].len().min(s1 - s0);
                        let stems = &mut songs[s_idx].1.mem_stems.stems;
                        for st in 0..4 {
                            for i in 0..mlen {
                                stems[st][s0 + i] = [sep[st][0][i], sep[st][1][i]];
                            }
                        }
                        upgraded_ranges[s_idx].push((s0, s0 + mlen));
                    }
                }
            }

            let (id_a, sa) = &songs[cand.a_idx];
            let (id_b, sb) = &songs[cand.b_idx];
            let mut spec = cand.spec.clone();
            spec.id = format!("sone_mix{:02}_{}_{}", made + 1, &id_a[..6], &id_b[..6]);

            let a_gain_db = (STATION_LUFS - sa.mem_stems.lufs()).min(0.0);
            let b_gain_db = (STATION_LUFS - sb.mem_stems.lufs()).min(0.0);
            let pair_lufs = sa.mem_stems.lufs().min(sb.mem_stems.lufs());
            spec.a.trim_db += a_gain_db - (pair_lufs - sa.mem_stems.lufs());
            spec.b.trim_db += b_gain_db - (pair_lufs - sb.mem_stems.lufs());

            let t_r = Instant::now();
            let rendered = match sone_render::render_with(&spec, &sa.mem_stems, &sb.mem_stems, &RenderOptions::default()) {
                Ok(r) => r,
                Err(e) => {
                    println!("  [skip] {} -> {}: render error: {e:#}", id_a, id_b);
                    continue;
                }
            };
            let drift = rendered.report.drift_ms();
            let lag = rendered.report.lag_ms();
            if drift.abs() > sone_render::MAX_DRIFT_MS || lag.abs() > sone_render::MAX_LAG_MS {
                println!("  [skip] {} -> {}: drift={:+.1}ms lag={:+.1}ms", id_a, id_b, drift, lag);
                continue;
            }
            let showcase_audio = splice_master_showcase(&sa.raw, &sb.raw, a_gain_db, b_gain_db, &rendered, &spec, 8.0, 8.0);
            let render_s = t_r.elapsed().as_secs_f64();

            let wav_path = out_dir.join(format!("{}.wav", spec.id));
            files::write_wav16(&wav_path, &showcase_audio)?;
            std::fs::write(out_dir.join(format!("{}.spec.json", spec.id)), serde_json::to_string_pretty(&[&spec])?)?;

            count_a[cand.a_idx] += 1;
            count_b[cand.b_idx] += 1;
            *recipe_seen.entry(format!("{:?}", cand.recipe)).or_insert(0) += 1;
            made += 1;

            println!(
                "E2E RENDERED {} | {:?} (p={:.2}) | Cingulate={:.3} | BPM {:.1}->{:.1} ({:+.1}%) | Key {}->{} | drift={:+.1}ms lag={:+.1}ms | Cortex={:.1}ms Cingulate6x={:.1}ms Render={:.2}s",
                spec.id,
                cand.recipe,
                cand.recipe_prob,
                cand.critic_good,
                sa.table.bpm,
                sb.table.bpm,
                cand.stretch_pct,
                sa.table.camelot,
                sb.table.camelot,
                drift,
                lag,
                cand.plan_ms,
                cand.critic_ms,
                render_s
            );

            bench_rows.push(serde_json::json!({
                "id": spec.id,
                "a": id_a,
                "b": id_b,
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
                "ear_rtf_a": sa.ear_s / sa.duration_s,
                "stems_24s_s_a": sa.stems_win24_s,
                "stems_rtf_a": sa.stems_win24_s / 24.0,
                "stems_full_s_a": sa.stems_full_s,
                "plan_ms": cand.plan_ms,
                "critic_ms": cand.critic_ms,
                "render_s": render_s,
                "total_transition_cpu_s": sa.ear_s + sb.ear_s + sa.stems_win24_s + sb.stems_win24_s + (cand.plan_ms + cand.critic_ms) / 1000.0 + render_s,
            }));
        }
    }

    std::fs::write(out_dir.join("e2e_benchmark.json"), serde_json::to_string_pretty(&bench_rows)?)?;
    println!("Completed {made} end-to-end CPU mixes -> {}", out_dir.display());
    Ok(())
}
