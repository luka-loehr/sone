//! End-to-end raw-audio CLI commands powered by the unified `model.safetensors` bundle:
//! - `sone transition`: beat-locked, stem-carved DJ transition between two WAV/FLAC tracks
//! - `sone stems`: phase-coherent 4-stem separation (`vocals`, `drums`, `bass`, `other`)

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use sone_analysis::{SongAnalyzer, measure::Block, structure::Beats};
use sone_beats::grid::{FPS, Grid, OnsetEnvelope, Votes, autocorrelation_period, fit_grid, median_period, prob_at};
use sone_core::{Label, Points, Recipe, Shape, Spec, Stem};
use sone_dsp::Stretcher;
use sone_infer::{
    BeatWorkspace, Cochlea, CochleaWorkspace, Sone, Tympanum, TympanumWorkspace, WeightFile,
    beat::RECIPE_ORDER,
};
use sone_plan::PlanOptions;
use sone_plan::keys::distance as key_distance;
use sone_render::{BeatGrid, MemoryStems, RenderOptions, StemSource, files::write_wav16};
use sone_sim::build::{A_AFTER, A_BEFORE, example_from_spec, recipe_of};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::args::Args;
use crate::songs::STATION_LUFS;

const SEAM: usize = 1024;

/// Decodes a 44.1 kHz mono or stereo WAV/FLAC file into stereo `[f32; 2]` frames.
pub fn decode_audio_44k(path: &Path) -> Result<Vec<[f32; 2]>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .with_context(|| format!("probing {} (supported: 44.1 kHz WAV or FLAC)", path.display()))?;
    let mut format = probed.format;
    let track = format.default_track().context("no audio track found")?;
    let track_id = track.id;
    let sr = track.codec_params.sample_rate.unwrap_or(0);
    ensure!(
        sr == sone_core::SAMPLE_RATE,
        "{}: sample rate is {sr} Hz, expected {} Hz (resample with `ffmpeg -i in.mp3 -ar 44100 out.wav`)",
        path.display(),
        sone_core::SAMPLE_RATE
    );
    let params = track.codec_params.clone();
    let mut decoder = symphonia::default::get_codecs().make(&params, &DecoderOptions::default())?;
    let mut out: Vec<[f32; 2]> = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let spec = *decoded.spec();
        let ch = spec.channels.count();
        let frames = decoded.frames();
        let b = buf.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        if (b.capacity() as u64) < decoded.capacity() as u64 {
            *b = SampleBuffer::new(decoded.capacity() as u64, spec);
        }
        b.copy_interleaved_ref(decoded);
        let s = b.samples();
        out.reserve(frames);
        for i in 0..frames {
            let l = s[i * ch];
            let r = s[i * ch + usize::from(ch > 1)];
            out.push([l, r]);
        }
    }
    ensure!(!out.is_empty(), "{}: decoded 0 audio frames", path.display());
    Ok(out)
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

fn apply_cortex_lanes(
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
    if recipe == Recipe::VoiceOverBeat && echo_cut_u.is_none_or(|cu| handoff_u > cu + 1.25) {
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

fn splice_masters(
    raw_a: &[[f32; 2]],
    raw_b: &[[f32; 2]],
    a_gain_db: f64,
    b_gain_db: f64,
    rendered: &sone_render::Rendered,
    spec: &Spec,
    window_s: Option<f64>,
) -> Vec<[f32; 2]> {
    let sr = 44_100.0;
    let ga = 10.0f64.powf(a_gain_db / 20.0) as f32;
    let gb = 10.0f64.powf(b_gain_db / 20.0) as f32;
    let a_from = ((rendered.a_zero_s + spec.clip[0] * rendered.bar_s) * sr).round().max(0.0) as usize;
    let seam = SEAM.min(rendered.audio.len() / 4);

    let a_lead_start = match window_s {
        Some(sec) => a_from.saturating_sub((sec * sr) as usize),
        None => 0,
    };
    let a_end = a_from.min(raw_a.len());
    let mut out: Vec<[f32; 2]> = Vec::new();
    if a_lead_start < a_end {
        out.extend(raw_a[a_lead_start..a_end].iter().map(|&f| [f[0] * ga, f[1] * ga]));
    }

    for (i, &cf) in rendered.audio.iter().enumerate() {
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

    let want_out = match window_s {
        Some(sec) => (sec * sr) as usize,
        None => raw_b.len().saturating_sub(b_resume),
    };
    if b_resume + 2048 < raw_b.len() && want_out > seam {
        let mut b_tail: Vec<[f32; 2]> = Vec::with_capacity(want_out + seam);
        if (rendered.ratio - 1.0).abs() < 1e-3 {
            let end_b = (b_resume + want_out).min(raw_b.len());
            b_tail.extend(raw_b[b_resume..end_b].iter().map(|&bf| [bf[0] * gb, bf[1] * gb]));
        } else {
            let mut stretcher = Stretcher::new(rendered.ratio);
            let mut read = b_resume;
            let glide_frames = (8.0 * rendered.bar_s * sr) as usize;
            while b_tail.len() < want_out && read + 2048 <= raw_b.len() {
                let p = (b_tail.len() as f64 / glide_frames.max(1) as f64).min(1.0);
                stretcher.set_ratio(rendered.ratio + (1.0 - rendered.ratio) * p);
                let chunk: Vec<f32> = raw_b[read..read + 2048].iter().flat_map(|f| [f[0] * gb, f[1] * gb]).collect();
                let made = stretcher.process(&chunk);
                b_tail.extend(made.as_chunks::<2>().0.iter().copied());
                read += 2048;
            }
            if window_s.is_none() && read < raw_b.len() {
                b_tail.extend(raw_b[read..].iter().map(|&bf| [bf[0] * gb, bf[1] * gb]));
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
            let take_end = b_tail.len().min(want_out);
            out.extend_from_slice(&b_tail[seam..take_end]);
        }
    }
    out
}

fn analyze_track(
    id: &str,
    stereo: Vec<[f32; 2]>,
    tympanum: &Tympanum,
    cochlea: &Cochlea,
) -> Result<(Vec<[f32; 2]>, sone_core::BeatTable, MemoryStems)> {
    let n = stereo.len();
    let duration_s = n as f64 / 44_100.0;
    let mono: Vec<f32> = stereo.iter().map(|s| 0.5 * (s[0] + s[1])).collect();

    let mut ear_ws = TympanumWorkspace::default();
    let ear_out = tympanum.analyze(&mut ear_ws, &mono);
    let beat_probs: Vec<f32> = ear_out.frames.iter().map(|f| f[0]).collect();
    let down_probs: Vec<f32> = ear_out.frames.iter().map(|f| f[1]).collect();
    let beats = fit_neuro_symbolic_beats(&mono, &beat_probs, &down_probs, duration_s);

    let left: Vec<f32> = stereo.iter().map(|s| s[0]).collect();
    let right: Vec<f32> = stereo.iter().map(|s| s[1]).collect();
    let hop = cochlea.config.hop;
    let chunk = (8 * 44_100 / hop) * hop;
    let overlap = (44_100 / hop) * hop;
    let mut sep: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
    let mut stems_ws = CochleaWorkspace::default();
    cochlea.separate(&mut stems_ws, &left, &right, chunk, overlap, &mut sep);

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
    let song = analyzer.finish().with_context(|| format!("analyzing {id}"))?;
    let stems_arr: [Vec<[f32; 2]>; 4] = std::array::from_fn(|s| (0..n).map(|i| [sep[s][0][i], sep[s][1][i]]).collect());
    let lufs = MemoryStems::measure_lufs(&stems_arr);
    let mem = MemoryStems {
        id: id.to_string(),
        stems: stems_arr,
        grid: BeatGrid::from_table(&song.table),
        lufs,
        clean_vocals: true,
        leak_free: true,
    };
    Ok((stereo, song.table, mem))
}

/// `sone transition <song_a.wav|flac> <song_b.wav|flac> --model <model.safetensors> --out <mix.wav> [--window-s SECONDS] [--spec FILE.json]`
pub fn transition(args: Args) -> Result<()> {
    let [path_a, path_b] = args.words.as_slice() else {
        bail!("usage: sone transition <song_a.wav|flac> <song_b.wav|flac> --model <model.safetensors> --out <mix.wav> [--window-s SECONDS] [--spec FILE.json]");
    };
    let model_path = args.path("model")?;
    let out_path = args.path("out")?;
    let window_s: Option<f64> = args.get("window-s")?;
    let spec_out = args.opt_path("spec");

    let t0 = Instant::now();
    let bytes = std::fs::read(&model_path).with_context(|| format!("reading {}", model_path.display()))?;
    let sone = Sone::load(WeightFile::from_vec(bytes)?)?;

    let pa = Path::new(path_a);
    let pb = Path::new(path_b);
    let id_a = pa.file_stem().and_then(|s| s.to_str()).unwrap_or("song_a").to_string();
    let id_b = pb.file_stem().and_then(|s| s.to_str()).unwrap_or("song_b").to_string();

    eprintln!("Decoding and analyzing Deck A ({}) & Deck B ({})...", pa.display(), pb.display());
    let (raw_a, raw_b) = (decode_audio_44k(pa)?, decode_audio_44k(pb)?);
    let tymp = &sone.tympanum;
    let coch = &sone.cochlea;
    let (res_a, res_b) = std::thread::scope(|sc| {
        let ha = sc.spawn(|| analyze_track(&id_a, raw_a, tymp, coch));
        let hb = sc.spawn(|| analyze_track(&id_b, raw_b, tymp, coch));
        (ha.join().expect("Deck A thread panicked"), hb.join().expect("Deck B thread panicked"))
    });
    let (raw_a, table_a, mem_a) = res_a?;
    let (raw_b, table_b, mem_b) = res_b?;
    ensure!(
        table_a.bars.len() >= 16 && table_b.bars.len() >= 16,
        "full-length songs required (got {} and {} bars; truncated preview clips lack real intros/outros)",
        table_a.bars.len(),
        table_b.bars.len()
    );

    let mut found_plan = None;
    for stretch_lim in [4.0, 6.8, 9.5] {
        let opt = PlanOptions {
            id: format!("{id_a}_into_{id_b}"),
            max_stretch_pct: stretch_lim,
            ..PlanOptions::default()
        };
        if let Ok(p) = sone_plan::plan(&table_a, &table_b, &opt)
            && !p.spec.no_sync
        {
            found_plan = Some(p);
            break;
        }
    }
    let rule_plan = found_plan.context("tracks do not share a compatible tempo within ±9.5% stretch (or 2:1 / 1:2 half-time lock)")?;

    let kd = key_distance(&table_a.camelot, &table_b.camelot);
    let rec = recipe_of(&rule_plan.spec);
    let stretch_pct = rule_plan.fit.stretch_pct as f32;
    let base_ex = example_from_spec(
        &rule_plan.spec.id,
        &rule_plan.spec,
        &table_a,
        &table_b,
        rec,
        stretch_pct,
        kd,
        Label { good: 1, issues: vec![], source: "cli".into() },
    );

    let mut beat_ws = BeatWorkspace::default();
    let ai_p = sone.cortex.cortex(&mut beat_ws, &base_ex);
    let mut r_idx: Vec<usize> = (0..4).collect();
    r_idx.sort_by(|&i, &j| ai_p.recipe_probs[j].total_cmp(&ai_p.recipe_probs[i]));

    let mut best_cand: Option<(f32, Spec, Recipe, f32)> = None;
    for &ri in r_idx.iter().take(2) {
        let cand_r = RECIPE_ORDER[ri];
        for sb_delta in [0.0f32, -2.0, 2.0] {
            let cand_start = (ai_p.start_bars + sb_delta).clamp(4.0, 16.0);
            let cand_spec = apply_cortex_lanes(&rule_plan.spec, &base_ex.b_beats, &ai_p.lanes, cand_r, cand_start, ai_p.swap_bars);
            let cand_ex = example_from_spec(
                &cand_spec.id,
                &cand_spec,
                &table_a,
                &table_b,
                cand_r,
                stretch_pct,
                kd,
                Label { good: 1, issues: vec![], source: "cli".into() },
            );
            let crit = sone.cingulate.cingulate(&mut beat_ws, &cand_ex);
            let worst_defect = crit.issues.iter().map(|&(_, p)| p).fold(0.0f32, f32::max);
            let stretch_pen = 0.008 * (stretch_pct.abs() - 2.5).max(0.0);
            let key_pen = if kd >= 2 { 0.03 } else { 0.0 };
            let rank = crit.good + 0.12 * ai_p.recipe_probs[ri] - 0.25 * worst_defect - stretch_pen - key_pen;
            if best_cand.as_ref().is_none_or(|(bs, _, _, _)| rank > *bs) {
                best_cand = Some((rank, cand_spec, cand_r, crit.good));
            }
        }
    }
    let (_, mut spec, chosen_recipe, critic_q) = best_cand.unwrap();

    let a_gain_db = (STATION_LUFS - mem_a.lufs()).min(0.0);
    let b_gain_db = (STATION_LUFS - mem_b.lufs()).min(0.0);
    let pair_lufs = mem_a.lufs().min(mem_b.lufs());
    spec.a.trim_db += a_gain_db - (pair_lufs - mem_a.lufs());
    spec.b.trim_db += b_gain_db - (pair_lufs - mem_b.lufs());

    let rendered = sone_render::render_with(&spec, &mem_a, &mem_b, &RenderOptions::default())?;
    let master = splice_masters(&raw_a, &raw_b, a_gain_db, b_gain_db, &rendered, &spec, window_s);

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    write_wav16(&out_path, &master)?;
    if let Some(sp) = spec_out {
        std::fs::write(sp, serde_json::to_string_pretty(&[&spec])?)?;
    }

    eprintln!(
        "Wrote {} ({:.1} s) in {:.2} s | {:?} | {:.1} -> {:.1} BPM ({:+.1}%) | {} -> {} | drift {:+.1} ms | Cingulate q={:.3}",
        out_path.display(),
        master.len() as f64 / 44_100.0,
        t0.elapsed().as_secs_f64(),
        chosen_recipe,
        table_a.bpm,
        table_b.bpm,
        stretch_pct,
        table_a.camelot,
        table_b.camelot,
        rendered.report.drift_ms(),
        critic_q,
    );
    Ok(())
}

/// `sone stems <song.wav|flac> --model <model.safetensors> --out <dir>`
pub fn stems(args: Args) -> Result<()> {
    let [song_arg] = args.words.as_slice() else {
        bail!("usage: sone stems <song.wav|flac> --model <model.safetensors> --out <dir>");
    };
    let model_path = args.path("model")?;
    let out_dir = args.path("out")?;
    std::fs::create_dir_all(&out_dir)?;

    let bytes = std::fs::read(&model_path).with_context(|| format!("reading {}", model_path.display()))?;
    let wf = WeightFile::from_vec(bytes)?;
    let cochlea = if wf.metadata.contains_key("organs") {
        Sone::load(wf)?.cochlea
    } else {
        Cochlea::load(wf)?
    };

    let song_path = Path::new(song_arg);
    let stereo = decode_audio_44k(song_path)?;
    let n = stereo.len();
    let dur_s = n as f64 / 44_100.0;
    let left: Vec<f32> = stereo.iter().map(|s| s[0]).collect();
    let right: Vec<f32> = stereo.iter().map(|s| s[1]).collect();

    let hop = cochlea.config.hop;
    let chunk = (8 * 44_100 / hop) * hop;
    let overlap = (44_100 / hop) * hop;
    let mut sep: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; 4];
    let mut ws = CochleaWorkspace::default();

    let t0 = Instant::now();
    cochlea.separate(&mut ws, &left, &right, chunk, overlap, &mut sep);
    let elapsed = t0.elapsed().as_secs_f64();

    let mut max_err = 0.0f32;
    for i in 0..n {
        let sl = sep[0][0][i] + sep[1][0][i] + sep[2][0][i] + sep[3][0][i];
        let sr = sep[0][1][i] + sep[1][1][i] + sep[2][1][i] + sep[3][1][i];
        max_err = max_err.max((sl - left[i]).abs()).max((sr - right[i]).abs());
    }
    let err_db = 20.0 * f64::from(max_err.max(1e-12)).log10();

    for (idx, name) in ["vocals", "drums", "bass", "other"].iter().enumerate() {
        let frames: Vec<[f32; 2]> = (0..n).map(|i| [sep[idx][0][i], sep[idx][1][i]]).collect();
        write_wav16(&out_dir.join(format!("{name}.wav")), &frames)?;
    }

    eprintln!(
        "Separated {:.1} s into {}/{{vocals,drums,bass,other}}.wav in {:.2} s ({:.2}x RT, mixture conservation {:.1} dB)",
        dur_s,
        out_dir.display(),
        elapsed,
        elapsed / dur_s.max(1e-6),
        err_db,
    );
    Ok(())
}
