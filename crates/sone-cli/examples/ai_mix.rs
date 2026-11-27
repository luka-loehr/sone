use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use sone_core::{BeatTable, Label, Points, Shape, Spec, Stem};
use sone_infer::{BeatNet, BeatWorkspace, WeightFile};
use sone_plan::keys::distance as key_distance;
use sone_plan::PlanOptions;
use sone_render::{MemoryStems, StemSource, files};
use sone_sim::build::{A_AFTER, A_BEFORE, example_from_spec, recipe_of};

const STATION_LUFS: f64 = -14.0;

fn load_stems(stems_dir: &Path, table: &BeatTable) -> Result<MemoryStems> {
    let dir = stems_dir.join(&table.id);
    let rd = |name: &str| files::read_flac(&dir.join(format!("{name}.flac")));
    let (v, d, b, o) = (rd("vocals")?, rd("drums")?, rd("bass")?, rd("other")?);
    let n = v.len().min(d.len()).min(b.len()).min(o.len());
    let mix: Vec<[f32; 2]> = (0..n).map(|i| std::array::from_fn(|c| ((v[i][c] + d[i][c]) + b[i][c]) + o[i][c])).collect();
    let cut = |mut x: Vec<[f32; 2]>| {
        x.resize(n, [0.0; 2]);
        x
    };
    let mut stems = [cut(v), cut(d), cut(b), cut(o)];
    let clean_vocals = dir.join("vocals_r.flac").exists();
    let mut leak_free = false;
    if clean_vocals {
        stems[0] = cut(rd("vocals_r")?);
        leak_free = dir.join("drums_i.flac").exists() && dir.join("bass_i.flac").exists();
        if leak_free {
            stems[1] = cut(rd("drums_i")?);
            stems[2] = cut(rd("bass_i")?);
        }
        stems[3] = (0..n).map(|i| std::array::from_fn(|c| ((mix[i][c] - stems[0][i][c]) - stems[1][i][c]) - stems[2][i][c])).collect();
    }
    let lufs = MemoryStems::measure_lufs(&stems);
    Ok(MemoryStems {
        id: table.id.clone(),
        stems,
        grid: sone_render::BeatGrid::from_table(table),
        lufs,
        clean_vocals,
        leak_free,
    })
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

fn apply_ai_lanes(
    base: &Spec,
    b_beats: &[Vec<f32>],
    ai_lanes: &[[f32; 8]],
    recipe: sone_core::Recipe,
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
    let handoff_u = if recipe == sone_core::Recipe::VoiceOverBeat {
        sustained_b_vocal_u(b_beats, b_earliest).clamp(b_earliest, -1.0)
    } else {
        b_earliest
    };
    spec.b.from = Some(handoff_u);
    spec.a.shape = Shape::Lin;
    spec.b.shape = Shape::Lin;
    spec.a.shapes.clear();
    spec.b.shapes.clear();

    // Align vocal exit + echo send with actual handoff_u if B's singing starts >1.25 bars after base echo
    let mut echo_cut_u: Option<f64> = base.a.sends.iter().find_map(|s| {
        if s.stem == Stem::Vocals {
            s.send.last().map(|pt| pt[0])
        } else {
            None
        }
    });
    if recipe == sone_core::Recipe::VoiceOverBeat {
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

            if recipe == sone_core::Recipe::VoiceOverBeat {
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exports = Path::new(args.first().map(String::as_str).unwrap_or("exports"));
    let stems_dir = Path::new(args.get(1).map(String::as_str).unwrap_or("data/stems"));
    let beats_dir = Path::new(args.get(2).map(String::as_str).unwrap_or("data/analysis/beats"));
    let out_dir = PathBuf::from(args.get(3).map(String::as_str).unwrap_or("out/ai_mixes"));
    let max_mixes: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    std::fs::create_dir_all(&out_dir)?;

    let plan_file = if exports.join("plan-v2-best-f16.safetensors").exists() {
        exports.join("plan-v2-best-f16.safetensors")
    } else {
        exports.join("plan-best-f16.safetensors")
    };
    let crit_file = if exports.join("critic-v2-best-f16.safetensors").exists() {
        exports.join("critic-v2-best-f16.safetensors")
    } else {
        exports.join("critic-best-f16.safetensors")
    };
    println!("Loading Plan={} Critic={}", plan_file.display(), crit_file.display());
    let plan_net = BeatNet::load(WeightFile::from_vec(std::fs::read(plan_file)?)?)?;
    let crit_net = BeatNet::load(WeightFile::from_vec(std::fs::read(crit_file)?)?)?;
    let mut ws = BeatWorkspace::default();

    // Load available song beat tables
    let mut songs: Vec<BeatTable> = Vec::new();
    for entry in std::fs::read_dir(stems_dir)? {
        let p = entry?.path();
        if !p.is_dir() {
            continue;
        }
        let id = p.file_name().unwrap().to_string_lossy().to_string();
        if p.join("vocals.flac").exists() && beats_dir.join(format!("{id}.json")).exists() {
            songs.push(BeatTable::load(&beats_dir.join(format!("{id}.json")))?);
        }
    }
    songs.sort_by(|a, b| a.id.cmp(&b.id));
    println!("Loaded {} songs with stems + beat tables", songs.len());

    // Score all ordered pairs using our neural Plan + Critic
    struct Cand {
        a_idx: usize,
        b_idx: usize,
        ai_spec: Spec,
        rule_recipe: sone_core::Recipe,
        ai_recipe: sone_core::Recipe,
        ai_recipe_prob: f32,
        critic_good: f32,
        top_issues: Vec<(String, f32)>,
        plan_ms: f64,
        critic_ms: f64,
    }

    let mut cands: Vec<Cand> = Vec::new();
    for i in 0..songs.len() {
        for j in 0..songs.len() {
            if i == j {
                continue;
            }
            let (ta, tb) = (&songs[i], &songs[j]);
            let opt = PlanOptions {
                id: format!("ai_{}_{}", &ta.id[..6], &tb.id[..6]),
                ..PlanOptions::default()
            };
            let Ok(rule_plan) = sone_plan::plan(ta, tb, &opt) else {
                continue;
            };
            if rule_plan.spec.no_sync {
                continue;
            }
            let kd = key_distance(&ta.camelot, &tb.camelot);
            let rec = recipe_of(&rule_plan.spec);
            let stretch_pct = ((ta.bpm - tb.bpm) / tb.bpm.max(1.0) * 100.0) as f32;
            let base_ex = example_from_spec(
                &rule_plan.spec.id,
                &rule_plan.spec,
                ta,
                tb,
                rec,
                stretch_pct,
                kd,
                Label { good: 1, issues: vec![], source: "ai".into() },
            );

            let t0 = Instant::now();
            let ai_p = plan_net.cortex(&mut ws, &base_ex);
            let plan_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let ai_spec = apply_ai_lanes(&rule_plan.spec, &base_ex.b_beats, &ai_p.lanes, ai_p.recipe, ai_p.start_bars, ai_p.swap_bars);
            let ai_ex = example_from_spec(
                &ai_spec.id,
                &ai_spec,
                ta,
                tb,
                ai_p.recipe,
                stretch_pct,
                kd,
                Label { good: 1, issues: vec![], source: "ai".into() },
            );

            let t1 = Instant::now();
            let crit = crit_net.cingulate(&mut ws, &ai_ex);
            let critic_ms = t1.elapsed().as_secs_f64() * 1000.0;

            let mut top_issues: Vec<(String, f32)> = crit
                .issues
                .iter()
                .filter(|(_, p)| *p >= 0.15)
                .map(|(iss, p)| (format!("{iss:?}"), *p))
                .collect();
            top_issues.sort_by(|a, b| b.1.total_cmp(&a.1));
            top_issues.truncate(3);

            cands.push(Cand {
                a_idx: i,
                b_idx: j,
                ai_spec,
                rule_recipe: rec,
                ai_recipe: ai_p.recipe,
                ai_recipe_prob: ai_p.recipe_probs.iter().cloned().fold(0.0f32, f32::max),
                critic_good: crit.good,
                top_issues,
                plan_ms,
                critic_ms,
            });
        }
    }

    cands.sort_by(|a, b| b.critic_good.total_cmp(&a.critic_good));
    println!("Evaluated {} syncable pairs with neural Plan+Critic", cands.len());

    // Render top distinct-song mixes that pass drift/lag checks
    let mut used_a: HashSet<usize> = HashSet::new();
    let mut used_b: HashSet<usize> = HashSet::new();
    let mut rendered_count = 0usize;
    let mut summaries: Vec<serde_json::Value> = Vec::new();

    for cand in &cands {
        if rendered_count >= max_mixes {
            break;
        }
        if used_a.contains(&cand.a_idx) || used_b.contains(&cand.b_idx) {
            continue;
        }
        let ta = &songs[cand.a_idx];
        let tb = &songs[cand.b_idx];
        let sa = load_stems(stems_dir, ta)?;
        let sb = load_stems(stems_dir, tb)?;

        let mut spec = cand.ai_spec.clone();
        spec.id = format!("ai_mix{:02}_{}_{}", rendered_count + 1, &ta.id[..6], &tb.id[..6]);
        let pair = sa.lufs().min(sb.lufs());
        let ga = (STATION_LUFS - sa.lufs()).min(0.0);
        let gb = (STATION_LUFS - sb.lufs()).min(0.0);
        spec.a.trim_db += ga - (pair - sa.lufs());
        spec.b.trim_db += gb - (pair - sb.lufs());

        let t_ren = Instant::now();
        let rendered = match sone_render::render_with(&spec, &sa, &sb, &sone_render::RenderOptions::default()) {
            Ok(r) => r,
            Err(e) => {
                println!("Render err {} ({} -> {}): {e:#}", spec.id, ta.id, tb.id);
                continue;
            }
        };
        let render_s = t_ren.elapsed().as_secs_f64();
        let drift = rendered.report.drift_ms();
        let lag = rendered.report.lag_ms();
        if drift.abs() > sone_render::MAX_DRIFT_MS || lag.abs() > sone_render::MAX_LAG_MS {
            println!("Skip {} (drift={:+.1}ms, lag={:+.1}ms)", spec.id, drift, lag);
            continue;
        }

        let wav_path = out_dir.join(format!("{}.wav", spec.id));
        files::write_wav16(&wav_path, &rendered.audio).with_context(|| format!("writing {}", wav_path.display()))?;
        std::fs::write(out_dir.join(format!("{}.spec.json", spec.id)), serde_json::to_string_pretty(&[&spec])?)?;

        println!(
            "RENDERED {} | {} ({:.1} BPM {}) -> {} ({:.1} BPM {}) | AI={:?} (p={:.2}, rule={:?}) | Critic={:.3} | drift={:+.1}ms lag={:+.1}ms | AI {:.1}+{:.1}ms, render {:.2}s",
            spec.id,
            ta.id,
            ta.bpm,
            ta.camelot,
            tb.id,
            tb.bpm,
            tb.camelot,
            cand.ai_recipe,
            cand.ai_recipe_prob,
            cand.rule_recipe,
            cand.critic_good,
            drift,
            lag,
            cand.plan_ms,
            cand.critic_ms,
            render_s
        );

        summaries.push(serde_json::json!({
            "id": spec.id,
            "a": ta.id,
            "b": tb.id,
            "bpm_a": ta.bpm,
            "bpm_b": tb.bpm,
            "camelot_a": ta.camelot,
            "camelot_b": tb.camelot,
            "ai_recipe": format!("{:?}", cand.ai_recipe),
            "ai_recipe_prob": cand.ai_recipe_prob,
            "rule_recipe": format!("{:?}", cand.rule_recipe),
            "critic_good": cand.critic_good,
            "top_issues": cand.top_issues,
            "drift_ms": drift,
            "lag_ms": lag,
            "stretch_pct": rendered.report.stretch_pct,
            "plan_ms": cand.plan_ms,
            "critic_ms": cand.critic_ms,
            "render_s": render_s,
            "duration_s": rendered.audio.len() as f64 / sone_dsp::SR,
        }));

        used_a.insert(cand.a_idx);
        used_b.insert(cand.b_idx);
        rendered_count += 1;
    }

    std::fs::write(out_dir.join("ai_summary.json"), serde_json::to_string_pretty(&summaries)?)?;
    println!("Wrote {} AI-planned mixes to {}", rendered_count, out_dir.display());
    Ok(())
}
