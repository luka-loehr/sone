//! The per-bar table of `reference/python/structure.py`: tempo fit, bar
//! phase from the tracker's downbeats, per-bar levels of every stem, the
//! vocal activity per beat and the eight-bar phrase phase from the novelty
//! at bar lines. Same rules, same rounding, so its JSON equals the Python
//! one's (levels within the 0.1 dB of the rounding).

use sone_core::analysis::{BarRow, GridResidual, TrackAnalysis};

use crate::measure::{HOP, Lane, Measurements, SR};
use crate::stats::{lstsq_line, median, percentile, round_to};

/// Beats as the tracker gives them.
#[derive(Debug, Clone, Default)]
pub struct Beats {
    pub times: Vec<f64>,
    pub downbeat: Vec<bool>,
}

impl Beats {
    pub fn from_flagged(v: &[(f64, bool)]) -> Self {
        Self { times: v.iter().map(|b| b.0).collect(), downbeat: v.iter().map(|b| b.1).collect() }
    }

    /// The Python reference's `.beats` text: "seconds [flag]" per line.
    pub fn parse(text: &str) -> Self {
        let mut b = Self::default();
        for line in text.lines() {
            let mut p = line.split_whitespace();
            let Some(t) = p.next().and_then(|v| v.parse::<f64>().ok()) else { continue };
            b.times.push(t);
            b.downbeat.push(p.next().and_then(|v| v.parse::<i64>().ok()).unwrap_or(0) == 1);
        }
        b
    }
}

/// The analysis with the unrounded values later stages need.
#[derive(Debug, Clone)]
pub struct Structure {
    pub json: TrackAnalysis,
    /// Downbeat times (every bar's start and the end of the last bar).
    pub bar_times: Vec<f64>,
    /// Index into the beats of each bar's downbeat.
    pub down: Vec<usize>,
    /// Beat numbers (a missed beat leaves a gap).
    pub idx: Vec<i64>,
    /// Loud level of the song, dB (unrounded).
    pub loud: f64,
    /// The 85th percentile of the bars' vocal levels relative to `loud`
    /// (the sung reference of `sung`).
    pub vref: f64,
    /// Beat period of the steady part.
    pub beat_s: f64,
    /// The tracker's downbeat flag per beat.
    pub flags: Vec<bool>,
}

/// Beat numbers, the tracker's bar phase, its downbeat votes and the beats
/// that start bars: what follows from the beats alone.
pub struct Counting {
    pub idx: Vec<i64>,
    pub bar_phase: usize,
    pub votes: [u32; 4],
    pub down: Vec<usize>,
}

pub fn count(beats: &Beats) -> Counting {
    let b = &beats.times;
    let iv: Vec<f64> = b.windows(2).map(|w| w[1] - w[0]).collect();
    let t = median(&iv);
    let mut idx = vec![0i64];
    for &d in &iv {
        let step = (d / t).round_ties_even().max(1.0) as i64;
        idx.push(idx.last().unwrap() + step);
    }
    let mut votes = [0u32; 4];
    for (i, &d) in beats.downbeat.iter().enumerate() {
        if d {
            votes[idx[i].rem_euclid(4) as usize] += 1;
        }
    }
    let bar_phase = argmax(&votes.map(f64::from));
    let down: Vec<usize> = (0..b.len()).filter(|&i| idx[i].rem_euclid(4) == bar_phase as i64).collect();
    Counting { idx, bar_phase, votes, down }
}

/// Sample indices where the per-bar and per-beat segments of [`analyse`]
/// start and end, for [`crate::measure::Measurer::with_cuts`].
pub fn cut_points(beats: &Beats) -> Vec<usize> {
    let c = count(beats);
    let mut cuts = Vec::new();
    for w in c.down.windows(2) {
        let (t0, t1) = (beats.times[w[0]], beats.times[w[1]]);
        for j in 0..5 {
            let t = t0 + (t1 - t0) * j as f64 / 4.0;
            cuts.push((t.max(0.0) * SR) as usize);
        }
    }
    cuts
}

pub fn analyse(id: &str, beats: &Beats, m: &Measurements) -> anyhow::Result<Structure> {
    let b = &beats.times;
    anyhow::ensure!(b.len() >= 40, "{id}: only {} beats", b.len());
    let iv: Vec<f64> = b.windows(2).map(|w| w[1] - w[0]).collect();
    let t = median(&iv);
    let Counting { idx, bar_phase, votes, down } = count(beats);
    // the longest run of regular intervals is the song's steady part
    let (mut best, mut cur, mut start) = ((0usize, 0usize), 0usize, 0usize);
    for (i, &d) in iv.iter().enumerate() {
        if (d - t).abs() < 0.035 {
            if cur == 0 {
                start = i;
            }
            cur += 1;
            if cur > best.1 - best.0 {
                best = (start, i + 1);
            }
        } else {
            cur = 0;
        }
    }
    let (s0, s1) = best;
    let xs: Vec<f64> = (0..=s1 - s0).map(|k| k as f64).collect();
    let (tfit, t0fit) = lstsq_line(&xs, &b[s0..=s1]);
    let resid: Vec<f64> = xs.iter().zip(&b[s0..=s1]).map(|(&k, &y)| y - (tfit * k + t0fit)).collect();
    let rms = (resid.iter().map(|r| r * r).sum::<f64>() / resid.len() as f64).sqrt();
    let rmax = resid.iter().fold(0.0f64, |a, r| a.max(r.abs()));
    let x32: Vec<f64> = (0..32).map(f64::from).collect();
    let mut local = Vec::new();
    let mut i = s0;
    while (i as i64) < s1 as i64 - 32 {
        let (slope, _) = lstsq_line(&x32, &b[i..i + 32]);
        local.push(round_to(60.0 / slope, 2) as f32);
        i += 16;
    }
    let delta = drum_delta(b, m);
    let bars: Vec<f64> = down.iter().map(|&i| b[i]).collect();
    anyhow::ensure!(bars.len() >= 8, "{id}: only {} bars", bars.len());
    let mix_db: Vec<f64> = bars.windows(2).map(|w| m.seg_db(Lane::Mix, w[0], w[1])).collect();
    let loud = percentile(&mix_db, 90.0);
    let lanes = [Lane::Mix, Lane::Vocals, Lane::Drums, Lane::Kick, Lane::Hats, Lane::Bass, Lane::Other];
    let mut rows = Vec::with_capacity(bars.len() - 1);
    for (i, w) in bars.windows(2).enumerate() {
        let (t0, t1) = (w[0], w[1]);
        let l: Vec<f32> = lanes.iter().map(|&lane| round_to(m.seg_db(lane, t0, t1) - loud, 1) as f32).collect();
        let bt: Vec<f64> = (0..5).map(|j| t0 + (t1 - t0) * j as f64 / 4.0).collect();
        let mut vb = [0.0f32; 4];
        for j in 0..4 {
            vb[j] = round_to(m.seg_db(Lane::Vocals, bt[j], bt[j + 1]) - loud, 1) as f32;
        }
        rows.push(BarRow {
            bar: i,
            t: round_to(t0, 3),
            len: round_to(t1 - t0, 4),
            mix: l[0],
            vocals: l[1],
            drums: l[2],
            kick: l[3],
            hats: l[4],
            bass: l[5],
            other: l[6],
            vbeats: vb,
            sung: [0; 4],
        });
    }
    let vlev: Vec<f64> = rows.iter().map(|r| f64::from(r.vocals)).collect();
    let vref = percentile(&vlev, 85.0);
    for r in &mut rows {
        for j in 0..4 {
            r.sung[j] = u8::from(f64::from(r.vbeats[j]) > vref - 14.0);
        }
    }
    let nov = novelty(&rows);
    let ph8: Vec<f64> = (0..8).map(|p| nov.iter().skip(p).step_by(8).sum()).collect();
    let ph4: Vec<f64> = (0..4).map(|p| nov.iter().skip(p).step_by(4).sum()).collect();
    let json = TrackAnalysis {
        id: id.to_string(),
        duration: round_to(m.len as f64 / SR, 2),
        bpm: round_to(60.0 / tfit, 3),
        beat_s: tfit,
        t0: t0fit,
        grid_resid_ms: GridResidual { rms: round_to(rms * 1000.0, 1) as f32, max: round_to(rmax * 1000.0, 1) as f32 },
        steady: vec![b[s0], b[s1]],
        local_bpm: local,
        drum_delta_ms: round_to(delta * 1000.0, 1) as f32,
        bar_phase: bar_phase as u8,
        downbeat_votes: votes.to_vec(),
        phrase8: argmax(&ph8) as u8,
        phrase8_scores: ph8.iter().map(|&v| round_to(v, 1) as f32).collect(),
        phrase4: argmax(&ph4) as u8,
        loud_db: round_to(loud, 2) as f32,
        bars: rows,
        novelty: nov.iter().map(|&v| round_to(v, 1) as f32).collect(),
        beats: b.iter().map(|&v| round_to(v, 4)).collect(),
        beat_idx: idx.clone(),
    };
    Ok(Structure { json, bar_times: bars, down, idx, loud, vref, beat_s: tfit, flags: beats.downbeat.clone() })
}

/// Change of the (rounded, −40-clamped) stem levels across each bar line:
/// the mean of the two bars after it against the two before.
pub fn novelty(rows: &[BarRow]) -> Vec<f64> {
    let feat: Vec<[f64; 6]> = rows
        .iter()
        .map(|r| [r.vocals, r.drums, r.kick, r.bass, r.other, r.hats].map(|v| f64::from(v).max(-40.0)))
        .collect();
    let mut nov = vec![0.0; rows.len()];
    for i in 2..rows.len().saturating_sub(2) {
        nov[i] = (0..6).map(|j| ((feat[i][j] + feat[i + 1][j]) / 2.0 - (feat[i - 2][j] + feat[i - 1][j]) / 2.0).abs()).sum();
    }
    nov
}

/// structure.py's shift of all beats (±50 ms) that best meets the drum
/// stem's onsets (log-energy rises at 128-sample hops).
fn drum_delta(beats: &[f64], m: &Measurements) -> f64 {
    const H: usize = 128;
    let per = H / HOP;
    let c = &m.cum[Lane::Drums as usize];
    let n = m.len / H;
    let e: Vec<f64> = (0..n).map(|i| (c[(i + 1) * per] - c[i * per]) / H as f64).collect();
    // np.convolve(e, ones(4) / 4, "same"): e[i-2..=i+1]
    let smooth: Vec<f64> = (0..n).map(|i| (i as i64 - 2..=i as i64 + 1).filter(|&j| j >= 0 && (j as usize) < n).map(|j| e[j as usize]).sum::<f64>() / 4.0).collect();
    let l: Vec<f64> = smooth.iter().map(|v| (v + 1e-7).log10()).collect();
    let env: Vec<f64> = (0..n).map(|i| if i == 0 { 0.0 } else { (l[i] - l[i - 1]).max(0.0) }).collect();
    let mut best = (f64::NEG_INFINITY, 0.0);
    for k in 0..=100 {
        let s = -0.05 + k as f64 * 0.001;
        let score: f64 = beats
            .iter()
            .filter_map(|&t| {
                let i = ((t + s) * SR / H as f64).round_ties_even() as i64;
                (i >= 0 && (i as usize) < n).then(|| env[i as usize])
            })
            .sum();
        if score > best.0 {
            best = (score, s);
        }
    }
    best.1
}

/// Index of the first maximum (numpy's argmax).
pub fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
