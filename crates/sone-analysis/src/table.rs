//! The song at beat resolution ([`BeatTable`]): what the planner and the
//! training data generator read, so they need no audio.
//!
//! - Beat times lie on a fitted grid: every beat of bar `b` on the local
//!   grid of that bar (handmix.py's `grid(b)`: a line through the tracker's
//!   beats, its phase on the drum attacks), fitted within two bars either
//!   side of it. A bar in which a tracked beat lies more than 40 ms off that
//!   grid has no steady beat (an intro without drums, a break) and keeps the
//!   tracker's beats. Beats the tracker
//!   missed are filled in by the grid, so the table has every beat from the
//!   first to the last.
//! - The bar lines are the analysis' (`TrackAnalysis`), moved by the
//!   bar-phase check ([`crate::phase`]): bar `b` of the table is bar `b` of
//!   the `TrackAnalysis` (never renumbered), moved by the shift of the
//!   section it lies in ([`BarPhase::sections`]: runs where the tracker's
//!   own downbeats and the check agree on another one), else by the song's
//!   `beat_shift`. The last analysis downbeat (the end of its last bar)
//!   moves with the last bar.
//! - What a reader may assume about bar lengths: bar `b` spans beats
//!   `bars[b]..bars[b + 1]` (the last one `bars[b]..` to the end), and
//!   `beat_in_bar` is 0 exactly on `bars[b]`. Most bars are 4 beats. Where a
//!   section begins or ends, the bar before its border is shorter or longer
//!   by the difference of the shifts (3 or 5 beats for −1, 2 or 6 for 2);
//!   where the tracker missed a downbeat, a bar spans 8 beats. In a bar of
//!   more than 4 beats `beat_in_bar` stays 3 on the extra beats; in a
//!   shorter one it simply ends early. Never assume `bars[b] = bars[0] +
//!   4 * b`; always index through `bars`. `bar_in_phrase8` counts `b`.
//! - With a shift of −1 the first bar may begin a beat before the
//!   tracker's first beat (its time is then extrapolated, possibly below 0).
//! - `bars` holds every downbeat of the analysis, including the one that
//!   ends its last bar.
//! - Per beat, the features of `sone_core::example::FEATURES`: levels in dB
//!   relative to the song's loud level (clamped at −60), `sung` (the beat's
//!   vocal level above the sung reference − 14 dB, as structure.py), and
//!   `line_end` (a sung line ends in this beat), position in the bar / 3,
//!   bar in the eight-bar phrase / 7, the novelty of structure.py on the
//!   first beat of each bar divided by 100 and capped at 1 (0 on the other
//!   beats; a clear section change scores 30–60), and the chroma of bass +
//!   other + vocals over the beat, normalised to unit sum (1/12 each when
//!   silent).

use sone_core::BeatTable;

use crate::Prepared;
use crate::grid::{Grid, grid};
use crate::key::Key;
use crate::measure::Lane;
use crate::phase::{Phase, Profile, decide};
use crate::stats::percentile;
use crate::voice;

/// Bars on either side of a bar that its grid in the table is fitted on.
pub const TABLE_SPAN_BARS: i64 = 2;

/// A bar keeps the tracker's beats when one of them is further than this
/// from the bar's grid.
pub const STEADY_S: f64 = 0.04;

/// Below this confidence a verdict other than 0 is not applied.
pub const APPLY_CONFIDENCE: f32 = 0.45;
/// A local section needs this many bars on which the tracker flagged the
/// same other downbeat.
const SECTION_BARS: usize = 4;

/// The bar-phase verdicts of a song.
#[derive(Debug, Clone)]
pub struct BarPhase {
    /// The song-level verdict (as the check found it).
    pub song: Phase,
    /// The shift applied to the whole song (the verdict, when confident enough).
    pub shift: i8,
    /// Sections (first bar, last bar, verdict) where the tracker's own
    /// downbeats disagree with the song's bar phase and the check confirms
    /// them; the table moves their bars by the section's shift.
    pub sections: Vec<(usize, usize, Phase)>,
    /// Verdicts of the single eight-bar windows (first bar, verdict).
    pub windows: Vec<(usize, Phase)>,
}

impl BarPhase {
    /// Shift the check finds for bar `b` (its section's, else the song's).
    pub fn shift_of(&self, b: usize) -> i8 {
        self.sections.iter().find(|s| s.0 <= b && b <= s.1).map_or(self.shift, |s| s.2.shift)
    }
}

fn add_profile(total: &mut Profile, p: &Profile) {
    for i in 0..4 {
        total.kick[i] += p.kick[i];
        total.clap[i] += p.clap[i];
        total.chords[i] += p.chords[i];
        total.n[i] += p.n[i];
        total.n_chords[i] += p.n_chords[i];
    }
}

/// The profile over eight-bar windows (stepping by four bars) of bars
/// `from..to` in which the drums play, each window on its own local grid
/// (fitted on the window's bars when `local`, else within 16 bars as
/// phasecheck.py does).
fn windows_profile(p: &Prepared, from: usize, to: usize, local: bool) -> (Profile, Vec<(usize, Phase)>) {
    let s = &p.structure;
    let rows = &s.json.bars;
    let to = to.min(rows.len());
    let mut total = Profile::default();
    let mut windows = Vec::new();
    let mut w = from;
    while w < to {
        let end = (w + 8).min(to);
        let drums = rows[w..end].iter().filter(|r| r.drums > -20.0).count();
        if end - w >= 4 && drums * 4 >= (end - w) * 3 {
            let g = grid(s, &p.env, w, 16, 0, local.then_some((w, end)));
            let mut prof = Profile::default();
            prof.add(&p.m, &p.chroma_bo, &g, 0..4 * (end - w) as i64);
            windows.push((w, decide(&prof)));
            add_profile(&mut total, &prof);
        }
        if end == to {
            break;
        }
        w += 4;
    }
    (total, windows)
}

/// The bar-phase check of a whole song: every eight-bar window with drums
/// summed into one verdict; then the runs of at least four bars on which the
/// tracker flagged another beat as the one (bars without a single flag do
/// not break a run), each checked on its own bars: a run whose check
/// confirms the tracker's shift with confidence ≥ [`APPLY_CONFIDENCE`]
/// becomes a section with that shift.
pub fn song_phase(p: &Prepared) -> BarPhase {
    let s = &p.structure;
    let (total, windows) = windows_profile(p, 0, s.json.bars.len(), true);
    let song = decide(&total);
    let shift = if song.shift != 0 && song.confidence < APPLY_CONFIDENCE { 0 } else { song.shift };
    // the tracker's flagged position per bar (None: no flag or several)
    let flagged: Vec<Option<i64>> = s
        .down
        .windows(2)
        .map(|w| {
            let pos: Vec<i64> = (w[0]..w[1]).filter(|&i| s.flags[i]).map(|i| s.idx[i] - s.idx[w[0]]).collect();
            (pos.len() == 1).then(|| pos[0])
        })
        .collect();
    let mut sections = Vec::new();
    let mut b = 0;
    while b < flagged.len() {
        let Some(pos) = flagged[b].filter(|&q| q != 0 && q < 4) else {
            b += 1;
            continue;
        };
        let (first, mut last, mut votes) = (b, b, 1);
        let mut j = b + 1;
        while j < flagged.len() {
            match flagged[j] {
                Some(q) if q == pos => {
                    last = j;
                    votes += 1;
                }
                None => {}
                Some(_) => break,
            }
            j += 1;
        }
        if votes >= SECTION_BARS {
            let (prof, _) = windows_profile(p, first, last + 1, true);
            let verdict = decide(&prof);
            let wanted = if pos == 3 { -1 } else { pos as i8 };
            if verdict.shift == wanted && verdict.confidence >= APPLY_CONFIDENCE && prof.n.iter().all(|&n| n > 0) {
                sections.push((first, last, verdict));
            }
        }
        b = last + 1;
    }
    BarPhase { song, shift, sections, windows }
}

/// Assemble the table, the analysis' bar lines moved by the phase check.
pub fn beat_table(p: &Prepared, phase: &BarPhase, key: &Key, line_ref_db: f64) -> BeatTable {
    let s = &p.structure;
    let idx = &s.idx;
    let down = &s.down;
    // table bar k is analysis bar k (never renumbered), moved by its
    // section's shift or else the song's
    let last_bar = down.len().saturating_sub(2);
    let bar_k: Vec<i64> = down.iter().enumerate().map(|(b, &i)| idx[i] + i64::from(phase.shift_of(b.min(last_bar)))).collect();
    let k_first = idx[0].min(bar_k[0]);
    let k_last = (*idx.last().expect("beats")).max(*bar_k.last().expect("bars"));
    // the local grid of every analysis bar, anchored on its own downbeat and
    // fitted within two bars: wider fits break where the tempo changes (or
    // the tracker switches to double time) inside their span
    let grids: Vec<Grid> = (0..down.len()).map(|b| grid(s, &p.env, b, TABLE_SPAN_BARS, 0, None)).collect();
    let old_down_k: Vec<i64> = down.iter().map(|&i| idx[i]).collect();
    // the tracker's own beat per beat number
    let mut tracked: Vec<Option<f64>> = vec![None; (k_last - k_first + 1) as usize];
    for (i, &k) in idx.iter().enumerate() {
        tracked[(k - k_first) as usize] = Some(s.json.beats[i]);
    }
    let bar_of = |k: i64| match old_down_k.binary_search(&k) {
        Ok(b) => b,
        Err(0) => 0,
        Err(b) => b - 1,
    };
    // a bar whose tracked beats stray more than 40 ms from its grid has no
    // steady beat (an intro without drums, a break): it keeps the tracker's
    // beats rather than a line that fits none of them
    let steady: Vec<bool> = (0..down.len())
        .map(|b| {
            (k_first..=k_last)
                .filter(|&k| bar_of(k) == b)
                .all(|k| tracked[(k - k_first) as usize].is_none_or(|t| (t - grids[b].at((k - old_down_k[b]) as f64)).abs() <= STEADY_S))
        })
        .collect();
    let mut times: Vec<Option<f64>> = (k_first..=k_last)
        .map(|k| {
            let b = bar_of(k);
            if steady[b] { Some(grids[b].at((k - old_down_k[b]) as f64)) } else { tracked[(k - k_first) as usize] }
        })
        .collect();
    // beats the tracker missed in an unsteady bar: between their neighbours
    // (or on the bar's grid at the ends)
    for i in 0..times.len() {
        if times[i].is_none() {
            let k = k_first + i as i64;
            let prev = (0..i).rev().find_map(|j| times[j].map(|t| (j, t)));
            let next = (i + 1..times.len()).find_map(|j| times[j].map(|t| (j, t)));
            times[i] = Some(match (prev, next) {
                (Some((a, ta)), Some((b, tb))) => ta + (tb - ta) * (i - a) as f64 / (b - a) as f64,
                _ => grids[bar_of(k)].at((k - old_down_k[bar_of(k)]) as f64),
            });
        }
    }
    let times: Vec<f64> = times.into_iter().map(|t| t.expect("filled")).collect();
    let n = times.len();
    let bars: Vec<usize> = bar_k.iter().map(|&k| (k - k_first) as usize).collect();
    // beats before the first bar count back from it; a longer bar (at a
    // section's border, or where the tracker missed a downbeat) keeps 3 on
    // its extra beats
    let beat_in_bar: Vec<u8> = (0..n)
        .map(|i| match bars.binary_search(&i) {
            Ok(_) => 0,
            Err(0) => (i as i64 - bars[0] as i64).rem_euclid(4) as u8,
            Err(b) => (i - bars[b - 1]).min(3) as u8,
        })
        .collect();
    let end_of = |i: usize| if i + 1 < n { times[i + 1] } else { times[i] + (times[i] - times[i.max(1) - 1]).max(0.2) };
    let loud = s.loud;
    let v16 = voice::sixteenths(&p.m, &times, line_ref_db);
    let (line_ends, line_starts) = voice::lines(&v16);
    let mut ends_in_beat = vec![false; n];
    for &e in &line_ends {
        ends_in_beat[e / 4] = true;
    }
    let lanes = [Lane::Vocals, Lane::Drums, Lane::Kick, Lane::Hats, Lane::Bass, Lane::Other, Lane::Mix];
    let nov = &s.json.novelty;
    let phrase8 = i64::from(s.json.phrase8);
    let mut features = Vec::with_capacity(n);
    for i in 0..n {
        let (t0, t1) = (times[i], end_of(i));
        let mut f: Vec<f32> = lanes.iter().map(|&l| ((p.m.level_db(l, t0, t1) - loud).max(-60.0)) as f32).collect();
        let vocal = p.m.level_db(Lane::Vocals, t0, t1) - loud;
        f.push(f32::from(u8::from(vocal > s.vref - 14.0)));
        f.push(f32::from(u8::from(ends_in_beat[i])));
        f.push(f32::from(beat_in_bar[i]) / 3.0);
        let bar = match bars.binary_search(&i) {
            Ok(b) => b as i64,
            Err(b) => b as i64 - 1,
        };
        f.push((bar - phrase8).rem_euclid(8) as f32 / 7.0);
        let on_bar_line = bars.binary_search(&i).is_ok();
        f.push(if on_bar_line && bar >= 0 && (bar as usize) < nov.len() { (nov[bar as usize] / 100.0).min(1.0) } else { 0.0 });
        let c = p.chroma_bov.energy(t0, t1);
        let sum: f64 = c.iter().sum();
        f.extend(c.iter().map(|&v| if sum > 1e-9 { (v / sum) as f32 } else { 1.0 / 12.0 }));
        features.push(f);
    }
    BeatTable {
        id: s.json.id.clone(),
        duration: s.json.duration,
        bpm: s.json.bpm,
        beat_s: s.beat_s,
        times,
        beat_in_bar,
        beat_shift: phase.shift,
        phase_confidence: if phase.shift == phase.song.shift { phase.song.confidence } else { 0.25 },
        bars,
        phrase8: s.json.phrase8,
        key: key.key.clone(),
        camelot: key.camelot.clone(),
        camelot_alt: key.alternative().unwrap_or_default(),
        tuning_cents: key.tuning_cents,
        loud_db: s.json.loud_db,
        features,
        v16,
        line_ends,
        line_starts,
    }
}

/// The sung reference of the vocal maps (`vmap.py`): the 85th percentile of
/// the bars' vocal levels, dB full scale.
pub fn sung_level(p: &Prepared) -> f64 {
    let bars = &p.structure.bar_times;
    let db: Vec<f64> = bars.windows(2).map(|w| p.m.seg_db(Lane::Vocals, w[0], w[1])).collect();
    percentile(&db, 85.0)
}
