//! Transitions broken on purpose. Every function takes a good example and
//! puts one fault into it that the owner named, or returns `None` when the
//! example offers no place for that fault.
//!
//! Lanes: 0-3 are song A's voice, drums, bass, melody; 4-7 song B's.

use sone_core::{Example, Issue, Recipe};

use crate::{build::A_BEFORE, rng::Rng};

const A_VOICE: usize = 0;
const A_DRUMS: usize = 1;
const A_BASS: usize = 2;
const A_OTHER: usize = 3;
const B_VOICE: usize = 4;
const B_DRUMS: usize = 5;
const B_BASS: usize = 6;
const B_OTHER: usize = 7;
const F_BASS: usize = 4;
const F_OTHER: usize = 5;
const F_SUNG: usize = 7;
const F_LINE_END: usize = 8;

/// Index of the ruler beat that starts on the one.
const ONE: usize = A_BEFORE as usize;

pub type Broken = (Example, Vec<Issue>);

pub const ALL: [(&str, fn(&Example, &mut Rng) -> Option<Broken>); 15] = [
    ("off_by_one", off_by_one),
    ("offset", offset),
    ("curve", curve),
    ("short_fade", short_fade),
    ("beat_hole", beat_hole),
    ("tail", tail),
    ("two_voices", two_voices),
    ("two_melodies", two_melodies),
    ("melody_at_once", melody_at_once),
    ("two_basses", two_basses),
    ("separately", separately),
    ("cut_in_word", cut_in_word),
    ("plain_cut", plain_cut),
    ("leak", leak),
    ("pitch_ride", pitch_ride),
];

fn first_tick(e: &Example) -> usize {
    ONE.saturating_sub((e.start_bars * 4.0).round() as usize)
}

fn b_sung(e: &Example, tick: usize) -> bool {
    e.b_beats.get(tick).is_some_and(|f| f[F_SUNG] > 0.5)
}

fn a_sung(e: &Example, tick: usize) -> bool {
    e.a_beats.get(tick).is_some_and(|f| f[F_SUNG] > 0.5)
}

/// Song B sits one of its beats late: kicks still meet, claps fight.
fn off_by_one(e: &Example, rng: &mut Rng) -> Option<Broken> {
    let ticks_per_beat = 1.0 / e.beat_ratio;
    if (ticks_per_beat - ticks_per_beat.round()).abs() > 1e-3 || !(1.0..=2.0).contains(&ticks_per_beat) {
        return None;
    }
    let k = ticks_per_beat.round() as usize;
    let late = rng.chance(0.5);
    let mut x = e.clone();
    let n = x.b_beats.len();
    x.b_beats = (0..n)
        .map(|i| {
            let from = if late { i.checked_sub(k) } else { Some(i + k) };
            from.and_then(|j| e.b_beats.get(j)).cloned().unwrap_or_else(|| e.b_beats[i].clone())
        })
        .collect();
    Some((x, vec![Issue::DownbeatOffByOne]))
}

fn offset(e: &Example, rng: &mut Rng) -> Option<Broken> {
    if e.recipe == Recipe::Gap {
        return None;
    }
    let mut x = e.clone();
    x.measures.beat_offset_ms = rng.range(12.0, 40.0) as f32 * if rng.chance(0.5) { 1.0 } else { -1.0 };
    Some((x, vec![Issue::BeatsOffset]))
}

/// A's fades as curves in dB that begin on a bar line: their start is heard.
fn curve(e: &Example, _: &mut Rng) -> Option<Broken> {
    let mut x = e.clone();
    let mut changed = false;
    for lane in [A_DRUMS, A_BASS, A_OTHER] {
        let full_until = (0..ONE).rev().find(|&i| e.lanes[i][lane] >= 0.97)?;
        let gone_from = (full_until..e.lanes.len()).find(|&i| e.lanes[i][lane] <= 0.02).unwrap_or(ONE);
        if gone_from < full_until + 8 {
            continue;
        }
        // Start on the bar line at or before the fade's start.
        let start = full_until / 4 * 4;
        for i in start..gone_from {
            let w = (i - start) as f32 / (gone_from - start) as f32;
            x.lanes[i][lane] = 10f32.powf(-24.0 * w / 20.0);
        }
        changed = true;
    }
    changed.then_some((x, vec![Issue::FadeCurve, Issue::AudibleFadeOnset]))
}

/// A's drums and bass are gone within a beat or so, somewhere inside the blend.
fn short_fade(e: &Example, rng: &mut Rng) -> Option<Broken> {
    let first = first_tick(e);
    if ONE < first + 12 {
        return None;
    }
    let at = first + 2 + rng.below(ONE - first - 10);
    let len = 1 + rng.below(3);
    let mut x = e.clone();
    for lane in [A_DRUMS, A_BASS] {
        for i in at..x.lanes.len() {
            let w = ((i - at) as f32 / len as f32).min(1.0);
            x.lanes[i][lane] = e.lanes[at][lane].max(0.8) * (1.0 - w);
        }
    }
    let mut issues = vec![Issue::FadeTooShort, Issue::FadeAtRandomPoint, Issue::AbruptStemExit];
    if (at..ONE).all(|i| x.lanes[i][B_DRUMS] < 0.3) {
        issues.extend([Issue::ABeatGoneBeforeOne, Issue::BeatHoleBeforeOne]);
    }
    Some((x, issues))
}

/// A's beat fades out early while B's beat only starts on the one.
fn beat_hole(e: &Example, rng: &mut Rng) -> Option<Broken> {
    if !(ONE - 8..ONE).all(|i| e.lanes[i][B_DRUMS] < 0.2) {
        return None;
    }
    let bars_before = 3 + rng.below(3);
    let from = ONE - bars_before * 4;
    let gone = ONE - 4 - rng.below(2) * 4;
    let mut x = e.clone();
    for lane in [A_DRUMS, A_BASS] {
        for i in from..ONE {
            let w = ((i - from) as f32 / (gone - from).max(1) as f32).min(1.0);
            x.lanes[i][lane] = e.lanes[from][lane] * (1.0 - w);
        }
    }
    Some((x, vec![Issue::ABeatGoneBeforeOne, Issue::BeatHoleBeforeOne]))
}

/// A's beat goes on after the one.
fn tail(e: &Example, rng: &mut Rng) -> Option<Broken> {
    if e.recipe == Recipe::Gap {
        return None;
    }
    let len = 4 + rng.below(5);
    let level = rng.range(0.6, 1.0) as f32;
    let mut x = e.clone();
    for lane in [A_DRUMS, A_BASS] {
        for i in ONE.saturating_sub(4)..ONE {
            x.lanes[i][lane] = x.lanes[i][lane].max(level);
        }
        for k in 0..len {
            if let Some(row) = x.lanes.get_mut(ONE + k) {
                row[lane] = level * (1.0 - k as f32 / len as f32);
            }
        }
    }
    Some((x, vec![Issue::TailAfterOne]))
}

fn two_voices(e: &Example, rng: &mut Rng) -> Option<Broken> {
    let first = first_tick(e);
    let both: Vec<usize> = (first..ONE + 8).filter(|&i| a_sung(e, i) && b_sung(e, i)).collect();
    if both.len() < 6 {
        return None;
    }
    let from = both[rng.below(both.len() / 2)];
    let to = (from + 8 + rng.below(12)).min(ONE + 8);
    let mut x = e.clone();
    for i in from..to {
        x.lanes[i][A_VOICE] = 1.0;
        x.lanes[i][B_VOICE] = 1.0;
    }
    Some((x, vec![Issue::TwoVoices]))
}

fn two_melodies(e: &Example, rng: &mut Rng) -> Option<Broken> {
    let from = ONE.checked_sub(12 + rng.below(3) * 4)?;
    let heard = |beats: &[Vec<f32>], i: usize| beats.get(i).is_some_and(|f| f[F_OTHER] > -32.0);
    if (from..ONE).filter(|&i| heard(&e.a_beats, i) && heard(&e.b_beats, i)).count() < 8 {
        return None;
    }
    let mut x = e.clone();
    for i in from..ONE {
        x.lanes[i][A_OTHER] = x.lanes[i][A_OTHER].max(0.85);
        x.lanes[i][B_OTHER] = x.lanes[i][B_OTHER].max(0.85);
    }
    let mut issues = vec![Issue::TwoMelodies];
    if e.recipe == Recipe::VoiceOverBeat {
        issues.push(Issue::BMelodyBeforeOne);
    }
    Some((x, issues))
}

fn melody_at_once(e: &Example, _: &mut Rng) -> Option<Broken> {
    let first = first_tick(e);
    if ONE < first + 8 || !(first..ONE).all(|i| e.lanes[i][B_OTHER] < 0.5) {
        return None;
    }
    let mut x = e.clone();
    for i in first..ONE {
        x.lanes[i][B_OTHER] = if i == first { 0.5 } else { 1.0 };
    }
    Some((x, vec![Issue::BMelodyEntersAtOnce, Issue::BMelodyBeforeOne]))
}

fn two_basses(e: &Example, rng: &mut Rng) -> Option<Broken> {
    let from = ONE.checked_sub(8 + rng.below(3) * 4)?;
    let heard = |beats: &[Vec<f32>], i: usize| beats.get(i).is_some_and(|f| f[F_BASS] > -25.0);
    if (from..ONE).filter(|&i| heard(&e.a_beats, i) && heard(&e.b_beats, i)).count() < 6 {
        return None;
    }
    let mut x = e.clone();
    for i in from..ONE {
        x.lanes[i][A_BASS] = 1.0;
        x.lanes[i][B_BASS] = 1.0;
    }
    Some((x, vec![Issue::TwoBasses]))
}

/// A's drums and bass leave, its melody stays.
fn separately(e: &Example, rng: &mut Rng) -> Option<Broken> {
    let first = first_tick(e);
    if ONE < first + 16 {
        return None;
    }
    let at = first + 4 + rng.below((ONE - first - 12) / 4 + 1) * 4;
    let mut x = e.clone();
    for i in at..x.lanes.len() {
        let w = ((i - at) as f32 / 5.0).min(1.0);
        x.lanes[i][A_DRUMS] = e.lanes[at][A_DRUMS].max(0.8) * (1.0 - w);
        x.lanes[i][A_BASS] = e.lanes[at][A_BASS].max(0.8) * (1.0 - w);
        if i < ONE {
            x.lanes[i][A_OTHER] = x.lanes[i][A_OTHER].max(0.8);
        }
    }
    Some((x, vec![Issue::StemsLeaveSeparately, Issue::AbruptStemExit]))
}

/// A's voice is cut where it is in the middle of a line.
fn cut_in_word(e: &Example, rng: &mut Rng) -> Option<Broken> {
    let exit = (1..e.lanes.len()).rev().find(|&i| e.lanes[i - 1][A_VOICE] >= 0.5 && e.lanes[i][A_VOICE] < 0.5)?;
    let inside: Vec<usize> = (exit.saturating_sub(10)..exit.saturating_sub(1)).filter(|&i| a_sung(e, i) && a_sung(e, i + 1) && e.a_beats[i][F_LINE_END] < 0.5).collect();
    if inside.is_empty() {
        return None;
    }
    let at = *rng.pick(&inside);
    let mut x = e.clone();
    for i in at + 1..x.lanes.len() {
        x.lanes[i][A_VOICE] = 0.0;
    }
    Some((x, vec![Issue::VoiceCutInWord]))
}

/// No blend at all: A stops, B starts.
fn plain_cut(e: &Example, _: &mut Rng) -> Option<Broken> {
    let mut x = e.clone();
    for (i, row) in x.lanes.iter_mut().enumerate() {
        let before = i < ONE;
        for (lane, v) in row.iter_mut().enumerate() {
            *v = if (lane < 4) == before { 1.0 } else { 0.0 };
        }
    }
    x.start_bars = 0.0;
    x.swap_bars = 0.0;
    Some((x, vec![Issue::CutNotTransition]))
}

fn leak(e: &Example, rng: &mut Rng) -> Option<Broken> {
    // Only heard where a voice is muted while its song plays on.
    let muted = (first_tick(e)..ONE).any(|i| (a_sung(e, i) && e.lanes[i][A_VOICE] < 0.2 && e.lanes[i][A_DRUMS] > 0.5) || (b_sung(e, i) && e.lanes[i][B_VOICE] < 0.2 && e.lanes[i][B_DRUMS] > 0.5));
    if !muted {
        return None;
    }
    let mut x = e.clone();
    x.measures.vocal_leak_db = rng.range(-28.0, -12.0) as f32;
    Some((x, vec![Issue::VocalLeak]))
}

fn pitch_ride(e: &Example, rng: &mut Rng) -> Option<Broken> {
    let mut x = e.clone();
    x.measures.stretch_pct = rng.range(6.0, 14.0) as f32 * if rng.chance(0.5) { 1.0 } else { -1.0 };
    x.tempo_b = x.tempo_a * x.beat_ratio / (1.0 + x.measures.stretch_pct / 100.0);
    Some((x, vec![Issue::PitchRiding]))
}
