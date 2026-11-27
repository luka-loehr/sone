//! From a spec and the two songs to a training example.

use sone_core::{BeatTable, Deck, Example, Label, Measures, Points, Recipe, Segment, Shape, Spec, Stem, example::BeatLanes};

/// Ruler beats (four per ruler bar) of song A before and after the one.
pub const A_BEFORE: i64 = 64;
pub const A_AFTER: i64 = 16;
pub const B_BEFORE: i64 = 64;
pub const B_AFTER: i64 = 64;
/// The lanes cover the same stretch as song A: 16 bars before the one, 4 after.
pub const LANE_TICKS: usize = (A_BEFORE + A_AFTER) as usize;

const FEATURES: usize = sone_core::example::FEATURES.len();

fn silence() -> Vec<f32> {
    let mut f = vec![-60.0; FEATURES];
    for v in &mut f[7..12] {
        *v = 0.0;
    }
    for v in &mut f[12..24] {
        *v = 0.0;
    }
    f
}

/// Beat of the song (index into its table, may be fractional or outside) that sits on the one.
pub fn one_beat(t: &BeatTable, d: &Deck) -> f64 {
    let bar = match t.bars.get(d.anchor_bar) {
        Some(&b) => b as f64,
        None => t.bars.last().map_or(0.0, |&b| (b + 4 * (d.anchor_bar + 1 - t.bars.len())) as f64),
    };
    // The table's bars already carry the song's own shift; the spec's shift is relative to the lab's analysis.
    bar + f64::from(d.beat_shift - i32::from(t.beat_shift)) - d.anchor_u * f64::from(d.beats_per_bar)
}

/// Which position of the song (ruler bars relative to the one) plays at ruler position `u`.
pub fn source(segments: &[Segment], u: f64) -> Option<f64> {
    if segments.is_empty() {
        return Some(u);
    }
    segments.iter().find(|s| u >= s.u0 - 1e-9 && s.u1.is_none_or(|u1| u < u1 - 1e-9)).map(|s| s.src + (u - s.u0))
}

fn features_at(t: &BeatTable, beat: f64) -> Vec<f32> {
    if beat < 0.0 {
        return silence();
    }
    t.features.get(beat.floor() as usize).cloned().unwrap_or_else(silence)
}

/// A lane's value with the shapes of the lab's renderer (without its 6 ms smoothing).
pub fn lane_at(points: &Points, shape: Shape, u: f64) -> f64 {
    match points.as_slice() {
        [] => 1.0,
        [first, ..] if u <= first[0] => first[1],
        [.., last] if u >= last[0] => last[1],
        _ => {
            let i = points.partition_point(|p| p[0] <= u);
            let (p, q) = (points[i - 1], points[i]);
            if q[0] <= p[0] {
                return q[1];
            }
            let w = (u - p[0]) / (q[0] - p[0]);
            match shape {
                Shape::Lin => p[1] + (q[1] - p[1]) * w,
                Shape::S => p[1] + (q[1] - p[1]) * (w * w * (3.0 - 2.0 * w)),
                Shape::Db => {
                    if p[1] <= 0.0 && q[1] <= 0.0 {
                        return 0.0;
                    }
                    let db = |v: f64| 20.0 * v.max(10f64.powf(-3.5)).log10();
                    let v = 10f64.powf((db(p[1]) + (db(q[1]) - db(p[1])) * w) / 20.0);
                    if v < 10f64.powf(-3.3) { 0.0 } else { v }
                }
            }
        }
    }
}

fn deck_gain(d: &Deck, stem: Stem, u: f64, is_a: bool) -> f64 {
    let present = if is_a { d.until.is_none_or(|until| u < until + 1e-9) || d.lanes.contains_key(&stem) } else { d.from.is_none_or(|from| u >= from - 1e-9) };
    if !present {
        return 0.0;
    }
    let lane = d.lanes.get(&stem).map_or(1.0, |p| lane_at(p, d.shape_of(stem), u));
    let trim = if d.gain_db.is_empty() { 1.0 } else { 10f64.powf(lane_at(&d.gain_db, Shape::Lin, u) / 20.0) };
    lane * trim
}

#[allow(clippy::too_many_arguments)]
pub fn example_from_spec(id: &str, spec: &Spec, a: &BeatTable, b: &BeatTable, recipe: Recipe, stretch_pct: f32, key_distance: u8, label: Label) -> Example {
    let (one_a, one_b) = (one_beat(a, &spec.a), one_beat(b, &spec.b));
    let (bpb_a, bpb_b) = (f64::from(spec.a.beats_per_bar), f64::from(spec.b.beats_per_bar));
    // Without sync B counts its own bars from where it starts.
    let b_shift = if spec.no_sync { spec.b.start_u.unwrap_or(0.0) - spec.b.from.unwrap_or(0.0) } else { 0.0 };
    let a_beats = (-A_BEFORE..A_AFTER)
        .map(|j| {
            let u = j as f64 / 4.0;
            match source(&spec.a.segments, u) {
                Some(src) => features_at(a, one_a + src * bpb_a),
                None => silence(),
            }
        })
        .collect();
    let b_beats = (-B_BEFORE..B_AFTER)
        .map(|j| {
            let u = j as f64 / 4.0 - b_shift;
            if spec.b.from.is_some_and(|from| u < from - 1e-9) {
                return silence();
            }
            match source(&spec.b.segments, u) {
                Some(src) => features_at(b, one_b + src * bpb_b),
                None => silence(),
            }
        })
        .collect();
    let lanes = (-A_BEFORE..A_AFTER)
        .map(|j| {
            // The middle of the ruler beat.
            let u = (j as f64 + 0.5) / 4.0;
            let mut row: BeatLanes = [0.0; 8];
            for (k, stem) in Stem::ALL.iter().enumerate() {
                row[k] = deck_gain(&spec.a, *stem, u, true) as f32;
                row[4 + k] = deck_gain(&spec.b, *stem, u - b_shift, false) as f32;
            }
            row
        })
        .collect();
    let start_bars = -spec.first_event as f32;
    let swap_bars = spec.swap_u.map_or(0.0, |s| -s as f32);
    Example {
        id: id.to_owned(),
        a: a.id.clone(),
        b: b.id.clone(),
        a_beats,
        b_beats,
        tempo_a: a.bpm as f32,
        tempo_b: b.bpm as f32,
        beat_ratio: (bpb_b / bpb_a) as f32,
        key_distance,
        recipe,
        start_bars,
        swap_bars,
        lanes,
        measures: Measures { beat_offset_ms: 0.0, stretch_pct, vocal_leak_db: -60.0, tuning_diff_cents: (a.tuning_cents - b.tuning_cents).abs() },
        label,
    }
}

/// The recipe a hand-written spec follows, read from what it does.
pub fn recipe_of(spec: &Spec) -> Recipe {
    if let Some(r) = spec.recipe {
        return r;
    }
    if spec.no_sync || spec.b.beats_per_bar == 6 || spec.a.beats_per_bar == 6 {
        return Recipe::Gap;
    }
    let b_lane = |stem: Stem, u: f64| spec.b.lanes.get(&stem).map_or(1.0, |p| lane_at(p, spec.b.shape_of(stem), u));
    let from = spec.b.from.unwrap_or(0.0);
    let mid = (from / 2.0).min(-1.0);
    if b_lane(Stem::Drums, mid) > 0.05 && b_lane(Stem::Drums, mid) > b_lane(Stem::Vocals, mid) {
        return Recipe::Sneak;
    }
    if b_lane(Stem::Vocals, mid) > 0.5 {
        return Recipe::VoiceOverBeat;
    }
    Recipe::OutroUnderIntro
}
