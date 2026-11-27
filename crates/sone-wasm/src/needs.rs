//! Which parts of both songs' stems a plan's rendering reads.

use sone_core::{BeatTable, Deck, Spec};
use sone_plan::{Plan, plan::beat_time};
use serde::Serialize;

/// Seconds added on both sides: the renderer fits its own grid and nudges song B by some milliseconds.
const MARGIN_S: f64 = 0.25;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Need {
    /// "all": every stem; "drums": the drum stem only (the grid is fitted on it).
    pub stems: &'static str,
    pub from_s: f64,
    pub to_s: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Needs {
    pub a: Vec<Need>,
    pub b: Vec<Need>,
}

/// Merges windows of the same kind that touch or overlap.
fn merged(mut v: Vec<Need>, duration: f64) -> Vec<Need> {
    for n in &mut v {
        n.from_s = (n.from_s - MARGIN_S).max(0.0);
        // The table's duration is rounded; the host knows the real end.
        n.to_s = (n.to_s + MARGIN_S).min(duration + 1.0);
    }
    v.retain(|n| n.to_s > n.from_s);
    v.sort_by(|x, y| x.stems.cmp(y.stems).then(x.from_s.total_cmp(&y.from_s)));
    let mut out: Vec<Need> = Vec::new();
    for n in v {
        match out.last_mut() {
            Some(last) if last.stems == n.stems && n.from_s <= last.to_s => last.to_s = last.to_s.max(n.to_s),
            _ => out.push(n),
        }
    }
    // Drums that lie inside a window of all stems are there already.
    let all: Vec<Need> = out.iter().filter(|n| n.stems == "all").cloned().collect();
    out.retain(|n| n.stems == "all" || !all.iter().any(|a| a.from_s <= n.from_s && n.to_s <= a.to_s));
    out
}

/// What one deck reads: `time(u)` is the song's time of a ruler position, the deck plays on `[from, to]`.
fn deck(deck: &Deck, table: &BeatTable, time: &dyn Fn(f64) -> f64, from: f64, to: f64, pairs: Option<[f64; 2]>) -> Vec<Need> {
    let mut v = Vec::new();
    if deck.segments.is_empty() {
        v.push(Need { stems: "all", from_s: time(from), to_s: time(to) });
    }
    for s in &deck.segments {
        let (u0, u1) = (s.u0.max(from), s.u1.unwrap_or(to).min(to));
        if u1 > u0 {
            // A splice reads 6 ms before its first frame.
            v.push(Need { stems: "all", from_s: time(s.src + (u0 - s.u0)) - 0.01, to_s: time(s.src + (u1 - s.u0)) });
        }
    }
    let bar = |k: usize| beat_time(table, table.bars.get(k).copied().unwrap_or(table.times.len()) as f64);
    match deck.fit_bars {
        Some([f0, f1]) => v.push(Need { stems: "drums", from_s: bar(f0) - 0.2, to_s: bar(f1) + 0.6 }),
        None => {
            let at = bar(deck.anchor_bar);
            v.push(Need { stems: "drums", from_s: at - 64.0 * table.beat_s, to_s: at + 64.0 * table.beat_s });
        }
    }
    if let Some([p0, p1]) = pairs {
        v.push(Need { stems: "drums", from_s: time(p0), to_s: time(p1) });
    }
    merged(v, table.duration)
}

pub fn needs(plan: &Plan, a: &BeatTable, b: &BeatTable) -> Needs {
    let spec: &Spec = &plan.spec;
    let a_time = |u: f64| plan.one_s + u * plan.bar_s;
    let b_one = beat_time(b, plan.b_one_beat as f64);
    // Song B's own seconds per ruler bar.
    let b_bar = f64::from(spec.b.beats_per_bar) * b.beat_s;
    let b_time = |u: f64| b_one + u * b_bar;
    let pairs = spec.align_pairs.as_ref();
    // Song A plays to `until` and rings out for four more bars; without pairs the drums are compared around the one.
    let a_to = spec.a.until.unwrap_or(0.0) + 4.0;
    let b_from = spec.b.from.unwrap_or(0.0);
    Needs {
        a: deck(&spec.a, a, &a_time, spec.clip[0], a_to.max(8.0_f64.min(spec.clip[1])), pairs.map(|p| p.a)),
        b: deck(&spec.b, b, &b_time, b_from.min(-8.0_f64.max(spec.clip[0])), spec.clip[1], pairs.map(|p| p.b)),
    }
}
