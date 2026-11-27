//! Plans the pairs of the transitions the owner accepted and compares the planner's decisions with
//! the hand-built ones: `plan_check <data dir> [version]`.
//!
//! Compared per pair, in bars of each song: the recipe, the one in song A (where the ruler's zero
//! lies, also past A's end when its bars are replayed), where A's voice leaves, the one in song B.
//! A hand-built spec counts its bars on `analysis/struct`; the planner on `analysis/beats`, whose bar
//! lines may be corrected by whole beats. Both are turned into song time before they are compared.

use std::collections::HashMap;

use anyhow::{Context, Result};
use sone_core::{BeatTable, Deck, Recipe, Spec, TrackAnalysis};
use sone_plan::{PairRules, PlanOptions, plan, plan::beat_time};

/// Song time of a bar position (a fraction of a bar, also before the song or past its end) on the lab's grid.
fn lab_time(lab: &TrackAnalysis, bar: f64) -> f64 {
    let down = lab.downbeats();
    let beat_no = |b: f64| -> f64 {
        let k = b.floor() as i64;
        let at = |k: i64| -> f64 {
            if k >= 0 && (k as usize) < down.len() {
                lab.beat_idx[down[k as usize]] as f64
            } else if k < 0 {
                lab.beat_idx[down[0]] as f64 + 4.0 * k as f64
            } else {
                lab.beat_idx[*down.last().unwrap()] as f64 + 4.0 * (k - down.len() as i64 + 1) as f64
            }
        };
        at(k) + (b - k as f64) * (at(k + 1) - at(k))
    };
    let number = beat_no(bar);
    match lab.beat_idx.iter().position(|&k| k as f64 >= number) {
        Some(i) if (lab.beat_idx[i] as f64 - number).abs() < 1e-9 => lab.beats[i],
        Some(0) => lab.beats[0] - (lab.beat_idx[0] as f64 - number) * lab.beat_s,
        Some(i) => lab.beats[i - 1] + (number - lab.beat_idx[i - 1] as f64) * lab.beat_s,
        None => lab.beats[lab.beats.len() - 1] + (number - lab.beat_idx[lab.beats.len() - 1] as f64) * lab.beat_s,
    }
}

/// The one of a deck, in bars of its song (the ruler's zero).
fn one_bar(deck: &Deck) -> f64 {
    deck.anchor_bar as f64 - deck.anchor_u * f64::from(deck.beats_per_bar) / 4.0 + f64::from(deck.beat_shift) / 4.0
}

/// Where deck A's voice leaves, in ruler bars: the last point of its lane at full level.
fn voice_exit_u(deck: &Deck) -> Option<f64> {
    let lane = deck.lanes.get(&sone_core::Stem::Vocals)?;
    let last_full = lane.iter().rposition(|p| p[1] >= 0.99)?;
    (last_full + 1 < lane.len()).then(|| lane[last_full][0])
}

/// Song time → position in bars of a beat table (fractional, also past its end).
fn table_bar(t: &BeatTable, time: f64) -> f64 {
    let bar_t = |k: usize| -> f64 {
        match t.bars.get(k) {
            Some(&b) => beat_time(t, b as f64),
            None => beat_time(t, (*t.bars.last().unwrap() + 4 * (k + 1 - t.bars.len())) as f64),
        }
    };
    let mut k = 0;
    while bar_t(k + 1) <= time && k < t.bars.len() + 64 {
        k += 1;
    }
    k as f64 + (time - bar_t(k)) / (bar_t(k + 1) - bar_t(k))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = std::path::PathBuf::from(args.get(1).map_or("data", String::as_str));
    let version = args.get(2).map_or("S7", String::as_str);
    let names: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(data.join("round0/data.json")).unwrap_or_else(|_| "{}".into()))?;
    let name = |id: &str| names["tracks"][id]["name"].as_str().unwrap_or(id).chars().take(22).collect::<String>();
    let rules = PairRules::load(&data.join("owner/pair_rules.json"))?;
    let mut tables: HashMap<String, BeatTable> = HashMap::new();
    for e in std::fs::read_dir(data.join("analysis/beats"))? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "json") {
            let t = BeatTable::load(&p).with_context(|| p.display().to_string())?;
            tables.insert(t.id.clone(), t);
        }
    }
    let specs = Spec::load_all(&data.join(format!("versions/{version}.json")))?;
    let mut same = [0usize; 4];
    println!("{:4} {:46} {:>32} {:>22} {:>18} {:>16}", "", "pair", "recipe planner / hand", "one in A (bar)", "A's voice leaves", "one in B (bar)");
    for hand in &specs {
        let (a, b) = (&tables[&hand.a.id], &tables[&hand.b.id]);
        let p = match plan(a, b, &rules.apply(&hand.a.id, &hand.b.id, PlanOptions { id: hand.id.clone(), ..PlanOptions::default() })) {
            Ok(p) => p,
            Err(e) => {
                println!("{:4} {:46} no transition: {e}", hand.id, format!("{} → {}", name(&hand.a.id), name(&hand.b.id)));
                continue;
            }
        };
        let lab = |id: &str| TrackAnalysis::load(&data.join(format!("analysis/struct/{id}.json")));
        let (la, lb) = (lab(&hand.a.id)?, lab(&hand.b.id)?);
        // Everything in bars of the planner's tables.
        let hand_a_one = table_bar(a, lab_time(&la, one_bar(&hand.a)));
        let hand_b_one = table_bar(b, lab_time(&lb, one_bar(&hand.b)));
        let mine_a_one = table_bar(a, p.one_s);
        let mine_b_one = table_bar(b, beat_time(b, p.b_one_beat as f64));
        let bars_a = |u: f64, deck: &Deck, one: f64| one + u * f64::from(deck.beats_per_bar) / 4.0;
        let hand_exit = voice_exit_u(&hand.a).map(|u| bars_a(u, &hand.a, hand_a_one));
        let mine_exit = voice_exit_u(&p.spec.a).map(|u| bars_a(u, &p.spec.a, mine_a_one));
        let hand_recipe = hand.recipe.unwrap_or(match hand.technique.as_str() {
            t if t.contains("schleicht") || t.contains("Intro aus B") => Recipe::Sneak,
            t if t.contains("Tempo-Sprung") => Recipe::Gap,
            t if t.contains("Outro-Beat") || t.contains("Gitarren-Intro") => Recipe::OutroUnderIntro,
            _ => Recipe::VoiceOverBeat,
        });
        let ok = [
            p.recipe == hand_recipe,
            (mine_a_one - hand_a_one).abs() < 0.13,
            match (mine_exit, hand_exit) {
                (Some(m), Some(h)) => (m - h).abs() < 0.5,
                (None, None) => true,
                _ => false,
            },
            (mine_b_one - hand_b_one).abs() < 0.13,
        ];
        for (n, o) in same.iter_mut().zip(ok) {
            *n += usize::from(o);
        }
        let mark = |o: bool| if o { "  " } else { "✗ " };
        let exit_txt = |e: Option<f64>| e.map_or("-".to_string(), |v| format!("{v:.2}"));
        println!(
            "{:4} {:46} {}{:>14} / {:<14} {}{:>8.2} / {:<8.2} {}{:>7} / {:<7} {}{:>6.2} / {:<6.2}",
            hand.id,
            format!("{} → {}", name(&hand.a.id), name(&hand.b.id)),
            mark(ok[0]),
            format!("{:?}", p.recipe),
            format!("{hand_recipe:?}"),
            mark(ok[1]),
            mine_a_one,
            hand_a_one,
            mark(ok[2]),
            exit_txt(mine_exit),
            exit_txt(hand_exit),
            mark(ok[3]),
            mine_b_one,
            hand_b_one,
        );
        if std::env::var_os("PLAN_NOTES").is_some() {
            for n in &p.notes {
                println!("       · {n}");
            }
        }
    }
    println!("\nsame as the owner's version, of {}: recipe {}, one in A {}, A's voice leaves {}, one in B {}", specs.len(), same[0], same[1], same[2], same[3]);
    Ok(())
}
