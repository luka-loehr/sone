//! From two songs to a transition.
//!
//! The ruler counts bars `u`, `u = 0` is the one. A ruler bar lasts
//! `bpb_a` beats of song A and `bpb_b` beats of song B (4 and 4 when both
//! are counted alike).

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use sone_core::{AlignPairs, BeatTable, Deck, Points, Recipe, Segment, Shape, Spec, Stem, Step};

use crate::{
    keys,
    lanes::{SNEAK_FLOOR, at_one, cut_in, echo, fade_to_cut, hold_cut, lin_fade, reverb, voice_out},
    tempo::{self, Lock, TempoFit},
    tts::{self, TtsBars},
    view::{self, F_DRUMS, F_HATS, F_MIX, F_OTHER, Opening, Song},
};

#[derive(Debug, Clone)]
pub struct PlanOptions {
    pub id: String,
    /// More stretch than this and the songs do not count as "same beat".
    pub max_stretch_pct: f64,
    /// Length of the blend in ruler bars (8; 4 when the look-ahead is short).
    pub lead_bars: f64,
    /// A line the DJ wants to say before song B starts, in seconds.
    pub tts_seconds: Option<f64>,
    /// Live: the part of song A (seconds) in which the one has to lie.
    pub window: Option<(f64, f64)>,
    /// Bars of the rendered clip before the first and after the last event.
    pub clip_before: f64,
    pub clip_after: f64,
    /// Take this recipe whatever the songs suggest (the training data generator uses it).
    pub force: Option<Recipe>,
    /// The owner heard song B's voice over song A's beat for this pair and did not want it: B's voice
    /// comes only on the one.
    pub no_voice_over_beat: bool,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self { id: "plan".into(), max_stretch_pct: 4.0, lead_bars: 8.0, tts_seconds: None, window: None, clip_before: 14.0, clip_after: 12.0, force: None, no_voice_over_beat: false }
    }
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub spec: Spec,
    pub recipe: Recipe,
    pub fit: TempoFit,
    pub key_distance: u8,
    /// 0..1: how well the songs offered what the recipe needs.
    pub confidence: f32,
    /// Beat of song A (index into its table, may lie past its last beat) on the one.
    pub a_one_beat: i64,
    pub b_one_beat: i64,
    /// Seconds per ruler bar.
    pub bar_s: f64,
    /// Time in song A of the one, seconds.
    pub one_s: f64,
    pub tts: Option<TtsBars>,
    /// What was decided and why, for the log.
    pub notes: Vec<String>,
}

impl Plan {
    /// Time in song A of a ruler position.
    pub fn a_time(&self, u: f64) -> f64 {
        self.one_s + u * self.bar_s
    }
}

/// Time of a beat, also before the first and after the last one of the table.
pub fn beat_time(t: &BeatTable, beat: f64) -> f64 {
    let n = t.times.len();
    if n == 0 {
        return beat * t.beat_s;
    }
    if beat <= 0.0 {
        return t.times[0] + beat * t.beat_s;
    }
    let last = (n - 1) as f64;
    if beat >= last {
        return t.times[n - 1] + (beat - last) * t.beat_s;
    }
    let i = beat.floor() as usize;
    t.times[i] + (t.times[i + 1] - t.times[i]) * (beat - i as f64)
}

struct Ruler {
    one: i64,
    bpb: i64,
}

impl Ruler {
    fn u_of_beat(&self, beat: f64) -> f64 {
        (beat - self.one as f64) / self.bpb as f64
    }
    /// End of a sixteenth note (a line ends when its last sixteenth is over).
    fn u_of_line_end(&self, sixteenth: usize) -> f64 {
        self.u_of_beat((sixteenth + 1) as f64 / 4.0)
    }
    fn u_of_line_start(&self, sixteenth: usize) -> f64 {
        self.u_of_beat(sixteenth as f64 / 4.0)
    }
    fn beat_of(&self, u: f64) -> f64 {
        self.one as f64 + u * self.bpb as f64
    }
}

struct ASide {
    one: i64,
    /// Where A's voice leaves, and whether a line really ended there.
    exit: Option<(f64, bool)>,
}

struct BSide {
    one: i64,
    from_u: f64,
    segments: Vec<Segment>,
    /// B's voice comes in here (ruler); `None`: B has no voice in reach.
    voice_u: Option<f64>,
    voice_on_line: bool,
    /// B's first bars with beat, its steady part for the tempo fit.
    fit_bars: Option<[usize; 2]>,
}

pub fn plan(a: &BeatTable, b: &BeatTable, opt: &PlanOptions) -> Result<Plan> {
    if a.bars.len() < 16 || b.bars.len() < 16 {
        bail!("a song shorter than 16 bars cannot be planned");
    }
    let (sa, sb) = (Song::new(a), Song::new(b));
    let mut notes = Vec::new();
    let mut confidence = 1.0f32;

    // The tempo each song has where the transition happens: some songs change it on the way (a half-time
    // outro, a slower intro), and their average says nothing about that part.
    let bpm_a = sa.local_bpm(sa.full_end_bar().saturating_sub(16), sa.full_end_bar()).unwrap_or(a.bpm);
    let b_from_bar = sb.arrival_bar().unwrap_or(0);
    let bpm_b = sb.local_bpm(b_from_bar, b_from_bar + 16).unwrap_or(b.bpm);
    let fit = tempo::fit(bpm_a, bpm_b, opt.max_stretch_pct);
    let key_distance = keys::song_distance(&a.camelot, &a.camelot_alt, &b.camelot, &b.camelot_alt);
    let detuned = (a.tuning_cents - b.tuning_cents).abs() > 25.0;
    let clash = key_distance >= 2 || detuned;
    let opening = view::opening(&sb);
    notes.push(format!("tempo {bpm_a:.2} → {bpm_b:.2} BPM, lock {} ({:+.2} %), keys {} → {} (distance {key_distance}{})", fit.ratio_name(), fit.stretch_pct, a.camelot, b.camelot, if detuned { ", tunings apart" } else { "" }));
    notes.push(format!("song B opens {opening:?}"));
    if opt.no_voice_over_beat && matches!(opening, Opening::VoiceFirst { .. }) {
        notes.push("the owner wants song B's voice only from the one for this pair".into());
    }

    // Without a common beat there is no transition the owner accepts ("only vocals and then a hard cut
    // to the new song ... this ain't one"): such a pair is not planned, the queue has to choose another song.
    if opt.force.is_none() && !fit.beat_on_beat() {
        bail!("{} and {} do not share a beat ({bpm_a:.1} against {bpm_b:.1} BPM): no transition", a.id, b.id);
    }
    if opt.force.is_none() && opening == Opening::NoBeat {
        bail!("song {} has no beat to take over with", b.id);
    }
    // Two beats may only lie on each other at the same count: at half against double tempo the faster
    // drums under the slower song sound "way too fast" (the owner, x29).
    let half_double = fit.bpb_a != fit.bpb_b;
    // B sings clearly early on: its voice alone can ride on A's beat (the owner asked for this "vocal
    // overlay" where B's beat sneaking in did not fit, x19, x26, x29).
    let overlay = !opt.no_voice_over_beat && sb.voice_bar().is_some_and(|v| v < 24 && sb.level(v, view::F_VOCALS) >= sb.typical(view::F_VOCALS) - 10.0);
    let recipe = opt.force.unwrap_or(match (fit.beat_on_beat(), clash, opening) {
        (false, _, _) | (_, _, Opening::NoBeat) => Recipe::Gap,
        (true, _, Opening::VoiceFirst { .. }) if !opt.no_voice_over_beat => Recipe::VoiceOverBeat,
        (true, false, Opening::MelodyFirst { .. }) => Recipe::OutroUnderIntro,
        // Half against double tempo: B's voice alone over A's beat when B sings early (two beats may not lie
        // on each other there). Clashing keys need nothing special any more: B's bass comes on the one in
        // every sneak, so before it only B's drums meet A.
        (true, _, _) if half_double && overlay => Recipe::VoiceOverBeat,
        (true, _, Opening::MelodyFirst { .. }) if half_double => Recipe::OutroUnderIntro,
        (true, _, _) if half_double => bail!("{} and {} only meet at half against double tempo and {} offers neither its voice nor a melody intro to lay over {}'s beat: no transition", a.id, b.id, b.id, a.id),
        (true, _, _) => Recipe::Sneak,
    });
    if recipe == Recipe::Gap {
        confidence *= 0.8;
    }
    if fit.stretch_pct.abs() > 2.5 {
        confidence *= 0.9;
    }
    confidence *= a.phase_confidence.clamp(0.5, 1.0) * b.phase_confidence.clamp(0.5, 1.0);

    let (bpb_a, bpb_b) = (i64::from(fit.bpb_a), i64::from(fit.bpb_b));
    let bar_s = 60.0 / bpm_a * bpb_a as f64;
    let mut lead = opt.lead_bars;
    // Room for the DJ's line: whole pairs of bars before song B's voice.
    let talk = opt.tts_seconds.map_or(0.0, |s| tts::bars_for(s, bar_s).min(8.0));

    // ── Song B: which of its bars sits on the one, and what plays before it ──
    let b_side = b_side(&sb, recipe, opening, lead, bpb_b, clash, &mut notes);

    // B's voice comes on the one (the owner: "would be cleaner if B was cropped so the voice started at
    // the 1 and not 1.5 seconds after"): when B's first line starts within a few bars after the one,
    // B's one moves onto it, a pickup of up to a bar before it allowed.
    let mut b_side = b_side;
    let mut b_moved = false;
    if matches!(recipe, Recipe::Sneak | Recipe::OutroUnderIntro) {
        // The owner accepted B's voice a bar after the one (h02, h03) and heard 1.3 bars as too late (b10).
        if let Some(v) = b_side.voice_u.filter(|&v| v > 1.25 && v <= 4.0) {
            let k = (v - 0.25).ceil();
            b_side.one += (k * bpb_b as f64) as i64;
            b_side.voice_u = Some(v - k);
            b_moved = true;
            notes.push(format!("song B's one moves {k} bar(s) on, so that its voice starts on the one"));
        }
    }
    if b_side.from_u > -lead + 0.5 && recipe != Recipe::Gap {
        // B has less before its one than the blend is long.
        lead = (-b_side.from_u).max(2.0);
        notes.push(format!("song B has only {lead} bars before its one, the blend is that long"));
        confidence *= 0.9;
    }
    if recipe == Recipe::Sneak && opt.window.is_none() {
        let fe = sa.snap_to_phrase(sa.full_end_bar());
        let want_fade = (lead * bpb_a as f64 / 4.0).ceil() as usize;
        if let Some(last_gap) = (fe.saturating_sub(want_fade)..fe).rev().find(|&k| !sa.carries(k)) {
            let clean_after = fe.saturating_sub(last_gap + 1);
            if clean_after >= 4 && !is_rising_buildup(&sa, fe) {
                let shrunk = ((clean_after / 2) * 2).max(4) as f64 * 4.0 / bpb_a as f64;
                if shrunk < lead {
                    lead = shrunk;
                    notes.push(format!("blend shortened to {lead} bars so song A finishes its chorus after the buildup break"));
                }
            }
        }
    }

    // ── Song A: where the one is ──
    let a_side = a_side(&sa, recipe, lead, talk, bpb_a, clash, opt, &mut notes)?;
    if matches!(a_side.exit, Some((_, false))) {
        confidence *= 0.8;
    }
    if !b_side.voice_on_line {
        confidence *= 0.9;
    }
    let ra = Ruler { one: a_side.one, bpb: bpb_a };
    let rb = Ruler { one: b_side.one, bpb: bpb_b };

    // ── Lanes ──
    let mut first_event;
    let mut last_event = 0.0f64;
    let mut a_lanes: BTreeMap<Stem, Points> = BTreeMap::new();
    let mut b_lanes: BTreeMap<Stem, Points> = BTreeMap::new();
    let mut a_hpf: BTreeMap<Stem, Points> = BTreeMap::new();
    let mut b_hpf: BTreeMap<Stem, Points> = BTreeMap::new();
    let mut a_sends = Vec::new();
    let mut a_until = 0.0;
    let mut swap_u = None;
    let mut steps = Vec::new();
    let mut tts_slot = None;
    let clip0 = |first: f64| first - opt.clip_before;
    let mut b_from = b_side.from_u;
    let b_voice = b_side.voice_u;
    let technique;

    match recipe {
        Recipe::VoiceOverBeat => {
            technique = "B's Stimme über A's Beat, B übernimmt auf der Eins";
            let bed = -(lead + talk);
            first_event = bed;
            let start = clip0(first_event);
            let (exit, ended) = a_side.exit.unwrap_or((bed, false));
            a_lanes.insert(Stem::Vocals, voice_out(start, exit, ended));
            // The echo repeats once a beat of the song (half a ruler beat when A is counted at half its felt tempo).
            a_sends.push(echo(Stem::Vocals, exit - 0.5, exit, if bpb_a == 2 { 0.5 } else { 1.0 }, 0.7));
            a_lanes.insert(Stem::Drums, hold_cut(start, 0.0, 1.0));
            a_lanes.insert(Stem::Bass, hold_cut(start, 0.0, 1.0));
            if clash {
                // The keys clash: under B's voice only A's beat, its bass and melody leave before B sings.
                a_lanes.insert(Stem::Bass, lin_fade(start, bed - 2.0, bed, 1.0));
                a_lanes.insert(Stem::Other, lin_fade(start, bed - 2.0, bed, 1.0));
            }
            // B's voice quiet for two bars or more before the one: B's melody fills from there,
            // and A's melody has to be gone by then (one melody at a time).
            let quiet_from = voice_quiet_from(&sb, &rb, b_from);
            match quiet_from {
                Some(q) if q <= -2.0 => {
                    if !clash {
                        a_lanes.insert(Stem::Other, lin_fade(start, bed, (q + 2.0).min(-1.0), 1.0));
                    }
                    b_lanes.insert(Stem::Other, vec![[b_from, 0.0], [q, 0.0], [(q + 4.0).min(-1.0), 1.0]]);
                    steps.push(step(q, "B's Stimme ist fertig, B's Melodie blendet langsam ein."));
                }
                _ => {
                    if !clash {
                        a_lanes.insert(Stem::Other, if key_distance == 0 { hold_cut(start, 0.0, 1.0) } else { lin_fade(start, bed, 0.0, 1.0) });
                    }
                    b_lanes.insert(Stem::Other, at_one(b_from, 0.0, 2.0));
                }
            }
            // B's voice is open from the first bar of the blend, as in every version the owner accepted (the
            // owner, x24: "would have been 10/10 if B's voice had started at -8 and not -6"), but it comes in
            // with a line, never inside a word (x25: "cuts in while in speak. if it started one sec before and
            // not in the middle of the word it would actually fit", x28): see `b_voice_entry`. A's voice must be
            // gone first, unless it rings out at the end of its line (x25).
            let a_gone = if ended { exit - PICKUP_BARS } else { exit };
            let entry = b_voice_entry(&sb, &rb, b_from, b_voice.filter(|_| b_side.voice_on_line), a_gone);
            if (entry - b_from).abs() > 1e-6 {
                notes.push(format!("song B is singing where the blend starts; its voice comes in at {entry:+.2} (a line or a breath)"));
            }
            if entry < b_from {
                b_from = entry;
                first_event = first_event.min(entry);
                b_lanes.insert(Stem::Vocals, vec![[b_from, 1.0]]);
            } else {
                b_lanes.insert(Stem::Vocals, cut_in(b_from, entry));
            }
            b_lanes.insert(Stem::Drums, cut_in(b_from, 0.0));
            b_lanes.insert(Stem::Bass, cut_in(b_from, 0.0));
            if talk > 0.0 {
                tts_slot = Some(TtsBars { from_u: bed, post_u: -lead, over: "A's Beat, niemand singt".into() });
                steps.push(step(bed, "A's Gesang endet. A's Beat läuft weiter, hier spricht der DJ."));
                steps.push(step(-lead, "B's Stimme setzt ein."));
            } else {
                steps.push(step(bed, "A's Gesang endet am Zeilenende (Echo). A's Drums und Bass laufen voll weiter. B's Stimme setzt direkt ein."));
            }
            steps.push(step(0.0, "Die Eins: B's Beat, Bass und Melodie setzen ein, A endet hart."));
        }
        Recipe::Sneak => {
            technique = "B's Beat schleicht sich unter A";
            let swap = -(lead / 2.0).max(2.0);
            swap_u = Some(swap);
            first_event = -lead;
            let start = clip0(first_event);
            let tease = tease_gain(&sa, &sb, &ra, &rb);
            b_lanes.insert(Stem::Drums, vec![[-lead, 0.0], [-lead + 1.0, 0.9 * tease.min(1.2)], [swap - 0.03, tease], [swap, 1.0]]);
            b_hpf.insert(Stem::Drums, vec![[-lead, 500.0], [swap - 0.05, 220.0], [swap, 20.0]]);
            // B's bass waits for the one: A's bass stays at half level until then (one bass at a time),
            // and before the one only B's drums meet A, so clashing keys do not matter either.
            b_lanes.insert(Stem::Bass, cut_in(b_from, 0.0));
            b_lanes.insert(Stem::Other, at_one(b_from, 0.0, 2.0));
            // B's voice comes on the one, or with its line a pickup earlier (the owner: "no reason not to
            // start B's voice at the one", "it started mid word").
            let a_swap_beat = ra.beat_of(swap).round() as i64;
            let a_instrumental_bed = (a_swap_beat..ra.one).all(|i| !sa.sung(i));
            let mut voice_at = if a_instrumental_bed {
                let earliest_16 = (rb.beat_of(swap).max(0.0).floor() as usize) * 4;
                let one_16 = (rb.one.max(0) as usize) * 4;
                let pre_line = sb
                    .t
                    .line_starts
                    .iter()
                    .copied()
                    .find(|&s| s >= earliest_16.saturating_sub(4) && s + 4 <= one_16)
                    .map(|s| rb.u_of_line_start(s).max(swap));
                pre_line.unwrap_or_else(|| match b_voice {
                    Some(v) if v >= -PICKUP_BARS => v,
                    Some(v) => v.max(0.0),
                    None => 0.0,
                })
            } else {
                match b_voice {
                    Some(v) if v >= -PICKUP_BARS => v,
                    Some(v) => v.max(0.0),
                    None => 0.0,
                }
            };
            let b_sings_at_one = (rb.one..rb.one + rb.bpb).any(|i| sb.sung(i));
            if let Some((exit, _)) = a_side.exit.filter(|e| e.0 > 0.0 && voice_at < e.0 + 0.1) {
                if b_sings_at_one {
                    // Song B sings right on the one (mix_006): never mute B's singer for a full bar after the drop!
                    voice_at = voice_at.min(0.0);
                } else {
                    let next_bar = (exit + 0.1).ceil();
                    voice_at = sb.line_start_from(rb.beat_of(exit + 0.1).ceil().max(0.0) as usize).map(|s| rb.u_of_line_start(s)).filter(|&u| u <= next_bar).unwrap_or(next_bar);
                }
            }
            b_lanes.insert(Stem::Vocals, cut_in(b_from, voice_at));
            let fade_from = swap - 2.0;
            // A's beat, bass and melody come down to half and are cut on the one: no hole before it.
            // When B's vocals overlap early over A's instrumental outro, tuck A's melody gently to 0.28 so B's singer shines.
            a_lanes.insert(Stem::Drums, fade_to_cut(start, fade_from, 0.0, SNEAK_FLOOR));
            a_lanes.insert(Stem::Bass, fade_to_cut(start, fade_from, 0.0, SNEAK_FLOOR));
            if voice_at < -0.5 {
                a_lanes.insert(Stem::Other, vec![[start, 1.0], [fade_from, 1.0], [voice_at, 0.28], [-0.015, 0.22], [0.0, 0.0]]);
            } else {
                a_lanes.insert(Stem::Other, fade_to_cut(start, fade_from, 0.0, SNEAK_FLOOR));
            }
            let (exit, ended) = a_side.exit.unwrap_or((0.0, false));
            // A's voice gives way to B's pickup or B singing on the one: one voice at a time.
            let (exit, ended) = if exit > voice_at - 0.02 && (voice_at < 0.0 || b_sings_at_one) {
                (voice_at.min(0.0), false)
            } else {
                (exit, ended)
            };
            a_lanes.insert(Stem::Vocals, voice_out(start, exit, ended));
            a_sends.push(echo(Stem::Vocals, exit - 0.5, exit, 1.0, 0.8));
            a_until = (exit + 0.1).max(0.0);
            last_event = voice_at;
            if talk > 0.0 {
                tts_slot = Some(TtsBars { from_u: -talk, post_u: 0.0, over: "B's Beat, niemand singt".into() });
            }
            steps.push(step(-lead, "B's Drums kommen leise dazu – nur die Höhen, ohne Kick."));
            steps.push(step(fade_from, "A's Drums, Bass und Melodie gehen linear auf die Hälfte zurück, bis genau zur Eins."));
            steps.push(step(swap, "B's Kick setzt ein; B's Bass wartet auf die Eins."));
            if talk > 0.0 {
                steps.push(step(-talk, "A's Gesang ist fertig, hier spricht der DJ."));
            }
            steps.push(step(0.0, "Die Eins: A endet hart. B's Bass, Melodie und Gesang beginnen hier."));
        }
        Recipe::OutroUnderIntro => {
            technique = "A's Beat unter dem Intro von B";
            first_event = -lead;
            let start = clip0(first_event);
            let (exit, ended) = a_side.exit.unwrap_or((-lead, false));
            if a_side.exit.is_some() {
                a_lanes.insert(Stem::Vocals, voice_out(start, exit, ended));
                a_sends.push(echo(Stem::Vocals, exit - 0.5, exit, 1.0, 0.7));
            } else {
                a_lanes.insert(Stem::Vocals, vec![[start, 1.0], [-lead, 1.0], [-lead + 0.1, 0.0]]);
            }
            a_lanes.insert(Stem::Drums, hold_cut(start, 0.0, 1.0));
            a_lanes.insert(Stem::Bass, hold_cut(start, 0.0, 1.0));
            a_lanes.insert(Stem::Other, vec![[start, 1.0], [-lead, 1.0], [-lead * 0.375, 0.3], [-0.015, 0.1], [0.0, 0.0]]);
            a_hpf.insert(Stem::Drums, vec![[-2.0, 20.0], [-1.0, 140.0], [-0.05, 800.0]]);
            b_lanes.insert(Stem::Other, vec![[b_from, 0.0], [(b_from + lead * 0.75).min(-1.0), 1.0]]);
            // A line of B that begins before the one comes in whole, up to a bar early (b10: "it's in the
            // middle of the singer saying a word"), not cut at a quarter bar before the one.
            b_lanes.insert(Stem::Vocals, cut_in(b_from, b_voice.unwrap_or(0.0).max(-PICKUP_BARS).max(b_from)));
            // Where B's one was moved onto its voice, B's beat may already have started: it waits for the one.
            b_lanes.insert(Stem::Drums, if b_moved { cut_in(b_from, 0.0) } else { vec![[b_from, 1.0]] });
            b_lanes.insert(Stem::Bass, if b_moved { cut_in(b_from, 0.0) } else { vec![[b_from, 1.0]] });
            if talk > 0.0 {
                tts_slot = Some(TtsBars { from_u: -talk.min(lead), post_u: 0.0, over: "A's Beat und B's Intro, niemand singt".into() });
            }
            steps.push(step(-lead, "A's Gesang ist vorbei, A's Beat läuft weiter. B's Melodie beginnt langsam einzublenden."));
            steps.push(step(-2.0, "A's Drums werden hochgefiltert."));
            steps.push(step(0.0, "Die Eins: B's Beat, Bass und Stimme setzen ein, A endet hart."));
        }
        Recipe::Gap => {
            technique = "Tempo-Sprung: Hook allein im Hall, B auf der Eins";
            let solo = if talk > 0.0 { talk.max(2.0) } else { 2.0 };
            first_event = -solo;
            let start = clip0(first_event);
            a_lanes.insert(Stem::Drums, vec![[start, 1.0], [-solo, 1.0], [-solo + 1.0, 0.0]]);
            a_lanes.insert(Stem::Bass, vec![[start, 1.0], [-solo, 1.0], [-solo + 1.0, 0.0]]);
            // A's hook carries the bars alone; where A's voice is too quiet to carry them, its melody stays
            // with it until the last beat.
            let voice_carries = mean_db(&sa, &ra, -solo, 0.0, view::F_VOCALS) > mean_db(&sa, &ra, -solo - 4.0, -solo, F_MIX) - 10.0;
            a_lanes.insert(Stem::Other, if voice_carries { vec![[start, 1.0], [-solo, 1.0], [-0.25, 0.0]] } else { vec![[start, 1.0], [-0.5, 1.0], [-0.25, 0.0]] });
            if talk > 0.0 {
                let (exit, ended) = a_side.exit.unwrap_or((-talk, false));
                a_lanes.insert(Stem::Vocals, voice_out(start, exit, ended));
                a_sends.push(reverb(Stem::Vocals, vec![[exit - 1.0, 0.0], [exit, 1.0], [exit + 0.1, 0.0]], 0.6));
                tts_slot = Some(TtsBars { from_u: -talk, post_u: 0.0, over: "A's Hall, niemand singt".into() });
                a_until = 0.0;
            } else {
                a_lanes.insert(Stem::Vocals, vec![[start, 1.0], [0.0, 1.0], [1.0, 0.0]]);
                a_sends.push(reverb(Stem::Vocals, vec![[-solo, 0.0], [-solo / 2.0, 0.6], [0.0, 1.0], [0.6, 0.5], [1.0, 0.0]], 0.6));
                // An echo that ticks in B's tempo counts B in.
                let beats = if fit.lock == Lock::None { 1.0 } else { bpb_a as f64 / bpb_b as f64 };
                let two_b_beats = 2.0 / bpb_b as f64;
                let mut e = echo(Stem::Vocals, -two_b_beats, 0.0, beats, 0.7);
                e.send = vec![[-two_b_beats - 0.01, 0.0], [-two_b_beats, 1.0], [-0.02, 1.0], [0.0, 0.0]];
                e.feedback = 0.5;
                e.hp = 500.0;
                e.lp = 5000.0;
                a_sends.push(e);
                a_until = 1.0;
            }
            // B's melody waits until A's voice is gone: its next four-bar line.
            let b_bar = 4.0 / bpb_b as f64;
            let melody_at = ((a_until + 1.0) / (4.0 * b_bar)).ceil() * 4.0 * b_bar;
            b_lanes.insert(Stem::Vocals, vec![[0.0, 1.0]]);
            b_lanes.insert(Stem::Drums, vec![[0.0, 1.0]]);
            b_lanes.insert(Stem::Bass, vec![[0.0, 1.0]]);
            b_lanes.insert(Stem::Other, at_one(0.0, 0.0, melody_at));
            last_event = melody_at;
            steps.push(step(-solo, "A's Drums und Bass blenden linear aus, A's Melodie kurz danach. Der Hall auf A's Stimme wächst."));
            steps.push(step(0.0, "Die Eins: B's Drums und Bass setzen hart ein. A's Stimme klingt im Hall aus."));
            steps.push(step(melody_at, "B's Melodie ist voll da, ab hier nur noch B."));
        }
    }
    steps.sort_by(|x, y| x.u.total_cmp(&y.u));
    // The transition ends where the last fader comes to rest.
    let rest = a_lanes.values().chain(b_lanes.values()).flat_map(|l| l.iter().map(|p| p[0])).fold(f64::MIN, f64::max);
    last_event = last_event.max(rest.min(16.0));

    // ── Beds must not have holes: a song's own break bar is replaced by another bar of the phrase ──
    let clip = [clip0(first_event), last_event.max(0.0) + opt.clip_after];
    let a_bed_until = match recipe {
        Recipe::Gap => a_until,
        _ => 0.0,
    };
    let a_segments = bed_segments(&sa, &ra, clip[0], first_event.min(-lead), a_bed_until, recipe != Recipe::Gap, &mut notes, "A");
    let b_segments = if b_side.segments.is_empty() && recipe == Recipe::Sneak {
        bed_segments(&sb, &rb, b_from, b_from, 0.0, true, &mut notes, "B")
    } else if recipe == Recipe::Sneak {
        // B's beat sneaks in from bars it plays twice: none of them may be a break.
        fill_holes(&sb, &rb, &b_side.segments, b_from, 0.0, &mut notes)
    } else {
        b_side.segments.clone()
    };

    // ── Levels: B must not land weaker than A's bed ──
    let a_level = mean_db(&sa, &ra, -lead, 0.0, F_MIX);
    let b_level = mean_db(&sb, &rb, 0.0, 4.0, F_MIX);
    let lift = f64::from((a_level - b_level - 1.0).clamp(0.0, 3.0));
    // The lift is gone again before the clip ends: after it the song plays as it is.
    let lift_until = last_event.max(0.0) + 1.0;
    let b_gain: Points = if lift > 0.4 { vec![[b_from, lift], [lift_until, lift], [(clip[1] - 0.5).max(lift_until + 1.0), 0.0]] } else { Vec::new() };
    if lift > 0.4 {
        notes.push(format!("song B lands {:.1} dB below A's bed, lifted by {lift:.1} dB", a_level - b_level));
    }

    // ── Where the beats of both songs are compared ──
    let align = align_windows(&sa, &sb, &ra, &rb, recipe, lead);

    let a_anchor = anchor(&sa, a_side.one, bpb_a);
    let b_anchor = anchor(&sb, b_side.one, bpb_b);
    let a_end = sa.beat_end_bar();
    let a_first = sa.first_beat_bar().unwrap_or(0);
    let one_s = beat_time(a, a_side.one as f64);
    let spec = Spec {
        id: opt.id.clone(),
        clip,
        first_event,
        last_event,
        swap_u,
        no_sync: fit.lock == Lock::None,
        align_pairs: align,
        recipe: Some(recipe),
        a: Deck {
            id: a.id.clone(),
            anchor_bar: a_anchor.0,
            anchor_u: a_anchor.1,
            // The table's bar lines are the corrected ones already.
            beat_shift: 0,
            fit_bars: Some({
                // The last bars of A through which its beat runs at the tempo of its end.
                let last = a_anchor.0.min(a_end.saturating_sub(1)).max(a_first);
                sa.steady_tempo_run(last, FIT_BARS, false, bpm_a)
            }),
            beats_per_bar: fit.bpb_a,
            tempo_bpm: None,
            until: Some(a_until),
            from: None,
            start_u: None,
            segments: a_segments,
            lanes: a_lanes,
            shape: Shape::Lin,
            shapes: BTreeMap::new(),
            hpf: a_hpf,
            lpf: BTreeMap::new(),
            sends: a_sends,
            gain_db: Vec::new(),
            trim_db: 0.0,
            nudge_ms: 0.0,
            auto_nudge: true,
            double: fit.bpb_a == 2,
        },
        b: Deck {
            id: b.id.clone(),
            anchor_bar: b_anchor.0,
            anchor_u: b_anchor.1,
            beat_shift: 0,
            fit_bars: b_side.fit_bars.map(|f| sb.steady_tempo_run(f[0], FIT_BARS, true, bpm_b)),
            beats_per_bar: fit.bpb_b,
            tempo_bpm: None,
            until: None,
            from: Some(b_from),
            start_u: (fit.lock == Lock::None).then_some(b_from),
            segments: b_segments,
            lanes: b_lanes,
            shape: Shape::Lin,
            shapes: BTreeMap::new(),
            hpf: b_hpf,
            lpf: BTreeMap::new(),
            sends: Vec::new(),
            gain_db: b_gain,
            trim_db: 0.0,
            nudge_ms: 0.0,
            auto_nudge: true,
            double: fit.bpb_b == 2,
        },
        technique: technique.into(),
        why: notes.join(" · "),
        answer: String::new(),
        key_txt: format!("{} → {}", a.key, b.key),
        steps,
        fx: serde_json::Value::Array(Vec::new()),
    };
    Ok(Plan { spec, recipe, fit, key_distance, confidence: confidence.clamp(0.0, 1.0), a_one_beat: a_side.one, b_one_beat: b_side.one, bar_s, one_s, tts: tts_slot, notes })
}

fn step(u: f64, text: &str) -> Step {
    Step { u, text: text.to_owned() }
}

/// The bar of the analysis the spec hangs on: the last one at or before the one that exists
/// in the song, and where it sits on the ruler.
fn anchor(s: &Song<'_>, one: i64, bpb: i64) -> (usize, f64) {
    let n = s.bars();
    let bar = (0..n).rev().find(|&k| s.bar_beat(k) as i64 <= one).unwrap_or(0);
    (bar, (s.bar_beat(bar) as i64 - one) as f64 / bpb as f64)
}

/// Mean level (dB, as power) of a feature over a stretch of the ruler.
fn mean_db(s: &Song<'_>, r: &Ruler, u0: f64, u1: f64, feature: usize) -> f32 {
    let (b0, b1) = (r.beat_of(u0).round() as i64, r.beat_of(u1).round() as i64);
    let levels: Vec<f32> = (b0..b1).map(|i| s.beat_level(i, feature)).collect();
    if levels.is_empty() {
        return -60.0;
    }
    let p = levels.iter().map(|d| 10f32.powf(d / 10.0)).sum::<f32>() / levels.len() as f32;
    10.0 * p.log10()
}

/// How loud B's high-passed drums have to be for the tease to be heard under A
/// (about 10 dB below A's mix when it is loudest).
fn tease_gain(sa: &Song<'_>, sb: &Song<'_>, ra: &Ruler, rb: &Ruler) -> f64 {
    let a_mix = mean_db(sa, ra, -8.0, -4.0, F_MIX);
    // What is left of the drums above the filter is roughly their hats plus the snare's top.
    let b_highs = mean_db(sb, rb, -8.0, -4.0, F_HATS).max(mean_db(sb, rb, 0.0, 4.0, F_HATS)) + 4.0;
    let gain = 10f64.powf(f64::from(a_mix - 10.0 - b_highs) / 20.0);
    gain.clamp(0.9, 2.4)
}

/// From where (ruler) B's voice stays quiet until the one, if it does.
fn voice_quiet_from(sb: &Song<'_>, rb: &Ruler, from_u: f64) -> Option<f64> {
    let one = rb.one;
    let first = rb.beat_of(from_u).round() as i64;
    // A pickup on the last beat before the one belongs to the line after it.
    let last_sung = (first..one - 1).rev().find(|&i| sb.sung(i))?;
    let quiet_beats = one - 1 - last_sung;
    (quiet_beats >= 2 * rb.bpb).then(|| ((last_sung + 1 - one) as f64 / rb.bpb as f64).ceil())
}

fn a_side(sa: &Song<'_>, recipe: Recipe, lead: f64, talk: f64, bpb: i64, clash: bool, opt: &PlanOptions, notes: &mut Vec<String>) -> Result<ASide> {
    let t = sa.t;
    // A's beat has to carry the bed, so the bed ends with the beat; when B brings its own beat the
    // song may play to where its content ends.
    let end_bar = match recipe {
        Recipe::VoiceOverBeat | Recipe::OutroUnderIntro => sa.beat_end_bar(),
        Recipe::Sneak | Recipe::Gap => sa.content_end_bar().max(sa.beat_end_bar()),
    };
    let end_beat = sa.bar_beat(end_bar) as i64;
    let lead_beats = ((lead + if recipe == Recipe::VoiceOverBeat { talk } else { 0.0 }) * bpb as f64).round() as i64;
    let in_window = |one: i64| match opt.window {
        Some((t0, t1)) => {
            let first = beat_time(t, (one - lead_beats) as f64);
            let at = beat_time(t, one as f64);
            first >= t0 && at <= t1
        }
        None => true,
    };
    // Bar starts that may carry the one.
    let bar_starts: Vec<i64> = (0..sa.bars() + 64).map(|k| sa.bar_beat(k) as i64).collect();
    let next_bar_start = |beat: f64| bar_starts.iter().copied().find(|&b| b as f64 >= beat - 0.26);

    let one = match recipe {
        Recipe::VoiceOverBeat | Recipe::OutroUnderIntro if opt.window.is_none() => {
            let gaps: Vec<f64> = if recipe == Recipe::VoiceOverBeat { vec![lead + talk] } else { vec![lead, (lead / 2.0).max(2.0)] };
            if let Some(c) = exit_and_one(sa, &gaps, bpb) {
                let r = Ruler { one: c.one, bpb };
                notes.push(format!(
                    "song A's voice leaves at {:.2} s ({}), the one is at {:.2} s, {} bar(s) after A's own beat is over",
                    beat_time(t, c.exit_beat),
                    c.kind,
                    beat_time(t, c.one as f64),
                    c.past_end
                ));
                return Ok(ASide { one: c.one, exit: Some((r.u_of_beat(c.exit_beat), c.kind != "bar line")) });
            }
            notes.push("song A has nowhere for its voice to leave; it ends on its last bar".into());
            return Ok(ASide { one: sa.bar_beat(sa.snap_to_phrase(sa.full_end_bar())) as i64, exit: None });
        }
        Recipe::VoiceOverBeat | Recipe::OutroUnderIntro => {
            // A's voice leaves on the end of a line, then A's beat carries alone for `lead` bars.
            let sung_late = (end_bar.saturating_sub(8)..end_bar).any(|k| sa.sung_share(k) >= 0.25);
            let mut best: Option<(i64, usize)> = None;
            for &s in &t.line_ends {
                let line_over = (s + 1) as f64 / 4.0;
                let Some(bed) = next_bar_start(line_over) else { continue };
                let one = bed + lead_beats;
                if !in_window(one) {
                    continue;
                }
                // Offline the song plays as long as it can: the one lies within its last bars
                // (the bed may run up to two bars past the beat, those are replayed). Live anything in the window is fine.
                let fits = opt.window.is_some() || (one <= end_beat + 2 * bpb && one >= end_beat - 8 * bpb);
                if fits && best.is_none_or(|(o, _)| if opt.window.is_some() { one < o } else { one > o }) {
                    best = Some((one, s));
                }
            }
            match best {
                Some((one, s)) => {
                    let r = Ruler { one, bpb };
                    notes.push(format!("song A's voice leaves on the line end at {:.2} s, the one is at {:.2} s", beat_time(t, (s + 1) as f64 / 4.0), beat_time(t, one as f64)));
                    return Ok(ASide { one, exit: Some((r.u_of_line_end(s), true)) });
                }
                None if !sung_late && opt.window.is_none() => {
                    notes.push("song A ends without voice, its last bars are the bed".into());
                    return Ok(ASide { one: end_beat, exit: None });
                }
                None => {
                    // No line end where it is needed: leave on a bar line, the bed is replayed if the song is over.
                    let one = match opt.window {
                        Some(_) => bar_starts.iter().copied().find(|&b| in_window(b)),
                        None => Some(end_beat),
                    };
                    let Some(one) = one else { bail!("no bar line for the one in the window") };
                    notes.push("song A has no line end where the voice has to leave, it leaves on the bar line".into());
                    return Ok(ASide { one, exit: Some((-(lead + talk), false)) });
                }
            }
        }
        Recipe::Sneak => match opt.window {
            Some(_) => bar_starts.iter().copied().filter(|&b| in_window(b)).find(|&b| sa.is_phrase_start(sa.t.bar_of(b.max(0) as usize), 4)).or_else(|| bar_starts.iter().copied().find(|&b| in_window(b))),
            // Where A's own beat is over, on its phrase: A's beat fades over its last full bars. When A
            // breaks down within those bars and comes back, the one comes before the break; replaying earlier
            // bars over it is heard as a cut (the owner, x30: "there's a cut in A at -8").
            None => {
                let mut one = sa.snap_to_phrase(sa.full_end_bar());
                let fade = (lead * bpb as f64 / 4.0).ceil() as usize;
                if let Some(last_gap) = (one.saturating_sub(fade)..one).rev().find(|&k| !sa.carries(k)) {
                    let clean_after = one.saturating_sub(last_gap + 1);
                    if clean_after < 4 || is_rising_buildup(sa, one) {
                        if let Some(first_gap) = (one.saturating_sub(fade)..one).find(|&k| !sa.carries(k)) {
                            if first_gap >= 16 {
                                let mut cand = sa.snap_to_phrase(first_gap);
                                if (cand.saturating_sub(fade)..cand).any(|k| !sa.carries(k)) {
                                    cand = first_gap;
                                }
                                if is_rising_buildup(sa, cand) && first_gap >= 20 {
                                    cand = sa.snap_to_phrase(first_gap.saturating_sub(2));
                                }
                                if !is_rising_buildup(sa, cand) {
                                    one = cand;
                                }
                            }
                        }
                    }
                }
                if is_rising_buildup(sa, one) {
                    if let Some(after) = (one + 4..=sa.bars()).find(|&k| sa.is_phrase_start(k, 4) && !is_rising_buildup(sa, k)) {
                        one = after;
                    }
                }
                Some(sa.bar_beat(one) as i64)
            }
        },
        Recipe::Gap => match opt.window {
            Some(_) => bar_starts.iter().copied().find(|&b| in_window(b)),
            // The last phrase start whose two bars before it still carry A's beat: the hook plays alone
            // there, so A must still be at full strength.
            None => {
                let fe = sa.full_end_bar();
                (fe.saturating_sub(16)..=fe)
                    .rev()
                    .find(|&k| k >= 2 && sa.is_phrase_start(k, 8) && sa.full(k - 1) && sa.full(k - 2))
                    .or_else(|| (end_bar.saturating_sub(8)..=end_bar).rev().find(|&k| sa.is_phrase_start(k, 8)))
                    .map(|k| sa.bar_beat(k) as i64)
                    .or(Some(end_beat))
            }
        },
    };
    let Some(one) = one else { bail!("no bar line for the one in the window") };
    let r = Ruler { one, bpb };
    // Where A's voice has to be gone at the latest.
    let latest = match recipe {
        Recipe::Sneak if talk > 0.0 => -talk,
        Recipe::Gap if talk > 0.0 => -talk,
        _ => 0.0,
    };
    let latest_beat = r.beat_of(latest);
    // When B's beat has sneaked in, A's last line may end just after the one and ring out over B's
    // first bar, rather than be cut inside a word.
    let past = if recipe == Recipe::Sneak && latest == 0.0 && !clash { 0.75 * bpb as f64 } else { 1.0 };
    let exit = t
        .line_ends
        .iter()
        .copied()
        .filter(|&s| {
            let over = (s + 1) as f64 / 4.0;
            over <= latest_beat + past && over >= latest_beat - 2.0 * bpb as f64
        })
        .next_back()
        .map(|s| (r.u_of_line_end(s), true));
    // No line ends there: a breath will do, before cutting the voice inside a word (the owner, x28:
    // "kind of ending abrupt since A is mid in singing").
    let exit = exit.or_else(|| {
        sa.breaths_from((r.beat_of(latest).max(0.0) as usize).saturating_sub((2 * bpb) as usize))
            .into_iter()
            .map(|s| s as f64 / 4.0)
            .filter(|&b| b <= latest_beat + past && b >= latest_beat - 2.0 * bpb as f64)
            .next_back()
            .map(|b| (r.u_of_beat(b), true))
    });
    let sings = (one - 2 * bpb..one).any(|i| sa.sung(i));
    let exit = match (exit, recipe) {
        (Some(e), _) => Some(e),
        (None, Recipe::Gap) if talk == 0.0 => None,
        (None, _) if sings || talk > 0.0 || clash => Some((latest, false)),
        (None, _) => None,
    };
    notes.push(format!("the one is at {:.2} s of song A{}", beat_time(t, one as f64), match exit { Some((u, true)) => format!(", its voice leaves on a line end at {u:+.2} bars"), Some((u, false)) => format!(", its voice has to leave at {u:+.2} bars without a line end"), None => String::new() }));
    Ok(ASide { one, exit })
}

/// B's voice may come in this many bars before the one, with the first words of its line (a pickup;
/// the owner: "would be cool if it started 1 sec earlier because it started mid word").
const PICKUP_BARS: f64 = 1.0;

/// Bars a deck's tempo is fitted on: those right where it plays in the blend, since some songs are
/// played by hand and their tempo wanders over longer stretches.
const FIT_BARS: usize = 8;

/// A bar line is a place for A's voice to leave when its voice there is this far under its typical level.
const QUIET_VOICE_DB: f32 = 3.0;

struct ExitAndOne {
    one: i64,
    exit_beat: f64,
    kind: &'static str,
    past_end: i64,
}

/// Offline, when song A's beat carries song B's intro: where A's voice leaves and where the one is.
///
/// A's voice leaves at the end of a line, at a breath, or (with an echo) on a phrase start. The one
/// follows `gap` ruler bars later. Scored as the owner's accepted transitions do it: a real end of a
/// line over a cut, the one on A's phrase grid, as much of A as its beat carries, and as few bars
/// replayed past its end as possible.
fn exit_and_one(sa: &Song<'_>, gaps: &[f64], bpb: i64) -> Option<ExitAndOne> {
    let t = sa.t;
    let fe = sa.full_end_bar();
    let target = sa.snap_to_phrase(fe) as i64;
    let lo = fe.saturating_sub(24);
    let bar_of = |beat: f64| t.bar_of(beat.max(0.0) as usize);
    let mut cands: Vec<(f64, &'static str)> = Vec::new();
    for &s in &t.line_ends {
        let e = (s + 1) as f64 / 4.0;
        if bar_of(e) >= lo && sa.carries(bar_of(e)) {
            cands.push((e, "end of a line"));
        }
    }
    for s in sa.breaths_from(sa.bar_beat(lo)) {
        let e = s as f64 / 4.0;
        if sa.carries(bar_of(e)) {
            cands.push((e, "breath"));
        }
    }
    // A voice is cut on a bar line only where it is quiet: the owner never accepted a voice cut while it sang.
    let typical_voice = sa.typical(view::F_VOCALS);
    for k in lo..=fe {
        if sa.is_phrase_start(k, 4) && sa.level(k, view::F_VOCALS) < typical_voice - QUIET_VOICE_DB {
            cands.push((sa.bar_beat(k) as f64, "bar line"));
        }
    }
    let phrase_points = |k: i64| -> f64 {
        if k < 0 {
            0.0
        } else if sa.is_phrase_start(k as usize, 8) {
            2.0
        } else if sa.is_phrase_start(k as usize, 4) {
            1.0
        } else {
            0.0
        }
    };
    let mut best: Option<(f64, ExitAndOne)> = None;
    for &(e, kind) in &cands {
        // The first bar line at or after the exit (an exit a sixteenth before a bar line counts for that bar).
        let Some(k_e) = (lo..sa.bars() + 16).find(|&k| sa.bar_beat(k) as f64 >= e - 0.26) else { continue };
        for &base_gap in gaps {
            let mut try_gaps = vec![base_gap];
            if base_gap >= 5.0 && (base_gap - 1.0 >= 4.0) {
                try_gaps.push(base_gap - 1.0);
            }
            for gap in try_gaps {
                let o_bar = k_e as i64 + (gap * bpb as f64 / 4.0).round() as i64;
                if o_bar > fe as i64 + 8 || o_bar + 16 < fe as i64 {
                    continue;
                }
                let mut score = match kind {
                    "end of a line" => 3.0,
                    "breath" => 2.5,
                    _ => phrase_points(k_e as i64),
                };
                score += phrase_points(o_bar);
                if (gap - base_gap).abs() > 1e-6 {
                    score -= 0.8;
                }
                if o_bar > fe as i64 {
                    // Strongly avoid running past `full_end_bar`, which forces a backward drum-loop jump!
                    score -= 5.5 * (o_bar - fe as i64) as f64;
                }
                if o_bar >= 0 && is_rising_buildup(sa, o_bar as usize) {
                    score -= 8.0;
                }
                score -= 3.5 * (o_bar - target).max(0) as f64 + 1.5 * (target - o_bar).max(0) as f64;
                if best.as_ref().is_none_or(|(b, _)| score > *b) {
                    let one = sa.bar_beat(o_bar.max(0) as usize) as i64;
                    best = Some((score, ExitAndOne { one, exit_beat: e, kind, past_end: o_bar - fe as i64 }));
                }
            }
        }
    }
    best.map(|(_, c)| c)
}

/// Where B's voice comes in when it is open from `b_from`: there, when nobody sings or B takes a breath there;
/// otherwise with a line that began up to a bar earlier (`line`), or at the breath nearest to `b_from`
/// (preferably before it), or where B is quietest. Not before `not_before` (A's voice). The owner, x25:
/// "cuts in while in speak. if it started one sec before and not in the middle of the word it would fit".
fn b_voice_entry(sb: &Song<'_>, rb: &Ruler, b_from: f64, line: Option<f64>, not_before: f64) -> f64 {
    let level = |s: usize| sb.t.v16.get(s / 4).map(|r| r[s % 4]);
    let s16 = |u: f64| (rb.beat_of(u) * 4.0).floor();
    let at = s16(b_from);
    if at < 0.0 || level(at as usize).is_none_or(|l| l <= view::BREATH_DB) {
        // Nobody sings at `b_from`: keep B's vocal stem gated at 0.0 until right before B's actual first
        // vocal line (mix_010: eliminates separated-stem synth/pad fuzz during pre-vocal bars!).
        if let Some(v) = line.filter(|&v| v > b_from + 0.12 && v < 0.0) {
            return (v - 0.05).max(b_from);
        }
        let next_s = sb.t.line_starts.iter().copied().find(|&s| s as f64 > at + 2.0);
        if let Some(s) = next_s {
            let u_s = rb.u_of_line_start(s);
            if u_s > b_from + 0.12 && u_s < 0.0 {
                return (u_s - 0.05).max(b_from);
            }
        }
        return b_from;
    }
    let lo = (b_from - PICKUP_BARS).max(not_before);
    if let Some(v) = line.filter(|&v| v >= lo && v <= b_from) {
        return v;
    }
    let first = s16(lo).ceil().max(0.0) as usize;
    let last = s16(b_from + 1.0) as usize;
    let cands: Vec<(usize, f32)> = (first..last).filter_map(|s| level(s).map(|l| (s, l))).collect();
    let dist = |s: usize| { let d = s as f64 - at; if d < 0.0 { -d } else { d + 0.5 } };
    let pick = cands.iter().filter(|c| c.1 <= view::BREATH_DB).min_by(|x, y| dist(x.0).total_cmp(&dist(y.0)))
        .or_else(|| cands.iter().min_by(|x, y| x.1.total_cmp(&y.1)));
    pick.map_or(b_from, |&(s, _)| rb.u_of_line_start(s).max(lo))
}

fn b_side(sb: &Song<'_>, recipe: Recipe, opening: Opening, lead: f64, bpb: i64, clash: bool, notes: &mut Vec<String>) -> BSide {
    let lead_beats = (lead * bpb as f64).round() as i64;
    // Ruler bars of B per bar of B.
    let per_bar = 4.0 / bpb as f64;
    let steady = |beat_bar: usize| Some(sb.steady_run(beat_bar, 24, true));
    let whole_bars_before = |one: i64| -> f64 { ((one as f64 / bpb as f64).floor()).min(lead) };
    // Where B's voice comes in, not before a ruler position: with a line. The answer says whether
    // it is a line's start.
    let voice_from = |one: i64, from_beat: i64, not_before_u: f64| -> (Option<f64>, bool) {
        let r = Ruler { one, bpb };
        let earliest = r.beat_of(not_before_u).max(from_beat as f64).max(0.0);
        let first = (earliest * 4.0).floor() as usize;
        let window = (2 * bpb * 4) as usize;
        // A line that starts within the next two bars; but a line already running that began at most a bar
        // earlier comes in whole as a pickup, rather than the voice waiting for the next one.
        let next = sb.t.line_starts.iter().copied().find(|&s| s >= first);
        let pickup = sb.t.line_starts.iter().copied().filter(|&s| s < first && s as f64 >= first as f64 - PICKUP_BARS * (bpb * 4) as f64).next_back();
        if let (Some(p), Some(n)) = (pickup, next) {
            if n > first + 1 && (first..n).any(|s| sb.voiced16(s)) {
                return (Some(r.u_of_line_start(p)), true);
            }
        }
        if let Some(s) = next.filter(|&s| s <= first + window) {
            return (Some(r.u_of_line_start(s)), true);
        }
        if !(first..first + window).any(|s| sb.voiced16(s)) {
            // Nobody sings there; the voice is open for whenever its next line comes.
            return (Some(not_before_u), true);
        }
        // Somebody is singing already. A line that began up to a bar earlier comes in whole.
        let running = sb.t.line_starts.iter().copied().filter(|&s| s < first && s + (bpb * 4) as usize >= first).next_back();
        if let Some(s) = running {
            return (Some(r.u_of_line_start(s)), true);
        }
        // Otherwise the voice comes in where it takes a breath, or, if it takes none, where it has to.
        match (first.max(1)..first + window).find(|&s| !sb.voiced16(s - 1) && sb.voiced16(s)) {
            Some(s) => (Some(r.u_of_line_start(s)), true),
            None => (Some(not_before_u), false),
        }
    };
    let loop_back = |bar: usize, notes: &mut Vec<String>| -> BSide {
        let one = sb.bar_beat(bar) as i64;
        notes.push(format!("song B's first {lead} bars with beat play twice: first without voice and melody under A, then for real from the one"));
        let (voice_u, on_line) = voice_from(one, one, 0.0);
        BSide { one, from_u: -lead, segments: vec![Segment { u0: -lead, u1: Some(0.0), src: 0.0 }, Segment { u0: 0.0, u1: None, src: 0.0 }], voice_u, voice_on_line: on_line, fit_bars: steady(bar) }
    };
    // B's beat runs from its start: the one is a phrase start with a whole blend of B's own beat before
    // it (B's voice waits for the one), unless B's voice comes so soon that its beat bars are looped.
    if let (Recipe::Sneak, Opening::BeatFirst { beat_bar, .. } | Opening::Together { bar: beat_bar }) = (recipe, opening) {
        let lead_bars = (lead * bpb as f64 / 4.0).ceil() as usize;
        let one_bar = match sb.voice_bar() {
            Some(v) if v <= beat_bar + 1 => Some(sb.phrase_start_from(beat_bar + lead_bars)),
            Some(v) if v >= beat_bar + 4 => Some(sb.phrase_start_from((beat_bar + lead_bars).max(v))),
            None => Some(sb.phrase_start_from(beat_bar + lead_bars)),
            Some(_) => None,
        };
        if let Some(one_bar) = one_bar.filter(|&k| k >= lead_bars && (k - lead_bars..k).all(|b| sb.carries(b))) {
            let one = sb.bar_beat(one_bar) as i64;
            notes.push(format!("song B's beat runs from bar {beat_bar}; its one is the phrase start at bar {one_bar}, its own bars before it sneak in"));
            let (voice_u, on_line) = voice_from(one, one, 0.0);
            return BSide { one, from_u: -lead, segments: Vec::new(), voice_u, voice_on_line: on_line, fit_bars: steady(beat_bar) };
        }
    }
    match (recipe, opening) {
        (Recipe::Gap, o) => {
            // After a tempo jump B has to land with its beat, not in a quiet intro.
            let bar = match o {
                Opening::NoBeat => 0,
                Opening::VoiceFirst { beat_bar } | Opening::MelodyFirst { beat_bar } | Opening::BeatFirst { beat_bar, .. } | Opening::Together { bar: beat_bar } => beat_bar,
            };
            let one = sb.bar_beat(bar) as i64;
            BSide { one, from_u: 0.0, segments: Vec::new(), voice_u: None, voice_on_line: true, fit_bars: sb.first_beat_bar().and_then(steady) }
        }
        (Recipe::VoiceOverBeat | Recipe::OutroUnderIntro, Opening::VoiceFirst { beat_bar } | Opening::MelodyFirst { beat_bar }) => {
            // Land on the true downbeat kick/drum drop (requiring beat 1 itself to carry drums, never a beat-4 pickup!).
            let downbeat_ok = |k: usize| {
                k < sb.bars()
                    && sb.full(k)
                    && sb.has_beat(k)
                    && sb.beat_level(sb.bar_beat(k) as i64, F_DRUMS) > -15.0
            };
            let early_drop = (1..beat_bar.min(sb.bars().saturating_sub(2)))
                .find(|&k| downbeat_ok(k) && sb.full(k + 1));
            let drop_bar = early_drop.unwrap_or_else(|| {
                (beat_bar..beat_bar + 6)
                    .find(|&k| downbeat_ok(k))
                    .unwrap_or(beat_bar)
            });
            let one = sb.bar_beat(drop_bar) as i64;
            let have = if recipe == Recipe::VoiceOverBeat {
                let vb = sb.voice_bar().unwrap_or(0).min(drop_bar);
                ((one - sb.bar_beat(vb) as i64) as f64 / bpb as f64).floor().min(lead)
            } else {
                whole_bars_before(one)
            };
            let from_u = -have.max(1.0).min(lead);
            let from_beat = one + (from_u * bpb as f64).round() as i64;
            let mut segments = Vec::new();
            if have < lead && recipe == Recipe::OutroUnderIntro && have >= 1.0 {
                // A short intro is played as often as it takes to fill the blend.
                let piece = if have >= 4.0 { 4.0 } else if have >= 2.0 { 2.0 } else { 1.0 };
                let mut u = -lead;
                while u < -piece - 1e-9 {
                    segments.push(Segment { u0: u, u1: Some(u + piece), src: -piece });
                    u += piece;
                }
                segments.push(Segment { u0: -piece, u1: None, src: -piece });
                notes.push(format!("song B's intro is {have} bars long, its last {piece} play {} times", (lead / piece) as usize));
                let (voice_u, on_line) = voice_from(one, one - (piece * bpb as f64) as i64, -0.25);
                return BSide { one, from_u: -lead, segments, voice_u, voice_on_line: on_line, fit_bars: steady(drop_bar) };
            }
            let (voice_u, on_line) = if recipe == Recipe::VoiceOverBeat { voice_from(one, from_beat, from_u) } else { voice_from(one, one - bpb, -0.25) };
            BSide { one, from_u, segments, voice_u, voice_on_line: on_line, fit_bars: steady(drop_bar) }
        }
        (Recipe::Sneak, Opening::BeatFirst { beat_bar, voice_bar }) if !clash || voice_bar > beat_bar => {
            let beat_only = (voice_bar - beat_bar) as f64 * per_bar;
            let one = sb.bar_beat(voice_bar) as i64;
            if beat_only >= lead {
                let (voice_u, on_line) = voice_from(one, one, 0.0);
                return BSide { one, from_u: -lead, segments: Vec::new(), voice_u, voice_on_line: on_line, fit_bars: steady(beat_bar) };
            }
            // Too few bars of beat before the voice: they are looped into an intro.
            let piece = if beat_only >= 4.0 { 4.0 } else if beat_only >= 2.0 { 2.0 } else { 1.0 };
            let mut segments = Vec::new();
            let mut u = -lead;
            while u < -piece - 1e-9 {
                segments.push(Segment { u0: u, u1: Some(u + piece), src: -piece });
                u += piece;
            }
            segments.push(Segment { u0: -piece, u1: None, src: -piece });
            notes.push(format!("song B has {beat_only} bars of beat before its voice, the last {piece} are looped into an intro of {lead}"));
            let (voice_u, on_line) = voice_from(one, one - (piece * bpb as f64) as i64, -0.5);
            BSide { one, from_u: -lead, segments, voice_u, voice_on_line: on_line, fit_bars: steady(beat_bar) }
        }
        (Recipe::VoiceOverBeat, Opening::BeatFirst { beat_bar, .. } | Opening::Together { bar: beat_bar }) => {
            // B's voice alone rides on A's beat: the one is a phrase start after B starts singing where B's
            // own beat plays at full strength (kick + drums, never a quiet riser or breakdown), and B's voice
            // enters right at the blend start.
            let lead_bars = (lead * bpb as f64 / 4.0).ceil() as usize;
            let voice_bar = sb.voice_bar().unwrap_or(beat_bar);
            let default_one = sb.phrase_start_from((voice_bar + lead_bars).max(beat_bar + 2));
            let mut cands_one = vec![default_one];
            for span in [lead_bars, 8usize, 6, 4, 3] {
                let raw = (voice_bar + span).max(beat_bar.max(1));
                let p4 = (raw..raw + 4).find(|&k| sb.is_phrase_start(k, 4)).unwrap_or(raw);
                cands_one.push(p4);
                cands_one.push(raw);
            }
            let one_bar = cands_one
                .iter()
                .copied()
                .find(|&k| k + 1 < sb.bars() && sb.full(k) && sb.has_beat(k) && sb.carries(k + 1) && k > voice_bar + 1)
                .or_else(|| {
                    cands_one
                        .into_iter()
                        .find(|&k| k + 1 < sb.bars() && sb.carries(k) && sb.carries(k + 1) && k > voice_bar + 1)
                })
                .unwrap_or(default_one);
            let actual_from = -(((one_bar.saturating_sub(voice_bar)) as f64) * per_bar).clamp(2.0, lead);
            let actual_beats = (-actual_from * bpb as f64).round() as i64;
            let one = sb.bar_beat(one_bar) as i64;
            notes.push(format!("song B's voice alone rides on A's beat for {:.1} bars from bar {voice_bar}; B's beat comes on the one, its bar {one_bar}", -actual_from));
            let (voice_u, on_line) = voice_from(one, one - actual_beats, actual_from);
            BSide { one, from_u: actual_from, segments: Vec::new(), voice_u, voice_on_line: on_line, fit_bars: steady(one_bar) }
        }
        (_, Opening::BeatFirst { beat_bar, .. } | Opening::Together { bar: beat_bar } | Opening::VoiceFirst { beat_bar } | Opening::MelodyFirst { beat_bar }) => {
            let _ = lead_beats;
            loop_back(beat_bar, notes)
        }
        (_, Opening::NoBeat) => BSide { one: sb.bar_beat(0) as i64, from_u: 0.0, segments: Vec::new(), voice_u: None, voice_on_line: true, fit_bars: None },
    }
}

/// True when `bar` sits right at the end of a rising pre-drop buildup in song A (so cutting A at `bar`
/// would chop off A's buildup vocals/riser right before A's own drop).
fn is_rising_buildup(sa: &Song<'_>, bar: usize) -> bool {
    if bar < 2 || bar + 2 > sa.bars() {
        return false;
    }
    let mix_pre = 0.5 * (sa.level(bar - 2, F_MIX) + sa.level(bar - 1, F_MIX));
    let mix_post = 0.5 * (sa.level(bar, F_MIX) + sa.level(bar + 1, F_MIX));
    let drm_pre = 0.5 * (sa.level(bar - 2, F_DRUMS) + sa.level(bar - 1, F_DRUMS));
    let drm_post = 0.5 * (sa.level(bar, F_DRUMS) + sa.level(bar + 1, F_DRUMS));
    if sa.carries(bar) && (mix_post - mix_pre > 2.8 || drm_post - drm_pre > 5.0) {
        return true;
    }
    false
}

/// The segments of a deck whose beat has to carry from `bed_from` to `bed_to`: the song's own bars
/// until the first one that does not carry its beat (a break, a fill-out, or the song is over); from
/// there the bed goes on with the bars four (eight, twelve, sixteen) bars earlier, in one piece, as a
/// DJ loops the last full phrase.
#[allow(clippy::too_many_arguments)]
fn bed_segments(s: &Song<'_>, r: &Ruler, deck_from: f64, bed_from: f64, bed_to: f64, carries: bool, notes: &mut Vec<String>, name: &str) -> Vec<Segment> {
    if !carries || bed_to <= bed_from {
        return Vec::new();
    }
    let typical = s.typical(F_DRUMS);
    if typical < -30.0 {
        // This song has no beat to carry anything (a quiet ending): leave it as it is.
        return Vec::new();
    }
    let bars: Vec<f64> = {
        let mut v = Vec::new();
        let mut u = bed_from;
        while u < bed_to - 1e-9 {
            v.push(u);
            u += 1.0;
        }
        v
    };
    let exists = |u: f64| r.beat_of(u + 1.0) <= s.t.times.len() as f64 && r.beat_of(u) >= 0.0;
    let ok = |u: f64| exists(u) && mean_db(s, r, u, u + 1.0, F_DRUMS) >= typical - view::GONE_DB;
    let Some(first_bad) = bars.iter().position(|&u| !ok(u)) else { return Vec::new() };
    let from = bars[first_bad];
    let Some(back) = [4.0, 8.0, 12.0, 16.0].into_iter().find(|back| bars[first_bad..].iter().all(|&u| ok(u - back))) else {
        notes.push(format!("song {name}'s bed has bars without beat and no earlier phrase to replay"));
        return Vec::new();
    };
    notes.push(format!("song {name}'s beat is gone from {from:+} bars; from there its bars of {back} bars earlier are replayed"));
    let mut segments = Vec::new();
    if deck_from < from {
        segments.push(Segment { u0: deck_from, u1: Some(from), src: deck_from });
    }
    segments.push(Segment { u0: from, u1: None, src: from - back });
    segments
}

/// Where a deck plays from its segments, bars of `[from, to)` whose source bar does not carry the beat
/// are played from the bar four (eight) bars earlier of the source instead.
fn fill_holes(s: &Song<'_>, r: &Ruler, segments: &[Segment], from: f64, to: f64, notes: &mut Vec<String>) -> Vec<Segment> {
    let typical = s.typical(F_DRUMS);
    let src_of = |u: f64| -> f64 {
        segments.iter().rev().find(|g| g.u0 <= u + 1e-9 && g.u1.is_none_or(|u1| u < u1 - 1e-9)).map_or(u, |g| g.src + (u - g.u0))
    };
    let ok = |src: f64| r.beat_of(src) >= 0.0 && r.beat_of(src + 1.0) <= s.t.times.len() as f64 && mean_db(s, r, src, src + 1.0, F_DRUMS) >= typical - view::GONE_DB;
    // Per bar: where it plays from.
    let mut per_bar: Vec<(f64, f64)> = Vec::new();
    let mut u = from;
    let mut filled = 0;
    while u < to - 1e-9 {
        let src = src_of(u);
        let src = if ok(src) {
            src
        } else if let Some(back) = [4.0, 8.0].into_iter().find(|b| ok(src - b)) {
            filled += 1;
            src - back
        } else {
            src
        };
        per_bar.push((u, src));
        u += 1.0;
    }
    if filled == 0 {
        return segments.to_vec();
    }
    notes.push(format!("{filled} bar(s) of song B's intro would be a break and are played from four bars earlier"));
    // Rebuild: the bars of the blend one by one where they changed, then the segments from `to` on.
    let mut out: Vec<Segment> = Vec::new();
    for &(u, src) in &per_bar {
        match out.last_mut() {
            Some(prev) if (prev.src + (u - prev.u0) - src).abs() < 1e-9 => prev.u1 = Some(u + 1.0),
            _ => out.push(Segment { u0: u, u1: Some(u + 1.0), src }),
        }
    }
    for g in segments.iter().filter(|g| g.u1.is_none_or(|u1| u1 > to + 1e-9)) {
        let u0 = g.u0.max(to);
        out.push(Segment { u0, u1: g.u1, src: g.src + (u0 - g.u0) });
    }
    out
}

fn align_windows(sa: &Song<'_>, sb: &Song<'_>, ra: &Ruler, rb: &Ruler, recipe: Recipe, lead: f64) -> Option<AlignPairs> {
    let has_beat = |s: &Song<'_>, r: &Ruler, u: f64| mean_db(s, r, u, u + 1.0, view::F_KICK).max(mean_db(s, r, u, u + 1.0, F_DRUMS)) > -16.0 && r.beat_of(u) >= 0.0 && r.beat_of(u + 1.0) <= s.t.times.len() as f64;
    let window = |s: &Song<'_>, r: &Ruler, candidates: &[f64]| -> Option<[f64; 2]> {
        for len in [8.0, 6.0, 4.0] {
            for &from in candidates {
                let mut u = from;
                let mut all = true;
                while u < from + len - 1e-9 {
                    all &= has_beat(s, r, u);
                    u += 1.0;
                }
                if all {
                    return Some([from, from + len]);
                }
            }
        }
        None
    };
    // Without B's beat under A the beats meet only at the hard cut: compare the bars on both sides of it.
    if recipe != Recipe::Sneak {
        let seam = |s: &Song<'_>, r: &Ruler, from: f64| (0..4).all(|i| has_beat(s, r, from + f64::from(i)));
        if seam(sa, ra, -4.0) && seam(sb, rb, 0.0) {
            return Some(AlignPairs { a: [-4.0, 0.0], b: [0.0, 4.0] });
        }
    }
    let a = window(sa, ra, &[-lead, -lead - 4.0, -lead - 8.0, -16.0, -24.0])?;
    let b_candidates: &[f64] = if recipe == Recipe::Sneak { &[-lead, 0.0, 4.0, 8.0] } else { &[0.0, 4.0, 8.0, 16.0] };
    let b = window(sb, rb, b_candidates)?;
    let len = (a[1] - a[0]).min(b[1] - b[0]);
    // Melody level is not needed here but the borrow keeps the feature list in one place.
    let _ = F_OTHER;
    Some(AlignPairs { a: [a[0], a[0] + len], b: [b[0], b[0] + len] })
}
