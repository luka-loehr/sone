use std::time::Duration;

use anyhow::Result;
use sone_core::{
    BeatTable, Event, SAMPLE_RATE, Stats, Stem, TtsSlot,
    event::{Keys, Tempo},
};
use sone_plan::{Plan, PlanOptions, keys, lanes, plan, tts::MARGIN_S};

use crate::talk::Talk;

const RATE: f64 = SAMPLE_RATE as f64;

/// A song as the engine reads it: its audio by position and what is known about it.
pub trait Song {
    /// Frames of the whole song.
    fn frames(&self) -> usize;
    /// Frames from `from` on; what lies past the end is silence.
    fn read(&self, from: usize, out: &mut [[f32; 2]]);
    fn table(&self) -> &BeatTable;
    /// The gain the station plays this song at (its loudness against the station's).
    /// The renderer has to use the same one inside the clip.
    fn gain(&self) -> f32 {
        1.0
    }
}

/// A rendered transition.
pub struct Clip {
    pub audio: Vec<[f32; 2]>,
    /// Frame of song A at which the clip takes over.
    pub a_from: usize,
    /// Frame of song B (its own time) that follows the clip's last frame.
    pub b_resume: f64,
    /// Output seconds per second of song B inside the clip.
    pub ratio: f64,
    /// Offset between the two songs' drum attacks as measured by the renderer, milliseconds.
    pub beat_offset_ms: f32,
    /// Largest dip of the mix below song A's level inside the clip, dB.
    pub level_dip_db: f32,
    /// Frames at the clip's end that are blended with the song that follows. `b_resume` is the
    /// frame of song B that belongs to the first of them.
    pub seam: usize,
}

pub trait Renderer {
    /// Renders the plan's clip from the stems of both songs.
    fn render(&mut self, plan: &Plan) -> Result<Clip>;
}

/// Plays a song at a tempo that may change slowly, pitch unchanged. Every moment of the song has to
/// come out where the ratios put it: an output frame made at ratio `r` moves the song on by `1 / r`
/// frames. The engine counts on that to know where in the song the glide is.
pub trait Glide {
    /// Output seconds per second of the song (1 is the song's own tempo).
    fn set_ratio(&mut self, ratio: f64);
    /// Takes frames of the song, returns what is ready.
    fn process(&mut self, input: &[[f32; 2]]) -> Vec<[f32; 2]>;
    /// A fresh glide starting at a ratio.
    fn restart(&mut self, ratio: f64);
}

#[derive(Debug, Clone)]
pub struct Config {
    /// How far ahead of the play position the song on air may be looked at, seconds.
    pub lookahead_s: f64,
    /// Bars over which the next song returns to its own tempo after the transition.
    pub glide_bars: f64,
    /// A plan has to leave this much time for the rendering, seconds.
    pub render_margin_s: f64,
    pub max_stretch_pct: f64,
    /// The audio of a plan is rendered somewhere else (another Worker): the engine announces the
    /// plan, hands it out with `take_requests` and goes on playing until `deliver` brings the clip.
    pub render_outside: bool,
    /// A line for the DJ is asked for this long before its slot begins (the voice has to be made first).
    pub talk_lead_s: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self { lookahead_s: 30.0, glide_bars: 16.0, render_margin_s: 1.0, max_stretch_pct: 4.0, render_outside: false, talk_lead_s: 3.0 }
    }
}

enum Phase {
    /// The song on air plays as it is.
    Song,
    /// A transition is rendered and waits for its first frame.
    Planned,
    /// The rendered transition plays.
    Clip { at: usize },
    /// The next song plays, still returning to its own tempo.
    Gliding {
        pending: Vec<[f32; 2]>,
        /// Next frame of the song the glide is given.
        read: usize,
        /// Where in the song the next frame the glide makes belongs.
        at: f64,
        /// Frames the glide makes before the song's first frame that is played (it runs in first).
        skip: usize,
        from_out: u64,
        until_out: u64,
        from_ratio: f64,
    },
}

/// The end of what played before, blended into what plays now.
struct Seam {
    tail: Vec<[f32; 2]>,
    at: usize,
}

/// Frames over which a clip comes in under the song on air.
const CLIP_HEAD: usize = 1024;
/// Frames of the song a glide is given before the first one that is played.
const GLIDE_RUN_IN: usize = 16384;
/// Frames over which the glide's last output is blended with the song itself.
const GLIDE_SEAM: usize = 1024;
/// Below this distance from 1 a ratio counts as the song's own tempo.
const OWN_TEMPO: f64 = 1e-4;

/// When the next song is to come.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum When {
    /// Where the song on air ends (the transition is planned once that part is inside the look-ahead).
    #[default]
    AtEnd,
    /// As soon as a transition fits (the listener skipped, or asked for a song).
    Now,
}

struct Next {
    song: Box<dyn Song>,
    talk_seconds: Option<f64>,
    voice: Option<Vec<[f32; 2]>>,
    when: When,
}

struct Planned {
    plan: Plan,
    clip: Clip,
    /// Output frame at which the clip starts.
    clip_out: u64,
    starts_out: u64,
    one_out: u64,
    ends_out: u64,
    started: bool,
    one: bool,
    ended: bool,
}

pub struct Live<R: Renderer, G: Glide> {
    cfg: Config,
    renderer: R,
    glide: G,
    on_air: Box<dyn Song>,
    /// Next frame of the song on air.
    pos: usize,
    /// Output frames so far.
    out: u64,
    phase: Phase,
    next: Option<Next>,
    planned: Option<Planned>,
    talk: Option<Talk>,
    seam: Option<Seam>,
    /// A plan whose audio is rendered outside and has not come yet.
    awaiting: Option<Plan>,
    requests: Vec<Plan>,
    /// Why the last announced plan will not happen; reported with the next block.
    dropped: Option<String>,
    /// A plan that failed is not tried again before this output frame.
    retry_after: u64,
    /// The longest blend still to be tried for the next song: a blend whose audio failed its checks
    /// (the drums drift apart) is tried again only shorter.
    max_lead: f64,
    /// The blend length of the plan whose audio is awaited.
    awaiting_lead: f64,
    last_stats: u64,
    /// CPU time spent since the last stats event.
    spent: Duration,
    pub log: Vec<String>,
}

impl<R: Renderer, G: Glide> Live<R, G> {
    pub fn new(on_air: Box<dyn Song>, from_frame: usize, renderer: R, glide: G, cfg: Config) -> Self {
        Self { cfg, renderer, glide, on_air, pos: from_frame, out: 0, phase: Phase::Song, next: None, planned: None, talk: None, seam: None, awaiting: None, requests: Vec::new(), dropped: None, retry_after: 0, max_lead: 8.0, awaiting_lead: 8.0, last_stats: 0, spent: Duration::ZERO, log: Vec::new() }
    }

    pub fn renderer(&self) -> &R {
        &self.renderer
    }

    pub fn renderer_mut(&mut self) -> &mut R {
        &mut self.renderer
    }

    /// Output time so far, seconds.
    pub fn now(&self) -> f64 {
        self.out as f64 / RATE
    }

    /// The song that comes next. `talk_seconds`: the DJ wants to say a line of about that length before it.
    pub fn queue(&mut self, song: Box<dyn Song>, talk_seconds: Option<f64>, when: When) {
        self.next = Some(Next { song, talk_seconds, voice: None, when });
        // A transition that has not started yet is planned again for the new song.
        if matches!(self.phase, Phase::Planned) {
            self.phase = Phase::Song;
            self.planned = None;
            self.talk = None;
            self.dropped = Some("another song was asked for".into());
        }
        if self.awaiting.take().is_some() {
            self.requests.clear();
            self.dropped = Some("another song was asked for".into());
        }
        self.retry_after = 0;
        self.max_lead = 8.0;
    }

    /// The DJ's line as audio. It is placed so that it ends at its post; `false`: it does not fit its slot.
    pub fn say(&mut self, voice: Vec<[f32; 2]>) -> bool {
        match (&self.planned, &mut self.next) {
            (Some(p), _) => match slot_frames(p) {
                Some((from, post, duck)) => {
                    self.talk = Talk::ending_at(voice, post, from.max(self.out), duck);
                    self.talk.is_some()
                }
                None => false,
            },
            (None, Some(next)) => {
                next.voice = Some(voice);
                true
            }
            _ => false,
        }
    }

    /// The next block of audio and the events that fall into it.
    pub fn pull(&mut self, out: &mut [[f32; 2]], cpu: &mut dyn FnMut() -> Duration) -> Vec<Event> {
        let started = cpu();
        let mut events = Vec::new();
        if let Some(reason) = self.dropped.take() {
            events.push(Event::PlanDropped { t: self.now(), reason });
        }
        // Audio that has not come by the time its first frame is due is not waited for any longer.
        if self.awaiting.as_ref().is_some_and(|p| p.a_time(p.spec.clip[0]) * RATE < (self.pos + out.len()) as f64) {
            self.awaiting = None;
            self.requests.clear();
            self.retry_after = self.out;
            events.push(Event::PlanDropped { t: self.now(), reason: "the audio of the transition was not there in time".into() });
        }
        if matches!(self.phase, Phase::Song) && self.next.is_some() && self.awaiting.is_none() && self.out >= self.retry_after {
            self.try_plan(&mut events);
        }
        let mut done = 0;
        while done < out.len() {
            done += self.fill(&mut out[done..], &mut events);
        }
        // The DJ over the music.
        if let Some(talk) = &self.talk {
            let first = self.out - out.len() as u64;
            for (i, frame) in out.iter_mut().enumerate() {
                let at = first + i as u64;
                let g = talk.music_gain(at, RATE);
                let v = talk.voice_at(at);
                *frame = [frame[0] * g + v[0], frame[1] * g + v[1]];
            }
            if self.out > talk.end() + RATE as u64 {
                self.talk = None;
            }
        }
        self.spent += cpu().saturating_sub(started);
        if self.out - self.last_stats >= RATE as u64 {
            let seconds = (self.out - self.last_stats) as f64 / RATE;
            events.push(Event::Stats { t: self.now(), stats: self.stats(self.spent.as_secs_f64() / seconds) });
            self.last_stats = self.out;
            self.spent = Duration::ZERO;
        }
        events
    }

    fn try_plan(&mut self, events: &mut Vec<Event>) {
        let Some(next) = &self.next else { return };
        let now_s = self.pos as f64 / RATE;
        let a = self.on_air.table();
        let b = next.song.table();
        let end_s = self.on_air.frames() as f64 / RATE;
        // Only what lies inside the look-ahead may be planned on, and the rendering needs its time.
        let lead_s = if next.talk_seconds.is_some() { self.cfg.render_margin_s.max(self.cfg.talk_lead_s) } else { self.cfg.render_margin_s };
        let window = (now_s + lead_s, (now_s + self.cfg.lookahead_s).min(end_s + 8.0 * a.beat_s * 4.0));
        let fit = sone_plan::tempo::fit(a.bpm, b.bpm, self.cfg.max_stretch_pct);
        let bar_s = a.beat_s * f64::from(fit.bpb_a);
        let options = |lead: f64, window: Option<(f64, f64)>, out: u64| PlanOptions { id: format!("live-{out}"), window, lead_bars: lead, tts_seconds: next.talk_seconds, max_stretch_pct: self.cfg.max_stretch_pct, clip_before: 0.0, clip_after: 4.0, ..PlanOptions::default() };
        let mut result = None;
        let mut used_lead = self.max_lead;
        if next.when == When::AtEnd && self.max_lead >= 2.0 {
            // The transition the song's end asks for, once all of it lies inside the look-ahead.
            match plan(a, b, &options(self.max_lead, None, self.out)) {
                Ok(p) => {
                    let (first, last) = (p.a_time(p.spec.clip[0]), p.a_time(p.spec.a.until.unwrap_or(0.0).max(0.0)));
                    if last > now_s + self.cfg.lookahead_s {
                        self.retry_after = self.out + RATE as u64;
                        return;
                    }
                    if first >= window.0 {
                        result = Some(p);
                    }
                    // Otherwise the song's end is too close for it: whatever still fits is planned below.
                }
                Err(e) => self.log.push(format!("{:.2} s: no plan for the song's end: {e}", self.now())),
            }
        }
        for lead in [8.0, 4.0, 2.0] {
            if result.is_some() {
                break;
            }
            if lead > self.max_lead {
                continue;
            }
            used_lead = lead;
            // The whole blend and the bars shown before it have to fit into the window.
            if (lead + next.talk_seconds.map_or(0.0, |s| sone_plan::tts::bars_for(s, bar_s))) * bar_s > window.1 - window.0 {
                continue;
            }
            match plan(a, b, &options(lead, Some(window), self.out)) {
                Ok(p) => result = Some(p),
                Err(e) => self.log.push(format!("{:.2} s: no plan with {lead} bars: {e}", self.now())),
            }
        }
        let Some(plan) = result else {
            // The window moves on with the song; try again in two seconds.
            self.retry_after = self.out + 2 * RATE as u64;
            return;
        };
        if self.cfg.render_outside {
            // Announced now; the audio comes with `deliver`.
            events.push(self.announce(&plan));
            self.requests.push(plan.clone());
            self.awaiting = Some(plan);
            self.awaiting_lead = used_lead;
            return;
        }
        let clip = match self.renderer.render(&plan) {
            Ok(c) => c,
            Err(e) => {
                self.log.push(format!("{:.2} s: the transition of {used_lead} bars failed: {e}", self.now()));
                self.max_lead = used_lead / 2.0;
                self.retry_after = self.out;
                return;
            }
        };
        let event = self.announce(&plan);
        match self.accept(plan, clip) {
            Ok(()) => events.push(event),
            Err(reason) => {
                self.log.push(format!("{:.2} s: {reason}, planning again", self.now()));
                self.retry_after = self.out + RATE as u64 / 2;
            }
        }
    }

    /// Output frame at which a moment of the song on air plays (while the song plays as it is).
    fn to_out(&self, a_time: f64) -> u64 {
        self.out + ((a_time * RATE) as i64 - self.pos as i64).max(0) as u64
    }

    /// Start, the one and end of a plan in output frames.
    fn frames_of(&self, plan: &Plan) -> (u64, u64, u64) {
        (self.to_out(plan.a_time(plan.spec.first_event)), self.to_out(plan.one_s), self.to_out(plan.a_time(plan.spec.last_event.max(0.0))))
    }

    /// The plan as the event that announces it.
    fn announce(&self, plan: &Plan) -> Event {
        let a = self.on_air.table();
        let b = self.next.as_ref().expect("a plan needs the next song").song.table();
        let (starts, one, ends) = self.frames_of(plan);
        let tts = plan.tts.as_ref().map(|slot| TtsSlot {
            slot_from: self.to_out(plan.a_time(slot.from_u)) as f64 / RATE,
            must_end_by: self.to_out(plan.a_time(slot.post_u)) as f64 / RATE - MARGIN_S,
            max_seconds: slot.max_seconds(plan.bar_s),
            over: slot.over.clone(),
            duck_db: sone_plan::tts::DUCK_DB,
        });
        Event::Plan {
            t: self.now(),
            a: a.id.clone(),
            b: b.id.clone(),
            recipe: plan.recipe,
            confidence: plan.confidence,
            starts_at: starts as f64 / RATE,
            one_at: one as f64 / RATE,
            ends_at: ends as f64 / RATE,
            tempo: Tempo { a: a.bpm, b: b.bpm, ratio: plan.fit.ratio_name().into(), stretch_pct: plan.fit.stretch_pct },
            key: Keys { a: a.camelot.clone(), b: b.camelot.clone(), fit: keys::fit_name(plan.key_distance, a.camelot == b.camelot).into() },
            tts,
        }
    }

    /// Takes the rendered clip of a plan. The song on air must not have passed the clip's first frame.
    fn accept(&mut self, plan: Plan, clip: Clip) -> std::result::Result<(), String> {
        if clip.a_from < self.pos {
            return Err(format!("the clip starts {:.2} s in the past", (self.pos - clip.a_from) as f64 / RATE));
        }
        let clip_out = self.out + (clip.a_from - self.pos) as u64;
        let (starts_out, one_out, ends_out) = self.frames_of(&plan);
        let now = self.now();
        self.log.extend(plan.notes.iter().map(|n| format!("{now:.2} s: {n}")));
        let planned = Planned { starts_out, one_out, ends_out, plan, clip, clip_out, started: false, one: false, ended: false };
        // A line that came before the plan is placed now.
        if let Some(voice) = self.next.as_mut().and_then(|n| n.voice.take()) {
            if let Some((from, post, duck)) = slot_frames(&planned) {
                self.talk = Talk::ending_at(voice, post, from.max(self.out), duck);
            }
        }
        self.planned = Some(planned);
        self.phase = Phase::Planned;
        Ok(())
    }

    /// With `render_outside`: the plans that wait for their audio since the last call.
    pub fn take_requests(&mut self) -> Vec<Plan> {
        std::mem::take(&mut self.requests)
    }

    /// With `render_outside`: the audio of a plan (by the id of its spec), or why there is none.
    /// Returns what there is to report.
    pub fn deliver(&mut self, id: &str, clip: Result<Clip>) -> Vec<Event> {
        let Some(plan) = self.awaiting.take_if(|p| p.spec.id == id) else {
            // Not waited for any more (another song was asked for, or it came too late).
            return Vec::new();
        };
        let result = match clip {
            Ok(clip) => self.accept(plan, clip),
            Err(e) => {
                self.max_lead = self.awaiting_lead / 2.0;
                Err(format!("the transition of {} bars failed: {e}", self.awaiting_lead))
            }
        };
        match result {
            Ok(()) => Vec::new(),
            Err(reason) => {
                self.retry_after = self.out + RATE as u64 / 2;
                vec![Event::PlanDropped { t: self.now(), reason }]
            }
        }
    }

    /// Fills as much of `out` as one phase gives; returns the frames written.
    fn fill(&mut self, out: &mut [[f32; 2]], events: &mut Vec<Event>) -> usize {
        let n = match &mut self.phase {
            Phase::Song if self.pos >= self.on_air.frames() && self.awaiting.is_none() && self.next.is_some() => {
                // The song is over and no transition could be planned (the songs share no beat): the next
                // song simply follows.
                let next = self.next.take().expect("next");
                self.log.push(format!("{:.2} s: the song is over without a transition, the next one starts", self.now()));
                self.on_air = next.song;
                self.pos = 0;
                return 0;
            }
            Phase::Song => {
                self.on_air.read(self.pos, out);
                scale(out, self.on_air.gain());
                self.pos += out.len();
                out.len()
            }
            Phase::Planned => {
                let p = self.planned.as_ref().expect("planned");
                let until_clip = (p.clip_out - self.out) as usize;
                if until_clip == 0 {
                    self.phase = Phase::Clip { at: 0 };
                    return 0;
                }
                let n = out.len().min(until_clip);
                self.on_air.read(self.pos, &mut out[..n]);
                scale(&mut out[..n], self.on_air.gain());
                self.pos += n;
                n
            }
            Phase::Clip { at } => {
                let p = self.planned.as_ref().expect("planned");
                let seam = p.clip.seam.min(p.clip.audio.len());
                let left = p.clip.audio.len() - seam - *at;
                if left == 0 {
                    // The next song takes over where the clip ended; the clip's last frames are blended into it.
                    let p = self.planned.take().expect("planned");
                    let next = self.next.take().expect("next");
                    if seam > 0 {
                        self.seam = Some(Seam { tail: p.clip.audio[p.clip.audio.len() - seam..].to_vec(), at: 0 });
                    }
                    self.on_air = next.song;
                    self.pos = p.clip.b_resume.round().max(0.0) as usize;
                    if (p.clip.ratio - 1.0).abs() < OWN_TEMPO {
                        self.phase = Phase::Song;
                    } else {
                        let glide_frames = (self.cfg.glide_bars * p.plan.bar_s * RATE) as u64;
                        self.glide.restart(p.clip.ratio);
                        // The glide starts some frames before the song's first frame that is played, so that
                        // it is in its stride there.
                        let run_in = GLIDE_RUN_IN.min(self.pos);
                        let skip = (run_in as f64 * p.clip.ratio).round() as usize;
                        let at = self.pos as f64 - skip as f64 / p.clip.ratio;
                        self.phase = Phase::Gliding { pending: Vec::new(), read: self.pos - run_in, at, skip, from_out: self.out, until_out: self.out + glide_frames, from_ratio: p.clip.ratio };
                    }
                    return 0;
                }
                let n = out.len().min(left);
                out[..n].copy_from_slice(&p.clip.audio[*at..*at + n]);
                if *at < CLIP_HEAD {
                    // The clip comes in under the song: where it goes on as the song did, nothing changes,
                    // where it starts with other bars of the song, there is no step.
                    let m = n.min(CLIP_HEAD - *at);
                    let mut song = vec![[0.0f32; 2]; m];
                    self.on_air.read(self.pos + *at, &mut song);
                    scale(&mut song, self.on_air.gain());
                    for (i, (o, s)) in out[..m].iter_mut().zip(&song).enumerate() {
                        let w = ((*at + i) as f32 + 0.5) / CLIP_HEAD as f32;
                        *o = [s[0] * (1.0 - w) + o[0] * w, s[1] * (1.0 - w) + o[1] * w];
                    }
                }
                *at += n;
                n
            }
            Phase::Gliding { pending, read, at, skip, from_out, until_out, from_ratio } => {
                if pending.is_empty() {
                    let back = self.out >= *until_out + RATE as u64;
                    let ratio = if back { 1.0 } else { *from_ratio + (1.0 - *from_ratio) * ((self.out - *from_out) as f64 / (*until_out - *from_out).max(1) as f64).min(1.0) };
                    self.glide.set_ratio(ratio);
                    // Back at its own tempo for a second: from here the song plays as it is, and what the
                    // glide would have played next is blended into it.
                    let wanted = if back { GLIDE_SEAM } else { 1 };
                    let mut made = Vec::new();
                    let mut block = vec![[0.0f32; 2]; 2048];
                    let song_from = at.round().max(0.0) as usize;
                    while made.len() < wanted + *skip && *read < self.on_air.frames() + (4 * RATE as usize) {
                        self.on_air.read(*read, &mut block);
                        *read += block.len();
                        made.extend(self.glide.process(&block));
                    }
                    *at += made.len() as f64 / ratio;
                    let dropped = (*skip).min(made.len());
                    made.drain(..dropped);
                    *skip -= dropped;
                    scale(&mut made, self.on_air.gain());
                    if back {
                        made.truncate(GLIDE_SEAM);
                        self.seam = (!made.is_empty()).then_some(Seam { tail: made, at: 0 });
                        self.pos = song_from;
                        self.phase = Phase::Song;
                        return 0;
                    }
                    if made.is_empty() {
                        // The song is over.
                        self.pos = song_from;
                        self.phase = Phase::Song;
                        return 0;
                    }
                    *pending = made;
                }
                let n = out.len().min(pending.len());
                out[..n].copy_from_slice(&pending[..n]);
                pending.drain(..n);
                n
            }
        };
        // What played before ends under what plays now.
        {
            if let Some(seam) = &mut self.seam {
                let len = seam.tail.len();
                for frame in out[..n].iter_mut() {
                    if seam.at >= len {
                        break;
                    }
                    let w = (seam.at as f32 + 0.5) / len as f32;
                    let old = seam.tail[seam.at];
                    *frame = [old[0] * (1.0 - w) + frame[0] * w, old[1] * (1.0 - w) + frame[1] * w];
                    seam.at += 1;
                }
                if seam.at >= len {
                    self.seam = None;
                }
            }
        }
        self.out += n as u64;
        if let Some(p) = &mut self.planned {
            let t = |frame: u64| frame as f64 / RATE;
            if !p.started && self.out >= p.starts_out {
                p.started = true;
                events.push(Event::TransitionStarted { t: t(p.starts_out) });
            }
            if !p.one && self.out >= p.one_out {
                p.one = true;
                events.push(Event::TheOne { t: t(p.one_out) });
            }
            if !p.ended && self.out >= p.ends_out {
                p.ended = true;
                events.push(Event::TransitionEnded { t: t(p.ends_out) });
            }
        }
        n
    }

    fn stats(&self, cpu_core_fraction: f64) -> Stats {
        let mut s = Stats { beat_offset_ms: 0.0, voices_at_once: 1, melodies_at_once: 1, basses_at_once: 1, level_dip_db: 0.0, lookahead_s: self.cfg.lookahead_s as f32, cpu_core_fraction: cpu_core_fraction as f32 };
        if let (Some(p), Phase::Clip { at }) = (&self.planned, &self.phase) {
            // Where on the ruler the clip is, and what the lanes let through there.
            let u = p.plan.spec.clip[0] + *at as f64 / RATE / p.plan.bar_s;
            let open = |deck: &sone_core::Deck, stem: Stem, is_a: bool| -> bool {
                let there = if is_a { deck.until.is_none_or(|until| u < until) } else { deck.from.is_none_or(|from| u >= from) };
                there && deck.lanes.get(&stem).map_or(1.0, |l| lanes::at(l, u)) > 0.25
            };
            let count = |stem: Stem| u8::from(open(&p.plan.spec.a, stem, true)) + u8::from(open(&p.plan.spec.b, stem, false));
            s.voices_at_once = count(Stem::Vocals);
            s.melodies_at_once = count(Stem::Other);
            s.basses_at_once = count(Stem::Bass);
            s.beat_offset_ms = p.clip.beat_offset_ms;
            s.level_dip_db = p.clip.level_dip_db;
        }
        s
    }
}

fn scale(x: &mut [[f32; 2]], gain: f32) {
    if gain != 1.0 {
        for f in x {
            *f = [f[0] * gain, f[1] * gain];
        }
    }
}

/// The slot of a planned transition in output frames: from, post (with its margin), ducking.
fn slot_frames(p: &Planned) -> Option<(u64, u64, f32)> {
    let slot = p.plan.tts.as_ref()?;
    let frame = |u: f64| -> u64 { (p.one_out as f64 + u * p.plan.bar_s * RATE).max(0.0) as u64 };
    Some((frame(slot.from_u), frame(slot.post_u).saturating_sub((MARGIN_S * RATE) as u64), sone_plan::tts::DUCK_DB))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use sone_core::{BeatTable, Event};
    use sone_plan::{Plan, plan::beat_time};

    use super::*;

    /// A song whose left channel is the frame's number and whose right channel names the song.
    struct Marked {
        table: BeatTable,
        mark: f32,
    }

    impl Song for Marked {
        fn frames(&self) -> usize {
            (self.table.duration * RATE) as usize
        }
        fn read(&self, from: usize, out: &mut [[f32; 2]]) {
            for (i, f) in out.iter_mut().enumerate() {
                *f = [(from + i) as f32, self.mark];
            }
        }
        fn table(&self) -> &BeatTable {
            &self.table
        }
    }

    /// Renders nothing, but places the clip where a real renderer would.
    struct Placeholder {
        b: BeatTable,
    }

    impl Renderer for Placeholder {
        fn render(&mut self, plan: &Plan) -> Result<Clip> {
            let a_from = (plan.a_time(plan.spec.clip[0]) * RATE).round() as usize;
            let frames = ((plan.spec.clip[1] - plan.spec.clip[0]) * plan.bar_s * RATE).round() as usize;
            let bpb_b = f64::from(plan.spec.b.beats_per_bar);
            let b_end = beat_time(&self.b, plan.b_one_beat as f64 + plan.spec.clip[1] * bpb_b);
            Ok(Clip { audio: vec![[-1.0, 9.0]; frames], a_from, b_resume: b_end * RATE, ratio: 1.0 / (1.0 + plan.fit.stretch_pct / 100.0), beat_offset_ms: 0.0, level_dip_db: 0.0, seam: 0 })
        }
    }

    /// Steps through the song at the ratio, taking the frame it stands on (the real one keeps the pitch).
    #[derive(Default)]
    struct Through {
        ratio: f64,
        held: Vec<[f32; 2]>,
        at: f64,
    }

    #[allow(non_upper_case_globals)]
    const Through: Through = Through { ratio: 1.0, held: Vec::new(), at: 0.0 };

    impl Glide for Through {
        fn set_ratio(&mut self, ratio: f64) {
            self.ratio = ratio;
        }
        fn process(&mut self, input: &[[f32; 2]]) -> Vec<[f32; 2]> {
            self.held.extend_from_slice(input);
            let mut out = Vec::new();
            while (self.at as usize) < self.held.len() {
                out.push(self.held[self.at as usize]);
                self.at += 1.0 / self.ratio;
            }
            let done = self.at as usize;
            self.held.drain(..done);
            self.at -= done as f64;
            out
        }
        fn restart(&mut self, ratio: f64) {
            *self = Through { ratio, held: Vec::new(), at: 0.0 };
        }
    }

    /// Talking Body → Yo No Lo Sé (both at 120 BPM), from the repo's beat tables.
    fn songs() -> (BeatTable, BeatTable) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/analysis/beats");
        let get = |id: &str| BeatTable::load(&dir.join(format!("{id}.json"))).expect("the repo's data/analysis/beats");
        (get("5vXnH99Tn0qAh84SWU5sfc"), get("5ihq4qbjaO2r5Ga9aGlsMj"))
    }

    #[test]
    fn one_transition_without_a_gap() {
        let (a, b) = songs();
        let start = (100.0 * RATE) as usize;
        let mut live = Live::new(Box::new(Marked { table: a, mark: 1.0 }), start, Placeholder { b: b.clone() }, Through, Config::default());
        let mut events = Vec::new();
        let mut audio = Vec::new();
        let mut block = vec![[0.0f32; 2]; 4410];
        let mut no_cpu = || Duration::ZERO;
        for k in 0..900 {
            if k == 20 {
                // Two seconds in, the listener asks for another song; the DJ wants five seconds for a line.
                live.queue(Box::new(Marked { table: b.clone(), mark: 2.0 }), Some(5.0), When::Now);
            }
            events.extend(live.pull(&mut block, &mut no_cpu));
            audio.extend_from_slice(&block);
        }
        let plan = events.iter().find_map(|e| match e {
            Event::Plan { t, starts_at, one_at, ends_at, tts, .. } => Some((*t, *starts_at, *one_at, *ends_at, tts.clone())),
            _ => None,
        });
        let (t, starts, one, ends, tts) = plan.expect("a plan");
        // Planned right when the next song was known, and everything lies inside the look-ahead.
        assert!((2.0..2.2).contains(&t), "{t}");
        assert!(starts > t && starts <= one && one <= ends, "{starts} {one} {ends}");
        assert!(one - t <= 30.0 + 1e-6, "the one is {:.1} s after the plan", one - t);
        let tts = tts.expect("a slot for the line");
        assert!(tts.max_seconds >= 5.0 && tts.must_end_by <= one + 1e-9 && tts.slot_from >= t, "{tts:?}");
        let order: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                Event::Plan { .. } => Some("plan"),
                Event::TransitionStarted { .. } => Some("started"),
                Event::TheOne { .. } => Some("one"),
                Event::TransitionEnded { .. } => Some("ended"),
                Event::PlanDropped { .. } => Some("dropped"),
                Event::Stats { .. } => None,
            })
            .collect();
        assert_eq!(order, ["plan", "started", "one", "ended"]);
        assert!(events.iter().filter(|e| matches!(e, Event::Stats { .. })).count() >= 88);
        // Song A up to the clip, the clip, song B after it: every frame in its place.
        let first_clip = audio.iter().position(|f| f[1] != 1.0).expect("the clip plays");
        let last_clip = audio.iter().rposition(|f| f[1] == 9.0).unwrap();
        // The clip comes in under the song within its first frames, then it plays alone.
        assert!(audio[first_clip..first_clip + CLIP_HEAD].windows(2).all(|w| w[1][1] > w[0][1] && w[1][1] < 9.0));
        assert!(audio[first_clip + CLIP_HEAD..=last_clip].iter().all(|f| f[1] == 9.0));
        assert!(audio[last_clip + 1..].iter().all(|f| f[1] == 2.0));
        for (i, f) in audio[..first_clip].iter().enumerate() {
            assert_eq!(f[0], (start + i) as f32);
        }
        // Song B plays on without a jump: through the glide a frame may be left out or played twice,
        // after it the song is there frame for frame.
        let b_frames: Vec<f32> = audio[last_clip + 1..].iter().map(|f| f[0]).collect();
        assert!(b_frames.windows(2).all(|w| (w[1] - w[0] - 1.0).abs() <= 1.01), "song B jumps");
        let end = &b_frames[b_frames.len() - 44100..];
        assert!(end.windows(2).all(|w| w[1] == w[0] + 1.0), "after the glide song B is not the song itself");
        // The clip starts where the plan said.
        assert!(((first_clip as f64 / RATE) - starts).abs() < 0.01, "{} {starts}", first_clip as f64 / RATE);
    }

    #[test]
    fn the_line_ends_on_its_post() {
        let (a, b) = songs();
        let mut live = Live::new(Box::new(Marked { table: a, mark: 1.0 }), (100.0 * RATE) as usize, Placeholder { b: b.clone() }, Through, Config::default());
        let mut block = vec![[0.0f32; 2]; 4410];
        let mut no_cpu = || Duration::ZERO;
        live.queue(Box::new(Marked { table: b, mark: 2.0 }), Some(5.0), When::Now);
        // The voice is a constant 100 on the left channel, so it can be told from the music's markers on the right.
        assert!(live.say(vec![[0.0, 100.0]; (4.0 * RATE) as usize]));
        let mut events = Vec::new();
        let mut audio = Vec::new();
        for _ in 0..600 {
            events.extend(live.pull(&mut block, &mut no_cpu));
            audio.extend_from_slice(&block);
        }
        let post = events.iter().find_map(|e| match e {
            Event::Plan { tts: Some(t), .. } => Some(t.must_end_by),
            _ => None,
        });
        let post = post.expect("a slot");
        let spoken: Vec<usize> = audio.iter().enumerate().filter(|(_, f)| f[1] > 50.0).map(|(i, _)| i).collect();
        let (first, last) = (*spoken.first().unwrap() as f64 / RATE, *spoken.last().unwrap() as f64 / RATE);
        assert!((last - post).abs() < 0.01, "the line ends at {last:.3}, its post is {post:.3}");
        assert!((last - first - 4.0).abs() < 0.01);
        // Under the line the music is 6 dB down.
        let mid = audio[((first + 2.0) * RATE) as usize];
        assert!(((mid[1] - 100.0) / 1.0 - 0.501).abs() < 0.01 || ((mid[1] - 100.0) / 9.0 - 0.501).abs() < 0.01, "{mid:?}");
    }

    #[test]
    fn a_queued_song_waits_for_the_end_of_the_song_on_air() {
        let (a, b) = songs();
        let natural = plan(&a, &b, &PlanOptions::default()).expect("a plan").one_s;
        let start = natural - 70.0;
        let mut live = Live::new(Box::new(Marked { table: a, mark: 1.0 }), (start * RATE) as usize, Placeholder { b: b.clone() }, Through, Config::default());
        let mut block = vec![[0.0f32; 2]; 4410];
        let mut no_cpu = || Duration::ZERO;
        live.queue(Box::new(Marked { table: b, mark: 2.0 }), None, When::AtEnd);
        let mut events = Vec::new();
        for _ in 0..1000 {
            events.extend(live.pull(&mut block, &mut no_cpu));
        }
        let Some(Event::Plan { t, one_at, .. }) = events.iter().find(|e| matches!(e, Event::Plan { .. })) else { panic!("no plan") };
        // The one is where the song's end puts it, and the plan was made when that came into sight.
        assert!((one_at - 70.0).abs() < 0.05, "the one at {one_at}");
        assert!(*t > 30.0 && one_at - t <= 30.0 + 1e-6, "planned at {t}");
        assert!(events.iter().any(|e| matches!(e, Event::TransitionEnded { .. })), "{:?}", events.iter().filter(|e| !matches!(e, Event::Stats { .. })).collect::<Vec<_>>());
    }

    #[test]
    fn audio_rendered_outside_arrives_later() {
        let (a, b) = songs();
        let cfg = Config { render_outside: true, ..Config::default() };
        let mut live = Live::new(Box::new(Marked { table: a, mark: 1.0 }), (100.0 * RATE) as usize, Placeholder { b: b.clone() }, Through, cfg);
        let mut outside = Placeholder { b: b.clone() };
        let mut block = vec![[0.0f32; 2]; 4410];
        let mut no_cpu = || Duration::ZERO;
        live.queue(Box::new(Marked { table: b.clone(), mark: 2.0 }), None, When::Now);
        let mut events = live.pull(&mut block, &mut no_cpu);
        assert!(matches!(events[0], Event::Plan { .. }), "announced at once: {events:?}");
        let requests = live.take_requests();
        assert_eq!(requests.len(), 1);
        // The audio takes 1.5 s to come; the song plays on meanwhile.
        let mut audio = Vec::new();
        for _ in 0..15 {
            events.extend(live.pull(&mut block, &mut no_cpu));
            audio.extend_from_slice(&block);
        }
        assert!(live.take_requests().is_empty(), "asked for once");
        events.extend(live.deliver(&requests[0].spec.id, outside.render(&requests[0])));
        for _ in 0..600 {
            events.extend(live.pull(&mut block, &mut no_cpu));
            audio.extend_from_slice(&block);
        }
        let names: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                Event::Plan { .. } => Some("plan"),
                Event::PlanDropped { .. } => Some("dropped"),
                Event::TransitionStarted { .. } => Some("started"),
                Event::TheOne { .. } => Some("one"),
                Event::TransitionEnded { .. } => Some("ended"),
                Event::Stats { .. } => None,
            })
            .collect();
        assert_eq!(names, ["plan", "started", "one", "ended"]);
        let (Some(Event::Plan { starts_at, .. }), Some(first_clip)) = (events.first(), audio.iter().position(|f| f[1] != 1.0)) else { panic!("no clip") };
        assert!(((first_clip + 4410) as f64 / RATE - starts_at).abs() < 0.01, "the clip plays where the plan said");
    }

    #[test]
    fn audio_that_comes_too_late_is_dropped_and_planned_again() {
        let (a, b) = songs();
        let cfg = Config { render_outside: true, ..Config::default() };
        // Far from the song's end, as after a skip: the transition is due within the look-ahead.
        let mut live = Live::new(Box::new(Marked { table: a, mark: 1.0 }), (100.0 * RATE) as usize, Placeholder { b: b.clone() }, Through, cfg);
        let mut block = vec![[0.0f32; 2]; 4410];
        let mut no_cpu = || Duration::ZERO;
        live.queue(Box::new(Marked { table: b, mark: 2.0 }), None, When::Now);
        let mut events = Vec::new();
        let mut asked = 0;
        for _ in 0..400 {
            events.extend(live.pull(&mut block, &mut no_cpu));
            asked += live.take_requests().len();
        }
        let dropped = events.iter().filter(|e| matches!(e, Event::PlanDropped { .. })).count();
        let plans = events.iter().filter(|e| matches!(e, Event::Plan { .. })).count();
        assert!(dropped >= 1 && plans == dropped + usize::from(plans > dropped) && asked == plans, "{plans} plans, {dropped} dropped, {asked} asked for");
        assert!(!events.iter().any(|e| matches!(e, Event::TransitionStarted { .. })), "nothing starts without audio");
    }

    #[test]
    fn a_line_too_long_for_its_slot_is_refused() {
        let (a, b) = songs();
        let mut live = Live::new(Box::new(Marked { table: a, mark: 1.0 }), (100.0 * RATE) as usize, Placeholder { b: b.clone() }, Through, Config::default());
        let mut block = vec![[0.0f32; 2]; 4410];
        let mut no_cpu = || Duration::ZERO;
        live.queue(Box::new(Marked { table: b, mark: 2.0 }), Some(4.0), When::Now);
        live.pull(&mut block, &mut no_cpu);
        assert!(!live.say(vec![[0.0; 2]; (25.0 * RATE) as usize]));
    }
}
