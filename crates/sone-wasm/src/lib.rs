//! Planner and renderer behind a plain C interface, for `wasm32-unknown-unknown`.
//!
//! The host (a Worker) keeps no audio in this module but what a transition
//! needs: it gives both songs' beat tables, asks for a plan, reads from the
//! plan which windows of the stems the rendering reads, hands those over as
//! 16-bit PCM, and gets the clip back as 16-bit PCM.
//!
//! ```text
//! a = sone_song(table json, lufs, frames)   b = sone_song(…)
//! sone_plan(a, b, options json)        → result: {plan, spec, needs}
//! sone_stems(a, stem, start, pcm)  …   for every window in needs
//! sone_render()                        → result: {report, placing}, audio: the clip
//! sone_forget(a)                       when the song is over
//! ```
//!
//! Every call returns 0 when it worked; then `sone_result_ptr/len` hold a JSON
//! text (after an error: its message) and, after `sone_render`,
//! `sone_audio_ptr/len` the clip (interleaved stereo, 16 bit, 44.1 kHz).

pub mod needs;
pub mod windows;

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use sone_core::{BeatTable, Stem};
use sone_plan::{Plan, PlanOptions};
use serde::Deserialize;

use crate::windows::WindowStems;

#[derive(Default)]
struct State {
    songs: HashMap<u32, WindowStems>,
    next: u32,
    plan: Option<(Plan, u32, u32)>,
    result: Vec<u8>,
    audio: Vec<i16>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Options {
    pub id: Option<String>,
    /// Seconds of song A in which the one has to lie (live: inside the look-ahead).
    pub window: Option<(f64, f64)>,
    pub lead_bars: Option<f64>,
    pub talk_seconds: Option<f64>,
    pub max_stretch_pct: Option<f64>,
    /// Bars of the clip before the first and after the last event (live: 0 and 4).
    pub clip_before: Option<f64>,
    pub clip_after: Option<f64>,
    /// The owner wants song B's voice only from the one (see `data/owner/pair_rules.json`).
    pub no_voice_over_beat: Option<bool>,
}

impl Options {
    fn plan_options(&self) -> PlanOptions {
        let d = PlanOptions::default();
        PlanOptions {
            id: self.id.clone().unwrap_or(d.id),
            window: self.window,
            lead_bars: self.lead_bars.unwrap_or(d.lead_bars),
            tts_seconds: self.talk_seconds,
            max_stretch_pct: self.max_stretch_pct.unwrap_or(d.max_stretch_pct),
            clip_before: self.clip_before.unwrap_or(0.0),
            clip_after: self.clip_after.unwrap_or(4.0),
            force: None,
            no_voice_over_beat: self.no_voice_over_beat.unwrap_or(false),
        }
    }
}

fn song(s: &mut State, table: &[u8], lufs: f64, frames: usize) -> Result<u32> {
    let table: BeatTable = serde_json::from_slice(table).context("beat table")?;
    s.next += 1;
    s.songs.insert(s.next, WindowStems::new(table, lufs, frames));
    Ok(s.next)
}

fn plan_call(s: &mut State, a: u32, b: u32, options: &[u8]) -> Result<String> {
    let opt: Options = if options.is_empty() { Options::default() } else { serde_json::from_slice(options).context("options")? };
    let (sa, sb) = (s.songs.get(&a).context("song a")?, s.songs.get(&b).context("song b")?);
    let plan = sone_plan::plan(&sa.table, &sb.table, &opt.plan_options())?;
    let needs = needs::needs(&plan, &sa.table, &sb.table);
    let text = serde_json::to_string(&serde_json::json!({
        "recipe": plan.recipe, "confidence": plan.confidence, "one_s": plan.one_s, "bar_s": plan.bar_s,
        "starts_s": plan.a_time(plan.spec.first_event), "ends_s": plan.a_time(plan.spec.last_event.max(0.0)),
        "stretch_pct": plan.fit.stretch_pct, "key_distance": plan.key_distance,
        "talk": plan.tts.as_ref().map(|t| serde_json::json!({"from_s": plan.a_time(t.from_u), "post_s": plan.a_time(t.post_u), "max_seconds": t.max_seconds(plan.bar_s), "over": t.over})),
        "notes": plan.notes, "spec": plan.spec, "needs": needs,
    }))?;
    s.plan = Some((plan, a, b));
    Ok(text)
}

fn render_call(s: &mut State) -> Result<String> {
    let Some((plan, a, b)) = &s.plan else { bail!("no plan") };
    let (sa, sb) = (s.songs.get(a).context("song a")?, s.songs.get(b).context("song b")?);
    let r = sone_render::render_with(&plan.spec, sa, sb, &sone_render::RenderOptions::default())?;
    let (missed_a, missed_b) = (sa.missed(), sb.missed());
    if missed_a + missed_b > 0 {
        bail!("the rendering read {missed_a} frames of song A and {missed_b} of song B that were not given (A: {}; B: {})", sa.first_miss(), sb.first_miss());
    }
    let bar_b = r.bar_s / r.ratio;
    let u_end = plan.spec.clip[1];
    let seg = plan.spec.b.segments.iter().rev().find(|s| s.u0 <= u_end && s.u1.is_none_or(|u1| u_end < u1));
    let b_end = r.b_zero_s + seg.map_or(u_end, |s| s.src + (u_end - s.u0)) * bar_b;
    s.audio.clear();
    s.audio.extend(r.audio.iter().flat_map(|f| f.iter().map(|v| (v * 32767.0).round().clamp(-32768.0, 32767.0) as i16)));
    Ok(serde_json::to_string(&serde_json::json!({
        "frames": r.audio.len(), "a_from_s": r.a_zero_s + plan.spec.clip[0] * r.bar_s, "b_resume_s": b_end, "ratio": r.ratio,
        "one_in_clip_s": -plan.spec.clip[0] * r.bar_s, "report": r.report,
    }))?)
}

fn answer(s: &mut State, r: Result<String>) -> i32 {
    match r {
        Ok(text) => {
            s.result = text.into_bytes();
            0
        }
        Err(e) => {
            s.result = format!("{e:#}").into_bytes();
            1
        }
    }
}

/// # Safety
/// `ptr` and `len` must describe memory of this module that holds initialised bytes (or `len` is 0).
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, len) } }
}

#[unsafe(no_mangle)]
pub extern "C" fn sone_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` must come from `sone_alloc(len)` and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sone_free(ptr: *mut u8, len: usize) {
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len.max(1)) });
}

/// A song by its beat table (JSON), its loudness and the length of its stems in frames.
/// Returns its handle, 0 when the table cannot be read.
///
/// # Safety
/// `ptr`, `len`: see [`bytes`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sone_song(ptr: *const u8, len: usize, lufs: f64, frames: usize) -> u32 {
    let table = unsafe { bytes(ptr, len) };
    STATE.with_borrow_mut(|s| match song(s, table, lufs, frames) {
        Ok(h) => h,
        Err(e) => {
            s.result = format!("{e:#}").into_bytes();
            0
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn sone_forget(song: u32) {
    STATE.with_borrow_mut(|s| {
        s.songs.remove(&song);
        if s.plan.as_ref().is_some_and(|p| p.1 == song || p.2 == song) {
            s.plan = None;
        }
    });
}

/// Forgets a song's audio, keeps its table.
#[unsafe(no_mangle)]
pub extern "C" fn sone_forget_stems(song: u32) {
    STATE.with_borrow_mut(|s| {
        if let Some(song) = s.songs.get_mut(&song) {
            song.clear();
        }
    });
}

/// A window of a stem (0 vocals, 1 drums, 2 bass, 3 other): `frames` frames of interleaved stereo
/// from frame `start` of the song on. The memory stays the host's.
///
/// # Safety
/// `ptr` must point at `frames * 2` samples inside this module's memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sone_stems(song: u32, stem: u32, start: i64, ptr: *const i16, frames: usize) -> i32 {
    let stem = match stem {
        0 => Stem::Vocals,
        1 => Stem::Drums,
        2 => Stem::Bass,
        3 => Stem::Other,
        _ => return 1,
    };
    let pcm = unsafe { std::slice::from_raw_parts(ptr, frames * 2) }.to_vec();
    STATE.with_borrow_mut(|s| match s.songs.get_mut(&song) {
        Some(song) => {
            song.add(stem, start, pcm);
            0
        }
        None => 1,
    })
}

/// # Safety
/// `ptr`, `len`: see [`bytes`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sone_plan(a: u32, b: u32, ptr: *const u8, len: usize) -> i32 {
    let options = unsafe { bytes(ptr, len) };
    STATE.with_borrow_mut(|s| {
        let r = plan_call(s, a, b, options);
        answer(s, r)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn sone_render() -> i32 {
    STATE.with_borrow_mut(|s| {
        let r = render_call(s);
        answer(s, r)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn sone_result_ptr() -> *const u8 {
    STATE.with_borrow(|s| s.result.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn sone_result_len() -> usize {
    STATE.with_borrow(|s| s.result.len())
}

#[unsafe(no_mangle)]
pub extern "C" fn sone_audio_ptr() -> *const i16 {
    STATE.with_borrow(|s| s.audio.as_ptr())
}

/// Samples (frames × 2).
#[unsafe(no_mangle)]
pub extern "C" fn sone_audio_len() -> usize {
    STATE.with_borrow(|s| s.audio.len())
}

/// Gives the clip's memory back.
#[unsafe(no_mangle)]
pub extern "C" fn sone_audio_done() {
    STATE.with_borrow_mut(|s| s.audio = Vec::new());
}
