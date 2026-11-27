//! What the live engine asks for, answered by the renderer and the stretcher.

use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use sone_core::Spec;
use sone_dsp::{SR, Stretcher};
use sone_live::{Clip, Glide, Renderer};
use sone_plan::Plan;
use sone_render::{Rendered, Report, StemSource};

use crate::songs::Loaded;

/// Frames at a clip's end that are blended with the song that follows (23 ms).
pub const SEAM: usize = 1024;

/// A rendered transition placed in both songs.
pub struct Placed {
    pub rendered: Rendered,
    /// The spec as rendered (the plan's, with the station's gains).
    pub spec: Spec,
    /// Frame of song A at which the clip takes over.
    pub a_from: usize,
    /// Frame of song B (its own time) at a ruler position.
    pub b_zero: f64,
    pub render_seconds: f64,
}

impl Placed {
    /// Frame of song B (its own time) that plays at ruler position `u`.
    pub fn b_frame(&self, u: f64) -> f64 {
        let bar_b = self.rendered.bar_s / self.rendered.ratio;
        let seg = self.spec.b.segments.iter().rev().find(|s| s.u0 <= u && s.u1.is_none_or(|u1| u < u1));
        let bars = seg.map_or(u, |s| s.src + (u - s.u0));
        (self.b_zero + bars * bar_b) * SR
    }
}

/// Renders `spec` so that both songs have the gains the station plays them at.
pub fn render_placed(spec: &Spec, a: &Loaded, b: &Loaded) -> Result<Placed> {
    let mut spec = spec.clone();
    // The renderer brings the louder song down to the quieter one; the station has its own gains.
    let pair = a.stems.lufs().min(b.stems.lufs());
    spec.a.trim_db += a.gain_db - (pair - a.stems.lufs());
    spec.b.trim_db += b.gain_db - (pair - b.stems.lufs());
    let started = Instant::now();
    let rendered = sone_render::render_with(&spec, &a.stems, &b.stems, &sone_render::RenderOptions::default()).with_context(|| format!("rendering {}", spec.id))?;
    let render_seconds = started.elapsed().as_secs_f64();
    // Rule 19 is about B's drums against A's in the blend; without them there, only the hard cut on the one counts.
    let drift = rendered.report.drift_ms();
    if !spec.no_sync && spec.beats_overlap() && drift.abs() > sone_render::MAX_DRIFT_MS {
        bail!("the drums drift {drift:+.1} ms apart across the blend");
    }
    let lag = rendered.report.lag_ms();
    if !spec.no_sync && lag.abs() > sone_render::MAX_LAG_MS {
        bail!("the drums lie {lag:+.1} ms apart");
    }
    let a_from = ((rendered.a_zero_s + spec.clip[0] * rendered.bar_s) * SR).round().max(0.0) as usize;
    let b_zero = rendered.b_zero_s;
    Ok(Placed { rendered, spec, a_from, b_zero, render_seconds })
}

pub fn beat_offset_ms(report: &Report) -> f32 {
    report.align.as_ref().map_or(0.0, |a| a.all.0 as f32)
}

pub fn level_dip_db(report: &Report) -> f32 {
    (report.shortterm_db.min - report.shortterm_db.reference).min(0.0) as f32
}

/// The renderer on the stems of the songs it was given.
#[derive(Default)]
pub struct StemRenderer {
    songs: HashMap<String, Rc<Loaded>>,
    /// Every transition rendered so far: the spec, its report, seconds the rendering took, seconds of audio.
    pub done: Vec<(Spec, Report, f64, f64)>,
}

impl StemRenderer {
    pub fn add(&mut self, song: Rc<Loaded>) {
        self.songs.insert(song.table.id.clone(), song);
    }
}

impl Renderer for StemRenderer {
    fn render(&mut self, plan: &Plan) -> Result<Clip> {
        let a = self.songs.get(&plan.spec.a.id).with_context(|| format!("no stems of {}", plan.spec.a.id))?;
        let b = self.songs.get(&plan.spec.b.id).with_context(|| format!("no stems of {}", plan.spec.b.id))?;
        let placed = render_placed(&plan.spec, a, b)?;
        let seam = SEAM.min(placed.rendered.audio.len());
        let u_seam = placed.spec.clip[1] - seam as f64 / SR / placed.rendered.bar_s;
        let clip = Clip {
            a_from: placed.a_from,
            b_resume: placed.b_frame(u_seam),
            ratio: placed.rendered.ratio,
            beat_offset_ms: beat_offset_ms(&placed.rendered.report),
            level_dip_db: level_dip_db(&placed.rendered.report),
            seam,
            audio: placed.rendered.audio,
        };
        self.done.push((placed.spec, placed.rendered.report, placed.render_seconds, clip.audio.len() as f64 / SR));
        Ok(clip)
    }
}

/// The key-locked stretcher as the engine's glide.
pub struct KeyLock(Stretcher);

impl Default for KeyLock {
    fn default() -> Self {
        Self(Stretcher::new(1.0))
    }
}

impl Glide for KeyLock {
    fn set_ratio(&mut self, ratio: f64) {
        self.0.set_ratio(ratio);
    }
    fn process(&mut self, input: &[[f32; 2]]) -> Vec<[f32; 2]> {
        let out = self.0.process(input.as_flattened());
        out.as_chunks::<2>().0.to_vec()
    }
    fn restart(&mut self, ratio: f64) {
        self.0 = Stretcher::new(ratio);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tone with clicks on a beat, 44.1 kHz stereo.
    fn music(frames: usize) -> Vec<[f32; 2]> {
        (0..frames)
            .map(|i| {
                let t = i as f64 / SR;
                let beat = (t * 2.0).fract();
                let click = if beat < 0.01 { (1.0 - beat / 0.01) * (t * 2.0 * std::f64::consts::PI * 1800.0).sin() } else { 0.0 };
                // Two tones and one that rises slowly, so that no two moments sound alike.
                let rising = 0.2 * (2.0 * std::f64::consts::PI * (500.0 * t + 4.0 * t * t)).sin();
                let tone = 0.3 * (t * 2.0 * std::f64::consts::PI * 220.0).sin() + 0.2 * (t * 2.0 * std::f64::consts::PI * 331.0).sin() + rising;
                let v = (tone + 0.5 * click) as f32;
                [v, v * 0.8]
            })
            .collect()
    }

    /// Where `part` lies in `whole` (frames), searched around `guess`, and how well it fits (1 = the same).
    fn locate(whole: &[[f32; 2]], part: &[[f32; 2]], guess: i64, reach: i64) -> (i64, f64) {
        let mut best = (0, f64::MIN);
        for d in -reach..=reach {
            let at = guess + d;
            if at < 0 || at as usize + part.len() > whole.len() {
                continue;
            }
            let w = &whole[at as usize..at as usize + part.len()];
            let (mut xy, mut xx, mut yy) = (0.0f64, 0.0f64, 0.0f64);
            for (a, b) in w.iter().zip(part) {
                xy += f64::from(a[0]) * f64::from(b[0]);
                xx += f64::from(a[0]) * f64::from(a[0]);
                yy += f64::from(b[0]) * f64::from(b[0]);
            }
            let r = xy / (xx * yy).sqrt().max(1e-12);
            if r > best.1 {
                best = (at, r);
            }
        }
        best
    }

    #[test]
    fn at_its_own_tempo_the_glide_gives_the_song_back() {
        let song = music(6 * 44100);
        let mut g = KeyLock::default();
        g.restart(1.0);
        let mut out = Vec::new();
        for block in song.chunks(2048) {
            out.extend(g.process(block));
        }
        assert!(out.len() > 4 * 44100, "{} frames came out", out.len());
        // Output frame k is input frame k.
        let part = &out[3 * 44100..3 * 44100 + 4096];
        let (at, fit) = locate(&song, part, 3 * 44100, 4000);
        assert!(fit > 0.999, "fit {fit}");
        assert_eq!(at, 3 * 44100, "the output is the song, frame for frame");
    }

    /// Quiet noise with loud clicks at uneven distances; returns the song and the clicks' first frames.
    fn clicks(frames: usize) -> (Vec<[f32; 2]>, Vec<usize>) {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let mut song: Vec<[f32; 2]> = (0..frames).map(|_| [0.01 * noise(), 0.01 * noise()]).collect();
        let mut at = Vec::new();
        let mut t = 9000usize;
        let mut k = 0usize;
        while t + 4000 < frames {
            at.push(t);
            for i in 0..3000 {
                let v = 0.8 * noise() * (-(i as f32) / 600.0).exp();
                song[t + i] = [v, v];
            }
            // The first frame is loud for sure.
            song[t] = [0.8, 0.8];
            k += 1;
            t += 13000 + (k * 7919) % 9000;
        }
        (song, at)
    }

    /// The engine's way of knowing where in the song a glide is: every output frame made at ratio
    /// `r` moves the song on by `1 / r` frames. Checked on the clicks of a click track.
    #[test]
    fn a_glide_is_where_its_ratios_put_it() {
        let (song, click_at) = clicks(40 * 44100);
        for from in [0.985, 1.02] {
            let mut g = KeyLock::default();
            g.restart(from);
            let start = 5 * 44100usize;
            let (mut read, mut at) = (start, start as f64);
            let glide = 12 * 44100;
            let mut made: Vec<[f32; 2]> = Vec::new();
            // Per call: the first output frame it made, where in the song that frame belongs, the ratio.
            let mut calls: Vec<(usize, f64, f64)> = Vec::new();
            while made.len() < glide + 2 * 44100 {
                let ratio = from + (1.0 - from) * (made.len() as f64 / glide as f64).min(1.0);
                g.set_ratio(ratio);
                let out = g.process(&song[read..read + 2048]);
                read += 2048;
                calls.push((made.len(), at, ratio));
                at += out.len() as f64 / ratio;
                made.extend(out);
            }
            let (mut worst, mut checked) = (0i64, 0);
            for &c in &click_at {
                // The output frame the click belongs on.
                let Some(&(first, song_at, ratio)) = calls.iter().rev().find(|call| call.1 <= c as f64) else { continue };
                let ideal = first as i64 + ((c as f64 - song_at) * ratio).round() as i64;
                // Not in the run-in (the engine does not play it), not past what was made.
                if ideal < 16384 || ideal as usize + 3000 > made.len() {
                    continue;
                }
                let from_frame = (ideal - 2000) as usize;
                let found = made[from_frame..from_frame + 4000].iter().position(|f| f[0].abs() > 0.3).map(|i| (from_frame + i) as i64);
                let found = found.unwrap_or_else(|| panic!("from {from}: the click of frame {c} is not within 45 ms of output frame {ideal}"));
                worst = worst.max((found - ideal).abs());
                checked += 1;
            }
            assert!(checked > 20, "{checked} clicks checked");
            assert!(worst <= 64, "from {from}: a click is {worst} frames from where the ratios put it");
        }
    }
}
