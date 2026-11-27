//! Tiny synthetic data for smoke runs and tests: songs with stems (a sine
//! bass, noise-burst drums, a vibrato voice with pauses, a sawtooth pad),
//! songs with Ear labels (clicks on beats, accented downbeats, a drone in a
//! known key, tone phrases as "voice"), and well-formed random `Example`s
//! whose labels follow simple rules, so a network can learn them.

use std::f32::consts::TAU;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use sone_core::example::{BeatFeatures, FEATURES};
use sone_core::spec::Recipe;
use sone_core::{Example, Issue, Label};
use sone_infer::random::Rng;

use crate::audio::{SAMPLE_RATE, write_wav};

const SR: f32 = SAMPLE_RATE as f32;

fn midi_hz(m: f32) -> f32 {
    440.0 * 2f32.powf((m - 69.0) / 12.0)
}

/// Four stereo stems of one synthetic song (vocals, drums, bass, other).
pub fn stems_song(seconds: f32, rng: &mut Rng) -> [[Vec<f32>; 2]; 4] {
    let n = (seconds * SR) as usize;
    let bpm = 100.0 + 30.0 * rng.uniform();
    let beat = 60.0 / bpm;
    let root = 36.0 + (rng.next_u64() % 12) as f32; // bass root, MIDI
    let scale = [0.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0];
    let mut out: [[Vec<f32>; 2]; 4] = std::array::from_fn(|_| [vec![0.0; n], vec![0.0; n]]);
    let beats = (seconds / beat) as usize + 1;
    let notes: Vec<f32> = (0..beats).map(|_| scale[(rng.next_u64() % 7) as usize]).collect();
    let voice_pan = 0.4 + 0.2 * rng.uniform();
    for i in 0..n {
        let t = i as f32 / SR;
        let b = (t / beat) as usize;
        let tb = t - b as f32 * beat; // time since the beat
        let bar = b / 4;
        // bass: one note per beat, decaying
        let f = midi_hz(root + notes[b]);
        let env = (-tb * 3.0).exp();
        let bass = 0.25 * env * ((TAU * f * t).sin() + 0.3 * (TAU * 2.0 * f * t).sin());
        out[2][0][i] = bass;
        out[2][1][i] = bass;
        // voice: sings two bars, rests two bars; vibrato 5.5 Hz, harmonics
        if bar % 4 < 2 {
            let f0 = midi_hz(root + 24.0 + notes[b.saturating_sub(1)] + 12.0 * ((bar % 2) as f32));
            let vib = 1.0 + 0.017 * (TAU * 5.5 * t).sin();
            let edge = (tb / 0.02).min(1.0) * ((beat - tb) / 0.02).min(1.0);
            let mut v = 0.0;
            for k in 1..7 {
                v += (TAU * f0 * vib * k as f32 * t).sin() / k as f32;
            }
            let v = 0.12 * edge * v;
            out[0][0][i] = v * (1.0 - voice_pan) * 2.0;
            out[0][1][i] = v * voice_pan * 2.0;
        }
        // pad: a triad per bar, sawtooth from eight harmonics, slightly detuned per side
        let chord = root + 12.0 + notes[(bar * 4).min(beats - 1)];
        for (c, detune) in [(0, 0.998f32), (1, 1.002)] {
            let mut p = 0.0;
            for iv in [0.0, 3.0, 7.0] {
                let f = midi_hz(chord + iv) * detune;
                for k in 1..9 {
                    p += (TAU * f * k as f32 * t).sin() / k as f32;
                }
            }
            out[3][c][i] = 0.02 * p;
        }
    }
    // drums: kick on 1 and 3, snare on 2 and 4, hats on eighths
    for b in 0..beats {
        let at = (b as f32 * beat * SR) as usize;
        let kind = b % 2;
        let len = (0.25 * SR) as usize;
        for j in 0..len {
            let i = at + j;
            if i >= n {
                break;
            }
            let tj = j as f32 / SR;
            let v = if kind == 0 {
                let f = 50.0 + 60.0 * (-tj * 30.0).exp();
                0.5 * (-tj * 12.0).exp() * (TAU * f * tj).sin()
            } else {
                0.25 * (-tj * 25.0).exp() * rng.sym(1.0)
            };
            out[1][0][i] += v;
            out[1][1][i] += v;
        }
        for half in 0..2 {
            let at = at + (half as f32 * beat * 0.5 * SR) as usize;
            let mut prev = 0.0;
            for j in 0..(0.04 * SR) as usize {
                let i = at + j;
                if i >= n {
                    break;
                }
                let noise = rng.sym(1.0);
                let v = 0.08 * (-(j as f32 / SR) * 90.0).exp() * (noise - prev);
                prev = noise;
                out[1][0][i] += v * 0.7;
                out[1][1][i] += v * 1.3;
            }
        }
    }
    out
}

/// `songs` directories `<out>/songNN/{mixture,vocals,drums,bass,other}.wav`.
pub fn cochlea_dataset(out: &Path, songs: usize, seconds: f32, seed: u64) -> Result<()> {
    let mut rng = Rng::new(seed);
    for s in 0..songs {
        let dir = out.join(format!("song{s:02}"));
        std::fs::create_dir_all(&dir)?;
        let stems = stems_song(seconds, &mut rng);
        let n = stems[0][0].len();
        let mix: [Vec<f32>; 2] = std::array::from_fn(|c| (0..n).map(|i| stems.iter().map(|st| st[c][i]).sum()).collect());
        write_wav(&dir.join("mixture.wav"), &mix[0], &mix[1])?;
        for (name, st) in crate::data_cochlea::STEM_NAMES.iter().zip(&stems) {
            write_wav(&dir.join(format!("{name}.wav")), &st[0], &st[1])?;
        }
    }
    Ok(())
}

/// Ear songs: `<out>/songNN.wav` + `<out>/songNN.json` (the label format of `cookbooks/ear.md`).
pub fn tympanum_dataset(out: &Path, songs: usize, seconds: f32, seed: u64) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let mut rng = Rng::new(seed);
    for s in 0..songs {
        let n = (seconds * SR) as usize;
        let bpm = 90.0 + 50.0 * rng.uniform();
        let beat = 60.0 / bpm;
        let offset = 0.1 + 0.4 * rng.uniform();
        let key = (rng.next_u64() % 24) as usize;
        let tuning = rng.sym(30.0);
        let tonic = 48.0 + (key % 12) as f32 + tuning / 100.0;
        let third = if key >= 12 { 3.0 } else { 4.0 };
        let mut audio = vec![0.0f32; n];
        // drone: tonic triad, low and steady, carries the key
        for (i, a) in audio.iter_mut().enumerate() {
            let t = i as f32 / SR;
            let mut v = 0.0;
            for iv in [0.0, third, 7.0, 12.0] {
                v += (TAU * midi_hz(tonic + iv) * t).sin();
            }
            *a = 0.03 * v;
        }
        let (mut beats, mut downbeats, mut phrases, mut vocals, mut line_ends) = (vec![], vec![], vec![], vec![], vec![]);
        let mut b = 0;
        loop {
            let t = offset + b as f32 * beat;
            if t > seconds - 0.3 {
                break;
            }
            beats.push(t);
            let down = b % 4 == 0;
            if down {
                downbeats.push(t);
            }
            if b % 32 == 0 {
                phrases.push(t);
            }
            // click: accented and lower on the downbeat
            let (f, amp) = if down { (1000.0, 0.5) } else { (2000.0, 0.3) };
            let at = (t * SR) as usize;
            for j in 0..(0.03 * SR) as usize {
                if at + j < n {
                    let tj = j as f32 / SR;
                    audio[at + j] += amp * (-tj * 150.0).exp() * (TAU * f * tj).sin();
                }
            }
            // a sung "line": beats 1-3 of every odd bar, a tone in the key
            if b % 8 == 4 {
                let end = t + 3.0 * beat;
                vocals.push([t as f64, end as f64]);
                line_ends.push(end);
                let f0 = midi_hz(tonic + 24.0 + 7.0);
                for j in 0..(3.0 * beat * SR) as usize {
                    if at + j < n {
                        let tj = j as f32 / SR;
                        let vib = 1.0 + 0.015 * (TAU * 5.0 * tj).sin();
                        audio[at + j] += 0.1 * ((TAU * f0 * vib * tj).sin() + 0.5 * (TAU * 2.0 * f0 * vib * tj).sin());
                    }
                }
            }
            b += 1;
        }
        let name = format!("song{s:02}");
        write_wav(&out.join(format!("{name}.wav")), &audio, &audio)?;
        let label = serde_json::json!({
            "audio": format!("{name}.wav"),
            "beats": beats, "downbeats": downbeats, "vocals": vocals, "line_ends": line_ends,
            "phrase_starts": phrases, "key": sone_infer::config::KEYS[key], "tuning_cents": tuning,
        });
        std::fs::write(out.join(format!("{name}.json")), serde_json::to_string_pretty(&label)?)?;
    }
    Ok(())
}

fn beat_row(rng: &mut Rng, vocal: bool, drums: bool, beat_in_bar: usize, bar: usize) -> BeatFeatures {
    let mut f = vec![0.0f32; FEATURES.len()];
    let db = |on: bool, rng: &mut Rng| if on { -6.0 + rng.sym(3.0) } else { -60.0 + rng.sym(5.0) };
    f[0] = db(vocal, rng);
    f[1] = db(drums, rng);
    f[2] = f[1] - 3.0;
    f[3] = f[1] - 6.0;
    f[4] = db(true, rng);
    f[5] = db(true, rng);
    f[6] = -3.0 + rng.sym(2.0);
    f[7] = if vocal { 1.0 } else { 0.0 };
    f[8] = 0.0;
    f[9] = beat_in_bar as f32;
    f[10] = (bar % 8) as f32;
    f[11] = rng.uniform();
    for c in 12..24 {
        f[c] = rng.uniform();
    }
    f
}

/// A random, well-formed `Example`. The recipe follows from B's intro (voice
/// without beat → VoiceOverBeat, beat before voice → Sneak, neither →
/// OutroUnderIntro, tempo gap → Gap); the bars and lanes follow the recipe.
/// Broken examples (`good == 0`) have one perturbation with its issue.
pub fn example(rng: &mut Rng, id: usize) -> Example {
    let gap = rng.uniform() < 0.15;
    let tempo_a = 118.0 + rng.sym(10.0);
    let tempo_b = if gap { tempo_a * 0.75 } else { tempo_a * (1.0 + rng.sym(0.03)) };
    let intro = (rng.next_u64() % 3) as usize; // 0 voice only, 1 beat only, 2 neither
    let recipe = if gap {
        Recipe::Gap
    } else {
        [Recipe::VoiceOverBeat, Recipe::Sneak, Recipe::OutroUnderIntro][intro]
    };
    let a_beats: Vec<BeatFeatures> = (0..80).map(|i| beat_row(rng, i < 56 && (i / 8) % 2 == 0, true, i % 4, i / 4)).collect();
    let b_beats: Vec<BeatFeatures> = (0..128)
        .map(|i| {
            let before = i < 64;
            let (v, d) = if before { (intro == 0, intro == 1) } else { (true, true) };
            beat_row(rng, v, d, i % 4, i / 4)
        })
        .collect();
    let (start_bars, swap_bars) = match recipe {
        Recipe::VoiceOverBeat => (8.0, 0.0),
        Recipe::Sneak => (8.0, 4.0),
        Recipe::OutroUnderIntro => (6.0, 0.0),
        Recipe::Gap => (2.0, 0.0),
    };
    let lanes: Vec<[f32; 8]> = (0..80)
        .map(|i| {
            let u = (i as f32 - 64.0) / 4.0; // bars relative to the one
            let before = u < 0.0;
            let ramp = |from: f32| if u < -from { 1.0 } else if before { (-u / from).clamp(0.0, 1.0) } else { 0.0 };
            let a_on = if before { 1.0 } else { 0.0 };
            let b_in = |at: f32| if u >= -at { 1.0 } else { 0.0 };
            match recipe {
                Recipe::VoiceOverBeat => [if u < -start_bars { 1.0 } else { 0.0 }, a_on, a_on, ramp(start_bars), b_in(start_bars), b_in(0.0), b_in(0.0), b_in(0.0)],
                Recipe::Sneak => [a_on, ramp(swap_bars + 2.0), ramp(swap_bars + 2.0), ramp(swap_bars + 2.0), b_in(0.0), b_in(start_bars), b_in(swap_bars), b_in(0.0)],
                Recipe::OutroUnderIntro => [0.0, a_on, a_on, 0.0, b_in(0.0), b_in(0.0), b_in(0.0), b_in(start_bars)],
                Recipe::Gap => [a_on, if u < -2.0 { 1.0 } else { 0.0 }, if u < -2.0 { 1.0 } else { 0.0 }, 0.0, b_in(0.0), b_in(0.0), b_in(0.0), b_in(0.0)],
            }
        })
        .collect();
    let mut ex = Example {
        id: format!("synth-{id}"),
        a: format!("a{}", id % 40),
        b: format!("b{}", (id * 7) % 40),
        a_beats,
        b_beats,
        tempo_a,
        tempo_b,
        beat_ratio: if gap { 0.75 } else { 1.0 },
        key_distance: (rng.next_u64() % 4) as u8,
        recipe,
        start_bars,
        swap_bars,
        lanes,
        measures: Default::default(),
        label: Label { good: 1, issues: vec![], source: "rules".into() },
    };
    if rng.uniform() < 0.5 {
        let issue = match rng.next_u64() % 4 {
            0 => {
                // B's drums in from the start
                ex.lanes.iter_mut().for_each(|l| l[5] = 1.0);
                Issue::BDrumsBeforeOne
            }
            1 => {
                // A's voice and B's voice together
                ex.lanes.iter_mut().for_each(|l| {
                    l[0] = 1.0;
                    l[4] = 1.0;
                });
                Issue::TwoVoices
            }
            2 => {
                // A's beat gone four bars before the one
                ex.lanes.iter_mut().skip(48).take(16).for_each(|l| l[1] = 0.0);
                Issue::ABeatGoneBeforeOne
            }
            _ => {
                // A keeps playing after the one
                ex.lanes.iter_mut().skip(64).for_each(|l| l[1] = 1.0);
                Issue::TailAfterOne
            }
        };
        ex.label = Label { good: 0, issues: vec![issue], source: "perturbed".into() };
    }
    ex
}

/// `n` examples as JSON lines.
pub fn examples_file(path: &Path, n: usize, seed: u64) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut rng = Rng::new(seed);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for i in 0..n {
        writeln!(f, "{}", serde_json::to_string(&example(&mut rng, i))?)?;
    }
    Ok(())
}
