//! Tests on synthetic songs (no audio files needed).

use sone_core::{AlignPairs, Deck, Segment, Shape, Spec, Stem};

use crate::deck::{Build, assemble, build_deck};
use crate::source::{BeatGrid, MemoryStems, StemSource};
use crate::{grid, render};

const SR: f64 = 44100.0;

/// A song at `bpm` whose beats are listed exactly; the drums are a click on every
/// beat `drum_delay` seconds late; the other stems are a ramp (vocals), silence and
/// a constant (so tests can tell where samples came from).
fn song(bpm: f64, secs: f64, drum_delay: f64) -> MemoryStems {
    let n = (secs * SR) as usize;
    let beat = 60.0 / bpm;
    let beats: Vec<f64> = (0..).map(|k| 0.5 + k as f64 * beat).take_while(|&t| t < secs - 0.5).collect();
    let beat_idx: Vec<i64> = (0..beats.len() as i64).collect();
    let mut drums = vec![[0.0f32; 2]; n];
    for &t in &beats {
        let i0 = ((t + drum_delay) * SR) as usize;
        for k in 0..400 {
            if i0 + k < n {
                // A bright decaying click.
                let v = if k % 2 == 0 { 0.5 } else { -0.5 } * (-(k as f32) / 60.0).exp();
                drums[i0 + k] = [v, v];
            }
        }
    }
    let vocals: Vec<[f32; 2]> = (0..n).map(|i| [i as f32 / n as f32, -(i as f32) / n as f32]).collect();
    let stems = [vocals, drums, vec![[0.0; 2]; n], vec![[0.25, 0.25]; n]];
    let lufs = MemoryStems::measure_lufs(&stems);
    MemoryStems {
        id: format!("song{bpm}"),
        stems,
        grid: BeatGrid { beats, beat_idx, bar_phase: 0, beat_s: beat, bars: None },
        lufs,
        clean_vocals: true,
        leak_free: true,
    }
}

fn deck(id: &str, anchor_bar: usize) -> Deck {
    serde_json::from_value(serde_json::json!({ "id": id, "anchor_bar": anchor_bar })).unwrap()
}

#[test]
fn segments_are_spliced_from_the_right_bars() {
    let s = song(120.0, 60.0, 0.0);
    let beat = 0.5;
    let bar = 4.0 * beat * SR; // 88200
    // Ruler 0..4 bars: bars 0-1 from the song's bar 8, bars 2-3 from its bar 20.
    let segs = [Segment { u0: 0.0, u1: Some(2.0), src: 8.0 }, Segment { u0: 2.0, u1: None, src: 20.0 }];
    let t_anchor = 1.0;
    let x = assemble(&s, Stem::Vocals, t_anchor, beat, &segs, 0.0, 4.0, 4.0);
    assert_eq!(x.len(), 4 * bar as usize);
    let n = s.len() as f32;
    let at = |t: f64| (t * SR).round() as usize;
    // First segment: straight copy from song time 1 s + 8 bars.
    let s0 = at(t_anchor + 8.0 * 2.0);
    assert_eq!(x[0][0], s0 as f32 / n);
    assert_eq!(x[1000][0], (s0 + 1000) as f32 / n);
    // Second segment starts at bar 2 with the song's bar 20, after a 264-sample equal-power crossfade.
    let d0 = 2 * bar as usize;
    let s1 = at(t_anchor + 20.0 * 2.0);
    assert_eq!(x[d0][0], s1 as f32 / n);
    assert_eq!(x[d0 + 5000][0], (s1 + 5000) as f32 / n);
    let old = (s0 + d0 - 264) as f32 / n;
    let new = (s1 - 264) as f32 / n;
    assert_eq!(x[d0 - 264][0], old); // cos(0) = 1, sin(0) = 0
    // Halfway through: old * cos + new * sin (equal power).
    let w = std::f64::consts::FRAC_PI_2 * 132.0 / 263.0;
    let want = (s0 + d0 - 132) as f64 / n as f64 * w.cos() + (s1 - 132) as f64 / n as f64 * w.sin();
    assert!((f64::from(x[d0 - 132][0]) - want).abs() < 1e-6, "{} vs {want}", x[d0 - 132][0]);
    let _ = new;
}

#[test]
fn deck_applies_lanes_gain_and_raw() {
    let s = song(120.0, 30.0, 0.0);
    let mut d = deck("song120", 2);
    // Other fades out linearly over bar 1; vocals muted; whole deck -6 dB.
    d.lanes.insert(Stem::Other, vec![[1.0, 1.0], [2.0, 0.0]]);
    d.lanes.insert(Stem::Vocals, vec![[0.0, 0.0]]);
    d.shape = Shape::Lin;
    d.gain_db = vec![[0.0, -6.0]];
    let out = build_deck(&s, 1.0, 0.5, &d, 0.0, 3.0, Build::default());
    let bar = 88200usize;
    let g = 10f32.powf(-6.0 / 20.0);
    // Drums are silent between clicks at 60000; other alone at 0.25 before the fade.
    assert!((out.total[60000][0] - 0.25 * g).abs() < 1e-6);
    let u = 1.0 + 60000.0 / 88200.0;
    assert!((out.total[bar + 60000][0] - 0.25 * (2.0 - u) as f32 * g).abs() < 1e-5);
    assert!(out.total[2 * bar + 60000][0].abs() < 1e-6);
    assert_eq!(out.lanes[&Stem::Vocals][500], 0.0);
    // Raw: every stem as it is.
    let raw = build_deck(&s, 1.0, 0.5, &d, 0.0, 3.0, Build { raw: true, ..Build::default() });
    let v = s.stems[0][44100 + 60000][0];
    assert!((raw.total[60000][0] - (v + 0.25)).abs() < 1e-6);
}

#[test]
fn grid_finds_tempo_and_drum_phase() {
    // Beats listed on time, the drum clicks sit 10 ms late: the grid follows the drums.
    let s = song(123.0, 90.0, 0.010);
    let g = grid(&s, 20, 16, 0, None).unwrap();
    assert!((g.beat - 60.0 / 123.0).abs() < 1e-9);
    // The attack envelope (1.5 kHz high-pass, 32-sample hops, smoothed) puts it within a few ms.
    assert!((g.shift - 0.010).abs() <= 0.003, "{}", g.shift);
    let down = s.grid.downbeats();
    assert!((g.t - (s.grid.beats[down[20]] + g.shift)).abs() < 1e-9);
    // beat_shift moves the bar line by whole beats.
    let g1 = grid(&s, 20, 16, 1, None).unwrap();
    assert!((g1.t - g.t - 60.0 / 123.0).abs() < 1e-6);
}

#[test]
fn render_locks_b_to_a() {
    let a = song(124.0, 120.0, 0.0);
    let b = song(122.0, 120.0, 0.004);
    let mut da = deck("a", 40);
    da.until = Some(0.0);
    da.lanes.insert(Stem::Vocals, vec![[0.0, 0.0]]);
    da.lanes.insert(Stem::Drums, vec![[-0.01, 1.0], [0.0, 0.0]]);
    da.lanes.insert(Stem::Other, vec![[-0.01, 1.0], [0.0, 0.0]]);
    let mut db = deck("b", 8);
    db.from = Some(-4.0);
    db.lanes.insert(Stem::Vocals, vec![[0.0, 0.0]]);
    let spec = Spec {
        id: "t1".into(),
        clip: [-8.0, 8.0],
        first_event: -4.0,
        last_event: 0.0,
        swap_u: None,
        no_sync: false,
        align_pairs: Some(AlignPairs { a: [-8.0, -4.0], b: [0.0, 4.0] }),
        recipe: None,
        a: da,
        b: db,
        technique: String::new(),
        why: String::new(),
        answer: String::new(),
        key_txt: String::new(),
        steps: Vec::new(),
        fx: serde_json::Value::Null,
    };
    let r = render(&spec, &a, &b).unwrap();
    let bar = 4.0 * 60.0 / 124.0;
    assert_eq!(r.audio.len(), (16.0 * bar * SR).round() as usize);
    assert!((r.report.tempo_a - 124.0).abs() < 1e-3 && (r.report.tempo_b - 122.0).abs() < 1e-3);
    assert!((r.ratio - 122.0 / 124.0).abs() < 1e-9);
    // Both grids follow their drums; with them B's clicks land on A's (lag 0 within a hop).
    let align = r.report.align.as_ref().unwrap();
    assert!(align.all.0.abs() <= 0.73, "{:?}", align);
    assert_eq!(r.report.bars.len(), 16);
    // A plays until the one, B from bar -4: the meter bridge shows it.
    assert!(r.report.bars[0].b <= -100.0 + 1e-9 && r.report.bars[0].a > -60.0);
    assert!(r.report.bars[15].a <= -100.0 + 1e-9 && r.report.bars[15].b > -60.0);
    assert_eq!(r.lanes.a.gain.len(), 4);
    assert!((r.bar_s - bar).abs() < 1e-12);
}

#[test]
fn explicit_bar_lines_with_a_short_bar() {
    // A beat table whose bar 3 has only three beats: bar 5 starts on beat 19, not 20.
    let mut s = song(120.0, 60.0, 0.0);
    let bars: Vec<usize> = [0, 4, 8, 12, 15, 19, 23, 27, 31, 35].into_iter().collect();
    s.grid.bars = Some(bars.clone());
    let g = grid(&s, 5, 16, 0, None).unwrap();
    let want = s.grid.beats[bars[5]];
    assert!((g.t - want).abs() <= 0.045 + 1e-9, "{} vs {want}", g.t);
    assert!((g.t - want - g.shift).abs() < 1e-9);
    // Without explicit bars, bar 5 would be beat 20.
    s.grid.bars = None;
    let g0 = grid(&s, 5, 16, 0, None).unwrap();
    assert!((g0.t - g0.shift - s.grid.beats[20]).abs() < 1e-9);
}

#[test]
fn bad_windows_do_not_panic() {
    let s = song(120.0, 60.0, 0.0);
    // One bar of fit_bars (1 beat): falls back to the usual window.
    let g = grid(&s, 20, 16, 0, Some([20, 20])).unwrap();
    assert!((g.beat - 0.5).abs() < 1e-9);
    assert!(grid(&s, 10_000, 16, 0, None).is_err());
    assert!(grid(&s, 20, 16, 0, Some([10_000, 10_001])).is_ok());
}
