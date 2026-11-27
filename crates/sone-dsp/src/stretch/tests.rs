use super::*;

/// Clicks (decaying noise bursts) every `every` samples over a quiet tone.
fn clicks(n: usize, every: usize) -> (Vec<[f32; 2]>, Vec<usize>) {
    let mut x: Vec<[f32; 2]> = (0..n).map(|i| [0.05 * (i as f32 * 0.031).sin(), 0.05 * (i as f32 * 0.031).sin()]).collect();
    let mut at = Vec::new();
    let mut seed = 12345u32;
    let mut i = 3000;
    while i + 3000 < n {
        for k in 0..800 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            let v = 0.8 * r * (-(k as f32) / 120.0).exp();
            x[i + k][0] += v;
            x[i + k][1] += v;
        }
        at.push(i);
        i += every;
    }
    (x, at)
}

/// First sample near each expected position where |y| jumps well above the tone.
fn find_attacks(y: &[[f32; 2]], near: &[f64]) -> Vec<f64> {
    near.iter()
        .map(|&t| {
            let lo = (t as usize).saturating_sub(300);
            let hi = ((t as usize) + 300).min(y.len());
            (lo..hi).find(|&i| y[i][0].abs() > 0.2).map_or(f64::NAN, |i| i as f64)
        })
        .collect()
}

fn tone(freq: f64, n: usize) -> Vec<[f32; 2]> {
    // Faded in over half a second (only its very start counts as an attack).
    (0..n)
        .map(|i| {
            let fade = (i as f64 / 22050.0).min(1.0);
            let v = (0.5 * fade * (std::f64::consts::TAU * freq * i as f64 / 44100.0).sin()) as f32;
            [v, v]
        })
        .collect()
}

#[test]
fn length_is_exact() {
    let (x, _) = clicks(300_000, 20_000);
    for r in [0.96, 0.965, 0.985, 1.0001, 1.016, 1.035, 1.04] {
        assert_eq!(stretch_with(&x, r, Method::Splice).0.len(), (300_000.0 * r).round() as usize, "{r}");
    }
}

#[test]
fn clicks_land_at_position_times_ratio() {
    let (x, at) = clicks(441_000, 19_000);
    for r in [0.96, 0.965, 0.985, 1.016, 1.035, 1.04] {
        let (y, st) = stretch_with(&x, r, Method::Splice);
        let want: Vec<f64> = at.iter().map(|&t| t as f64 * r).collect();
        let got = find_attacks(&y, &want);
        // The burst's first loud sample comes a few samples after its start in both.
        for ((w, g), t) in want.iter().zip(&got).zip(&at) {
            let lead = (*t..*t + 50).find(|&i| x[i][0].abs() > 0.2).unwrap() - *t;
            assert!((w + lead as f64 - g).abs() <= 8.0, "ratio {r}: want {w}, got {g}, stats {st:?}");
        }
        assert!(st.missed == 0, "{st:?}");
    }
}

#[test]
fn attacks_are_copied_as_they_were() {
    let (x, at) = clicks(441_000, 19_000);
    for r in [0.965, 1.035] {
        let y = stretch_with(&x, r, Method::Splice).0;
        let got = find_attacks(&y, &at.iter().map(|&t| t as f64 * r).collect::<Vec<_>>());
        for (&t, &g) in at.iter().zip(&got) {
            let g = g as usize;
            let x0 = (t..t + 50).find(|&i| x[i][0].abs() > 0.2).unwrap();
            // The first 25 ms of every burst are the input's own samples (both bands in step,
            // up to float rounding of the band split).
            let err = (0..1100).map(|k| (y[g + k][0] - x[x0 + k][0]).abs()).fold(0.0, f32::max);
            assert!(err < 1e-5, "ratio {r}, burst at {t}: max error {err}");
        }
    }
}

#[test]
fn streaming_follows_ratio_changes() {
    let (x, at) = clicks(441_000, 19_000);
    let mut s = Stretcher::with_method(1.02, Method::Splice);
    let flat: Vec<f32> = x.iter().flat_map(|v| [v[0], v[1]]).collect();
    let mut out = Vec::new();
    let mut switched_at = 0;
    for (i, chunk) in flat.chunks(1000).enumerate() {
        if i == 300 {
            switched_at = out.len() / 2;
            s.set_ratio(1.0);
        }
        out.extend(s.process(chunk));
    }
    out.extend(s.flush());
    let y: Vec<[f32; 2]> = out.as_chunks::<2>().0.iter().map(|c| [c[0], c[1]]).collect();
    assert!(switched_at > 50_000);
    // Output before the switch ran at 1.02, the rest at 1.0.
    let in_before = switched_at as f64 / 1.02;
    let want_len = switched_at as f64 + (441_000.0 - in_before);
    assert!((y.len() as f64 - want_len).abs() < 50.0, "{} vs {want_len}", y.len());
    // Every burst lands where the ratio in force put it (bursts near the switch may use the old plan).
    let map = |t: f64| if t <= in_before { t * 1.02 } else { switched_at as f64 + (t - in_before) };
    let got = find_attacks(&y, &at.iter().map(|&t| map(t as f64)).collect::<Vec<_>>());
    for (&t, g) in at.iter().zip(&got) {
        if (t as f64 - in_before).abs() < 25_000.0 {
            continue;
        }
        let x0 = (t..t + 50).find(|&i| x[i][0].abs() > 0.2).unwrap() as f64;
        assert!((map(x0) - g).abs() <= 10.0, "burst {t}: want {}, got {g}", map(x0));
    }
}

/// Smallest and largest RMS over windows of ten periods of a tone.
fn level_range(y: &[[f32; 2]], freq: f64) -> (f32, f32) {
    let w = (10.0 * 44100.0 / freq).round() as usize;
    y.windows(w)
        .step_by(w / 4)
        .map(|c| (c.iter().map(|v| v[0] * v[0]).sum::<f32>() / c.len() as f32).sqrt())
        .fold((f32::MAX, 0.0f32), |(a, b), v| (a.min(v), b.max(v)))
}

/// Largest sample-to-sample step after the fade-in, relative to a pure tone's steepest one.
fn roughness(y: &[[f32; 2]], freq: f64) -> f32 {
    let steep = 0.5 * std::f32::consts::TAU * freq as f32 / 44100.0;
    y[22050..y.len() - 4000].windows(2).map(|w| (w[1][0] - w[0][0]).abs()).fold(0.0, f32::max) / steep
}

#[test]
fn held_tones_splice_seamlessly() {
    for freq in [80.0, 220.0, 1000.0] {
        let x = tone(freq, 200_000);
        for r in [0.965, 1.035] {
            let (y, st) = stretch_with(&x, r, Method::Splice);
            assert!(st.coarse > 0 && st.min_rho > 0.99, "{freq} Hz at {r}: {st:?}");
            let rough = roughness(&y, freq);
            assert!(rough < 1.03, "{freq} Hz at {r}: {rough} {st:?}");
        }
    }
}

#[test]
fn phase_vocoder_length_timing_and_tones() {
    let (x, at) = clicks(441_000, 19_000);
    for r in [0.96, 0.985, 1.016, 1.04] {
        let (y, st) = stretch_with(&x, r, Method::PhaseVocoder);
        assert_eq!(y.len(), (441_000.0 * r).round() as usize);
        let want: Vec<f64> = at.iter().map(|&t| t as f64 * r).collect();
        let got = find_attacks(&y, &want);
        for ((w, g), t) in want.iter().zip(&got).zip(&at) {
            let lead = (*t..*t + 50).find(|&i| x[i][0].abs() > 0.2).unwrap() - *t;
            assert!((w + lead as f64 - g).abs() <= 8.0, "ratio {r}: want {w}, got {g}, {st:?}");
        }
        assert!(st.resets > 0, "{st:?}");
    }
    for freq in [80.0, 220.0, 1000.0] {
        let x = tone(freq, 200_000);
        for r in [0.965, 1.035] {
            let (y, _) = stretch_with(&x, r, Method::PhaseVocoder);
            let (lo, hi) = level_range(&y[22050..y.len() - 8000], freq);
            assert!(20.0 * (hi / lo).log10() < 0.5, "{freq} Hz at {r}: level {lo}..{hi}");
            let rough = roughness(&y, freq);
            assert!(rough < 1.03, "{freq} Hz at {r}: {rough}");
        }
    }
}

#[test]
fn phase_vocoder_streams_like_offline() {
    let (x, _) = clicks(200_000, 19_000);
    let flat: Vec<f32> = x.iter().flat_map(|v| [v[0], v[1]]).collect();
    let mut s = Stretcher::with_method(1.02, Method::PhaseVocoder);
    let mut out = Vec::new();
    for c in flat.chunks(3000) {
        out.extend(s.process(c));
    }
    out.extend(s.flush());
    let off = stretch_with(&x, 1.02, Method::PhaseVocoder).0;
    assert!((out.len() as f64 / 2.0 - off.len() as f64).abs() <= 1.0, "{} vs {}", out.len() / 2, off.len());
    let n = off.len().min(out.len() / 2);
    let err = (0..n).map(|i| (out[2 * i] - off[i][0]).abs()).fold(0.0, f32::max);
    assert!(err < 1e-5, "{err}");
}

#[test]
#[ignore]
fn pv_tone_levels() {
    for freq in [80.0, 81.0, 220.0, 1000.0] {
        let x = tone(freq, 200_000);
        for r in [0.965, 1.0, 1.035] {
            let (y, _) = stretch_with(&x, r, Method::PhaseVocoder);
            let (lo, hi) = level_range(&y[22050..y.len() - 8000], freq);
            eprintln!("{freq} Hz r {r}: rms {lo:.4}..{hi:.4} ({:.2} dB)", 20.0 * (hi / lo).log10());
        }
    }
}
