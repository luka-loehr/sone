//! The moves a lane is made of. Every shape here is one the owner accepted:
//! plain linear fades, hard cuts on the one, B's melody arriving on the one.

use sone_core::{Points, Send, SendKind, Stem};

/// How long before a bar line a hard cut starts to fall (in bars): about 25 ms, so the
/// downbeat that belongs to the other song is not touched.
pub const CUT: f64 = 0.015;

/// Full until `from`, then a straight line to silence at `to`.
pub fn lin_fade(start: f64, from: f64, to: f64, level: f64) -> Points {
    vec![[start, level], [from, level], [to, 0.0]]
}

/// Full until the bar line `at`, gone on it.
pub fn hold_cut(start: f64, at: f64, level: f64) -> Points {
    vec![[start, level], [at - CUT, level], [at, 0.0]]
}

/// Full until `from`, a straight line down to `level` just before the bar line `at`, gone on it: a
/// fade that does not leave a hole (the owner: "melody, bass and drums must not go completely down,
/// only 50 percent"), then the hard cut on the one.
pub fn fade_to_cut(start: f64, from: f64, at: f64, level: f64) -> Points {
    vec![[start, 1.0], [from, 1.0], [at - CUT, level], [at, 0.0]]
}

/// How far A's drums, bass and melody come down before the one when B's beat sneaks in.
pub const SNEAK_FLOOR: f64 = 0.5;

/// Silent before `at`, there on it.
pub fn cut_in(first: f64, at: f64) -> Points {
    if at <= first { vec![[first, 1.0]] } else { vec![[first, 0.0], [at - 0.03, 0.0], [at, 1.0]] }
}

/// B's melody: silent before `at`, there at once but soft, full at `full`.
pub fn at_one(first: f64, at: f64, full: f64) -> Points {
    vec![[first, 0.0], [at - 0.02, 0.0], [at, 0.4], [full, 1.0]]
}

/// A voice that leaves at `exit`. After a real end of a line the fade comes after it,
/// otherwise it has to be done before the bar line.
pub fn voice_out(start: f64, exit: f64, line_ended: bool) -> Points {
    if line_ended { vec![[start, 1.0], [exit, 1.0], [exit + 0.1, 0.0]] } else { vec![[start, 1.0], [exit - 0.12, 1.0], [exit, 0.0]] }
}

/// The echo on a voice's last words (the owner: "LOVE THE ECHO AT THE GESANG. do this more often").
pub fn echo(stem: Stem, from: f64, to: f64, beats: f64, level: f64) -> Send {
    Send {
        kind: SendKind::Echo,
        stem,
        send: vec![[from - 0.1, 0.0], [from, 1.0], [to - 0.05, 1.0], [to, 0.0]],
        beats,
        feedback: 0.55,
        level: Some(level),
        hp: 350.0,
        lp: 4500.0,
        rt60: 2.4,
        post: false,
    }
}

pub fn reverb(stem: Stem, send: Points, level: f64) -> Send {
    Send { kind: SendKind::Reverb, stem, send, beats: 0.75, feedback: 0.55, level: Some(level), hp: 350.0, lp: 4500.0, rt60: 2.6, post: false }
}

/// Value of a lane at `u` (straight lines between the points, flat outside).
pub fn at(points: &Points, u: f64) -> f64 {
    match points.as_slice() {
        [] => 1.0,
        [first, ..] if u <= first[0] => first[1],
        [.., last] if u >= last[0] => last[1],
        _ => {
            let i = points.partition_point(|p| p[0] <= u);
            let (p, q) = (points[i - 1], points[i]);
            if q[0] <= p[0] { q[1] } else { p[1] + (q[1] - p[1]) * (u - p[0]) / (q[0] - p[0]) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        let f = lin_fade(-22.0, -6.0, 0.0, 1.0);
        assert_eq!(at(&f, -10.0), 1.0);
        assert!((at(&f, -3.0) - 0.5).abs() < 1e-9);
        assert_eq!(at(&f, 0.0), 0.0);
        let c = hold_cut(-22.0, 0.0, 1.0);
        assert_eq!(at(&c, -0.02), 1.0);
        assert_eq!(at(&c, 0.0), 0.0);
        let m = at_one(-8.0, 0.0, 2.0);
        assert_eq!(at(&m, -0.5), 0.0);
        assert!((at(&m, 0.0) - 0.4).abs() < 1e-9);
        assert_eq!(at(&m, 2.0), 1.0);
    }
}
