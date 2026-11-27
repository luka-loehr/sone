//! How two tempos fit. "Same beat" is the owner's first rule: equal tempo, or
//! one song counted at half or double the other's pace, with a little
//! stretch at most. 2:3 also locks bar lines, but beats of the two songs then
//! interleave, so nothing rhythmic of both may sound together.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lock {
    /// Beat on beat.
    Same,
    /// Every beat of A is every second beat of B (A is counted in half time).
    AHalf,
    /// Every beat of B is every second beat of A (B is counted in half time).
    BHalf,
    /// Two beats of A last as long as three of B.
    TwoToThree,
    /// Three beats of A last as long as two of B.
    ThreeToTwo,
    /// No common grid within the allowed stretch.
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TempoFit {
    pub lock: Lock,
    /// Beats of A per ruler bar.
    pub bpb_a: u32,
    /// Beats of B per ruler bar.
    pub bpb_b: u32,
    /// How much faster (+) or slower (-) B has to play, percent.
    pub stretch_pct: f64,
}

impl TempoFit {
    /// Do the two beats coincide (so that drums of both may overlap)?
    pub fn beat_on_beat(&self) -> bool {
        matches!(self.lock, Lock::Same | Lock::AHalf | Lock::BHalf)
    }

    pub fn ratio_name(&self) -> &'static str {
        match self.lock {
            Lock::Same => "1:1",
            Lock::AHalf => "1:2",
            Lock::BHalf => "2:1",
            Lock::TwoToThree => "2:3",
            Lock::ThreeToTwo => "3:2",
            Lock::None => "none",
        }
    }

    /// Beats of B per beat of A on the ruler.
    pub fn beat_ratio(&self) -> f32 {
        self.bpb_b as f32 / self.bpb_a as f32
    }
}

/// The best lock of `bpm_b` onto `bpm_a` within `max_stretch_pct`.
/// Beat-on-beat locks win over 2:3 when both are possible.
pub fn fit(bpm_a: f64, bpm_b: f64, max_stretch_pct: f64) -> TempoFit {
    // (lock, beats of A per ruler bar, beats of B per ruler bar)
    let options = [
        (Lock::Same, 4u32, 4u32),
        (Lock::AHalf, 2, 4),
        (Lock::BHalf, 4, 2),
        (Lock::TwoToThree, 4, 6),
        (Lock::ThreeToTwo, 6, 4),
    ];
    let mut best: Option<TempoFit> = None;
    for (lock, bpb_a, bpb_b) in options {
        // One ruler bar lasts bpb_a beats of A. B has to fit bpb_b of its beats into it.
        let wanted_b = bpm_a * f64::from(bpb_b) / f64::from(bpb_a);
        let stretch_pct = (wanted_b / bpm_b - 1.0) * 100.0;
        if stretch_pct.abs() > max_stretch_pct {
            continue;
        }
        let candidate = TempoFit { lock, bpb_a, bpb_b, stretch_pct };
        let better = match &best {
            None => true,
            Some(b) => {
                let rank = |f: &TempoFit| (!f.beat_on_beat(), (f.stretch_pct.abs() * 1000.0) as i64);
                rank(&candidate) < rank(b)
            }
        };
        if better {
            best = Some(candidate);
        }
    }
    best.unwrap_or(TempoFit { lock: Lock::None, bpb_a: 4, bpb_b: 4, stretch_pct: 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lab_pairs() {
        // Talking Body → Yo No Lo Sé
        let f = fit(120.0, 120.01, 4.0);
        assert_eq!(f.lock, Lock::Same);
        assert!(f.stretch_pct.abs() < 0.05);
        // Mr. Saxobeat → Don't Stop The Music: B has to play 3.5 % faster
        let f = fit(127.0, 122.7, 4.0);
        assert_eq!(f.lock, Lock::Same);
        assert!((f.stretch_pct - 3.5).abs() < 0.05, "{}", f.stretch_pct);
        // Sprinter (139) → Harleys In Hawaii (counted at 70)
        let f = fit(138.97, 70.0, 4.0);
        assert_eq!((f.lock, f.bpb_a, f.bpb_b), (Lock::BHalf, 4, 2));
        // Girl You Loud (counted at 64.75) → Where Them Girls At (130)
        let f = fit(64.75, 130.0, 4.0);
        assert_eq!((f.lock, f.bpb_a, f.bpb_b), (Lock::AHalf, 2, 4));
        // Empire State Of Mind → Sexy Bitch: 2:3
        let f = fit(86.65, 130.0, 4.0);
        assert_eq!((f.lock, f.bpb_b), (Lock::TwoToThree, 6));
        assert!(!f.beat_on_beat());
        // Pump It Up (130) → Fast Car, slowed (96.8): nothing fits
        assert_eq!(fit(130.0, 96.8, 4.0).lock, Lock::None);
    }
}
