//! The DJ's voice over the music: placed so that it ends at its post, with
//! the music stepping back under it.

/// Ramp of the ducking, seconds.
const RAMP_S: f64 = 0.2;

pub struct Talk {
    /// Stereo frames of the voice.
    pub audio: Vec<[f32; 2]>,
    /// Output sample at which its first frame plays.
    pub from: u64,
    pub duck_db: f32,
}

impl Talk {
    /// Places a line so that it ends at `must_end_by` (output samples). `None` when it would have
    /// to start before `not_before`: then it is too long for its slot.
    pub fn ending_at(audio: Vec<[f32; 2]>, must_end_by: u64, not_before: u64, duck_db: f32) -> Option<Self> {
        let from = must_end_by.checked_sub(audio.len() as u64)?;
        (from >= not_before).then_some(Self { audio, from, duck_db })
    }

    pub fn end(&self) -> u64 {
        self.from + self.audio.len() as u64
    }

    /// Gain of the music at an output sample.
    pub fn music_gain(&self, at: u64, rate: f64) -> f32 {
        let ramp = (RAMP_S * rate) as i64;
        let depth = 10f32.powf(self.duck_db / 20.0);
        let (at, from, end) = (at as i64, self.from as i64, self.end() as i64);
        let down = ((at - (from - ramp)) as f32 / ramp as f32).clamp(0.0, 1.0);
        let up = ((at - end) as f32 / ramp as f32).clamp(0.0, 1.0);
        let under = down * (1.0 - up);
        1.0 + (depth - 1.0) * under
    }

    pub fn voice_at(&self, at: u64) -> [f32; 2] {
        at.checked_sub(self.from).and_then(|i| self.audio.get(i as usize)).copied().unwrap_or([0.0; 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_on_its_post() {
        let rate = 44_100.0;
        let t = Talk::ending_at(vec![[0.5; 2]; 44_100 * 3], 10 * 44_100, 5 * 44_100, -6.0).unwrap();
        assert_eq!(t.end(), 10 * 44_100);
        assert_eq!(t.from, 7 * 44_100);
        assert!((t.music_gain(6 * 44_100, rate) - 1.0).abs() < 1e-6);
        assert!((t.music_gain(8 * 44_100, rate) - 0.501).abs() < 1e-3);
        assert!((t.music_gain(11 * 44_100, rate) - 1.0).abs() < 1e-6);
        // Too long for its slot.
        assert!(Talk::ending_at(vec![[0.0; 2]; 44_100 * 6], 10 * 44_100, 5 * 44_100, -6.0).is_none());
    }
}
