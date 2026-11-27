//! Where the DJ's voice fits. Radio calls it hitting the post: talk over music
//! on which nobody sings and be done when the singer comes in. Here the post
//! is the moment song B starts for real.

use serde::{Deserialize, Serialize};

/// How long before its post the voice has to be done, seconds.
pub const MARGIN_S: f64 = 0.3;
/// How far the music steps back under the voice.
pub const DUCK_DB: f32 = -6.0;

/// The slot on the bar ruler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsBars {
    pub from_u: f64,
    /// The post: B's voice comes in here, or the one.
    pub post_u: f64,
    pub over: String,
}

impl TtsBars {
    pub fn max_seconds(&self, bar_s: f64) -> f64 {
        ((self.post_u - self.from_u) * bar_s - MARGIN_S).max(0.0)
    }
}

/// Ruler bars a line of `seconds` needs: whole pairs of bars, so that phrases stay intact.
pub fn bars_for(seconds: f64, bar_s: f64) -> f64 {
    let bars = ((seconds + MARGIN_S) / bar_s).ceil();
    (bars / 2.0).ceil() * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn room() {
        // 130 BPM: a bar lasts 1.846 s
        assert_eq!(bars_for(6.0, 1.846), 4.0);
        assert_eq!(bars_for(7.5, 1.846), 6.0);
        let slot = TtsBars { from_u: -4.0, post_u: 0.0, over: String::new() };
        assert!((slot.max_seconds(1.846) - 7.084).abs() < 1e-3);
    }
}
