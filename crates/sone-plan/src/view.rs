//! A song as the planner looks at it: bars, where its beat starts and ends,
//! where somebody sings.

use sone_core::BeatTable;

// Indices into `sone_core::example::FEATURES`.
pub const F_VOCALS: usize = 0;
pub const F_DRUMS: usize = 1;
pub const F_KICK: usize = 2;
pub const F_HATS: usize = 3;
pub const F_BASS: usize = 4;
pub const F_OTHER: usize = 5;
pub const F_MIX: usize = 6;
pub const F_SUNG: usize = 7;

/// A kick this far below the song's loud level still counts as "the beat is there".
const BEAT_THERE_DB: f32 = -14.0;
/// A bar's drums within this of their typical level: the song's beat has arrived.
const FULL_DB: f32 = 6.0;
/// A bar's drums more than this below their typical level: the song's beat is over (or has a hole).
pub const GONE_DB: f32 = 12.0;
/// A sixteenth this far under the song's sung level is a breath.
pub const BREATH_DB: f32 = -9.0;
/// A sixteenth note counts as sung above this level, dB against the song's sung level.
const VOICED_DB: f32 = -20.0;

pub struct Song<'a> {
    pub t: &'a BeatTable,
}

impl<'a> Song<'a> {
    pub fn new(t: &'a BeatTable) -> Self {
        Self { t }
    }

    pub fn bars(&self) -> usize {
        self.t.bars.len()
    }

    /// First beat of a bar; bars past the end are continued on the grid.
    pub fn bar_beat(&self, bar: usize) -> usize {
        match self.t.bars.get(bar) {
            Some(&b) => b,
            None => self.t.bars.last().map_or(0, |&b| b + 4 * (bar + 1 - self.t.bars.len())),
        }
    }

    /// Mean of a feature over the beats of a bar, as power when it is a level in dB.
    pub fn level(&self, bar: usize, feature: usize) -> f32 {
        let from = self.bar_beat(bar);
        let beats: Vec<f32> = (from..from + 4).filter_map(|i| self.t.features.get(i).map(|f| f[feature])).collect();
        if beats.is_empty() {
            return -60.0;
        }
        let power: f32 = beats.iter().map(|d| 10f32.powf(d / 10.0)).sum::<f32>() / beats.len() as f32;
        (10.0 * power.log10()).max(-60.0)
    }

    pub fn beat_level(&self, beat: i64, feature: usize) -> f32 {
        usize::try_from(beat).ok().and_then(|i| self.t.features.get(i)).map_or(-60.0, |f| f[feature])
    }

    pub fn sung(&self, beat: i64) -> bool {
        if let Ok(b) = usize::try_from(beat) {
            if let Some(r) = self.t.v16.get(b) {
                if r.iter().filter(|&&x| x > VOICED_DB).count() >= 2 {
                    return true;
                }
            }
        }
        usize::try_from(beat).ok().and_then(|i| self.t.features.get(i)).is_some_and(|f| f[F_SUNG] > 0.5)
    }

    /// Whether the voice sounds on a sixteenth note (beat * 4 + sixteenth). Without sixteenths in
    /// the table the beat's own flag answers.
    pub fn voiced16(&self, sixteenth: usize) -> bool {
        match self.t.v16.get(sixteenth / 4) {
            Some(row) => row[sixteenth % 4] > VOICED_DB,
            None => self.sung((sixteenth / 4) as i64),
        }
    }

    /// Share of a bar's beats on which somebody sings.
    pub fn sung_share(&self, bar: usize) -> f32 {
        let from = self.bar_beat(bar) as i64;
        (from..from + 4).filter(|&i| self.sung(i)).count() as f32 / 4.0
    }

    pub fn has_beat(&self, bar: usize) -> bool {
        self.level(bar, F_KICK) > BEAT_THERE_DB || self.level(bar, F_DRUMS) > BEAT_THERE_DB + 2.0
    }

    /// The first bar from which the beat stays for four bars.
    pub fn first_beat_bar(&self) -> Option<usize> {
        (0..self.bars().saturating_sub(3)).find(|&b| (b..b + 4).all(|k| self.has_beat(k)))
    }

    /// The bar after the last one of the song's beat (its closing bar with thinned drums does not count).
    pub fn beat_end_bar(&self) -> usize {
        let n = self.bars();
        let typical = {
            let mut levels: Vec<f32> = (0..n).map(|b| self.level(b, F_DRUMS)).collect();
            levels.sort_by(f32::total_cmp);
            levels.get(n * 3 / 4).copied().unwrap_or(-60.0)
        };
        (0..n).rev().find(|&b| self.has_beat(b) && self.level(b, F_DRUMS) > typical - 4.0).map_or(n, |b| b + 1)
    }

    /// Bars through which the beat runs without a break: what a tempo grid can be fitted on. At most
    /// `max` bars, from `bar` on forward (or up to `bar`, looking backward). Where the beat does not
    /// run at `bar` itself (a break, an ending without drums), the nearest such stretch of at least
    /// four bars in that direction.
    pub fn steady_run(&self, bar: usize, max: usize, forward: bool) -> [usize; 2] {
        const ENOUGH: usize = 4;
        let n = self.bars();
        if n == 0 {
            return [0, 0];
        }
        let bar = bar.min(n - 1);
        let forward_from = |start: usize| {
            let mut last = start;
            while last + 1 < n && last + 1 < start + max && self.has_beat(last + 1) {
                last += 1;
            }
            [start, last]
        };
        let backward_from = |end: usize| {
            let mut first = end;
            while first > 0 && first + max > end + 1 && self.has_beat(first - 1) {
                first -= 1;
            }
            [first, end]
        };
        let long_enough = |r: &[usize; 2]| r[1] - r[0] + 1 >= ENOUGH;
        let found = if forward {
            (bar..n).filter(|&b| b == bar || self.has_beat(b)).map(forward_from).find(long_enough)
        } else {
            (0..=bar).rev().filter(|&b| b == bar || self.has_beat(b)).map(backward_from).find(long_enough)
        };
        // A song without any such stretch: the bars next to `bar`, whatever plays in them.
        found.unwrap_or(if forward { [bar, (bar + 7).min(n - 1)] } else { [bar.saturating_sub(7), bar] })
    }

    /// The first bar from which somebody sings on at least half the beats of two bars in a row.
    pub fn first_voice_bar(&self) -> Option<usize> {
        (0..self.bars().saturating_sub(1)).find(|&b| self.sung_share(b) >= 0.5 && self.sung_share(b + 1) >= 0.5)
    }

    /// The first bar from which the voice stays: a verse, not a shout or a whispered hook over the intro.
    pub fn first_verse_bar(&self) -> Option<usize> {
        (0..self.bars().saturating_sub(3)).find(|&b| self.sung_share(b) >= 0.5 && (b..b + 4).map(|k| self.sung_share(k)).sum::<f32>() >= 2.8)
    }

    /// The bar after the last one in which the song still plays at its level (before it fades or stops).
    pub fn content_end_bar(&self) -> usize {
        let n = self.bars();
        (0..n).rev().find(|&b| self.level(b, F_MIX) > -8.0).map_or(n, |b| b + 1)
    }

    /// Tempo of a bar (BPM, from its beats), `None` past the song's end.
    pub fn bar_bpm(&self, bar: usize) -> Option<f64> {
        let from = *self.t.bars.get(bar)?;
        let to = self.t.bars.get(bar + 1).copied().unwrap_or(self.t.times.len().saturating_sub(1));
        if to <= from || to >= self.t.times.len() {
            return None;
        }
        Some(60.0 * (to - from) as f64 / (self.t.times[to] - self.t.times[from]))
    }

    /// The tempo the song plays at over `[from, to)` bars: the median of its bars' tempos.
    pub fn local_bpm(&self, from: usize, to: usize) -> Option<f64> {
        let mut v: Vec<f64> = (from..to).filter_map(|k| self.bar_bpm(k)).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        Some(v[v.len() / 2])
    }

    /// Bars around `bar` that carry the beat and keep `bpm` (within 1.5 %): what a tempo grid can be
    /// fitted on. At most `max` bars, from `bar` on forward or up to it backward; where `bar` itself
    /// does not fit, the nearest such stretch of at least four bars in that direction.
    pub fn steady_tempo_run(&self, bar: usize, max: usize, forward: bool, bpm: f64) -> [usize; 2] {
        let n = self.bars();
        if n == 0 {
            return [0, 0];
        }
        let fits = |k: usize| self.has_beat(k) && self.bar_bpm(k).is_some_and(|b| (b / bpm - 1.0).abs() < 0.015);
        let bar = bar.min(n - 1);
        let run_from = |start: usize| -> [usize; 2] {
            if forward {
                let mut last = start;
                while last + 1 < n && last + 1 < start + max && fits(last + 1) {
                    last += 1;
                }
                [start, last]
            } else {
                let mut first = start;
                while first > 0 && first + max > start + 1 && fits(first - 1) {
                    first -= 1;
                }
                [first, start]
            }
        };
        let starts: Vec<usize> = if forward { (bar..n).collect() } else { (0..=bar).rev().collect() };
        starts.into_iter().filter(|&k| fits(k)).map(run_from).find(|r| r[1] - r[0] + 1 >= 4).unwrap_or_else(|| self.steady_run(bar, max, forward))
    }

    /// The level a feature has in the loud part of the song: its 75th percentile over the bars.
    pub fn typical(&self, feature: usize) -> f32 {
        let mut v: Vec<f32> = (0..self.bars()).map(|b| self.level(b, feature)).collect();
        if v.is_empty() {
            return -60.0;
        }
        v.sort_by(f32::total_cmp);
        v[v.len() * 3 / 4]
    }

    /// A bar in which the song's beat plays at full strength (not a break, a fill-out, a fade or an intro).
    pub fn full(&self, bar: usize) -> bool {
        self.drums_within(bar, FULL_DB)
    }

    /// A bar that has the song's beat at all: drums no more than `GONE_DB` below their typical level.
    pub fn carries(&self, bar: usize) -> bool {
        self.drums_within(bar, GONE_DB)
    }

    fn drums_within(&self, bar: usize, db: f32) -> bool {
        bar < self.bars() && self.level(bar, F_DRUMS) >= self.typical(F_DRUMS) - db
    }

    /// The bar after the last run of at least four bars that carry the song's beat: where its own beat is over.
    pub fn full_end_bar(&self) -> usize {
        let full: Vec<bool> = (0..self.bars()).map(|b| self.carries(b)).collect();
        let mut run = 0;
        let mut end = None;
        for (b, &f) in full.iter().enumerate() {
            run = if f { run + 1 } else { 0 };
            if run >= 4 {
                end = Some(b + 1);
            }
        }
        end.unwrap_or(self.bars())
    }

    /// The first bar of the first run of at least four full bars: where the song's beat has arrived.
    pub fn arrival_bar(&self) -> Option<usize> {
        (0..self.bars().saturating_sub(3)).find(|&b| (b..b + 4).all(|k| self.full(k)))
    }

    /// The first bar in which somebody sings on at least half the beats.
    pub fn voice_bar(&self) -> Option<usize> {
        (0..self.bars()).find(|&b| self.sung_share(b) >= 0.5)
    }

    /// `bar` moved onto a phrase start of eight bars when one lies within a bar of it.
    pub fn snap_to_phrase(&self, bar: usize) -> usize {
        for d in [0i64, -1, 1] {
            let k = bar as i64 + d;
            if k >= 0 && self.is_phrase_start(k as usize, 8) {
                return k as usize;
            }
        }
        bar
    }

    /// The first phrase start (eight bars) at or after `bar`.
    pub fn phrase_start_from(&self, bar: usize) -> usize {
        (bar..bar + 8).find(|&k| self.is_phrase_start(k, 8)).unwrap_or(bar)
    }

    /// Breaths: sixteenth notes from which the voice stays at least 9 dB under its sung level for three
    /// sixteenths, right after it sang. The line detector does not count them as line ends, but a voice
    /// can leave there.
    pub fn breaths_from(&self, beat: usize) -> Vec<usize> {
        let v: Vec<f32> = self.t.v16.iter().flat_map(|row| row.iter().copied()).collect();
        let quiet = |i: usize| v.get(i).is_some_and(|&x| x <= BREATH_DB);
        let mut out = Vec::new();
        let mut i = (beat * 4).max(1);
        while i + 2 < v.len() {
            if quiet(i) && quiet(i + 1) && quiet(i + 2) && !quiet(i - 1) {
                out.push(i);
                i += 3;
            } else {
                i += 1;
            }
        }
        out
    }

    /// Sixteenth notes (beat * 4 + sixteenth) on which a line starts, at or after a beat.
    pub fn line_start_from(&self, beat: usize) -> Option<usize> {
        self.t.line_starts.iter().copied().find(|&s| s >= beat * 4)
    }

    /// The last line end before a beat (exclusive), not earlier than `not_before`.
    pub fn line_end_before(&self, beat: usize, not_before: usize) -> Option<usize> {
        self.t.line_ends.iter().copied().filter(|&s| s < beat * 4 && s >= not_before * 4).next_back()
    }

    pub fn is_phrase_start(&self, bar: usize, every: usize) -> bool {
        (bar + every - usize::from(self.t.phrase8) % every) % every == 0
    }
}

/// What song B offers at its beginning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opening {
    /// Somebody sings before the beat comes: the voice can ride on A's beat.
    VoiceFirst { beat_bar: usize },
    /// Melody without beat and without voice: it can grow under A's beat.
    MelodyFirst { beat_bar: usize },
    /// The beat plays for some bars before the voice: it can sneak under A.
    BeatFirst { beat_bar: usize, voice_bar: usize },
    /// Beat and voice start together: the first bars are played twice, once without the voice.
    Together { bar: usize },
    /// No beat in sight.
    NoBeat,
}

pub fn opening(b: &Song<'_>) -> Opening {
    // Where the beat arrives at its own level; a few soft hats in an intro are not the beat yet.
    let Some(beat_bar) = b.arrival_bar().or_else(|| b.first_beat_bar()) else {
        return Opening::NoBeat;
    };
    // A voice before the beat counts from its first sound, a voice over the beat only when it stays.
    let early = b.first_voice_bar().filter(|&v| v + 2 <= beat_bar);
    let voice_bar = early.or_else(|| b.first_verse_bar().filter(|&v| v >= beat_bar).or_else(|| b.first_verse_bar()));
    match voice_bar {
        Some(v) if v + 2 <= beat_bar => Opening::VoiceFirst { beat_bar },
        _ if beat_bar >= 2 && (0..beat_bar).all(|k| b.sung_share(k) < 0.5) => {
            // Is there anything to hear before the beat?
            let melody = (0..beat_bar).map(|k| b.level(k, F_OTHER)).fold(-60.0f32, f32::max);
            if melody > -30.0 { Opening::MelodyFirst { beat_bar } } else { Opening::BeatFirst { beat_bar, voice_bar: voice_bar.unwrap_or(beat_bar + 8) } }
        }
        Some(v) if v >= beat_bar + 1 => Opening::BeatFirst { beat_bar, voice_bar: v },
        Some(_) => Opening::Together { bar: beat_bar },
        None => Opening::BeatFirst { beat_bar, voice_bar: beat_bar + 8 },
    }
}
