//! What a song's stems tell: per-bar levels (`TrackAnalysis`, equal to
//! `reference/python/structure.py`), whether the bar line is on the one
//! ([`phase`]), the voice per sixteenth note and where sung lines end and
//! begin ([`voice`]), key and tuning ([`key`]), and all of it per beat
//! ([`table`], `BeatTable`).
//!
//! Everything runs from one streaming pass over the four stems
//! ([`SongAnalyzer`]: push blocks of mono samples of the four stems, then
//! finish), without threads or files, so it runs in a Worker; with the
//! feature `files`, [`io`] reads FLAC stems from disk. The beats come from
//! the tracker (`sone-beats`, or the Python reference's `.beats` files).

pub mod chroma;
pub mod filter;
pub mod grid;
#[cfg(feature = "files")]
pub mod io;
pub mod key;
pub mod measure;
pub mod phase;
pub mod report;
pub mod stats;
pub mod structure;
pub mod table;
pub mod voice;

use sone_core::{BeatTable, TrackAnalysis};

use chroma::Chroma;
use measure::{Block, Measurements, Measurer};
use structure::{Beats, Structure};

/// A song measured and counted: what every later stage reads.
pub struct Prepared {
    pub structure: Structure,
    pub m: Measurements,
    /// Drum attack envelope for the grid's phase.
    pub env: Vec<f32>,
    pub tuning_cents: f64,
    /// Chroma of bass + other, and of bass + other + vocals.
    pub chroma_bo: Chroma,
    pub chroma_bov: Chroma,
}

pub fn prepare(id: &str, beats: &Beats, m: Measurements) -> anyhow::Result<Prepared> {
    let structure = structure::analyse(id, beats, &m)?;
    let env = grid::attack_envelope(&m);
    let tuning_cents = m.tuning_cents();
    let chroma_bo = Chroma::new(&m, &m.fine_bo, tuning_cents);
    let chroma_bov = Chroma::new(&m, &m.fine_bov, tuning_cents);
    Ok(Prepared { structure, m, env, tuning_cents, chroma_bo, chroma_bov })
}

/// Everything about one song.
pub struct Song {
    pub track: TrackAnalysis,
    pub table: BeatTable,
    pub phase: table::BarPhase,
    pub key: key::Key,
}

pub fn analyse(id: &str, beats: &Beats, m: Measurements) -> anyhow::Result<Song> {
    let p = prepare(id, beats, m)?;
    let phase = table::song_phase(&p);
    let key_chroma = Chroma::new(&p.m, &p.m.fine_key, p.tuning_cents);
    let key = key::estimate(&key::song_chroma(&key_chroma), p.tuning_cents);
    let table = table::beat_table(&p, &phase, &key, table::sung_level(&p));
    Ok(Song { track: p.structure.json, table, phase, key })
}

/// The streaming entry point: the beats first, then the stems in blocks.
pub struct SongAnalyzer {
    id: String,
    beats: Beats,
    measurer: Measurer,
}

impl SongAnalyzer {
    /// `seconds`: about how long the song is (to reserve memory once).
    pub fn new(id: &str, beats: Beats, seconds: f64) -> Self {
        let cuts = structure::cut_points(&beats);
        Self { id: id.to_string(), measurer: Measurer::with_cuts(seconds, cuts), beats }
    }

    /// Mono blocks of the four stems (equal lengths).
    pub fn push(&mut self, block: Block<'_>) {
        self.measurer.push(block);
    }

    pub fn finish(self) -> anyhow::Result<Song> {
        analyse(&self.id, &self.beats, self.measurer.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::SR;

    /// A synthetic song at 120 BPM: a kick on every beat, a snare on 2 and
    /// 4 of the true bar, a bass whose note changes on every true one, and a
    /// voice that sings three beats (releasing 30 ms early) and rests one in
    /// every bar from bar 2 on.
    /// The tracker's downbeats are a beat late (on the true 2).
    fn song(bars: usize) -> (Beats, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let beat = 0.5;
        let lead = 0.25;
        let n = ((lead + bars as f64 * 4.0 * beat + 1.0) * SR) as usize;
        let (mut v, mut d, mut b, o) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        let mut seed = 1u32;
        let mut noise = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        };
        let roots = [55.0, 73.42, 61.74, 82.41];
        let mut beats = Beats::default();
        for k in 0..bars * 4 {
            let t = lead + k as f64 * beat;
            beats.times.push(t);
            beats.downbeat.push(k % 4 == 1);
            let s0 = (t * SR) as usize;
            for i in 0..(0.12 * SR) as usize {
                let x = i as f32 / SR as f32;
                d[s0 + i] += 0.8 * (-x / 0.03).exp() * (2.0 * std::f32::consts::PI * 55.0 * x).sin();
                if k % 2 == 1 {
                    d[s0 + i] += 0.3 * (-x / 0.02).exp() * noise();
                }
            }
            let f = roots[(k / 4) % 4];
            let sung = k >= 8 && k % 4 != 3;
            let release = if k % 4 == 2 { ((beat - 0.03) * SR) as usize } else { usize::MAX };
            for i in 0..(beat * SR) as usize {
                let x = (s0 + i) as f32 / SR as f32;
                b[s0 + i] = 0.3 * (2.0 * std::f32::consts::PI * f * x).sin();
                if sung && i < release {
                    v[s0 + i] = 0.2 * (2.0 * std::f32::consts::PI * 440.0 * x).sin();
                }
            }
        }
        (beats, v, d, b, o)
    }

    #[test]
    fn a_synthetic_song_is_counted_and_its_bar_line_corrected() {
        let bars = 24;
        let (beats, v, d, b, o) = song(bars);
        let mut a = SongAnalyzer::new("synthetic", beats, v.len() as f64 / SR);
        for i in (0..v.len()).step_by(10_000) {
            let j = (i + 10_000).min(v.len());
            a.push(Block { vocals: &v[i..j], drums: &d[i..j], bass: &b[i..j], other: &o[i..j] });
        }
        let s = a.finish().expect("analysis");
        // the tracker's one is the true 2: the bar line moves back a beat
        assert_eq!(s.phase.song.shift, -1, "{:?}", s.phase.song);
        assert_eq!(s.table.beat_shift, -1);
        assert!((s.track.bpm - 120.0).abs() < 0.01);
        assert_eq!(s.track.bars.len() + 1, s.table.bars.len());
        let t = &s.table;
        // the first bar now starts a beat before the tracker's first downbeat
        assert!((t.times[t.bars[0]] - 0.25).abs() < 0.01, "{}", t.times[t.bars[0]]);
        assert!(t.features.iter().all(|f| f.len() == sone_core::example::FEATURES.len()));
        // beat k of the table is the k-th click
        for (k, &time) in t.times.iter().enumerate() {
            assert!((time - (0.25 + 0.5 * k as f64)).abs() < 0.005, "beat {k} at {time}");
        }
        // a line ends on the last sung beat of every bar (the true 3)
        for &e in &t.line_ends {
            assert_eq!(t.beat_in_bar[e / 4], 2, "line end at sixteenth {e}");
        }
        assert!(t.line_ends.len() >= bars - 4, "{:?}", t.line_ends);
    }

    /// Sections the written tables apply: song, first bar, last bar, shift.
    const SECTIONS: [(&str, usize, usize, i64); 3] =
        [("78nx0HDJIFD5xDq2L5420Z", 11, 37, -1), ("57508shL0obX0KUOUL8CJk", 0, 15, -1), ("1YwWHm6Ad1fizVPaQ8nrco", 43, 54, 2)];

    /// The tables in `data/analysis/beats` (written by `analyse`) satisfy all
    /// structural invariants, and the three `SECTIONS` reference tracks agree
    /// bar-for-bar with their legacy `struct_final` analyses when present.
    #[test]
    fn written_tables_are_consistent() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/analysis");
        let Ok(dir) = std::fs::read_dir(root.join("beats")) else { return };
        for e in dir {
            let p = e.unwrap().path();
            let t = BeatTable::load(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert!(t.times.windows(2).all(|w| w[1] > w[0]), "{}", t.id);
            assert!(t.features.iter().all(|f| f.len() == sone_core::example::FEATURES.len()), "{}", t.id);
            assert_eq!((t.v16.len(), t.beat_in_bar.len()), (t.times.len(), t.times.len()), "{}", t.id);
            for &bi in &t.bars[..t.bars.len().saturating_sub(1)] {
                if bi < t.beat_in_bar.len() {
                    assert_eq!(t.beat_in_bar[bi], 0, "{}", t.id);
                }
            }
            let sections: Vec<_> = SECTIONS.iter().filter(|s| s.0 == t.id).collect();
            if sections.is_empty() {
                continue;
            }
            let Ok(a) = TrackAnalysis::load(&root.join("struct_final").join(p.file_name().unwrap())) else { continue };
            assert_eq!(t.bars.len(), a.bars.len() + 1, "{}", t.id);
            let down = a.downbeats();
            let base = |b: usize| a.beat_idx[down[b + 1]] - a.beat_idx[down[b]];
            let shift = |b: usize| sections.iter().find(|s| s.1 <= b && b <= s.2).map_or(i64::from(t.beat_shift), |s| s.3);
            for (b, row) in a.bars.iter().enumerate() {
                let off = (t.times[t.bars[b]] - row.t) / t.beat_s;
                let want = shift(b) as f64;
                let steady = (row.len / (4.0 * t.beat_s) - 1.0).abs() < 0.03;
                assert!((off - want).abs() < 1.5 && (!steady || (off - want).abs() < 0.2), "{} bar {b}: {off}", t.id);
                let len = (t.bars[b + 1] - t.bars[b]) as i64;
                let next = if b + 1 < a.bars.len() { shift(b + 1) } else { shift(b) };
                assert_eq!(len, base(b) + next - shift(b), "{} bar {b}", t.id);
                assert!(base(b) == 4 || base(b) == 8, "{} bar {b}: {} beats", t.id, base(b));
                assert_eq!(t.beat_in_bar[t.bars[b]], 0, "{} bar {b}", t.id);
            }
            if t.id == "78nx0HDJIFD5xDq2L5420Z" {
                assert!((t.times[t.bars[21]] - 29.56).abs() <= 0.02, "Blessings bar 21 at {}", t.times[t.bars[21]]);
            }
        }
    }
}
