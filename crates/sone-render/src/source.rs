//! Where a song's stems come from.

use sone_core::Stem;

/// What the renderer needs of a song's beat analysis (`struct/<id>.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct BeatGrid {
    /// Beat times, seconds.
    pub beats: Vec<f64>,
    /// Beat numbers (a missed beat leaves a gap).
    pub beat_idx: Vec<i64>,
    /// Which beat (mod 4) is the one.
    pub bar_phase: u8,
    /// Beat period of the steady part, seconds.
    pub beat_s: f64,
    /// The bar lines one by one (indices into `beats`), where they do not follow from one bar
    /// phase: a beat table's, with its corrected sections.
    pub bars: Option<Vec<usize>>,
}

impl BeatGrid {
    pub fn from_analysis(a: &sone_core::TrackAnalysis) -> Self {
        BeatGrid { beats: a.beats.clone(), beat_idx: a.beat_idx.clone(), bar_phase: a.bar_phase, beat_s: a.beat_s, bars: None }
    }

    /// The grid of a beat table: its beats have no gaps, its bar lines are taken as they are
    /// (the planner counts bars this way).
    pub fn from_table(t: &sone_core::BeatTable) -> Self {
        BeatGrid {
            beats: t.times.clone(),
            beat_idx: (0..t.times.len() as i64).collect(),
            bar_phase: t.bars.first().map_or(0, |&b| (b % 4) as u8),
            beat_s: t.beat_s,
            bars: Some(t.bars.iter().copied().filter(|&b| b < t.times.len()).collect()),
        }
    }

    /// Indices into `beats` of the downbeats, in bar order.
    pub fn downbeats(&self) -> Vec<usize> {
        if let Some(bars) = &self.bars {
            return bars.clone();
        }
        (0..self.beats.len()).filter(|&i| self.beat_idx[i].rem_euclid(4) == i64::from(self.bar_phase)).collect()
    }
}

/// The four stems of a song (stereo, 44.1 kHz; they add up to the song), its beats
/// and its loudness. Reads are by window so a source may keep only what a
/// transition needs in memory.
pub trait StemSource {
    fn id(&self) -> &str;
    /// Length in samples per channel.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Fills `out` with the stem from sample `start` on; zeros outside the song.
    fn read(&self, stem: Stem, start: i64, out: &mut [[f32; 2]]);
    fn beats(&self) -> &BeatGrid;
    /// Integrated loudness of the song (the sum of the stems), LUFS.
    fn lufs(&self) -> f64;
    /// The report's `clean_vocals` / `leak_free` flags (BS-RoFormer voice, drums and bass
    /// separated from the song without its voice).
    fn clean_vocals(&self) -> bool {
        true
    }
    fn leak_free(&self) -> bool {
        true
    }
}

/// Stems held in memory.
pub struct MemoryStems {
    pub id: String,
    /// vocals, drums, bass, other, all the same length.
    pub stems: [Vec<[f32; 2]>; 4],
    pub grid: BeatGrid,
    pub lufs: f64,
    pub clean_vocals: bool,
    pub leak_free: bool,
}

impl MemoryStems {
    /// Loudness as the reference measures it: of the stems' sum, added in float32
    /// in the order vocals, drums, bass, other.
    pub fn measure_lufs(stems: &[Vec<[f32; 2]>; 4]) -> f64 {
        let n = stems[0].len();
        let mix: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let mut s = [0.0f32; 2];
                for st in stems {
                    s[0] += st[i][0];
                    s[1] += st[i][1];
                }
                s
            })
            .collect();
        sone_dsp::integrated_loudness(&mix, sone_dsp::SR)
    }
}

fn stem_index(stem: Stem) -> usize {
    match stem {
        Stem::Vocals => 0,
        Stem::Drums => 1,
        Stem::Bass => 2,
        Stem::Other => 3,
    }
}

impl StemSource for MemoryStems {
    fn id(&self) -> &str {
        &self.id
    }
    fn len(&self) -> usize {
        self.stems[0].len()
    }
    fn read(&self, stem: Stem, start: i64, out: &mut [[f32; 2]]) {
        let src = &self.stems[stem_index(stem)];
        for (k, o) in out.iter_mut().enumerate() {
            let i = start + k as i64;
            *o = if i >= 0 && (i as usize) < src.len() { src[i as usize] } else { [0.0; 2] };
        }
    }
    fn beats(&self) -> &BeatGrid {
        &self.grid
    }
    fn lufs(&self) -> f64 {
        self.lufs
    }
    fn clean_vocals(&self) -> bool {
        self.clean_vocals
    }
    fn leak_free(&self) -> bool {
        self.leak_free
    }
}
