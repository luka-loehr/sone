//! A song of which only windows are in memory: the stems around a transition,
//! 16 bit, as the host cut them out of its cache.

use std::cell::Cell;

use sone_core::{BeatTable, Stem};
use sone_render::{BeatGrid, StemSource};

pub struct Window {
    pub start: i64,
    /// Interleaved stereo.
    pub pcm: Vec<i16>,
}

impl Window {
    fn frames(&self) -> i64 {
        (self.pcm.len() / 2) as i64
    }
}

pub struct WindowStems {
    pub table: BeatTable,
    grid: BeatGrid,
    frames: usize,
    lufs: f64,
    windows: [Vec<Window>; 4],
    /// Frames asked for that lie inside the song but in no window.
    missed: Cell<u64>,
    /// The first such read: stem, first frame, frames.
    first_miss: Cell<Option<(usize, i64, usize)>>,
}

fn index(stem: Stem) -> usize {
    match stem {
        Stem::Vocals => 0,
        Stem::Drums => 1,
        Stem::Bass => 2,
        Stem::Other => 3,
    }
}

impl WindowStems {
    /// `frames`: the length of the stems (0: as long as the table says the song is).
    pub fn new(table: BeatTable, lufs: f64, frames: usize) -> Self {
        let frames = if frames > 0 { frames } else { (table.duration * sone_dsp::SR) as usize };
        Self { grid: BeatGrid::from_table(&table), table, frames, lufs, windows: Default::default(), missed: Cell::new(0), first_miss: Cell::new(None) }
    }

    pub fn add(&mut self, stem: Stem, start: i64, pcm: Vec<i16>) {
        self.windows[index(stem)].push(Window { start, pcm });
    }

    pub fn missed(&self) -> u64 {
        self.missed.get()
    }

    /// The first read that found no audio, in words.
    pub fn first_miss(&self) -> String {
        match self.first_miss.get() {
            Some((stem, start, frames)) => format!("{} from {:.3} s for {:.3} s", ["vocals", "drums", "bass", "other"][stem], start as f64 / sone_dsp::SR, frames as f64 / sone_dsp::SR),
            None => String::new(),
        }
    }

    /// Bytes of audio held.
    pub fn bytes(&self) -> usize {
        self.windows.iter().flatten().map(|w| w.pcm.len() * 2).sum()
    }

    pub fn clear(&mut self) {
        self.windows = Default::default();
        self.missed.set(0);
        self.first_miss.set(None);
    }
}

impl StemSource for WindowStems {
    fn id(&self) -> &str {
        &self.table.id
    }
    fn len(&self) -> usize {
        self.frames
    }
    fn read(&self, stem: Stem, start: i64, out: &mut [[f32; 2]]) {
        out.fill([0.0; 2]);
        let end = start + out.len() as i64;
        let mut covered = 0i64;
        for w in &self.windows[index(stem)] {
            let (from, to) = (start.max(w.start), end.min(w.start + w.frames()));
            if to <= from {
                continue;
            }
            covered += to - from;
            let src = &w.pcm[((from - w.start) * 2) as usize..((to - w.start) * 2) as usize];
            for (o, s) in out[(from - start) as usize..(to - start) as usize].iter_mut().zip(src.as_chunks::<2>().0) {
                *o = [f32::from(s[0]) / 32768.0, f32::from(s[1]) / 32768.0];
            }
        }
        let inside = (end.min(self.frames as i64) - start.max(0)).max(0);
        if inside > covered {
            self.missed.set(self.missed.get() + (inside - covered) as u64);
            if self.first_miss.get().is_none() {
                self.first_miss.set(Some((index(stem), start, out.len())));
            }
        }
    }
    fn beats(&self) -> &BeatGrid {
        &self.grid
    }
    fn lufs(&self) -> f64 {
        self.lufs
    }
}
