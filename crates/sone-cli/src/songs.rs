//! Songs on disk: the stems (`<stems>/<id>/*.flac`), the beat table
//! (`<beats>/<id>.json`), and the song itself as the sum of its stems.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context, Result};
use sone_core::BeatTable;
use sone_render::{MemoryStems, StemSource, files};

/// Loudness the station plays at, LUFS. Songs louder than this come down to it; quieter ones stay.
pub const STATION_LUFS: f64 = -14.0;

pub struct Loaded {
    pub stems: MemoryStems,
    pub table: BeatTable,
    /// The song (what is on air outside a transition).
    pub mix: Vec<[f32; 2]>,
    pub gain_db: f64,
}

impl Loaded {
    pub fn gain(&self) -> f32 {
        10f64.powf(self.gain_db / 20.0) as f32
    }
}

pub struct Library {
    stems: PathBuf,
    beats: PathBuf,
    station_lufs: f64,
    songs: HashMap<String, Rc<Loaded>>,
}

impl Library {
    pub fn new(stems: &Path, beats: &Path, station_lufs: f64) -> Self {
        Self { stems: stems.into(), beats: beats.into(), station_lufs, songs: HashMap::new() }
    }

    pub fn table(&self, id: &str) -> Result<BeatTable> {
        let path = self.beats.join(format!("{id}.json"));
        BeatTable::load(&path).with_context(|| format!("beat table {}", path.display()))
    }

    pub fn song(&mut self, id: &str) -> Result<Rc<Loaded>> {
        if let Some(s) = self.songs.get(id) {
            return Ok(s.clone());
        }
        let table = self.table(id)?;
        let stems = load_stems(&self.stems, &table).with_context(|| format!("stems of {id}"))?;
        let n = stems.len();
        let mut mix = vec![[0.0f32; 2]; n];
        for s in &stems.stems {
            for (m, v) in mix.iter_mut().zip(s) {
                m[0] += v[0];
                m[1] += v[1];
            }
        }
        let gain_db = (self.station_lufs - stems.lufs()).min(0.0);
        let song = Rc::new(Loaded { stems, table, mix, gain_db });
        self.songs.insert(id.into(), song.clone());
        Ok(song)
    }
}

/// The song's stems as the lab's renderer loads them (`vocals_r`, `drums_i`, `bass_i` where they
/// exist, `other` is what is left of the song), with the bar lines of the beat table.
fn load_stems(stems_dir: &Path, table: &BeatTable) -> Result<MemoryStems> {
    let dir = stems_dir.join(&table.id);
    let rd = |name: &str| files::read_flac(&dir.join(format!("{name}.flac")));
    let (v, d, b, o) = (rd("vocals")?, rd("drums")?, rd("bass")?, rd("other")?);
    let n = v.len().min(d.len()).min(b.len()).min(o.len());
    let mix: Vec<[f32; 2]> = (0..n).map(|i| std::array::from_fn(|c| ((v[i][c] + d[i][c]) + b[i][c]) + o[i][c])).collect();
    let cut = |mut x: Vec<[f32; 2]>| {
        x.resize(n, [0.0; 2]);
        x
    };
    let mut stems = [cut(v), cut(d), cut(b), cut(o)];
    let clean_vocals = dir.join("vocals_r.flac").exists();
    let mut leak_free = false;
    if clean_vocals {
        stems[0] = cut(rd("vocals_r")?);
        leak_free = dir.join("drums_i.flac").exists() && dir.join("bass_i.flac").exists();
        if leak_free {
            stems[1] = cut(rd("drums_i")?);
            stems[2] = cut(rd("bass_i")?);
        }
        stems[3] = (0..n).map(|i| std::array::from_fn(|c| ((mix[i][c] - stems[0][i][c]) - stems[1][i][c]) - stems[2][i][c])).collect();
    }
    let lufs = MemoryStems::measure_lufs(&stems);
    Ok(MemoryStems { id: table.id.clone(), stems, grid: grid_of(table), lufs, clean_vocals, leak_free })
}

/// The renderer's beat grid with the table's bar lines.
fn grid_of(table: &BeatTable) -> sone_render::BeatGrid {
    sone_render::BeatGrid::from_table(table)
}

/// A loaded song as the live engine reads it.
pub struct OnAir(pub Rc<Loaded>);

impl sone_live::Song for OnAir {
    fn frames(&self) -> usize {
        self.0.mix.len()
    }
    fn read(&self, from: usize, out: &mut [[f32; 2]]) {
        let mix = &self.0.mix;
        for (i, o) in out.iter_mut().enumerate() {
            *o = mix.get(from + i).copied().unwrap_or([0.0; 2]);
        }
    }
    fn table(&self) -> &BeatTable {
        &self.0.table
    }
    fn gain(&self) -> f32 {
        self.0.gain()
    }
}
