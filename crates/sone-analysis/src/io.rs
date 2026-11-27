//! Reading stems from disk (feature `files`): the FLAC files are decoded in
//! lockstep and fed to the [`Measurer`] in blocks, so no stem is ever held
//! whole in memory.

use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::measure::{Block, Measurements, Measurer};

/// Which stems stand for vocals, drums, bass and other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StemSet {
    /// Demucs' four stems (`vocals`, `drums`, `bass`, `other`), as structure.py.
    Original,
    /// `vocals_r` (BS-RoFormer), `drums_i` and `bass_i` (separated from the
    /// song without its voice) and other = mix − these three, where mix is
    /// the sum of the four original stems (as handmix.py).
    Final,
}

struct Flac {
    blocks: claxon::FlacReader<BufReader<File>>,
    scale: f32,
    channels: usize,
    buffer: Vec<i32>,
    queue: VecDeque<f32>,
    done: bool,
    path: PathBuf,
}

impl Flac {
    fn open(path: &Path) -> Result<Self> {
        let reader = claxon::FlacReader::new(BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?))
            .with_context(|| format!("reading {}", path.display()))?;
        let info = reader.streaminfo();
        if info.sample_rate != 44_100 {
            bail!("{}: {} Hz, not 44.1 kHz", path.display(), info.sample_rate);
        }
        Ok(Self {
            scale: 1.0 / (1u64 << (info.bits_per_sample - 1)) as f32,
            channels: info.channels as usize,
            blocks: reader,
            buffer: Vec::new(),
            queue: VecDeque::new(),
            done: false,
            path: path.to_path_buf(),
        })
    }

    /// Decode until `want` mono samples are queued or the file ends.
    fn fill(&mut self, want: usize) -> Result<()> {
        while !self.done && self.queue.len() < want {
            let buffer = std::mem::take(&mut self.buffer);
            let mut blocks = self.blocks.blocks();
            match blocks.read_next_or_eof(buffer).with_context(|| format!("decoding {}", self.path.display()))? {
                Some(block) => {
                    let n = block.duration() as usize;
                    // soundfile's float32 of the channels' mean
                    let c = self.channels;
                    for i in 0..n {
                        let mut s = 0.0f32;
                        for ch in 0..c {
                            s += block.sample(ch as u32, i as u32) as f32 * self.scale;
                        }
                        self.queue.push_back(if c == 2 { s * 0.5 } else { s / c as f32 });
                    }
                    self.buffer = block.into_buffer();
                }
                None => self.done = true,
            }
        }
        Ok(())
    }

    /// The next `n` samples (zeros past the end).
    fn take(&mut self, n: usize, out: &mut Vec<f32>) {
        out.clear();
        let k = n.min(self.queue.len());
        out.extend(self.queue.drain(..k));
        out.resize(n, 0.0);
    }
}

fn samples(path: &Path) -> Result<u64> {
    let reader = claxon::FlacReader::open(path).with_context(|| format!("reading {}", path.display()))?;
    reader.streaminfo().samples.with_context(|| format!("{}: no length in its header", path.display()))
}

const BLOCK: usize = 1 << 16;

/// Measure the stems of one song in `dir` (`<dir>/<name>.flac`).
pub fn measure_dir(dir: &Path, set: StemSet, cuts: Vec<usize>) -> Result<Measurements> {
    Ok(measure_dir_timed(dir, set, cuts)?.0)
}

/// [`measure_dir`], and the seconds spent in the measurement itself (without
/// decoding the FLAC files).
pub fn measure_dir_timed(dir: &Path, set: StemSet, cuts: Vec<usize>) -> Result<(Measurements, f64)> {
    let mut busy = std::time::Duration::ZERO;
    let names: &[&str] = match set {
        StemSet::Original => &["vocals", "drums", "bass", "other"],
        StemSet::Final => &["vocals", "drums", "bass", "other", "vocals_r", "drums_i", "bass_i"],
    };
    let mut files = names.iter().map(|n| Flac::open(&dir.join(format!("{n}.flac")))).collect::<Result<Vec<_>>>()?;
    // the song is as long as the original mix (handmix.py cuts the new stems to it)
    let len = samples(&dir.join("drums.flac"))? as usize;
    let mut m = Measurer::with_cuts(len as f64 / 44_100.0, cuts);
    let mut bufs: Vec<Vec<f32>> = vec![Vec::with_capacity(BLOCK); names.len()];
    let mut other = Vec::with_capacity(BLOCK);
    let mut done = 0;
    while done < len {
        let n = BLOCK.min(len - done);
        for (f, buf) in files.iter_mut().zip(bufs.iter_mut()) {
            f.fill(n)?;
            f.take(n, buf);
        }
        let started = std::time::Instant::now();
        match set {
            StemSet::Original => m.push(Block { vocals: &bufs[0], drums: &bufs[1], bass: &bufs[2], other: &bufs[3] }),
            StemSet::Final => {
                other.clear();
                other.extend((0..n).map(|i| bufs[0][i] + bufs[1][i] + bufs[2][i] + bufs[3][i] - bufs[4][i] - bufs[5][i] - bufs[6][i]));
                m.push(Block { vocals: &bufs[4], drums: &bufs[5], bass: &bufs[6], other: &other });
            }
        }
        busy += started.elapsed();
        done += n;
    }
    let started = std::time::Instant::now();
    let m = m.finish();
    Ok((m, (busy + started.elapsed()).as_secs_f64()))
}
