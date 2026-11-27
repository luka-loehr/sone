//! Songs on disk in the layout of atlas: `stems/<id>/*.flac` and the beat
//! analysis `struct/<id>.json`.

use std::path::Path;

use anyhow::{Context, bail};
use sone_core::TrackAnalysis;

use crate::source::{BeatGrid, MemoryStems};

/// Decodes a FLAC file to stereo float32 the way libsndfile does (integer / 2^(bits-1)).
pub fn read_flac(path: &Path) -> anyhow::Result<Vec<[f32; 2]>> {
    let mut reader = claxon::FlacReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let info = reader.streaminfo();
    if info.channels != 2 {
        bail!("{}: {} channels, need 2", path.display(), info.channels);
    }
    let scale = 1.0 / (1u64 << (info.bits_per_sample - 1)) as f32;
    let mut out = Vec::with_capacity(info.samples.unwrap_or(0) as usize);
    let mut blocks = reader.blocks();
    let mut block = claxon::Block::empty();
    while let Some(next) = blocks.read_next_or_eof(block.into_buffer())? {
        block = next;
        out.extend(block.stereo_samples().map(|(l, r)| [l as f32 * scale, r as f32 * scale]));
    }
    Ok(out)
}

/// Loads a song as the reference's `Track` does: the four Demucs stems add up to
/// the song; when `vocals_r.flac` (BS-RoFormer) exists it replaces the voice, and
/// `drums_i.flac` / `bass_i.flac` (separated from the song without its voice) the
/// drums and bass; `other` takes whatever is left, so the four still add up to the song.
pub fn load_song(stems_dir: &Path, struct_dir: &Path, id: &str) -> anyhow::Result<MemoryStems> {
    let analysis = TrackAnalysis::load(&struct_dir.join(format!("{id}.json")))
        .with_context(|| format!("analysis of {id} in {}", struct_dir.display()))?;
    load_stems(stems_dir, id, BeatGrid::from_analysis(&analysis))
}

/// The same with the bar lines of the song's beat table (`beats/<id>.json`), as the
/// planner counts bars (local corrections of the bar line included).
pub fn load_song_with_table(stems_dir: &Path, beats_dir: &Path, id: &str) -> anyhow::Result<MemoryStems> {
    let table = sone_core::BeatTable::load(&beats_dir.join(format!("{id}.json")))
        .with_context(|| format!("beat table of {id} in {}", beats_dir.display()))?;
    load_stems(stems_dir, id, BeatGrid::from_table(&table))
}

fn load_stems(stems_dir: &Path, id: &str, grid: BeatGrid) -> anyhow::Result<MemoryStems> {
    let dir = stems_dir.join(id);
    let rd = |name: &str| read_flac(&dir.join(format!("{name}.flac")));
    let (v, d, b, o) = (rd("vocals")?, rd("drums")?, rd("bass")?, rd("other")?);
    let n = v.len().min(d.len()).min(b.len()).min(o.len());
    // numpy's sum(): ((vocals + drums) + bass) + other, in float32.
    let mix: Vec<[f32; 2]> = (0..n)
        .map(|i| std::array::from_fn(|c| ((v[i][c] + d[i][c]) + b[i][c]) + o[i][c]))
        .collect();
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
        stems[3] = (0..n)
            .map(|i| std::array::from_fn(|c| ((mix[i][c] - stems[0][i][c]) - stems[1][i][c]) - stems[2][i][c]))
            .collect();
    }
    let lufs = MemoryStems::measure_lufs(&stems);
    Ok(MemoryStems { id: id.to_string(), stems, grid, lufs, clean_vocals, leak_free })
}

/// Writes 16-bit PCM like soundfile (`round(x * 32767)`, clipped).
pub fn write_wav16(path: &Path, x: &[[f32; 2]]) -> anyhow::Result<()> {
    let spec = hound::WavSpec { channels: 2, sample_rate: 44100, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec)?;
    for v in x {
        for &s in v {
            w.write_sample((s * 32767.0).round_ties_even().clamp(-32768.0, 32767.0) as i16)?;
        }
    }
    w.finalize()?;
    Ok(())
}

/// Writes raw little-endian float32, interleaved stereo.
pub fn write_f32(path: &Path, x: &[[f32; 2]]) -> anyhow::Result<()> {
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.iter().flat_map(|s| s.to_le_bytes())).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}
