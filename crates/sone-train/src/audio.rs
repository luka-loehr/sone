//! Reading stereo 44.1 kHz audio in pieces: FLAC and WAV through symphonia
//! (seeking, so a random 3 s chunk of a song costs a few ms, not a decode of
//! the whole file), and raw interleaved little-endian f32 (`.f32`). Mono files are duplicated to both channels.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const SAMPLE_RATE: u32 = sone_core::SAMPLE_RATE;

/// Samples per channel.
pub fn length(path: &Path) -> Result<u64> {
    if is_raw(path) {
        return Ok(std::fs::metadata(path)?.len() / 8);
    }
    let (format, track_id) = open(path)?;
    let track = format.tracks().iter().find(|t| t.id == track_id).context("no track")?;
    let rate = track.codec_params.sample_rate.unwrap_or(0);
    ensure!(rate == SAMPLE_RATE, "{}: sample rate {rate}, expected {SAMPLE_RATE} (convert first)", path.display());
    track.codec_params.n_frames.with_context(|| format!("{}: unknown length", path.display()))
}

fn is_raw(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "f32")
}

fn open(path: &Path) -> Result<(Box<dyn symphonia::core::formats::FormatReader>, u32)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .with_context(|| format!("probing {}", path.display()))?;
    let format = probed.format;
    let track_id = format.default_track().context("no audio track")?.id;
    Ok((format, track_id))
}

/// `len` samples per channel from `start`, zero-padded past the end.
pub fn read(path: &Path, start: u64, len: usize) -> Result<[Vec<f32>; 2]> {
    let mut l = vec![0.0f32; len];
    let mut r = vec![0.0f32; len];
    if is_raw(path) {
        let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let total = f.metadata()?.len() / 8;
        if start >= total {
            return Ok([l, r]);
        }
        let n = (len as u64).min(total - start) as usize;
        f.seek(SeekFrom::Start(start * 8))?;
        let mut bytes = vec![0u8; n * 8];
        f.read_exact(&mut bytes)?;
        for (i, frame) in bytes.as_chunks::<8>().0.iter().enumerate() {
            l[i] = f32::from_le_bytes(frame[..4].try_into()?);
            r[i] = f32::from_le_bytes(frame[4..].try_into()?);
        }
        return Ok([l, r]);
    }
    let (mut format, track_id) = open(path)?;
    let params = format.tracks().iter().find(|t| t.id == track_id).context("no track")?.codec_params.clone();
    let mut decoder = symphonia::default::get_codecs().make(&params, &DecoderOptions::default())?;
    let mut at = 0u64;
    if start > 0 {
        match format.seek(SeekMode::Accurate, SeekTo::TimeStamp { ts: start, track_id }) {
            Ok(s) => at = s.actual_ts,
            Err(SymError::SeekError(_)) => return Ok([l, r]),
            Err(e) => return Err(e.into()),
        }
        decoder.reset();
    }
    let end = start + len as u64;
    let mut buf: Option<SampleBuffer<f32>> = None;
    while at < end {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        at = packet.ts();
        let decoded = decoder.decode(&packet)?;
        let spec = *decoded.spec();
        let channels = spec.channels.count();
        let frames = decoded.frames();
        let b = buf.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        if (b.capacity() as u64) < decoded.capacity() as u64 {
            *b = SampleBuffer::new(decoded.capacity() as u64, spec);
        }
        b.copy_interleaved_ref(decoded);
        let s = b.samples();
        for i in 0..frames {
            let t = at + i as u64;
            if t < start || t >= end {
                continue;
            }
            let j = (t - start) as usize;
            l[j] = s[i * channels];
            r[j] = s[i * channels + if channels > 1 { 1 } else { 0 }];
        }
        at += frames as u64;
    }
    Ok([l, r])
}

/// The whole file.
pub fn read_all(path: &Path) -> Result<[Vec<f32>; 2]> {
    let n = length(path)?;
    if n > 60 * 60 * SAMPLE_RATE as u64 {
        bail!("{}: longer than an hour", path.display());
    }
    read(path, 0, n as usize)
}

/// 32-bit float stereo WAV.
pub fn write_wav(path: &Path, l: &[f32], r: &[f32]) -> Result<()> {
    let spec = hound::WavSpec { channels: 2, sample_rate: SAMPLE_RATE, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut w = hound::WavWriter::create(path, spec).with_context(|| format!("creating {}", path.display()))?;
    for (a, b) in l.iter().zip(r) {
        w.write_sample(*a)?;
        w.write_sample(*b)?;
    }
    w.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_and_raw_segments() {
        let dir = std::env::temp_dir().join(format!("sone-train-audio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = 100_000;
        let l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        let r: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002).cos() * 0.5).collect();
        let wav = dir.join("a.wav");
        write_wav(&wav, &l, &r).unwrap();
        let raw = dir.join("a.f32");
        let bytes: Vec<u8> = l.iter().zip(&r).flat_map(|(a, b)| a.to_le_bytes().into_iter().chain(b.to_le_bytes())).collect();
        std::fs::write(&raw, bytes).unwrap();
        for p in [&wav, &raw] {
            assert_eq!(length(p).unwrap(), n as u64);
            for start in [0u64, 12_345, 99_000] {
                let [a, b] = read(p, start, 4000).unwrap();
                for i in 0..4000 {
                    let t = start as usize + i;
                    let (wl, wr) = if t < n { (l[t], r[t]) } else { (0.0, 0.0) };
                    assert_eq!((a[i], b[i]), (wl, wr), "{} at {t}", p.display());
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
