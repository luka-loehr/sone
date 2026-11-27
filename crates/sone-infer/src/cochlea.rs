//! Cochlea: a tonotopic 42-band source separator inspired by basilar-membrane
//! frequency decomposition (after Band-Split RNN, Luo & Yu 2023, with the
//! recurrent layers replaced by dilated depthwise time convolutions and
//! cross-band attention; see `cookbooks/cochlea.toml`).
//!
//! Chunk of stereo audio → STFT (2048 / 512, Hann) → per band: RMSNorm and a
//! projection to `dim` → blocks of (time convolution, attention across bands,
//! feed-forward) → per stem and band an MLP that predicts a complex mask for
//! both channels → mask × mixture STFT → inverse STFT. "other" is the mixture
//! minus the three predicted stems, so the four always sum to the song.
//!
//! Activations are `(frame, band, dim)`.

use anyhow::Result;

use crate::config::CochleaConfig;
use crate::dsp::{Stft, frames};
use crate::layers::{Attention, FeedForward, GROUP_ROWS, Linear, Scratch, TimeBlock};
use crate::ops::{BAND_EPS, EPS, gelu, linear, rms_norm_copy, rms_norm_rows};
use crate::st::WeightFile;

struct BandIn {
    norm: Vec<f32>,
    proj: Linear,
}

struct Block {
    time: TimeBlock,
    t_attn: Option<Attention>,
    band: Attention,
    ff: FeedForward,
}

struct MaskHead {
    l1: Linear,
    l2: Linear,
}

pub struct Cochlea {
    pub config: CochleaConfig,
    file: WeightFile,
    split: Vec<BandIn>,
    band_emb: Vec<f32>,
    blocks: Vec<Block>,
    norm: Vec<f32>,
    /// `[stem][band]`
    masks: Vec<Vec<MaskHead>>,
}

/// Buffers reused across chunks: the mixture STFT, estimated STFTs, the
/// activations and the layer scratch. Their size follows the chunk length.
#[derive(Default)]
pub struct CochleaWorkspace {
    spec: Vec<f32>,
    est: Vec<f32>,
    x: Vec<f32>,
    h: Vec<f32>,
    s: Scratch,
    stft: Option<Stft>,
    /// One chunk's four stems, both channels (for [`Cochlea::separate`]).
    chunk_out: Vec<[Vec<f32>; 2]>,
    chunk_in: [Vec<f32>; 2],
}

impl Cochlea {
    pub fn load(mut file: WeightFile) -> Result<Self> {
        let config: CochleaConfig = file.config_any(&["cochlea", "stems"])?;
        config.validate()?;
        let (d, k) = (config.dim, config.bands.len());
        let t = &mut file;
        let split = (0..k)
            .map(|b| {
                let w4 = 4 * config.width(b);
                Ok(BandIn { norm: t.vec(&format!("split.{b}.norm"), &[w4])?, proj: Linear::load(t, &format!("split.{b}"), d, w4)? })
            })
            .collect::<Result<Vec<_>>>()?;
        let band_emb = t.vec("band_emb", &[k, d])?;
        let blocks = (0..config.blocks)
            .map(|i| {
                Ok(Block {
                    time: TimeBlock::load(t, &format!("block{i}.time"), d, config.ff(), config.kernel, config.dilations[i])?,
                    t_attn: if config.time_attn {
                        Some(Attention::load(t, &format!("block{i}.t_attn"), d, config.heads)?)
                    } else {
                        None
                    },
                    band: Attention::load(t, &format!("block{i}.band"), d, config.heads)?,
                    ff: FeedForward::load(t, &format!("block{i}.ff"), d, config.ff())?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = t.vec("norm", &[d])?;
        let masks = config
            .stem_names()
            .iter()
            .map(|stem| {
                (0..k)
                    .map(|b| {
                        Ok(MaskHead {
                            l1: Linear::load(t, &format!("mask.{stem}.{b}.l1"), config.mask_hidden, d)?,
                            l2: Linear::load(t, &format!("mask.{stem}.{b}.l2"), 4 * config.width(b), config.mask_hidden)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        file.finish()?;
        Ok(Self { config, file, split, band_emb, blocks, norm, masks })
    }

    pub fn weights_bytes(&self) -> usize {
        self.file.byte_len()
    }

    /// Separate one chunk. `len` (= `left.len()`) must be a multiple of the
    /// hop. `out[s]` receives stem `s` (vocals, drums, bass, other), both
    /// channels, `len` samples each.
    pub fn separate_chunk(&self, ws: &mut CochleaWorkspace, left: &[f32], right: &[f32], out: &mut [[Vec<f32>; 2]]) {
        let c = &self.config;
        let len = left.len();
        assert!(len % c.hop == 0 && right.len() == len && out.len() >= 4);
        let (d, k, bins) = (c.dim, c.bands.len(), c.bins());
        let t_n = frames(len, c.hop);
        let CochleaWorkspace { spec, x, h, s, stft, .. } = ws;
        let stft = stft.get_or_insert_with(|| Stft::new(c.n_fft, c.hop));
        stft.forward_stereo(left, right, spec);

        // band split: every band's features are contiguous in (frame, bin, 4)
        x.clear();
        x.resize(t_n * k * d, 0.0);
        for (b, band) in self.split.iter().enumerate() {
            let w4 = 4 * c.width(b);
            let f0 = c.bands[b][0];
            band.proj.w.load(&self.file, &mut s.w1);
            let frames_per = (GROUP_ROWS * 4 / w4).max(1);
            let mut t0 = 0;
            while t0 < t_n {
                let g = frames_per.min(t_n - t0);
                s.rows.resize(g * w4, 0.0);
                rms_norm_copy(&spec[(t0 * bins + f0) * 4..], g, w4, bins * 4, &band.norm, BAND_EPS, &mut s.rows);
                let dst = &mut x[(t0 * k + b) * d..];
                linear(&s.rows, g, w4, w4, &s.w1, d, Some(&band.proj.b), dst, k * d);
                for i in 0..g {
                    let row = &mut x[((t0 + i) * k + b) * d..][..d];
                    for (v, e) in row.iter_mut().zip(&self.band_emb[b * d..(b + 1) * d]) {
                        *v += e;
                    }
                }
                t0 += g;
            }
        }

        for block in &self.blocks {
            block.time.apply(&self.file, x, h, t_n, k, s);
            if let Some(ta) = &block.t_attn {
                h.resize(k * t_n * d, 0.0);
                for t in 0..t_n {
                    for b in 0..k {
                        h[(b * t_n + t) * d..(b * t_n + t + 1) * d]
                            .copy_from_slice(&x[(t * k + b) * d..(t * k + b + 1) * d]);
                    }
                }
                ta.apply(&self.file, h, k, t_n, s);
                for b in 0..k {
                    for t in 0..t_n {
                        x[(t * k + b) * d..(t * k + b + 1) * d]
                            .copy_from_slice(&h[(b * t_n + t) * d..(b * t_n + t + 1) * d]);
                    }
                }
            }
            block.band.apply(&self.file, x, t_n, k, s);
            block.ff.apply(&self.file, x, t_n * k, s);
        }
        rms_norm_rows(x, d, &self.norm, EPS);

        let mut sum_l = std::mem::take(&mut ws.chunk_in[0]);
        let mut sum_r = std::mem::take(&mut ws.chunk_in[1]);
        sum_l.clear();
        sum_l.resize(len, 0.0);
        sum_r.clear();
        sum_r.resize(len, 0.0);
        let CochleaWorkspace { spec, est, x, s, stft, .. } = ws;
        let stft = stft.as_mut().expect("created above");
        let m = c.mask_hidden;
        let spec_len = t_n * bins * 4;
        let bound = c.mask_bound;

        if c.four_stems {
            est.clear();
            est.resize(4 * spec_len, 0.0);
            for (si, heads) in self.masks.iter().enumerate() {
                let est_s = &mut est[si * spec_len..(si + 1) * spec_len];
                for (b, head) in heads.iter().enumerate() {
                    let (f0, w4) = (c.bands[b][0], 4 * c.width(b));
                    head.l1.w.load(&self.file, &mut s.w1);
                    head.l2.w.load(&self.file, &mut s.w2);
                    let frames_per = (GROUP_ROWS / 2).max(1);
                    let mut t0 = 0;
                    while t0 < t_n {
                        let g = frames_per.min(t_n - t0);
                        s.inner.resize(g * m, 0.0);
                        linear(&x[(t0 * k + b) * d..], g, d, k * d, &s.w1, m, Some(&head.l1.b), &mut s.inner, m);
                        gelu(&mut s.inner);
                        linear(&s.inner, g, m, m, &s.w2, w4, Some(&head.l2.b), &mut est_s[(t0 * bins + f0) * 4..], bins * 4);
                        t0 += g;
                    }
                }
                if bound > 0.0 {
                    let inv_b = 1.0 / bound;
                    for v in est_s.iter_mut() {
                        *v = bound * (*v * inv_b).tanh();
                    }
                }
                for (e, x_bin) in est_s.as_chunks_mut::<2>().0.iter_mut().zip(spec.as_chunks::<2>().0) {
                    let (mr, mi) = (e[0], e[1]);
                    e[0] = mr * x_bin[0] - mi * x_bin[1];
                    e[1] = mr * x_bin[1] + mi * x_bin[0];
                }
            }
            // Stereo-coherent energy-weighted mixture-consistent projection:
            // E_s' = E_s + w_s * (X - sum_k E_k), where w_s = (|E_s|^2 + eps) / sum_k(|E_k|^2 + eps).
            let n_tf = t_n * bins;
            for idx in 0..n_tf {
                let base = idx * 4;
                let mut p = [0.0f32; 4];
                for si in 0..4 {
                    let sb = si * spec_len + base;
                    let (e0, e1, e2, e3) = (est[sb], est[sb + 1], est[sb + 2], est[sb + 3]);
                    p[si] = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3 + 1e-8;
                }
                let inv_p = 1.0 / (p[0] + p[1] + p[2] + p[3]);
                for ch in 0..4 {
                    let s_sum = est[base + ch]
                        + est[spec_len + base + ch]
                        + est[2 * spec_len + base + ch]
                        + est[3 * spec_len + base + ch];
                    let res = spec[base + ch] - s_sum;
                    for si in 0..4 {
                        est[si * spec_len + base + ch] += (p[si] * inv_p) * res;
                    }
                }
            }
            for si in 0..3 {
                let est_s = &est[si * spec_len..(si + 1) * spec_len];
                let [ol, or] = &mut out[si];
                ol.resize(len, 0.0);
                or.resize(len, 0.0);
                stft.inverse_stereo(est_s, len, ol, or);
                for i in 0..len {
                    sum_l[i] += ol[i];
                    sum_r[i] += or[i];
                }
            }
        } else {
            for (si, heads) in self.masks.iter().enumerate() {
                est.clear();
                est.resize(spec_len, 0.0);
                for (b, head) in heads.iter().enumerate() {
                    let (f0, w4) = (c.bands[b][0], 4 * c.width(b));
                    head.l1.w.load(&self.file, &mut s.w1);
                    head.l2.w.load(&self.file, &mut s.w2);
                    let frames_per = (GROUP_ROWS / 2).max(1);
                    let mut t0 = 0;
                    while t0 < t_n {
                        let g = frames_per.min(t_n - t0);
                        s.inner.resize(g * m, 0.0);
                        // band b's rows are strided by k·d in (frame, band, dim)
                        linear(&x[(t0 * k + b) * d..], g, d, k * d, &s.w1, m, Some(&head.l1.b), &mut s.inner, m);
                        gelu(&mut s.inner);
                        // the mask goes straight into the band's slots of `est`
                        linear(&s.inner, g, m, m, &s.w2, w4, Some(&head.l2.b), &mut est[(t0 * bins + f0) * 4..], bins * 4);
                        t0 += g;
                    }
                }
                if bound > 0.0 {
                    let inv_b = 1.0 / bound;
                    for v in est.iter_mut() {
                        *v = bound * (*v * inv_b).tanh();
                    }
                }
                // estimate = mask ⊗ mixture, per bin and channel
                for (e, x_bin) in est.as_chunks_mut::<2>().0.iter_mut().zip(spec.as_chunks::<2>().0) {
                    let (mr, mi) = (e[0], e[1]);
                    e[0] = mr * x_bin[0] - mi * x_bin[1];
                    e[1] = mr * x_bin[1] + mi * x_bin[0];
                }
                let [ol, or] = &mut out[si];
                ol.resize(len, 0.0);
                or.resize(len, 0.0);
                stft.inverse_stereo(est, len, ol, or);
                for i in 0..len {
                    sum_l[i] += ol[i];
                    sum_r[i] += or[i];
                }
            }
        }
        let [ol, or] = &mut out[3];
        ol.clear();
        or.clear();
        ol.extend(left.iter().zip(&sum_l).map(|(m, s)| m - s));
        or.extend(right.iter().zip(&sum_r).map(|(m, s)| m - s));
        ws.chunk_in = [sum_l, sum_r];
    }

    /// Separate a whole signal in chunks of `chunk` samples (a multiple of
    /// the hop) that overlap by `overlap` and are cross-faded linearly.
    /// "other" is again the mixture minus the three, so the sum is exact.
    pub fn separate(&self, ws: &mut CochleaWorkspace, left: &[f32], right: &[f32], chunk: usize, overlap: usize, out: &mut [[Vec<f32>; 2]]) {
        let len = left.len();
        let plan = crate::chunk::Plan::new(len, chunk, overlap);
        for o in out.iter_mut().take(4) {
            for ch in o.iter_mut() {
                ch.clear();
                ch.resize(len, 0.0);
            }
        }
        let mut chunk_out = std::mem::take(&mut ws.chunk_out);
        chunk_out.resize_with(4, Default::default);
        let (mut cl, mut cr) = (Vec::new(), Vec::new());
        for (i, &start) in plan.starts.iter().enumerate() {
            cl.clear();
            cr.clear();
            cl.extend((start..start + chunk).map(|t| left.get(t).copied().unwrap_or(0.0)));
            cr.extend((start..start + chunk).map(|t| right.get(t).copied().unwrap_or(0.0)));
            self.separate_chunk(ws, &cl, &cr, &mut chunk_out);
            for (s, o) in chunk_out.iter().enumerate().take(3) {
                for j in 0..chunk.min(len.saturating_sub(start)) {
                    let w = plan.weight(i, j);
                    out[s][0][start + j] += w * o[0][j];
                    out[s][1][start + j] += w * o[1][j];
                }
            }
        }
        for j in 0..len {
            out[3][0][j] = left[j] - out[0][0][j] - out[1][0][j] - out[2][0][j];
            out[3][1][j] = right[j] - out[0][1][j] - out[1][1][j] - out[2][1][j];
        }
        ws.chunk_out = chunk_out;
    }
}
