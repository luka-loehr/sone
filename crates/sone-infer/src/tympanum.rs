//! Tympanum: rhythmic, vocal-breath, phrase, and pitch/key perception from a
//! 50 Hz log-mel spectrogram (hop 882 at 44.1 kHz). Per-mel-bin affine input,
//! projection, dilated depthwise time-convolution blocks, a bidirectional GRU
//! stack, then per frame five logits (beat, downbeat, voice, line end, phrase
//! start) and per chunk the key (24 classes) and tuning (cents).

use anyhow::Result;

use crate::config::{TYMPANUM_FRAME_HEADS, TympanumConfig};
use crate::dsp::LogMel;
use crate::layers::{Linear, Scratch, TimeBlock};
use crate::ops::{gelu, linear_into, sigmoid, softmax};
use crate::st::{Mat, WeightFile};

/// PyTorch's GRU cell: gates `r, z, n` stacked in that order in `w_ih (3h, in)`
/// and `w_hh (3h, h)`; `n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))`,
/// `h' = (1 - z) ⊙ n + z ⊙ h`.
struct GruDir {
    w_ih: Mat,
    w_hh: Mat,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

struct GruLayer {
    fwd: GruDir,
    bwd: GruDir,
    input: usize,
}

pub struct Tympanum {
    pub config: TympanumConfig,
    file: WeightFile,
    in_scale: Vec<f32>,
    in_shift: Vec<f32>,
    proj: Linear,
    convs: Vec<TimeBlock>,
    grus: Vec<GruLayer>,
    frame_head: Linear,
    key_head: Linear,
    tuning_head: Linear,
}

#[derive(Default)]
pub struct TympanumWorkspace {
    x: Vec<f32>,
    h: Vec<f32>,
    seq: Vec<f32>,
    gates: Vec<f32>,
    s: Scratch,
    mel: Option<LogMel>,
    logmel: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TympanumOutput {
    /// Per frame, the probabilities of [`TYMPANUM_FRAME_HEADS`].
    pub frames: Vec<[f32; 5]>,
    /// Probabilities of [`crate::config::KEYS`].
    pub key: [f32; 24],
    pub tuning_cents: f32,
}

impl TympanumOutput {
    pub fn key_name(&self) -> &'static str {
        let best = (0..24).max_by(|&a, &b| self.key[a].total_cmp(&self.key[b])).unwrap_or(0);
        crate::config::KEYS[best]
    }
}

/// Tuning is predicted as `TUNING_RANGE · tanh(head)` cents.
pub const TUNING_RANGE: f32 = 50.0;

impl Tympanum {
    pub fn load(mut file: WeightFile) -> Result<Self> {
        let config: TympanumConfig = file.config_any(&["tympanum", "ear"])?;
        let (c, h, mels) = (config.channels, config.gru_hidden, config.n_mels);
        let t = &mut file;
        let in_scale = t.vec("in_scale", &[mels])?;
        let in_shift = t.vec("in_shift", &[mels])?;
        let proj = Linear::load(t, "proj", c, mels)?;
        let convs = config
            .dilations
            .iter()
            .enumerate()
            .map(|(i, &dil)| TimeBlock::load(t, &format!("conv{i}"), c, 2 * c, config.kernel, dil))
            .collect::<Result<Vec<_>>>()?;
        let mut grus = Vec::new();
        for l in 0..config.gru_layers {
            let input = if l == 0 { c } else { 2 * h };
            let mut dir = |name: &str| -> Result<GruDir> {
                let p = format!("gru{l}.{name}");
                Ok(GruDir {
                    w_ih: t.mat(&format!("{p}.w_ih"), 3 * h, input)?,
                    w_hh: t.mat(&format!("{p}.w_hh"), 3 * h, h)?,
                    b_ih: t.vec(&format!("{p}.b_ih"), &[3 * h])?,
                    b_hh: t.vec(&format!("{p}.b_hh"), &[3 * h])?,
                })
            };
            grus.push(GruLayer { fwd: dir("fwd")?, bwd: dir("bwd")?, input });
        }
        let frame_head = Linear::load(t, "frame_head", TYMPANUM_FRAME_HEADS.len(), 2 * h)?;
        let key_head = Linear::load(t, "key_head", 24, 2 * h)?;
        let tuning_head = Linear::load(t, "tuning_head", 1, 2 * h)?;
        file.finish()?;
        Ok(Self { config, file, in_scale, in_shift, proj, convs, grus, frame_head, key_head, tuning_head })
    }

    /// Log-mel of a mono signal (the network's input), `(frames, n_mels)`.
    pub fn logmel<'a>(&self, ws: &'a mut TympanumWorkspace, mono: &[f32]) -> &'a [f32] {
        let mel = ws.mel.get_or_insert_with(|| LogMel::for_tympanum(&self.config));
        mel.compute(mono, &mut ws.logmel);
        &ws.logmel
    }

    /// Run a mono chunk end to end.
    pub fn analyze(&self, ws: &mut TympanumWorkspace, mono: &[f32]) -> TympanumOutput {
        self.logmel(ws, mono);
        let mel = std::mem::take(&mut ws.logmel);
        let out = self.run(ws, &mel);
        ws.logmel = mel;
        out
    }

    /// The raw outputs for a log-mel `(frames, n_mels)`: frame logits
    /// `(frames, 5)`, key logits `(24)`, tuning head `(1)` before tanh.
    pub fn logits(&self, ws: &mut TympanumWorkspace, logmel: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        let c = &self.config;
        let (mels, ch, hid) = (c.n_mels, c.channels, c.gru_hidden);
        let t_n = logmel.len() / mels;
        let TympanumWorkspace { x, h, seq, gates, s, .. } = ws;
        s.rows.clear();
        s.rows.extend(logmel.iter().enumerate().map(|(i, &v)| v * self.in_scale[i % mels] + self.in_shift[i % mels]));
        let w = self.proj.w.load(&self.file, &mut s.w1).to_vec();
        linear_into(&s.rows, t_n, mels, &w, ch, Some(&self.proj.b), x);
        gelu(x);
        for conv in &self.convs {
            conv.apply(&self.file, x, h, t_n, 1, s);
        }
        // bidirectional GRU stack: `x` holds the layer input, `seq` the output (frames, 2h)
        for layer in &self.grus {
            seq.clear();
            seq.resize(t_n * 2 * hid, 0.0);
            for (d, dir) in [&layer.fwd, &layer.bwd].into_iter().enumerate() {
                dir.w_ih.load(&self.file, &mut s.w1);
                dir.w_hh.load(&self.file, &mut s.w2);
                linear_into(x, t_n, layer.input, &s.w1, 3 * hid, Some(&dir.b_ih), gates);
                let mut state = vec![0.0f32; hid];
                let mut hp = vec![0.0f32; 3 * hid];
                for step in 0..t_n {
                    let t = if d == 0 { step } else { t_n - 1 - step };
                    // hp = W_hh h + b_hh
                    for (o, (row, b)) in hp.iter_mut().zip(s.w2.chunks_exact(hid).zip(&dir.b_hh)) {
                        *o = b + row.iter().zip(&state).map(|(w, v)| w * v).sum::<f32>();
                    }
                    let xp = &gates[t * 3 * hid..(t + 1) * 3 * hid];
                    for j in 0..hid {
                        let r = sigmoid(xp[j] + hp[j]);
                        let z = sigmoid(xp[hid + j] + hp[hid + j]);
                        let n = (xp[2 * hid + j] + r * hp[2 * hid + j]).tanh();
                        state[j] = (1.0 - z) * n + z * state[j];
                    }
                    seq[t * 2 * hid + d * hid..][..hid].copy_from_slice(&state);
                }
            }
            std::mem::swap(x, seq);
        }
        let feat = 2 * hid;
        let mut frame_logits = Vec::new();
        let fw = self.frame_head.w.load(&self.file, &mut s.w1).to_vec();
        linear_into(x, t_n, feat, &fw, TYMPANUM_FRAME_HEADS.len(), Some(&self.frame_head.b), &mut frame_logits);
        let mut pooled = vec![0.0f32; feat];
        for row in x.chunks_exact(feat).take(t_n) {
            for (p, v) in pooled.iter_mut().zip(row) {
                *p += v;
            }
        }
        pooled.iter_mut().for_each(|p| *p /= t_n.max(1) as f32);
        let mut key = Vec::new();
        let kw = self.key_head.w.load(&self.file, &mut s.w1).to_vec();
        linear_into(&pooled, 1, feat, &kw, 24, Some(&self.key_head.b), &mut key);
        let mut tuning = Vec::new();
        let tw = self.tuning_head.w.load(&self.file, &mut s.w1).to_vec();
        linear_into(&pooled, 1, feat, &tw, 1, Some(&self.tuning_head.b), &mut tuning);
        (frame_logits, key, tuning[0])
    }

    pub fn run(&self, ws: &mut TympanumWorkspace, logmel: &[f32]) -> TympanumOutput {
        let (frames, mut key, tuning) = self.logits(ws, logmel);
        softmax(&mut key);
        TympanumOutput {
            frames: frames.as_chunks::<5>().0.iter().map(|f| f.map(sigmoid)).collect(),
            key: key.try_into().expect("24 keys"),
            tuning_cents: TUNING_RANGE * tuning.tanh(),
        }
    }
}
