//! The Beat This! network (Foscarin, Schlüter, Widmer, ISMIR 2024;
//! github.com/CPJKU/beat_this, `beat_this/model/beat_tracker.py` and
//! `roformer.py`) in plain Rust: a log-mel spectrogram in, beat and downbeat
//! logits per 20 ms frame out. No ML runtime: the network is a handful of
//! layer types (convolution, linear, RMSNorm, GELU, rotary attention with
//! sigmoid gates), each a loop or a `matrixmultiply::sgemm`, so it builds for
//! wasm32 unchanged. The weights come from `export.py` (the Beat This
//! checkpoint with its BatchNorms folded into scale/shift and the
//! convolutions) as safetensors, F16 or F32.
//!
//! Layout of the frontend activations: `(freq, time, channel)`, so a
//! time-directed attention sees contiguous `(time, channel)` sequences and a
//! convolution's output rows are channels.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail, ensure};

/// Frames per chunk the model was trained on (30 s at 50 fps).
pub const CHUNK: usize = 1500;
/// Frames discarded at both edges of a chunk (`split_predict_aggregate`).
pub const BORDER: usize = 6;
pub const N_MELS: usize = 128;
const HEAD_DIM: usize = 32;

/// Chunk starts for a piece of `len` frames (`split_piece` with
/// `avoid_short_end`): chunks overlap by the border, the last one ends at the
/// end of the piece.
pub fn chunk_starts(len: usize) -> Vec<i64> {
    if len == 0 {
        return Vec::new();
    }
    let step = (CHUNK - 2 * BORDER) as i64;
    let mut starts: Vec<i64> = (0..)
        .map(|i| -(BORDER as i64) + i * step)
        .take_while(|&s| s < len as i64 - BORDER as i64)
        .collect();
    if len > CHUNK - 2 * BORDER {
        *starts.last_mut().expect("a chunk") = len as i64 - (CHUNK - BORDER) as i64;
    }
    starts
}

/// Weights of one attention block (`roformer.Attention`, gated).
struct Attention {
    norm: Vec<f32>,
    /// `(3 * heads * 32, dim)`
    qkv: Half,
    /// `(heads, dim)`
    gates_w: Half,
    gates_b: Vec<f32>,
    /// `(dim, heads * 32)`
    out: Half,
    dim: usize,
    heads: usize,
}

/// `roformer.FeedForward`: RMSNorm, Linear, GELU, Linear.
struct FeedForward {
    norm: Vec<f32>,
    w1: Half,
    b1: Vec<f32>,
    w2: Half,
    b2: Vec<f32>,
    dim: usize,
    inner: usize,
}

/// A convolution with kernel `(kf, 3)`, stride `(kf, 1)`, padding `(0, 1)`
/// and its BatchNorm folded in, followed by GELU.
struct Conv {
    /// Reordered to `(kf * 3 * cin, cout)`: row `(j * 3 + dt) * cin + i`.
    w: Half,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    kf: usize,
}

struct Block {
    attn_f: Attention,
    ff_f: FeedForward,
    attn_t: Attention,
    ff_t: FeedForward,
    conv: Conv,
}

pub struct BeatThis {
    in_scale: Vec<f32>,
    in_shift: Vec<f32>,
    stem: Conv,
    blocks: Vec<Block>,
    /// `(dim, channels * freqs)`
    linear_w: Half,
    linear_b: Vec<f32>,
    layers: Vec<(Attention, FeedForward)>,
    norm: Vec<f32>,
    head_w: Half,
    head_b: Vec<f32>,
    dim: usize,
    /// Rotary frequencies for a 32-dim head (`RotaryEmbedding(32)`).
    freqs: [f32; HEAD_DIM / 2],
}

/// The tensors of a safetensors file: shape, F16 or F32, the raw bytes.
struct Tensors(HashMap<String, (Vec<usize>, bool, &'static [u8])>);

impl Tensors {
    fn parse(bytes: &'static [u8]) -> Result<Self> {
        ensure!(bytes.len() >= 8, "weights: truncated");
        let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        let data_start = 8 + header_len;
        ensure!(bytes.len() >= data_start, "weights: truncated header");
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes[8..data_start])?;
        let data = &bytes[data_start..];
        let mut out = HashMap::new();
        for (name, info) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info["dtype"].as_str().unwrap_or_default();
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .ok_or_else(|| anyhow!("weights: {name} has no shape"))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let offsets = info["data_offsets"].as_array().ok_or_else(|| anyhow!("weights: {name} has no offsets"))?;
            let (start, end) = (offsets[0].as_u64().unwrap_or(0) as usize, offsets[1].as_u64().unwrap_or(0) as usize);
            let raw = data.get(start..end).ok_or_else(|| anyhow!("weights: {name} out of range"))?;
            let f16 = match dtype {
                "F32" => false,
                "F16" => true,
                other => bail!("weights: {name} has dtype {other}"),
            };
            let size = if f16 { 2 } else { 4 };
            ensure!(raw.len() == size * shape.iter().product::<usize>(), "weights: {name} size mismatch");
            out.insert(name, (shape, f16, raw));
        }
        Ok(Self(out))
    }

    fn take(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (shape, f16, raw) = self.0.remove(name).with_context(|| format!("weights: {name} missing"))?;
        let values = if f16 {
            raw.as_chunks::<2>().0.iter().map(|&b| f16_to_f32(u16::from_le_bytes(b))).collect()
        } else {
            raw.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect()
        };
        Ok((shape, values))
    }

    /// A matrix as [`Half`]: the file's own bytes when they are F16.
    fn take_half(&mut self, name: &str) -> Result<(Vec<usize>, Half)> {
        let (shape, f16, raw) = self.0.remove(name).with_context(|| format!("weights: {name} missing"))?;
        if f16 {
            return Ok((shape, Half::Borrowed(raw)));
        }
        let values: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
        Ok((shape, Half::from_f32(&values)))
    }

    fn vec(&mut self, name: &str) -> Result<Vec<f32>> {
        Ok(self.take(name)?.1)
    }

    fn half(&mut self, name: &str) -> Result<Half> {
        Ok(self.take_half(name)?.1)
    }

    fn attention(&mut self, prefix: &str) -> Result<Attention> {
        let (shape, qkv) = self.take_half(&format!("{prefix}.qkv"))?;
        let dim = shape[1];
        let heads = shape[0] / 3 / HEAD_DIM;
        Ok(Attention {
            norm: self.vec(&format!("{prefix}.norm"))?,
            qkv,
            gates_w: self.half(&format!("{prefix}.gates.w"))?,
            gates_b: self.vec(&format!("{prefix}.gates.b"))?,
            out: self.half(&format!("{prefix}.out"))?,
            dim,
            heads,
        })
    }

    fn feed_forward(&mut self, prefix: &str) -> Result<FeedForward> {
        let (shape, w1) = self.take_half(&format!("{prefix}.w1"))?;
        Ok(FeedForward {
            norm: self.vec(&format!("{prefix}.norm"))?,
            w1,
            b1: self.vec(&format!("{prefix}.b1"))?,
            w2: self.half(&format!("{prefix}.w2"))?,
            b2: self.vec(&format!("{prefix}.b2"))?,
            dim: shape[1],
            inner: shape[0],
        })
    }

    fn conv(&mut self, prefix: &str) -> Result<Conv> {
        let (shape, w) = self.take(&format!("{prefix}.w"))?;
        let [cout, cin, kf, kt] = shape[..] else { bail!("weights: {prefix}.w is not 4-d") };
        ensure!(kt == 3, "weights: {prefix}.w kernel width {kt}");
        let mut reordered = vec![0.0; kf * 3 * cin * cout];
        for o in 0..cout {
            for i in 0..cin {
                for j in 0..kf {
                    for dt in 0..3 {
                        reordered[((j * 3 + dt) * cin + i) * cout + o] = w[((o * cin + i) * kf + j) * 3 + dt];
                    }
                }
            }
        }
        Ok(Conv { w: Half::from_f32(&reordered), b: self.vec(&format!("{prefix}.b"))?, cin, cout, kf })
    }
}

impl BeatThis {
    /// Load a network from a downloaded weights file (e.g. `final0` F16,
    /// 40 MB, from R2). The bytes are kept for the life of the process (the
    /// model reads its F16 matrices in place, so load it once and keep it,
    /// e.g. in a `OnceLock`); no f32 copy is made.
    pub fn from_vec(bytes: Vec<u8>) -> Result<Self> {
        Self::from_safetensors(Box::leak(bytes.into_boxed_slice()))
    }

    /// Load the network from `export.py`'s safetensors (any model size).
    /// F16 matrices are used in place (the bytes must live as long as the
    /// program: `include_bytes!`, or a leaked file).
    pub fn from_safetensors(bytes: &'static [u8]) -> Result<Self> {
        let mut t = Tensors::parse(bytes)?;
        let in_scale = t.vec("stem.in_scale")?;
        ensure!(in_scale.len() == N_MELS, "weights: expected {N_MELS} mel bins");
        let in_shift = t.vec("stem.in_shift")?;
        let stem = t.conv("stem.conv")?;
        let mut blocks = Vec::new();
        for i in 0..3 {
            blocks.push(Block {
                attn_f: t.attention(&format!("block{i}.attn_f"))?,
                ff_f: t.feed_forward(&format!("block{i}.ff_f"))?,
                attn_t: t.attention(&format!("block{i}.attn_t"))?,
                ff_t: t.feed_forward(&format!("block{i}.ff_t"))?,
                conv: t.conv(&format!("block{i}.conv"))?,
            });
        }
        let (shape, linear_w) = t.take_half("frontend.linear.w")?;
        let dim = shape[0];
        let linear_b = t.vec("frontend.linear.b")?;
        let mut layers = Vec::new();
        while t.0.contains_key(&format!("layer{}.attn.qkv", layers.len())) {
            let n = layers.len();
            layers.push((t.attention(&format!("layer{n}.attn"))?, t.feed_forward(&format!("layer{n}.ff"))?));
        }
        ensure!(!layers.is_empty(), "weights: no transformer layers");
        let mut freqs = [0.0; HEAD_DIM / 2];
        for (k, f) in freqs.iter_mut().enumerate() {
            *f = (1.0 / 10_000f64.powf((2 * k) as f64 / HEAD_DIM as f64)) as f32;
        }
        Ok(Self {
            in_scale,
            in_shift,
            stem,
            blocks,
            linear_w,
            linear_b,
            layers,
            norm: t.vec("norm")?,
            head_w: t.half("head.w")?,
            head_b: t.vec("head.b")?,
            dim,
            freqs,
        })
    }

    /// Steps of one chunk for [`BeatThis::chunk_step`]: the stem, two per
    /// frontend block, the projection, one per transformer layer, the head.
    pub fn chunk_steps(&self) -> usize {
        2 + 2 * self.blocks.len() + self.layers.len() + 1
    }

    /// Run the chunk that starts at frame `start` (of [`chunk_starts`]) and
    /// write its logits, without the borders, into `beat` and `down`
    /// (`split_predict_aggregate`). `mel` is the whole piece, `(frames, 128)`.
    pub fn predict_chunk(&self, ws: &mut Workspace, mel: &[f32], start: i64, beat: &mut [f32], down: &mut [f32]) {
        for step in 0..self.chunk_steps() {
            self.chunk_step(ws, mel, start, step, beat, down);
        }
    }

    /// Step `step` of [`BeatThis::predict_chunk`] (the steps run in order; a
    /// caller can yield between them: the longest is a time-directed
    /// attention, about a tenth of a chunk).
    pub fn chunk_step(&self, ws: &mut Workspace, mel: &[f32], start: i64, step: usize, beat: &mut [f32], down: &mut [f32]) {
        let len = mel.len() / N_MELS;
        let blocks = self.blocks.len();
        let Workspace { a, b, cols, scratch, h, t, freqs, channels } = ws;
        if step == 0 {
            // the chunk, zero-padded at the piece's edges; BatchNorm1d over
            // the mel bins; (freq, time, 1); the stem
            let from = start.max(0) as usize;
            let to = ((start + CHUNK as i64).max(0) as usize).min(len);
            let left = (-start).max(0) as usize;
            let right = (start + CHUNK as i64 - len as i64).clamp(0, BORDER as i64) as usize;
            *t = left + (to - from) + right;
            let input = &mut scratch.input;
            input.clear();
            input.resize(N_MELS * *t, 0.0);
            for frame in from..to {
                let at = frame - from + left;
                for f in 0..N_MELS {
                    input[f * *t + at] = mel[frame * N_MELS + f] * self.in_scale[f] + self.in_shift[f];
                }
            }
            for at in (0..left).chain(left + to - from..*t) {
                for f in 0..N_MELS {
                    input[f * *t + at] = self.in_shift[f];
                }
            }
            let rows = N_MELS / self.stem.kf;
            a.clear();
            a.resize(rows * *t * self.stem.cout, 0.0);
            let w = self.stem.w.load(&mut scratch.w1);
            for fo in 0..rows {
                self.stem.row(&scratch.input, *t, fo, w, cols, b);
                a[fo * *t * self.stem.cout..][..*t * self.stem.cout].copy_from_slice(b);
            }
            *freqs = N_MELS / self.stem.kf;
            *channels = self.stem.cout;
            return;
        }
        let t = *t;
        if step <= 2 * blocks {
            let block = &self.blocks[(step - 1) / 2];
            let (f_n, c_n) = (*freqs, *channels);
            if step % 2 == 1 {
                // frequency-directed: sequences over freq for every frame, a
                // block of frames at a time gathered into (frame, freq, channel)
                let frames = (GROUP_ROWS / f_n).max(1);
                let mut t0 = 0;
                while t0 < t {
                    let g = frames.min(t - t0);
                    b.clear();
                    b.resize(g * f_n * c_n, 0.0);
                    for i in 0..g {
                        for f in 0..f_n {
                            b[(i * f_n + f) * c_n..][..c_n].copy_from_slice(&a[(f * t + t0 + i) * c_n..][..c_n]);
                        }
                    }
                    block.attn_f.apply(b, g, f_n, &self.freqs, scratch);
                    block.ff_f.apply(b, g * f_n, scratch);
                    for i in 0..g {
                        for f in 0..f_n {
                            a[(f * t + t0 + i) * c_n..][..c_n].copy_from_slice(&b[(i * f_n + f) * c_n..][..c_n]);
                        }
                    }
                    t0 += g;
                }
            } else {
                // time-directed: sequences over time for every freq
                block.attn_t.apply(a, f_n, t, &self.freqs, scratch);
                block.ff_t.apply(a, f_n * t, scratch);
                // the convolution in place: output row fo takes exactly the
                // memory of the input rows it reads (kf · cin = cout)
                let conv = &block.conv;
                assert_eq!(conv.kf * conv.cin, conv.cout, "in-place convolution");
                let w = conv.w.load(&mut scratch.w1);
                for fo in 0..f_n / conv.kf {
                    conv.row(a, t, fo, w, cols, b);
                    a[fo * t * conv.cout..][..t * conv.cout].copy_from_slice(b);
                }
                a.truncate(f_n / conv.kf * t * conv.cout);
                *freqs /= block.conv.kf;
                *channels = block.conv.cout;
            }
            return;
        }
        if step == 2 * blocks + 1 {
            // "b c f t -> b t (c f)", then the projection to the transformer width
            let (freqs, channels) = (*freqs, *channels);
            let features = channels * freqs;
            let w = self.linear_w.load(&mut scratch.w1);
            h.clear();
            let frames = (GROUP_ROWS * 4 / features).max(1);
            let mut t0 = 0;
            while t0 < t {
                let g = frames.min(t - t0);
                b.clear();
                b.resize(g * features, 0.0);
                for f in 0..freqs {
                    for i in 0..g {
                        let src = &a[(f * t + t0 + i) * channels..][..channels];
                        let row = &mut b[i * features..][..features];
                        for (c, &v) in src.iter().enumerate() {
                            row[c * freqs + f] = v;
                        }
                    }
                }
                linear_into(b, g, features, w, self.dim, Some(&self.linear_b), &mut scratch.inner);
                h.extend_from_slice(&scratch.inner);
                t0 += g;
            }
            return;
        }
        let layer = step - 2 * blocks - 2;
        if let Some((attn, ff)) = self.layers.get(layer) {
            attn.apply(h, 1, t, &self.freqs, scratch);
            ff.apply(h, t, scratch);
            return;
        }
        // the head; SumHead: beat = beat + downbeat
        rms_norm_rows(h, self.dim, &self.norm);
        linear_into(h, t, self.dim, self.head_w.load(&mut scratch.w1), 2, Some(&self.head_b), b);
        for i in BORDER..t.saturating_sub(BORDER) {
            let at = start + i as i64;
            if at >= 0 && (at as usize) < len {
                beat[at as usize] = b[2 * i] + b[2 * i + 1];
                down[at as usize] = b[2 * i + 1];
            }
        }
    }
}

/// Buffers reused across layers and chunks (wasm memory never shrinks, so
/// the peak is what counts): the frontend activations `a` (`(freq, t,
/// channel)`, 1024 floats per frame, updated in place), a block buffer `b`
/// (at most one convolution output row, `t · 256`), the convolution columns,
/// the transformer activations and the attention scratch.
#[derive(Default)]
pub struct Workspace {
    a: Vec<f32>,
    b: Vec<f32>,
    cols: Vec<f32>,
    scratch: Scratch,
    /// The transformer's `(t, dim)` activations.
    h: Vec<f32>,
    /// Frames of the chunk in flight, and its current frequency bins and
    /// channels.
    t: usize,
    freqs: usize,
    channels: usize,
}

/// Frames per im2col block of a convolution.
const CONV_FRAMES: usize = 256;

impl Conv {
    /// Output row `fo` (`(t, cout)`, GELU applied) of `x (freqs, t, cin)`
    /// into `out`; `w` is the widened weight matrix.
    fn row(&self, x: &[f32], t: usize, fo: usize, w: &[f32], cols: &mut Vec<f32>, out: &mut Vec<f32>) {
        let k = self.kf * 3 * self.cin;
        out.clear();
        for _ in 0..t {
            out.extend_from_slice(&self.b);
        }
        let mut t0 = 0;
        while t0 < t {
            let n = CONV_FRAMES.min(t - t0);
            // im2col: row per frame, columns (j, dt, i)
            cols.clear();
            cols.resize(n * k, 0.0);
            for i in 0..n {
                let frame = t0 + i;
                let row = &mut cols[i * k..][..k];
                for j in 0..self.kf {
                    let f = fo * self.kf + j;
                    for dt in 0..3 {
                        let src_t = frame as isize + dt as isize - 1;
                        if src_t >= 0 && src_t < t as isize {
                            row[(j * 3 + dt) * self.cin..][..self.cin]
                                .copy_from_slice(&x[(f * t + src_t as usize) * self.cin..][..self.cin]);
                        }
                    }
                }
            }
            let dst = &mut out[t0 * self.cout..][..n * self.cout];
            // SAFETY: the dimensions and strides describe `cols`, `w` and `dst`.
            unsafe {
                matrixmultiply::sgemm(
                    n, k, self.cout, 1.0,
                    cols.as_ptr(), k as isize, 1,
                    w.as_ptr(), self.cout as isize, 1,
                    1.0, dst.as_mut_ptr(), self.cout as isize, 1,
                );
            }
            t0 += n;
        }
        gelu(out);
    }
}

/// Reusable buffers for the attention blocks.
#[derive(Default)]
struct Scratch {
    norm: Vec<f32>,
    qkv: Vec<f32>,
    gates: Vec<f32>,
    heads_out: Vec<f32>,
    scores: Vec<f32>,
    sums: Vec<f32>,
    inner: Vec<f32>,
    /// The chunk's normalised spectrogram, `(128, t)`.
    input: Vec<f32>,
    /// The layer's weights as f32 (they are kept as f16).
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
}

/// Rows processed together (attention groups whole sequences up to this;
/// the feed-forward blocks rows): bounds the scratch memory, which in wasm
/// never shrinks, to a few MB.
const GROUP_ROWS: usize = 512;
/// Queries per attention score block (`QUERY_BLOCK × seq` scores in cache).
const QUERY_BLOCK: usize = 64;

impl Attention {
    /// `x` is `(batch, seq, dim)`; `x += attention(x)` per sequence.
    fn apply(&self, x: &mut [f32], batch: usize, seq: usize, freqs: &[f32; HEAD_DIM / 2], s: &mut Scratch) {
        let (cos, sin) = rotary_tables(seq, freqs);
        self.qkv.load(&mut s.w1);
        self.gates_w.load(&mut s.w2);
        self.out.load(&mut s.w3);
        let group = (GROUP_ROWS / seq).max(1);
        let mut b0 = 0;
        while b0 < batch {
            let n = group.min(batch - b0);
            let rows = &mut x[b0 * seq * self.dim..(b0 + n) * seq * self.dim];
            self.apply_group(rows, n, seq, &cos, &sin, s);
            b0 += n;
        }
    }

    fn apply_group(&self, x: &mut [f32], batch: usize, seq: usize, cos: &[f32], sin: &[f32], s: &mut Scratch) {
        let rows = batch * seq;
        let dim = self.dim;
        let inner = self.heads * HEAD_DIM;
        s.norm.clear();
        s.norm.extend_from_slice(&x[..rows * dim]);
        rms_norm_rows(&mut s.norm, dim, &self.norm);
        linear_into(&s.norm, rows, dim, &s.w2, self.heads, Some(&self.gates_b), &mut s.gates);
        for g in s.gates.iter_mut() {
            *g = 1.0 / (1.0 + (-*g).exp());
        }
        s.heads_out.clear();
        s.heads_out.resize(rows * inner, 0.0);
        let block = QUERY_BLOCK.min(seq);
        s.scores.resize(block * seq, 0.0);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        // one head at a time: its q, k, v are (rows, 32) each, side by side
        let stride = 3 * HEAD_DIM;
        for h in 0..self.heads {
            s.qkv.clear();
            s.qkv.resize(rows * stride, 0.0);
            for (part, offset) in [(0, 0), (1, HEAD_DIM), (2, 2 * HEAD_DIM)] {
                let w = &s.w1[(part * inner + h * HEAD_DIM) * dim..][..HEAD_DIM * dim];
                // SAFETY: norm is (rows, dim), w (32, dim) read transposed,
                // the destination a (rows, 32) view with row stride `stride`.
                unsafe {
                    matrixmultiply::sgemm(
                        rows, dim, HEAD_DIM, 1.0,
                        s.norm.as_ptr(), dim as isize, 1,
                        w.as_ptr(), 1, dim as isize,
                        0.0, s.qkv[offset..].as_mut_ptr(), stride as isize, 1,
                    );
                }
            }
            // rotary position embedding on q and k (position = index in the sequence)
            for (r, row) in s.qkv.chunks_exact_mut(stride).enumerate() {
                let pos = r % seq;
                let (cos, sin) = (&cos[pos * HEAD_DIM / 2..][..HEAD_DIM / 2], &sin[pos * HEAD_DIM / 2..][..HEAD_DIM / 2]);
                for head in row[..2 * HEAD_DIM].as_chunks_mut::<HEAD_DIM>().0 {
                    for k in 0..HEAD_DIM / 2 {
                        let (a, b) = (head[2 * k], head[2 * k + 1]);
                        head[2 * k] = a * cos[k] - b * sin[k];
                        head[2 * k + 1] = b * cos[k] + a * sin[k];
                    }
                }
            }
            for b in 0..batch {
                let base = b * seq;
                let k = s.qkv[base * stride + HEAD_DIM..].as_ptr();
                let v = s.qkv[base * stride + 2 * HEAD_DIM..].as_ptr();
                let mut q0 = 0;
                while q0 < seq {
                    let nq = block.min(seq - q0);
                    let q = s.qkv[(base + q0) * stride..].as_ptr();
                    let scores = &mut s.scores[..nq * seq];
                    // SAFETY: q (nq, 32), k (seq, 32) and v (seq, 32) are
                    // views with row stride `stride` inside s.qkv; scores is
                    // (nq, seq); out a (nq, 32) view with row stride `inner`.
                    unsafe {
                        matrixmultiply::sgemm(
                            nq, HEAD_DIM, seq, scale,
                            q, stride as isize, 1,
                            k, 1, stride as isize,
                            0.0, scores.as_mut_ptr(), seq as isize, 1,
                        );
                    }
                    s.sums.clear();
                    s.sums.extend(scores.chunks_exact_mut(seq).map(softmax_unnormalised));
                    let out = s.heads_out[(base + q0) * inner + h * HEAD_DIM..].as_mut_ptr();
                    // SAFETY: as above.
                    unsafe {
                        matrixmultiply::sgemm(
                            nq, seq, HEAD_DIM, 1.0,
                            scores.as_ptr(), seq as isize, 1,
                            v, stride as isize, 1,
                            0.0, out, inner as isize, 1,
                        );
                    }
                    for (i, &sum) in s.sums.iter().enumerate() {
                        let row = &mut s.heads_out[(base + q0 + i) * inner + h * HEAD_DIM..][..HEAD_DIM];
                        let inv = 1.0 / sum;
                        row.iter_mut().for_each(|v| *v *= inv);
                    }
                    q0 += nq;
                }
            }
        }
        for (row, gates) in s.heads_out.chunks_exact_mut(inner).zip(s.gates.chunks_exact(self.heads)) {
            for (head, &g) in row.as_chunks_mut::<HEAD_DIM>().0.iter_mut().zip(gates) {
                head.iter_mut().for_each(|v| *v *= g);
            }
        }
        // x += heads_out · outᵀ
        // SAFETY: heads_out is (rows, inner), out (dim, inner), x (rows, dim).
        unsafe {
            matrixmultiply::sgemm(
                rows, inner, dim, 1.0,
                s.heads_out.as_ptr(), inner as isize, 1,
                s.w3.as_ptr(), 1, inner as isize,
                1.0, x.as_mut_ptr(), dim as isize, 1,
            );
        }
    }
}

impl FeedForward {
    /// `x += ff(x)` for `(rows, dim)`, in blocks of rows.
    fn apply(&self, x: &mut [f32], rows: usize, s: &mut Scratch) {
        self.w1.load(&mut s.w1);
        self.w2.load(&mut s.w2);
        for block in x[..rows * self.dim].chunks_mut(GROUP_ROWS * self.dim) {
            self.apply_block(block, block.len() / self.dim, s);
        }
    }

    fn apply_block(&self, x: &mut [f32], rows: usize, s: &mut Scratch) {
        s.norm.clear();
        s.norm.extend_from_slice(&x[..rows * self.dim]);
        rms_norm_rows(&mut s.norm, self.dim, &self.norm);
        linear_into(&s.norm, rows, self.dim, &s.w1, self.inner, Some(&self.b1), &mut s.inner);
        gelu(&mut s.inner);
        for row in x.chunks_exact_mut(self.dim).take(rows) {
            for (v, b) in row.iter_mut().zip(&self.b2) {
                *v += b;
            }
        }
        // SAFETY: inner is (rows, inner), w2 (dim, inner), x (rows, dim).
        unsafe {
            matrixmultiply::sgemm(
                rows, self.inner, self.dim, 1.0,
                s.inner.as_ptr(), self.inner as isize, 1,
                s.w2.as_ptr(), 1, self.inner as isize,
                1.0, x.as_mut_ptr(), self.dim as isize, 1,
            );
        }
    }
}

fn rotary_tables(seq: usize, freqs: &[f32; HEAD_DIM / 2]) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(seq * HEAD_DIM / 2);
    let mut sin = Vec::with_capacity(seq * HEAD_DIM / 2);
    for pos in 0..seq {
        for &f in freqs {
            let angle = pos as f32 * f;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

/// `x (rows, k) · wᵀ (w: (n, k)) + bias` → `out (rows, n)`.
fn linear_into(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize, bias: Option<&[f32]>, out: &mut Vec<f32>) {
    out.clear();
    match bias {
        Some(bias) => {
            for _ in 0..rows {
                out.extend_from_slice(bias);
            }
        }
        None => out.resize(rows * n, 0.0),
    }
    // SAFETY: x is (rows, k), w (n, k) read transposed, out (rows, n).
    unsafe {
        matrixmultiply::sgemm(
            rows, k, n, 1.0,
            x.as_ptr(), k as isize, 1,
            w.as_ptr(), 1, k as isize,
            1.0, out.as_mut_ptr(), n as isize, 1,
        );
    }
}

/// `roformer.RMSNorm`: `F.normalize(x) * sqrt(dim) * gamma` per row.
fn rms_norm_rows(x: &mut [f32], dim: usize, gamma: &[f32]) {
    let scale = (dim as f32).sqrt();
    for row in x.chunks_exact_mut(dim) {
        let norm = sum_squares(row).sqrt().max(1e-12);
        let k = scale / norm;
        for (v, g) in row.iter_mut().zip(gamma) {
            *v *= k * g;
        }
    }
}

fn sum_squares(x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (chunks, rest) = x.as_chunks::<8>();
    for c in chunks {
        for i in 0..8 {
            acc[i] += c[i] * c[i];
        }
    }
    acc.iter().sum::<f32>() + rest.iter().map(|v| v * v).sum::<f32>()
}

/// GELU (`nn.GELU()`, the erf form): `x/2 · (1 + erf(x/√2))`.
fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2));
    }
}

/// erf, branch-free so it vectorises: Abramowitz & Stegun 7.1.26
/// (|error| < 1.5e-7, in f32 < 5e-7).
#[inline(always)]
fn erf(x: f32) -> f32 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let y = 1.0
        - t * (0.254_829_6 + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))))
            * fast_exp(-a * a);
    y.copysign(x)
}

/// `row = exp(row - max(row))`; returns the sum (the caller divides, after
/// the product with V: 32 divisions per row instead of `seq`).
fn softmax_unnormalised(row: &mut [f32]) -> f32 {
    let mut lanes = [f32::NEG_INFINITY; 8];
    let (chunks, rest) = row.as_chunks::<8>();
    for c in chunks {
        for i in 0..8 {
            lanes[i] = if c[i] > lanes[i] { c[i] } else { lanes[i] };
        }
    }
    let max = rest.iter().chain(&lanes).fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    let mut acc = [0.0f32; 8];
    let (chunks, rest) = row.as_chunks_mut::<8>();
    for c in chunks {
        for i in 0..8 {
            c[i] = fast_exp(c[i] - max);
            acc[i] += c[i];
        }
    }
    let mut sum = 0.0f32;
    for v in rest {
        *v = fast_exp(*v - max);
        sum += *v;
    }
    sum + acc.iter().sum::<f32>()
}

/// `exp` for x ≤ 0 (softmax), branch-free so it vectorises; ~1 ulp.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    let x = x.max(-87.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    // Cody-Waite: ln 2 in two parts
    let r = x - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_67 + r * (0.041_666_668 + r * (0.008_333_452 + r * (0.001_388_949_4 + r * 0.000_198_527_62))))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// A weight matrix as f16 — the embedded file's own bytes, or converted —
/// widened into a scratch buffer right before its GEMM (wasm memory never
/// shrinks, so no f32 copy of the model is ever made).
enum Half {
    /// Little-endian f16 bytes of the weights file.
    Borrowed(&'static [u8]),
    Owned(Vec<u16>),
}

impl Half {
    fn from_f32(values: &[f32]) -> Self {
        Self::Owned(values.iter().map(|&v| f32_to_f16(v)).collect())
    }

    fn load<'a>(&self, out: &'a mut Vec<f32>) -> &'a [f32] {
        out.clear();
        match self {
            Self::Borrowed(bytes) => out.extend(bytes.as_chunks::<2>().0.iter().map(|&b| f16_to_f32(u16::from_le_bytes(b)))),
            Self::Owned(values) => out.extend(values.iter().map(|&h| f16_to_f32(h))),
        }
        out
    }
}

/// Round to nearest even; the weights are F16 already, so this is exact for
/// them (F32 exports lose what F16 cannot hold).
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // subnormal
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = u32::from(rest > mid || (rest == mid && half & 1 == 1));
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    let round = u32::from(rest > 0x1000 || (rest == 0x1000 && half & 1 == 1));
    sign | (half + round) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000,
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips() {
        for h in 0..=u16::MAX {
            let v = f16_to_f32(h);
            if v.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(v), h, "{h:#x}");
        }
        assert_eq!(f32_to_f16(1.0 + 1.0 / 4096.0), 0x3c00); // ties to even
        assert_eq!(f32_to_f16(1e6), 0x7c00);
    }

    #[test]
    fn f16_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }

    #[test]
    fn erf_is_accurate() {
        for i in -4000..4000 {
            let x = i as f32 * 0.001;
            let (a, b) = (erf(x), libm::erff(x));
            assert!((a - b).abs() <= 5e-7, "{x}: {a} vs {b}");
        }
    }

    #[test]
    fn fast_exp_is_accurate() {
        for i in 0..2000 {
            let x = -(i as f32) * 0.04;
            let (a, b) = (fast_exp(x), x.exp());
            assert!((a - b).abs() <= b * 2e-6 + 1e-38, "{x}: {a} vs {b}");
        }
    }
}
