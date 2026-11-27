//! Hyper-parameters of the four networks and the names and shapes of their
//! tensors. Both sides use these lists: `sone-models` exports exactly these
//! names (a test checks it), `sone-infer` loads exactly these. The config
//! itself travels in every weights file (`__metadata__.config`, JSON).
//!
//! Conventions: a linear layer is `{p}.w` `(out, in)` plus `{p}.b` `(out)`
//! (PyTorch layout); an RMSNorm gain is `{p}.norm` `(dim)`.

use serde::{Deserialize, Serialize};

/// Stems in output order. "other" is not predicted: it is the mixture minus these.
pub const STEMS: [&str; 3] = ["vocals", "drums", "bass"];

/// Shapes of the tensors of a network, in a stable order.
pub type Shapes = Vec<(String, Vec<usize>)>;

fn linear(out: &mut Shapes, p: &str, n_out: usize, n_in: usize) {
    out.push((format!("{p}.w"), vec![n_out, n_in]));
    out.push((format!("{p}.b"), vec![n_out]));
}

/// Pre-norm feed-forward: norm, `w1 (inner, dim)`, GELU, `w2 (dim, inner)`, residual.
fn feed_forward(out: &mut Shapes, p: &str, dim: usize, inner: usize) {
    out.push((format!("{p}.norm"), vec![dim]));
    linear(out, &format!("{p}.l1"), inner, dim);
    linear(out, &format!("{p}.l2"), dim, inner);
}

/// Pre-norm self-attention: norm, `qkv (3·dim, dim)` without bias, `out (dim, dim)` with bias.
fn attention(out: &mut Shapes, p: &str, dim: usize) {
    out.push((format!("{p}.norm"), vec![dim]));
    out.push((format!("{p}.qkv"), vec![3 * dim, dim]));
    linear(out, &format!("{p}.out"), dim, dim);
}

/// Time block: norm, depthwise dilated convolution along time
/// (`conv.w (dim, kernel)`, `conv.b (dim)`), then a feed-forward without its own norm.
fn time_block(out: &mut Shapes, p: &str, dim: usize, inner: usize, kernel: usize) {
    out.push((format!("{p}.norm"), vec![dim]));
    out.push((format!("{p}.conv.w"), vec![dim, kernel]));
    out.push((format!("{p}.conv.b"), vec![dim]));
    linear(out, &format!("{p}.l1"), inner, dim);
    linear(out, &format!("{p}.l2"), dim, inner);
}

fn count(shapes: &Shapes) -> usize {
    shapes.iter().map(|(_, s)| s.iter().product::<usize>()).sum()
}

// ---------------------------------------------------------------- Stems

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CochleaConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    /// `[first_bin, end_bin)` of every band, ascending and gap-free over all `n_fft/2 + 1` bins.
    pub bands: Vec<[usize; 2]>,
    /// Width of the band feature vectors.
    pub dim: usize,
    pub blocks: usize,
    /// Heads of the attention across bands (and across time when `time_attn` is true).
    pub heads: usize,
    /// Feed-forward width = `dim · ff_mult`.
    pub ff_mult: usize,
    /// Kernel of the depthwise time convolution (odd).
    pub kernel: usize,
    /// Dilation of every block's time convolution.
    pub dilations: Vec<usize>,
    /// Hidden width of each band's mask MLP.
    pub mask_hidden: usize,
    /// Dual-axis BS-RoFormer-style self-attention along time per band (`block{i}.t_attn`).
    #[serde(default)]
    pub time_attn: bool,
    /// Predict all 4 stems (`vocals`, `drums`, `bass`, `other`) with stereo-coherent
    /// energy-weighted mixture-consistent residual projection instead of dumping error into `other`.
    #[serde(default)]
    pub four_stems: bool,
    /// Smoothly bound raw complex masks via `b * tanh(raw / b)` (`0.0` = unbounded legacy).
    #[serde(default)]
    pub mask_bound: f32,
}

pub const STEMS_4: [&str; 4] = ["vocals", "drums", "bass", "other"];

impl CochleaConfig {
    /// `small`, `base`, `large`, `v2-base`, or `pro`.
    pub fn size(name: &str) -> Option<Self> {
        let cycle = [1, 2, 4, 8, 16, 32];
        match name {
            "small" | "base" | "large" => {
                let (dim, blocks, heads, mask_hidden) = match name {
                    "small" => (48, 4, 4, 128),
                    "base" => (64, 6, 4, 192),
                    "large" => (96, 8, 6, 256),
                    _ => unreachable!(),
                };
                Some(Self {
                    sample_rate: 44_100,
                    n_fft: 2048,
                    hop: 512,
                    bands: default_bands(2048),
                    dim,
                    blocks,
                    heads,
                    ff_mult: 2,
                    kernel: 5,
                    dilations: (0..blocks).map(|i| cycle[i % cycle.len()]).collect(),
                    mask_hidden,
                    time_attn: false,
                    four_stems: false,
                    mask_bound: 0.0,
                })
            }
            "v2-base" | "pro" | "ultra" => {
                let (dim, blocks, heads, mask_hidden, bands) = match name {
                    "v2-base" => (80, 6, 4, 192, studio_bands(4096)),
                    "pro" => (128, 8, 8, 256, studio_bands(4096)),
                    "ultra" => (160, 10, 8, 320, studio_bands(4096)),
                    _ => unreachable!(),
                };
                Some(Self {
                    sample_rate: 44_100,
                    n_fft: 4096,
                    hop: 1024,
                    bands,
                    dim,
                    blocks,
                    heads,
                    ff_mult: 2,
                    kernel: 5,
                    dilations: (0..blocks).map(|i| cycle[i % cycle.len()]).collect(),
                    mask_hidden,
                    time_attn: true,
                    four_stems: true,
                    mask_bound: 2.0,
                })
            }
            _ => None,
        }
    }

    pub fn stem_names(&self) -> &'static [&'static str] {
        if self.four_stems { &STEMS_4 } else { &STEMS }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn ff(&self) -> usize {
        self.dim * self.ff_mult
    }

    pub fn width(&self, band: usize) -> usize {
        self.bands[band][1] - self.bands[band][0]
    }

    /// Runs of consecutive bands with equal width: `(first band, count, width)`.
    /// Training batches each run into one matrix product.
    pub fn band_groups(&self) -> Vec<(usize, usize, usize)> {
        let mut out: Vec<(usize, usize, usize)> = Vec::new();
        for k in 0..self.bands.len() {
            let w = self.width(k);
            match out.last_mut() {
                Some(g) if g.2 == w => g.1 += 1,
                _ => out.push((k, 1, w)),
            }
        }
        out
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.kernel % 2 == 1, "stems: kernel must be odd");
        anyhow::ensure!(self.dilations.len() == self.blocks, "stems: one dilation per block");
        anyhow::ensure!(self.dim % self.heads == 0, "stems: dim must divide into heads");
        anyhow::ensure!(!self.bands.is_empty() && self.bands[0][0] == 0, "stems: bands must start at bin 0");
        for w in self.bands.windows(2) {
            anyhow::ensure!(w[0][1] == w[1][0] && w[0][0] < w[0][1], "stems: bands must be contiguous");
        }
        anyhow::ensure!(self.bands.last().unwrap()[1] == self.bins(), "stems: bands must end at the last bin");
        Ok(())
    }

    /// Every tensor of the network. Band features are `4·width` long:
    /// index `j·4 + c·2 + ri` for bin `j` of the band, channel `c`, real/imaginary `ri`.
    pub fn tensor_shapes(&self) -> Shapes {
        let (d, k) = (self.dim, self.bands.len());
        let mut s = Shapes::new();
        for b in 0..k {
            let w4 = 4 * self.width(b);
            s.push((format!("split.{b}.norm"), vec![w4]));
            linear(&mut s, &format!("split.{b}"), d, w4);
        }
        s.push(("band_emb".into(), vec![k, d]));
        for i in 0..self.blocks {
            time_block(&mut s, &format!("block{i}.time"), d, self.ff(), self.kernel);
            if self.time_attn {
                attention(&mut s, &format!("block{i}.t_attn"), d);
            }
            attention(&mut s, &format!("block{i}.band"), d);
            feed_forward(&mut s, &format!("block{i}.ff"), d, self.ff());
        }
        s.push(("norm".into(), vec![d]));
        for &stem in self.stem_names() {
            for b in 0..k {
                linear(&mut s, &format!("mask.{stem}.{b}.l1"), self.mask_hidden, d);
                linear(&mut s, &format!("mask.{stem}.{b}.l2"), 4 * self.width(b), self.mask_hidden);
            }
        }
        s
    }

    pub fn num_params(&self) -> usize {
        count(&self.tensor_shapes())
    }

    /// Multiply-adds of the network per STFT frame (the transforms not included).
    pub fn macs_per_frame(&self) -> u64 {
        let (d, k, ff, m) = (self.dim as u64, self.bands.len() as u64, self.ff() as u64, self.mask_hidden as u64);
        let bins4 = 4 * self.bins() as u64;
        let split = bins4 * d;
        let head_dim = d / self.heads as u64;
        let time = k * (self.kernel as u64 * d + 2 * d * ff);
        // qkv, scores and weighted sum over k bands, output projection
        let band = k * (3 * d * d + d * d) + self.heads as u64 * 2 * k * k * head_dim;
        let ffn = k * 2 * d * ff;
        let t_attn = if self.time_attn { k * (3 * d * d + d * d) } else { 0 };
        let n_stems = self.stem_names().len() as u64;
        let masks = n_stems * (k * d * m + m * bins4);
        split + self.blocks as u64 * (time + t_attn + band + ffn) + masks
    }

    /// Multiply-adds of the STFT and the inverse STFTs per frame, both
    /// channels, counting a radix-2 FFT of N points as 2·N·log2(N) real
    /// multiply-adds (its N/2·log2 N complex butterflies).
    pub fn transform_macs_per_frame(&self) -> u64 {
        let n = self.n_fft as u64;
        let fft = 2 * n * n.trailing_zeros() as u64;
        // one packed complex FFT carries both channels; one forward, three inverse
        4 * fft
    }
}

/// High-resolution 64-band psychoacoustic vocal-harmonic split for `n_fft = 4096`
/// (10.77 Hz/bin at 44.1 kHz): 21.5 Hz sub-bass/kick bands (`0..172 Hz`),
/// 43 Hz vocal-fundamental bands (`172..1033 Hz`), 129 Hz vocal-formant bands
/// (`1.0..4.1 kHz`), 689 Hz sibilance/hat bands (`4.1..11 kHz`), and air bands.
pub fn studio_bands(n_fft: usize) -> Vec<[usize; 2]> {
    let bins = n_fft / 2 + 1;
    let scale = n_fft as f64 / 4096.0;
    let table = [(16, 2), (96, 4), (384, 12), (1024, 64), (1536, 512), (bins, 513)];
    let mut out: Vec<[usize; 2]> = Vec::new();
    let mut at = 0;
    for (end, width) in table {
        let end = ((end as f64 * scale).round() as usize).min(bins);
        let width = ((width as f64 * scale).round() as usize).max(1);
        while at < end {
            let stop = (at + width).min(end);
            if stop - at < width.div_ceil(2) && !out.is_empty() {
                out.last_mut().unwrap()[1] = stop;
            } else {
                out.push([at, stop]);
            }
            at = stop;
        }
    }
    if let Some(last) = out.last_mut() {
        last[1] = bins;
    }
    out
}

/// Ultra-fine 83-band vocal-formant harmonic split for `Cochlea-Ultra` (`n_fft = 4096`):
/// 24 sub-bass/fundamental bands at 21.5 Hz (`0..517 Hz`), 36 core vocal F1/F2 bands
/// at 43 Hz (`517..2067 Hz`), 12 presence bands at 172 Hz (`2.1..4.1 kHz`), plus sibilance & air.
pub fn ultra_bands(n_fft: usize) -> Vec<[usize; 2]> {
    let bins = n_fft / 2 + 1;
    let scale = n_fft as f64 / 4096.0;
    let table = [(48, 2), (192, 4), (384, 16), (896, 64), (1408, 256), (bins, 641)];
    let mut out: Vec<[usize; 2]> = Vec::new();
    let mut at = 0;
    for (end, width) in table {
        let end = ((end as f64 * scale).round() as usize).min(bins);
        let width = ((width as f64 * scale).round() as usize).max(1);
        while at < end {
            let stop = (at + width).min(end);
            if stop - at < width.div_ceil(2) && !out.is_empty() {
                out.last_mut().unwrap()[1] = stop;
            } else {
                out.push([at, stop]);
            }
            at = stop;
        }
    }
    if let Some(last) = out.last_mut() {
        last[1] = bins;
    }
    out
}

/// Bands narrow at the bottom, wide at the top (after BSRNN's split for
/// 44.1 kHz, with fewer bands): 86 Hz up to 1 kHz, 258 Hz up to 4 kHz,
/// 517 Hz up to 8 kHz, 1 kHz up to 16 kHz, then about 2.8 kHz.
pub fn default_bands(n_fft: usize) -> Vec<[usize; 2]> {
    let bins = n_fft / 2 + 1;
    // (end bin, width in bins) at n_fft 2048; scaled for other sizes
    let scale = n_fft as f64 / 2048.0;
    let table = [(48, 4), (192, 12), (384, 24), (768, 48), (bins, 128)];
    let mut out: Vec<[usize; 2]> = Vec::new();
    let mut at = 0;
    for (end, width) in table {
        let end = ((end as f64 * scale) as usize).min(bins);
        let width = ((width as f64 * scale).round() as usize).max(1);
        while at < end {
            let stop = (at + width).min(end);
            // a leftover narrower than half a band joins the band before it
            if stop - at < width.div_ceil(2) && !out.is_empty() {
                out.last_mut().unwrap()[1] = stop;
            } else {
                out.push([at, stop]);
            }
            at = stop;
        }
    }
    out
}

// ---------------------------------------------------------------- Ear

/// Frame heads of the Ear, in output order.
pub const TYMPANUM_FRAME_HEADS: [&str; 5] = ["beat", "downbeat", "voice", "line_end", "phrase_start"];

/// The 24 keys: index = pitch class (C = 0) + 12 for minor.
pub const KEYS: [&str; 24] = [
    "C major", "C# major", "D major", "D# major", "E major", "F major", "F# major", "G major", "G# major", "A major", "A# major", "B major",
    "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor", "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TympanumConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    /// 882 samples at 44.1 kHz: 50 frames per second.
    pub hop: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub channels: usize,
    pub kernel: usize,
    /// One per convolution block.
    pub dilations: Vec<usize>,
    pub gru_layers: usize,
    /// Per direction.
    pub gru_hidden: usize,
}

impl Default for TympanumConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            n_fft: 2048,
            hop: 882,
            n_mels: 128,
            fmin: 30.0,
            fmax: 11_000.0,
            channels: 96,
            kernel: 5,
            dilations: vec![1, 2, 4, 8],
            gru_layers: 2,
            gru_hidden: 64,
        }
    }
}

impl TympanumConfig {
    pub fn fps(&self) -> f64 {
        self.sample_rate as f64 / self.hop as f64
    }

    pub fn tensor_shapes(&self) -> Shapes {
        let (c, h) = (self.channels, self.gru_hidden);
        let mut s = Shapes::new();
        s.push(("in_scale".into(), vec![self.n_mels]));
        s.push(("in_shift".into(), vec![self.n_mels]));
        linear(&mut s, "proj", c, self.n_mels);
        for i in 0..self.dilations.len() {
            time_block(&mut s, &format!("conv{i}"), c, 2 * c, self.kernel);
        }
        for l in 0..self.gru_layers {
            let input = if l == 0 { c } else { 2 * h };
            for dir in ["fwd", "bwd"] {
                let p = format!("gru{l}.{dir}");
                s.push((format!("{p}.w_ih"), vec![3 * h, input]));
                s.push((format!("{p}.w_hh"), vec![3 * h, h]));
                s.push((format!("{p}.b_ih"), vec![3 * h]));
                s.push((format!("{p}.b_hh"), vec![3 * h]));
            }
        }
        linear(&mut s, "frame_head", TYMPANUM_FRAME_HEADS.len(), 2 * h);
        linear(&mut s, "key_head", 24, 2 * h);
        linear(&mut s, "tuning_head", 1, 2 * h);
        s
    }

    pub fn num_params(&self) -> usize {
        count(&self.tensor_shapes())
    }
}

// ---------------------------------------------------------------- Cortex and Cingulate

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BeatKind {
    #[serde(alias = "plan")]
    Cortex,
    #[serde(alias = "critic")]
    Cingulate,
}

/// A transformer encoder over the beat tokens of both songs, with a global
/// token first: `[global, A's 80 beats, B's 128 beats]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatConfig {
    pub kind: BeatKind,
    /// A's beats: 64 before the one, 16 after (the lanes' ruler, beat for beat).
    pub a_len: usize,
    pub b_len: usize,
    /// `sone_core::example::FEATURES`.
    pub features: usize,
    /// Gains per beat (A's four stems, B's four): an input of the critic, an output of the planner.
    pub lanes: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub ff: usize,
}

/// Global inputs of the planner: tempo A, tempo B, beat ratio, key distance.
/// The critic also gets the four `sone_core::Measures` (beat offset, stretch,
/// vocal leak, tuning difference): measured on the audio, they are the only
/// way it can hear a leaking voice or drifting beats.
pub const GLOBALS: usize = 4;
pub const MEASURES: usize = 4;
pub const RECIPES: usize = 4;

impl BeatConfig {
    pub fn cortex() -> Self {
        Self { kind: BeatKind::Cortex, a_len: 80, b_len: 128, features: 24, lanes: 8, dim: 128, layers: 6, heads: 4, ff: 512 }
    }

    pub fn cingulate() -> Self {
        Self { kind: BeatKind::Cingulate, a_len: 80, b_len: 128, features: 24, lanes: 8, dim: 96, layers: 4, heads: 4, ff: 384 }
    }

    /// Width of an A token's input.
    pub fn a_features(&self) -> usize {
        match self.kind {
            BeatKind::Cortex => self.features,
            BeatKind::Cingulate => self.features + self.lanes,
        }
    }

    /// Width of the global token's input.
    pub fn globals(&self) -> usize {
        match self.kind {
            BeatKind::Cortex => GLOBALS,
            BeatKind::Cingulate => GLOBALS + MEASURES,
        }
    }

    pub fn tokens(&self) -> usize {
        1 + self.a_len + self.b_len
    }

    pub fn issues(&self) -> usize {
        sone_core::Issue::ALL.len()
    }

    pub fn tensor_shapes(&self) -> Shapes {
        let d = self.dim;
        let mut s = Shapes::new();
        // input standardisation, fitted on the training data, not trained
        s.push(("input.a_mean".into(), vec![self.a_features()]));
        s.push(("input.a_std".into(), vec![self.a_features()]));
        s.push(("input.b_mean".into(), vec![self.features]));
        s.push(("input.b_std".into(), vec![self.features]));
        s.push(("input.g_mean".into(), vec![self.globals()]));
        s.push(("input.g_std".into(), vec![self.globals()]));
        linear(&mut s, "embed_a", d, self.a_features());
        linear(&mut s, "embed_b", d, self.features);
        linear(&mut s, "embed_g", d, self.globals());
        s.push(("pos_a".into(), vec![self.a_len, d]));
        s.push(("pos_b".into(), vec![self.b_len, d]));
        for i in 0..self.layers {
            attention(&mut s, &format!("layer{i}.attn"), d);
            feed_forward(&mut s, &format!("layer{i}.ff"), d, self.ff);
        }
        s.push(("norm".into(), vec![d]));
        match self.kind {
            BeatKind::Cortex => {
                linear(&mut s, "head_recipe", RECIPES, d);
                linear(&mut s, "head_bars", 2, d);
                linear(&mut s, "head_lanes", self.lanes, d);
            }
            BeatKind::Cingulate => linear(&mut s, "head", 1 + self.issues(), d),
        }
        s
    }

    pub fn num_params(&self) -> usize {
        count(&self.tensor_shapes())
    }
}

/// `start_bars` and `swap_bars` are predicted as `BARS_SCALE ·` the head's output.
pub const BARS_SCALE: f32 = 8.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bands_cover_the_spectrum() {
        let b = default_bands(2048);
        assert_eq!(b.len(), 42);
        assert_eq!(b[0], [0, 4]);
        assert_eq!(*b.last().unwrap(), [896, 1025]);
        for name in ["small", "base", "large"] {
            CochleaConfig::size(name).unwrap().validate().unwrap();
        }
        let groups = CochleaConfig::size("base").unwrap().band_groups();
        assert_eq!(groups.iter().map(|g| g.1).sum::<usize>(), 42);
    }
}
