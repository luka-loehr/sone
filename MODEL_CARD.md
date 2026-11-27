# Model card: Sone

## Summary

| | |
|---|---|
| **Family** | `Sone-6.5M` (6,520,451 params, 13.12 MB), `Sone-v2-Base` (13,606,935 params, 27.36 MB), `Sone-Pro` (26,372,391 params, 52.90 MB), `Sone-Ultra` (35,481,159 params, 71.12 MB) |
| **Wire format** | Single-file zero-copy IEEE-754 FP16 `.safetensors` (`sone-infer/1`, loaded via `sone_infer::Sone::load`; standalone per-organ `.safetensors` and FP32 exports also available) |
| **Task** | 44.1 kHz stereo track pair -> beat-locked (`<= 0.7 ms` rendered drift), stem-carved, vocal-onset-aligned DJ transition spliced bit-for-bit into untouched studio masters outside the ~24 s boundary window `[u_0, u_1]` |
| **Author** | Luka Löhr |
| **Engine code** | Pure Rust and WebAssembly (`sone-infer`, `sone-dsp`, `sone-render`, `sone-live`, Apache-2.0, zero external ML runtime dependencies) |
| **Weights licence** | Non-commercial research use (CC BY-NC-SA 4.0, inherited from MUSDB18-HQ; zero audio retained in weights or code) |

## Architecture

Named after Stevens' psychoacoustic *sone* unit of perceived loudness, **Sone** packs four namespaced neural organs into one `model.safetensors` file:

| Tier | `cochlea.*` (Separator) | `tympanum.*` (7-Head Sensor) | `cortex.*` (Beat Planner) | `cingulate.*` (Conflict Monitor) | Tensors | Total Params | FP16 Size |
|---|---|---|---|---|---:|---:|---:|
| **`Sone-6.5M`** | 4,519,376 (2048-FFT, 42b, 6B, `d=64`) | 304,510 (128-mel, 4C+2xBiGRU-64) | 1,221,494 (6L/128D) | 475,071 (4L/96D, 11 heads) | 910 | **6,520,451** | 13.12 MB |
| **`Sone-v2-Base`** | 11,605,860 (4096-FFT, 64b Dual-Axis, 6B, `d=80`) | 304,510 (128-mel, equiv.) | 1,221,494 (6L/128D, `v4-Base`) | 475,071 (4L/96D, 30 heads) | 1,520 | **13,606,935** | 27.36 MB |
| **`Sone-Pro`** | 20,069,012 (4096-FFT, 64b Dual-Axis, 8B, `d=128`) | 1,473,406 (160-mel, 6C+3xBiGRU-128) | 3,604,726 (8L/192D, `v4-Pro`) | 1,225,247 (6L/128D, 30 heads) | 1,618 | **26,372,391** | 52.90 MB |
| **`Sone-Ultra`** | 29,177,780 (4096-FFT, 64b Dual-Axis, 10B, `d=160` Net2Net) | 1,473,406 (160-mel, 6C+3xBiGRU-128) | 3,604,726 (8L/192D, `v4-Pro`) | 1,225,247 (6L/128D, 30 heads) | 1,658 | **35,481,159** | 71.12 MB |

- **`cochlea.*`**: Band-split Dual-Axis Time+Band Transformer predicting bounded 4-stem complex masks (`2 * tanh(raw / 2)`) projected onto exact stereo mixture conservation (`< -148 dB`).
- **`tympanum.*`**: 50 Hz dilated depthwise TCN + BiGRU predicting beats, downbeats, vocal activity, inter-phrase breaths, 8-bar phrase starts, 24 musical keys, and sub-semitone tuning cents, paired at inference with a multi-octave spectral-flux comb grid.
- **`cortex.*`**: Pre-RMSNorm Beat Transformer over 209 tokens (`[global; A_80; B_128]`, 24 features per beat) predicting transition recipe, normalized bar offsets `(u_start, u_swap)`, and `80 x 8` stem gain curves.
- **`cingulate.*`**: Conflict Transformer scoring candidate plans with acceptability `q in [0, 1]` and up to 30 independent DJ defect heads for Best-of-6 reranking.

See `docs/ORGANS.md`, `crates/sone-infer/`, and `paper/main.pdf` for exact equations and wire metadata.

## Training

- **Supervision:**
  - `Cochlea`: Vocal-weighted complex STFT L1, waveform L1, 5-scale multi-resolution STFT loss, 128-mel perceptual loss, active-vocal cosine phase-coherence loss, direct time-domain SDR, and `8x` silent-vocal pre-onset suppression.
  - `Tympanum`: Exact vinyl-speed equivariant resampling (`alpha = 2^(delta_s / 12)`) transforming all five temporal targets, key class, and tuning cents analytically.
  - `Cortex v4` & `Cingulate v4`: Sharp-cut `7x` edge-weighted fader supervision, accompaniment-floor regularisation, post-drop unity lock, pre-vocal silence gating, and 50% vectorized 12-mode GPU counterfactual defect synthesis (20,000 steps per organ).
- **Data:**
  - `Cochlea`: 150 MUSDB18-HQ multitracks (138 train, 12 held-out) + 12 broadcast residual multitracks (6 train, 6 unseen test) + 1,101 Stage-1 `HTDemucs-ft` Creative Commons 48 s 4-stem clips + 372 two-stage `BS-RoFormer-1297` (12.98 dB vocal SDR) + `HTDemucs-ft` 48 s clips from MTG-Jamendo and Song Describer (streamed in ephemeral GPU RAM).
  - `Tympanum`: 584 training songs / 16 held-out (MUSDB18-HQ isolated stems + 450 stem-distilled Creative Commons tracks + 12 broadcast 7-head tracks).
  - `Cortex` & `Cingulate`: 26,904 simulated training transitions (`sone-sim v4`), 5,932 strictly unseen validation transitions (`val.jsonl.gz`), and 152 creator-audited transitions (`owner.jsonl`).
- **Compute:**
  - Dual NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (`sm_120`, 96 GB GDDR7, BF16 + `torch.compile`, 37.5 to 44.8 six-second stereo examples/s) for Stage-1 distillation, `Cochlea-v2-Base`, and `Cochlea-Pro`.
  - Dual NVIDIA L4 GPUs (`sm_89`, 24 GB GDDR6, BF16 FlashAttention) for two-stage `BS-RoFormer-1297` distillation, `Cochlea-Pro v2`, and Net2Net-expanded `Cochlea-Ultra`.
  - NVIDIA A100-SXM4-40GB (`sm_80`) for scaling sweeps; NVIDIA Tesla T4 (`sm_75`, 16 GB, FP16 Tensor Cores) for `Cortex v4`, `Cingulate v4`, and `Tympanum`.
  - Production single-core CPU host: 2-vCPU Intel Xeon VM (Model 151, AVX2/FMA/F16C, 3.8 GiB RAM, `rustc 1.99.1 -C opt-level=3`, single thread).

## Evaluation

### Stem separation (held-out MUSDB18-HQ, n = 12, 60 s stereo)

| Model (`cochlea.*`) | Vocals ↑ | Drums ↑ | Bass ↑ | Other ↑ | Mean SDR ↑ | Vocal leak ↓ | 1-vCPU RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Cochlea v1` (4.52 M) | 3.57 | 3.50 | 2.75 | 1.33 | 2.79 | -6.3 dB | 0.13x |
| `Cochlea-v2-Base` (11.61 M) | 6.73 | 6.06 | 4.23 | 2.60 | 4.90 | -9.3 dB | 0.40x |
| `Cochlea-Pro` (20.07 M, Stage-1) | 7.84 | 6.76 | 4.41 | 3.19 | 5.55 | -10.8 dB | 0.96x |
| `Cochlea-Pro v2` (20.07 M, +RoFormer-1297) | **7.98** | 6.99 | **4.63** | **3.31** | **5.73** | **-10.9 dB** | 0.96x |
| `Cochlea-Ultra` (29.18 M, 10B Net2Net) | 7.95 | **7.03** | 4.57 | 3.30 | 5.71 | -10.9 dB | 1.30x |
| *GPU Teacher: `BS-RoFormer-1297`* | *12.98* | *–* | *–* | *–* | *–* | *-21.4 dB* | *GPU offline* |

On the 6 unseen broadcast test songs, `Cochlea-Ultra` reaches 6.93 dB mean SDR with `-29.3 dB` silent-intro suppression. All checkpoints conserve the stereo mixture to `< -148 dB` with FP32-vs-FP16 max waveform deviation `<= 2.39e-4`.

### Structural perception, planning, and conflict monitoring

| Organ | Metric | Value | 1-vCPU latency |
|---|---|---|---:|
| `Tympanum v2-Base` (305 K) | Beat F1 / Downbeat F1 / Voice Acc / Tuning MAE | 0.865 / 0.465 / 81.6% / 6.56 c | 0.0024x RT |
| `Tympanum-Pro` (1.47 M) | Beat F1 / Downbeat F1 / Voice Acc / Tuning MAE | 0.862 / 0.506 / 82.3% / 6.41 c | 0.0085x RT |
| `Cortex v4-Pro` (3.60 M) | Held-out Recipe / Bar MAE / 8-Lane MAE / Owner Align | 98.16% / 0.127 bars / 0.0157 / 100.0% | 38.03 ms |
| `Cingulate v4-Pro` (1.23 M) | Held-out Acc (`n = 5,932`) / Owner Prec (`n = 152`) | 96.61% / 100.0% (FP = 0) | 16.66 ms |

### Single-threaded CPU budget per transition (2 x 24 s boundary + master splice)

| Tier | Total 1-core CPU / mix | Amortized load (180 s song) |
|---|---:|---:|
| `Sone-6.5M` | 8.6 s | 4.8% |
| `Sone-v2-Base` | 21.5 s | 11.9% |
| `Sone-Pro` | 49.5 s | 27.5% |
| `Sone-Ultra` | 65.5 s | 36.4% |

## Intended use

- Automated radio stations, playlist transition engines, and local or browser WebAssembly DJ players that require untouched master stereo outside transitions and low single-core CPU cost.
- Research on compact phase-coherent band-split separation, equivariant metrical analysis, and neuro-symbolic audio control.

## Limitations

- **Truncated preview clips vs. full-length songs:** `Tympanum` and `Cortex` are trained on complete songs to locate real outro phrases and vocal exits on Deck A alongside natural beat or verse openings on Deck B and overlap them. Passing preview clips chopped off mid-song (for example 30 s store snippets or 2 min dataset previews cut mid-chorus) removes the natural ending and degrades transition quality into fallback cuts.
- **Continuous-vocal outros:** When Deck A sings up to its final downbeat with no instrumental outro or breath gap, `Cortex` selects a clean on-beat cut or minimal sneak rather than risking a double-vocal clash (`Cingulate` flags `CutNotTransition`).
- **Single-pass CPU window vs. offline GPU ensembles:** Two-stage `BS-RoFormer-1297` + `HTDemucs-ft` distillation and 10-block Net2Net `Cochlea-Ultra` sharpen vocal isolation, but heavily drenched stereo reverb tails remain bounded by single-pass 6 s chunked capacity compared with `>100M` full-song GPU RoPE ensembles.
- **Global BSSEval SDR on silent boundary stems:** Standard full-window SDR diverges toward `-inf dB` on instrumental windows with `-108 dBFS` ground-truth vocals even when residual vocal energy is `< -77 dBFS`. Evaluate DJ windows with MDX23 active-chunk median SDR alongside silent-stem RMS floors.
- **4/4 bar grammar:** `Cortex` plans within a 4-beat bar grammar (`±6.8%` key-locked stretch) and does not synthesize turntable scratch routines, stutter edits, or 3/4 meter transitions.
- **Single-creator preference labels:** `owner.jsonl` (`n = 152`) reflects one station creator's blind listening judgments rather than a multi-listener MUSHRA panel.

## Usage

```rust
use sone_infer::{Sone, WeightFile};

let bytes = std::fs::read("model.safetensors")?;
let sone = Sone::load(WeightFile::from_vec(bytes)?)?;
```

```bash
cargo test --release --workspace
cargo build --release -p sone
```
