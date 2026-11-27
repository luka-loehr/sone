---
license: cc-by-nc-sa-4.0
library_name: safetensors
tags:
  - audio
  - music-information-retrieval
  - source-separation
  - beat-tracking
  - automatic-dj
  - rust
  - cpu-inference
  - webassembly
---

# Model Card: Sone (`Sone-6.5M` · `Sone-v2-Base` · `Sone-Pro`)

## Summary

| | |
|---|---|
| **Family** | `Sone-6.5M` (`6,520,451` params, `13.12 MB`), `Sone-v2-Base` (`13,606,935` params, `27.36 MB`), `Sone-Pro` (`26,372,391` params, `52.90 MB`) |
| **Wire format** | Unified IEEE-754 FP16 `model.safetensors` (`sone-infer/1`, zero-copy loaded via `sone_infer::Sone::load`, plus optional standalone organ slices) |
| **Task** | Raw `44.1 kHz` stereo song pair → beat-locked, stem-carved, vocal-onset-aligned DJ transition spliced bit-for-bit into untouched studio masters outside `[u_0, u_1]` |
| **Author** | Luka Löhr |
| **Engine code** | Pure Rust + WebAssembly (`sone-infer`, `sone-dsp`, `sone-render`, `sone-live`, Apache-2.0, zero external ML runtime dependencies) |
| **Weights license** | Non-commercial research use (`CC BY-NC-SA 4.0`, inherited from `MUSDB18-HQ`; zero audio retained in weights or repo) |

## Unified 4-Organ Architecture (`model.safetensors`)

Named after Stevens' psychoacoustic *sone* unit of human perceived loudness, **Sone** models the human auditory-to-executive pathway through four namespaced organs inside a single `.safetensors` file:

| Tier | `cochlea.*` (Separator) | `tympanum.*` (7-Head Sensor) | `cortex.*` (Beat Planner) | `cingulate.*` (Defect Monitor) | Tensors | Total Params | FP16 Size |
|---|---|---|---|---|---:|---:|---:|
| **`Sone-6.5M`** | `4,519,376` (`2048`-FFT, 42b, 6B) | `304,510` (`128`-mel, 4C+2×BiGRU-64) | `1,221,494` (`6L/128D`) | `475,071` (`4L/96D`, 11 heads) | `910` | **`6,520,451`** | `13.12 MB` |
| **`Sone-v2-Base`** | `11,605,860` (`4096`-FFT, 64b Dual-Axis, 6B) | `304,510` (`128`-mel, vinyl-equiv.) | `1,221,494` (`6L/128D`, `v3`) | `475,071` (`4L/96D`, 30 heads) | `1,520` | **`13,606,935`** | `27.36 MB` |
| **`Sone-Pro`** | `20,069,012` (`4096`-FFT, 64b Dual-Axis, 8B) | `1,473,406` (`160`-mel, 6C+3×BiGRU-128) | `3,604,726` (`8L/192D`, `v3-Pro`) | `1,225,247` (`6L/128D`, 30 heads) | `1,618` | **`26,372,391`** | `52.90 MB` |

## Training & Hardware Inventory

- **`Cochlea` (`v2-Base` & `Pro`)**: Trained on **2× parallel NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs** (`G4`, `sm_120`, `96 GB` GDDR7, CUDA `13.0`, PyTorch `2.14.1+cu130`, BF16 + `torch.compile`; `81.8 GB` peak VRAM at `bs=48` / `45.5 ex/s` for `v2-Base`, `70.5 GB` peak VRAM at `bs=32` / `37.3 ex/s` for `Pro`). Supervised on `150` `MUSDB18-HQ` multitracks + `1,101` Creative Commons `48 s` 4-stem tracks distilled in GPU memory from `HTDemucs-ft` (`MTG-Jamendo` + `Song Describer`) under vocal-prioritized multi-res STFT, `128`-mel perceptual, active-vocal cosine phase-coherence, and bidirectional anti-bleed losses.
- **`Tympanum` (`v1` & `v2/Pro`)**: Trained on NVIDIA Tesla `T4` (`16 GB`) and NVIDIA `L4` (`24 GB`) across `MUSDB18-HQ` isolated stems + `450` stem-distilled Creative Commons tracks under analytical vinyl-speed pitch/key/tuning/tempo equivariance (`alpha = 2^(delta_s / 12)`), paired at inference with a deterministic multi-octave spectral-flux comb grid.
- **`Cortex v3` & `Cingulate v3` (`Base` & `Pro`)**: Trained on NVIDIA `L4` (`24 GB`, BF16 FlashAttention SDPA) over `25,962` simulated transitions (`sone-sim`), `142` human-audited transitions (`owner.jsonl`), and `50%` on-GPU 7-mode counterfactual DJ defect synthesis (`20,000` steps per organ).
- **Single-Core CPU Evaluation Host**: `2`-vCPU Intel Xeon VM (Model 151, AVX2/FMA/F16C, `3.8 GiB` RAM, Linux `6.1.167`, `rustc 1.99.1 -C opt-level=3`, single thread).

## Evaluation

### 1. Cortical Planning (`Cortex`) & Counterfactual Defect Monitoring (`Cingulate`)

Verified end-to-end in pure single-threaded Rust (`cortex_cingulate_verify`) on strictly unseen `val.jsonl.gz` (`n = 5,707`) and human-audited `owner.jsonl` (`n = 142`, zero-shot in `v1/v2`, preference-aligned in `v3`):

| Organ | Variant | Unseen `val` Metric | `owner.jsonl` Alignment | FP32-vs-FP16 $\Delta_{\max}$ | 1-Thread CPU |
|---|---|---|---|---:|---:|
| **`Cortex`** | `v1` (`6L/128D`) | `95.28%` rec · `0.429` bars MAE | `82.18%` rec (zero-shot) | `0.0049` | `16.16 ms` |
| **`Cortex`** | `v2` (`6L/128D`, sharp-cut) | `96.77%` rec · `0.320` bars MAE | `86.14%` rec (zero-shot) | `0.0052` | `16.74 ms` |
| **`Cortex`** | `v3-Base` (`6L/128D`) | `96.89%` rec (`780/805`) · `0.178` bars (`0.0182` lane) | **`99.01%`** (`100/101`, `0.04` bars) | `0.0029` | `17.30 ms` |
| **`Cortex`** | **`v3-Pro` (`8L/192D`)** | **`98.63%`** rec (`794/805`) · **`0.165` bars** (**`0.0170` lane**) | **`99.01%`** (`100/101`, `0.05` bars) | **`0.0016`** | `38.03 ms` |
| **`Cingulate`** | `v2` (`4L/96D`, 11 heads) | `97.39%` good/bad acc | `89.36%` precision (zero-shot) | `0.0011` | `8.84 ms` |
| **`Cingulate`** | **`v3-Base` (`4L/96D`, 30 heads)** | **`97.28%`** acc (`5,552/5,707`) | **`100.0%` precision (`TP=91, FP=0`)** | `0.0053` | **`8.71 ms`** |
| **`Cingulate`** | **`v3-Pro` (`6L/128D`, 30 heads)** | **`97.00%`** acc (`50%` GPU CF) | **`98.95%` precision (`TP=94, FP=1`)** | `0.0028` | `16.66 ms` |

### 2. Boundary Separation (`Cochlea`) & Single-Core CPU Latency Budget

All `Cochlea` checkpoints enforce exact stereo mixture conservation (`||v + d + b + o - mix||_inf in [-151.4, -148.3] dB`) and `<= 2.39e-4` FP32-vs-FP16 max waveform deviation in `sone-infer`:

| Tier | MUSDB18-HQ Vocal / Mean SDR | Radio Mean SDR | `Cochlea` 1-vCPU RTF | `2 × 24 s` Boundary CPU | Total 1-Core CPU / Mix | Avg. 1-vCPU Load (180 s Song) |
|---|---|---|---:|---:|---:|---:|
| **`Sone-6.5M`** | `3.57 dB` / `2.79 dB` | `3.31 dB` | `0.13×` | `6.24 s` | **`8.6 s`** | **`4.8%`** |
| **`Sone-v2-Base`** | **`6.73 dB`** / **`4.90 dB`** | **`5.20 dB`** | `0.40×` | `19.10 s` | **`21.5 s`** | **`11.9%`** |
| **`Sone-Pro`** | **`7.84 dB`** / **`5.55 dB`** | **`6.84 dB`** | **`0.95×`** | `45.56 s` | **`48.8 s`** | **`27.1%`** |

## Intended Use

- Real-time automated radio stations, playlist transition engines, and browser-local WebAssembly DJ players requiring untouched master audio outside transitions and fractional-vCPU operating costs.
- Research on compact phase-coherent band-split separation, equivariant music structure analysis, and neuro-symbolic audio control.

## Limitations

- **Continuous-vocal outros**: When Track A sings up to its final beat with no instrumental outro or breath gap, `Cortex` selects a clean on-beat cut or short sneak rather than creating a double-vocal clash (`Cingulate` flags `cut_not_transition`).
- **Distillation teacher ceiling**: Student separation on Creative Commons tracks is bounded by `HTDemucs-ft` pseudo-labels; dense stereo vocal reverb tails can leave faint high-mid shimmer relative to `>100M` offline GPU ensembles.
- **Global BSSEval SDR on silent boundary stems**: Standard full-window SDR diverges toward `-inf dB` on instrumental windows with `-106 dBFS` ground-truth vocals even when residual vocal energy is inaudible (`< -36 dBFS`); evaluate DJ windows with MDX23 active-chunk median SDR alongside silent-stem RMS floors.
- **4/4 bar grammar**: `Cortex` plans within a 4-beat bar grammar (`±6%` key-locked stretch during vocal overlap) and does not synthesize turntable scratch routines or `3/4` meter transitions.
- **Single-creator preference labels**: `owner.jsonl` (`n = 142`) reflects one station creator's blind listening judgments rather than a multi-listener MUSHRA panel.

## Usage

```rust
use sone_infer::{Sone, WeightFile};

let bytes = std::fs::read("model.safetensors")?;
let sone = Sone::load(WeightFile::from_vec(bytes)?)?;
```
