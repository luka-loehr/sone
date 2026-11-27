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

# Model Card: Sone (`Sone-6.5M` · `Sone-v2-Base` · `Sone-Pro` · `Sone-Ultra`)

## Summary

| | |
|---|---|
| **Family** | `Sone-6.5M` (`6,520,451` params, `13.12 MB`), `Sone-v2-Base` (`13,606,935` params, `27.36 MB`), `Sone-Pro` (`26,372,287` params, `52.90 MB`), `Sone-Ultra` (`35,481,055` params, `71.12 MB`) |
| **Wire format** | Unified IEEE-754 FP16 `model.safetensors` (`sone-infer/1`, zero-copy loaded via `sone_infer::Sone::load`, plus optional standalone organ slices) |
| **Task** | Raw `44.1 kHz` stereo song pair → beat-locked (`<= 0.7 ms` rendered drift, `<= 2.2 ms` sync gate), stem-carved, vocal-onset-aligned DJ transition spliced bit-for-bit into untouched studio masters outside `[u_0, u_1]` |
| **Author** | Luka Löhr |
| **Engine code** | Pure Rust + WebAssembly (`sone-infer`, `sone-dsp`, `sone-render`, `sone-live`, Apache-2.0, zero external ML runtime dependencies) |
| **Weights license** | Non-commercial research use (`CC BY-NC-SA 4.0`, inherited from `MUSDB18-HQ`; zero audio retained in weights or repo) |

## Unified 4-Organ Architecture (`model.safetensors`)

Named after Stevens' psychoacoustic *sone* unit of human perceived loudness, **Sone** models the human auditory-to-executive pathway through four namespaced organs inside a single `.safetensors` file:

| Tier | `cochlea.*` (Separator) | `tympanum.*` (7-Head Sensor) | `cortex.*` (Beat Planner) | `cingulate.*` (Defect Monitor) | Total Params | FP16 Size |
|---|---|---|---|---|---:|---:|
| **`Sone-6.5M`** | `4,519,376` (`2048`-FFT, 42b, 6B) | `304,510` (`128`-mel, 4C+2×BiGRU-64) | `1,221,494` (`6L/128D`) | `475,071` (`4L/96D`, 11 heads) | **`6,520,451`** | `13.12 MB` |
| **`Sone-v2-Base`** | `11,605,860` (`4096`-FFT, 64b Dual-Axis, 6B) | `304,510` (`128`-mel, vinyl-equiv.) | `1,221,494` (`6L/128D`, `v4-Base`) | `475,071` (`4L/96D`, 30 heads) | **`13,606,935`** | `27.36 MB` |
| **`Sone-Pro`** | `20,069,012` (`4096`-FFT, 64b Dual-Axis, 8B, `7.98 dB` voc) | `1,473,406` (`160`-mel, 6C+3×BiGRU-128) | `3,604,622` (`8L/192D`, `v4-Pro`) | `1,225,247` (`6L/128D`, 30 heads) | **`26,372,287`** | `52.90 MB` |
| **`Sone-Ultra`** | `29,177,780` (`4096`-FFT, 64b Dual-Axis, 10B Net2Net) | `1,473,406` (`160`-mel, 6C+3×BiGRU-128) | `3,604,622` (`8L/192D`, `v4-Pro`) | `1,225,247` (`6L/128D`, 30 heads) | **`35,481,055`** | `71.12 MB` |

## Training & Hardware Inventory

- **`Cochlea` (`v2-Base`, `Pro v2`, & `Ultra`)**: Trained across **2× NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs** (`G4`, `sm_120`, `96 GB` GDDR7, `44.8 ex/s` `v2-Base`, `36.4 ex/s` `Pro`) and **2× NVIDIA `L4` GPUs** (`sm_89`, `24 GB`, BF16 FlashAttention). Supervised on `150` `MUSDB18-HQ` multitracks + `1,101` `HTDemucs-ft` Creative Commons `48 s` 4-stem tracks + `372` two-stage **`BS-RoFormer-1297` (`12.98 dB` vocal SDR) + `HTDemucs-ft`** `48 s` 4-stem tracks under vocal-prioritized multi-res STFT, `128`-mel perceptual, active-vocal cosine phase-coherence, direct time-domain SDR, and `8×` silent-vocal pre-onset suppression losses.
- **`Tympanum` (`Base` & `Pro`)**: Trained on NVIDIA `T4` (`16 GB`) and `L4` (`24 GB`) across `MUSDB18-HQ` isolated stems + `450` stem-distilled Creative Commons tracks under exact vinyl-speed pitch/key/tuning/tempo equivariance (`alpha = 2^(delta_s / 12)`), paired at inference with a multi-octave spectral-flux comb grid and closed-form Newton drum-slope alignment (`<= 2.2 ms` beat drift).
- **`Cortex v4` & `Cingulate v4` (`Base` & `Pro`)**: Trained on NVIDIA `L4` and `T4` (hardware Tensor Cores) over `26,904` simulated transitions (`sone-sim`), `5,932` strictly unseen validation transitions, `152` human-audited transitions (`owner.jsonl`), and `50%` vectorized 12-mode GPU counterfactual DJ defect synthesis (`20,000` steps per organ).
- **Single-Core CPU Evaluation Host**: `2`-vCPU Intel Xeon VM (Model 151, AVX2/FMA/F16C, `3.8 GiB` RAM, `rustc 1.99.1 -C opt-level=3`, single thread).

## Intended Use

- Real-time automated radio stations, playlist transition engines, and browser-local WebAssembly DJ players requiring untouched master audio outside transitions and fractional-vCPU operating costs.
- Research on compact phase-coherent band-split separation, equivariant music structure analysis, and neuro-symbolic audio control.

## Limitations

- **Continuous-vocal outros**: When Track A sings up to its final beat with no instrumental outro or breath gap, `Cortex` selects a clean on-beat cut or short sneak rather than creating a double-vocal clash (`Cingulate` flags `cut_not_transition`).
- **Single-pass CPU window vs. offline GPU ensembles**: Two-stage `BS-RoFormer-1297` + `HTDemucs-ft` distillation and 10-block Net2Net `Cochlea-Ultra` substantially sharpen vocal isolation, yet heavily drenched stereo reverb tails remain bounded by single-pass `6 s` chunked capacity compared with `>100M` full-song GPU ensembles.
- **Global BSSEval SDR on silent boundary stems**: Standard full-window SDR diverges toward `-inf dB` on instrumental windows with `-108 dBFS` ground-truth vocals even when residual vocal energy is `< -77 dBFS`; evaluate DJ windows with MDX23 active-chunk median SDR alongside silent-stem RMS floors.
- **4/4 bar grammar**: `Cortex` plans within a 4-beat bar grammar (`±6.8%` key-locked stretch) and does not synthesize turntable scratch routines or `3/4` meter transitions.
- **Single-creator preference labels**: `owner.jsonl` (`n = 152`) reflects one station creator's blind listening judgments rather than a multi-listener MUSHRA panel.

## Usage

```rust
use sone_infer::{Sone, WeightFile};

let bytes = std::fs::read("model.safetensors")?;
let sone = Sone::load(WeightFile::from_vec(bytes)?)?;
```
