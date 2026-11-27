# Memory and Single-Core CPU Profile

Cloudflare Workers cap each isolate at `128 MB` of linear WebAssembly memory, and streaming radio servers budget a fraction of one vCPU core per station. Every figure below was measured single-threaded on x86_64 (`rustc 1.99.1 -C opt-level=3`, Intel Xeon Model 151 with AVX2/FMA/F16C, 3.8 GiB RAM) or inside `sone-wasm` on Node 22.

## Why Boundary-Only Separation Fits Sub-Core Budgets

Separating two full 3-minute songs would spend ~85% of compute on audio that never overlaps. **Sone** runs `Cochlea` strictly on the `2 x 24 s` transition boundary (`48 s` total per song) and splices untouched master stereo everywhere else, reducing separation compute by **~7.5x**.

## End-to-End 1-Core CPU Budget per Transition

Measured single-threaded across all four unified `model.safetensors` tiers (`2 x 24 s` stereo separation window + `K = 6` cortical candidate reranking + DSP render, Newton drum-slope lock, and master splice):

| Stage | Scope per transition | `Sone-6.5M` (13.12 MB) | `Sone-v2-Base` (27.36 MB) | `Sone-Pro` (52.90 MB) | `Sone-Ultra` (71.12 MB) |
|---|---|---:|---:|---:|---:|
| **`Tympanum` + Comb-Grid** | `2 x` track structure (50 Hz) | 0.35 s | 0.35 s | 1.18 s | 1.18 s |
| **`Cochlea`** | `2 x 24 s` boundary stems | 6.24 s (0.13x) | 19.10 s (0.40x) | 46.19 s (0.96x) | 62.28 s (1.30x) |
| **`Cortex`** | 209 beat tokens | 16.7 ms | 17.3 ms | 38.0 ms | 38.0 ms |
| **`Cingulate`** (`K = 6`) | `6 x 209` candidate tokens | 53.0 ms | 52.3 ms | 100.0 ms | 100.0 ms |
| **DSP + Newton Lock + Splice** | Stretch + slope lock + cosine seam | 1.95 s | 1.95 s | 1.95 s | 1.95 s |
| **Total 1-core CPU / mix** | Full raw-stereo-to-master mix | **8.6 s** | **21.5 s** | **49.5 s** | **65.5 s** |
| **Amortized load (180 s song)** | Fraction of 1 cloud vCPU core | **4.8%** | **11.9%** | **27.5%** | **36.4%** |

All four tiers run well below one CPU core on average over a standard 3-minute song (`Sone-Ultra` finishes in `< 17 s` when parallelized across 4 CPU cores).

## WebAssembly and Worker Memory Footprint

| Component | Peak RAM | CPU | Environment | Fits 128 MB Worker |
|---|---:|---:|---|---|
| `sone-wasm` module + reverb IRs | 3.3 MB | – | `wasm32` | yes |
| `Sone-6.5M` weights (`sone-infer/1`, FP16) | 13.12 MB (zero-copy) | – | native / `wasm32` | yes |
| `Sone-v2-Base` weights (`sone-infer/1`, FP16) | 27.36 MB (zero-copy) | – | native / `wasm32` | yes |
| `Sone-Pro` weights (`sone-infer/1`, FP16) | 52.90 MB (zero-copy) | – | native / `wasm32` | yes (dedicated isolate) |
| `Sone-Ultra` weights (`sone-infer/1`, FP16) | 71.12 MB (zero-copy) | – | native / `wasm32` | native / container |
| Two `BeatTable` structs in memory | +0.3 MB (58 KB each) | – | `wasm32` | yes |
| `Cortex` + `Cingulate` planning (`K = 6`) | `< 2 MB` scratch | 70 to 138 ms | native / `wasm32` | yes |
| Live station playback (`sone-live`) | ~5 MB (176 KB/s PCM16 clip) | `< 0.1%` core | native / `wasm32` | yes |
| Whole-buffer `sone-render` (with meter curves) | 243 to 394 MB | 0.4 to 1.8 s | `wasm32` | native / container only |

Weights stay in 16-bit IEEE-754 half-precision inside the shared `WeightFile` buffer. `sone-infer` widens one tile or layer at a time into reusable FP32 scratch workspaces (`matrixmultiply::sgemm`), avoiding 2x weight duplication in RAM.

## Recommended Worker Split at Scale

`sone-live` supports external asynchronous rendering via `render_outside` (`take_requests` / `deliver`):

1. **Station Durable Object (`< 25 MB`)**: Runs `sone-live`, `Cortex`, and `Cingulate`, streaming untouched masters and emitting real-time events.
2. **Boundary Analysis and Render Worker**: Runs `Tympanum` + `Cochlea` on the `24 s` boundary windows up to `30 s` ahead of `u = 0`, returning the rendered transition clip before its first sample is due.
