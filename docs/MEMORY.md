# Memory and Single-Core CPU Profile

Cloudflare Workers cap each isolate at `128 MB` of linear WebAssembly memory, and streaming radio servers budget a small fraction of one vCPU core per station. Every figure below specifies how it was measured on single-threaded x86_64 (`rustc 1.99.1 -C opt-level=3`, Intel Xeon AVX2/FMA/F16C) or inside `sone-wasm` on Node 22.

## Why Boundary-Only Separation Fits Sub-Core Budgets

Separating two full 3-minute songs would waste ~85% of compute on audio that never overlaps. **Sone** runs `Cochlea` strictly on the `2 × 24 s` transition boundary (`48 s` total per song) and splices untouched master stereo everywhere else, reducing separation compute by **~7.5×**.

## End-to-End 1-Core CPU Budget per Transition

Measured single-threaded on our Intel Xeon cloud worker across all three unified `model.safetensors` tiers (`2 × 24 s` stereo separation window + `K = 6` cortical candidate reranking + DSP render & master splice):

| Stage | Scope per transition | `Sone-6.5M` (`13.1 MB`) | `Sone-v2-Base` (`27.3 MB`) | `Sone-Pro` (`52.9 MB`) |
|---|---|---:|---:|---:|
| **`Tympanum` + Comb-Grid** | `2 ×` track structure (`50 Hz`) | `0.35 s` (`0.0024×` RT) | `0.35 s` (`0.0024×` RT) | `1.18 s` (`0.0085×` RT) |
| **`Cochlea`** | `2 × 24 s` boundary stems | `6.24 s` (`0.13×` RT) | `20.00 s` (`0.42×` RT) | `49.78 s` (`1.04×` RT) |
| **`Cortex`** | `209` beat tokens | `16.2 ms` | `17.3 ms` | `38.0 ms` |
| **`Cingulate`** (`K = 6`) | `6 × 209` candidate tokens | `53.0 ms` (`8.8 ms` / cand) | `52.3 ms` (`8.7 ms` / cand) | `100.0 ms` (`16.7 ms` / cand) |
| **DSP Render + Master Splice** | Key-locked stretch + pocket + echo | `1.95 s` | `1.95 s` | `1.95 s` |
| **Total 1-core CPU / mix** | Full raw-stereo-to-master transition | **`8.6 s`** | **`22.4 s`** | **`53.0 s`** |
| **Amortized load (180 s song)** | Fraction of 1 cloud vCPU core | **`4.8%`** | **`12.4%`** | **`29.5%`** |

All three tiers execute comfortably faster than real-time over a standard 3-minute track on a single CPU thread.

## WebAssembly & Worker Memory Footprint

| Component | Peak RAM | CPU | Environment | Fits 128 MB Worker |
|---|---:|---:|---|---|
| `sone-wasm` module + reverb IRs | `3.3 MB` | — | `wasm32` | yes |
| `Sone-6.5M` weights (`sone-infer/1`, FP16) | `13.04 MB` (zero-copy) | — | native / `wasm32` | yes |
| `Sone-v2-Base` weights (`sone-infer/1`, FP16) | `27.34 MB` (zero-copy) | — | native / `wasm32` | yes |
| `Sone-Pro` weights (`sone-infer/1`, FP16) | `52.90 MB` (zero-copy) | — | native / `wasm32` | yes (dedicated isolate) |
| Two `BeatTable` structs in memory | `+0.3 MB` (`58 KB` each) | — | `wasm32` | yes |
| `Cortex` + `Cingulate` planning (`K = 6`) | `< 2 MB` scratch | `70–138 ms` | native / `wasm32` | yes |
| Live station playback (`sone-live`) | `~5 MB` (`176 KB/s` PCM16 clip) | `< 0.1%` core | native / `wasm32` | yes |
| Whole-buffer `sone-render` (with meter curves) | `243–394 MB` | `0.4–1.8 s` | `wasm32` | native / container only |

Weights remain in 16-bit IEEE-754 half-precision inside the shared `WeightFile` buffer; `sone-infer` widens one tile/layer at a time into reusable FP32 scratch workspaces (`matrixmultiply::sgemm`), avoiding 2× weight duplication in RAM.

## Recommended Worker Split at Scale

`sone-live` supports external asynchronous rendering via `render_outside` (`take_requests` / `deliver`):

1. **Station Durable Object (`< 25 MB`)**: Runs `sone-live`, `Cortex`, and `Cingulate`, streaming untouched masters and emitting real-time events.
2. **Boundary Analysis & Render Worker**: Runs `Tympanum` + `Cochlea` on the `24 s` boundary windows up to `30 s` ahead of `u_0`, returning the rendered transition clip well before its first sample is due.
