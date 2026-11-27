# Architecture

## Overview

A transition in **Sone** is not generated through a full-track neural audio codec; it is perceived, planned, audited, and mixed exclusively across a ~24 s boundary window `[u_0, u_1]`. Four compact anatomical neural organs packaged inside a single `model.safetensors` checkpoint (`sone-infer/1`) mirror the human auditory-to-executive pathway, while deterministic Rust signal processing renders the blend and splices untouched studio master stereo bit-for-bit outside `[u_0, u_1]`.

```text
Deck A master ─┐
               ├─► boundary window ─► Cochlea  (4096-FFT, 64-band 4-stem Wiener separator)
Deck B master ─┘     (~24 s each)  ─► Tympanum (50 Hz 7-head sensor + comb-grid beat tracker)
                                          │
                                          ▼
                                      Cortex   (Beat Transformer: recipe, bars, 8 stem lanes)
                                          │
                                          ▼
                                      Cingulate (30-defect counterfactual monitor, Best-of-K)
                                          │
                                          ▼
                    untouched A ──► cosine seam ──► DSP blend ──► cosine seam + glide ──► untouched B
```

## Workspace Crates

| Crate | Purpose | `wasm32` |
|---|---|---|
| `sone-core` | Shared transition `Spec`, `BeatTable`, `Event`, training `Example`, and 30-class `Issue` enum | yes |
| `sone-dsp` | Linkwitz-Riley filters, stem lanes, tempo-synced echo, reverb, limiter, BS.1770-4 LUFS, key-locked stretch | yes |
| `sone-beats` | Neuro-symbolic multi-octave comb-grid and neural beat/downbeat tracking in pure Rust | yes |
| `sone-analysis` | Streaming stem & metrical feature extractor (`SongAnalyzer` → 24-feature `BeatTable`) | yes |
| `sone-plan` | Metrical alignment, Camelot harmonic rules, vocal-onset lock, and candidate spec generation | yes |
| `sone-render` | Phase-locked deterministic transition renderer (`StemSource`) with per-bar telemetry | yes |
| `sone-sim` | Synthetic transition corpus & structured DJ defect simulator | no |
| `sone-infer` | Dependency-free FP16/FP32 `.safetensors` runtime (`Sone::load`, `sone-infer/1`, zero-copy) | yes |
| `sone-models` | Backend-generic Burn definitions of `Cochlea`, `Tympanum`, `Cortex`, and `Cingulate` | no |
| `sone-train` | Native Rust training, checkpoint export, and numerical parity CLI (`sone-train`) | no |
| `sone-live` | Pull-driven streaming radio/player station engine within a 30 s look-ahead buffer | yes |
| `sone-wasm` | Compact WebAssembly C ABI (`sone_alloc`, `sone_song`, `sone_plan`, `sone_stems`, `sone_render`) | target |
| `sone-cli` | Unified `sone` command-line binary (`analyse`, `plan`, `render`, `mix`, `live`, `simulate`) | native |

## The Four Anatomical Organs (`model.safetensors`)

All three tiers (**`Sone-6.5M`**, **`Sone-v2-Base`**, and **`Sone-Pro`**) serialize into one unified IEEE-754 FP16 `.safetensors` file under schema `sone-infer/1`:

| Organ | Namespace | Role | `Sone-6.5M` | `Sone-v2-Base` | `Sone-Pro` |
|---|---|---|---:|---:|---:|
| **Cochlea** | `cochlea.*` | Tonotopic stem separator (`vocals`, `drums`, `bass`, `other`) | `4,519,376` | `11,605,860` | `20,069,012` |
| **Tympanum** | `tympanum.*` | 50 Hz 7-head beat, downbeat, vocal, breath, 8-bar phrase, key & tuning sensor | `304,510` | `304,510` | `1,473,406` |
| **Cortex** | `cortex.*` | Beat-synchronous Transformer planner (recipe, bar offsets, 8 stem gain lanes) | `1,221,494` | `1,221,494` | `3,604,726` |
| **Cingulate** | `cingulate.*` | Conflict & error monitor scoring acceptability + up to 30 DJ defect modes | `475,071` | `475,071` | `1,225,247` |
| **Unified** | `model.safetensors` | Zero-copy single-file checkpoint (`Sone::load`) | **`6,520,451`** | **`13,606,935`** | **`26,372,391`** |

See [`ORGANS.md`](ORGANS.md) for mathematical definitions, loss functions, and wire-format details.

## Bar Ruler (`u`) and Transition Recipes

Every event in a transition is addressed on a continuous four-beat bar ruler `u`, where `u = 0` is **the one**—the downbeat on which Deck B assumes rhythmic and harmonic ownership:

| Deck B opening | Tempo lock | Harmonic fit | Recipe |
|---|---|---|---|
| Singing before full drums | `1:1`, `1:2`, or `2:1` | Compatible Camelot | **`VoiceOverBeat`**: B's vocal enters at `u_handoff` over A's rhythm pocket; A swaps out hard at `u = 0` |
| Instrumental beat before vocal | `1:1` | Compatible Camelot | **`Sneak`**: B's hats (`u = -8`) and kick (`u = -4`) sneak under A; A fades to 50% and cuts cleanly on `u = 0` |
| Melodic beatless intro | `1:1`, `1:2`, or `2:1` | Compatible Camelot | **`OutroUnderIntro`**: A's outro beat carries B's instrumental intro, filtering out onto `u = 0` |
| Beat before vocal | `1:1` | Clashing keys | **`Sneak` (unpitched overlap)**: Only B's unpitched drums meet A prior to `u = 0`; bass/melody swap strictly on `u = 0` |

## Streaming Station Engine (`sone-live`)

`sone-live` has no internal threads, wall-clock timers, or filesystem calls; the host pulls PCM blocks on demand and receives structured JSON events (`sone_core::Event`):

- **Queueing**: `When::AtEnd` plans the natural phrase-aligned outro transition once inside the 30 s `lookahead_s` window; `When::Now` handles listener skips with 8-, 4-, or 2-bar blends.
- **Untouched master splicing**: Before the transition clip, Deck A leaves the engine sample-for-sample identical to the studio master; a 1024-sample (23.2 ms) raised-cosine seam enters the rendered blend; after the blend, Deck B glides smoothly from `r_blend` back to `1.0x` native tempo over 8–16 bars and thereafter streams bit-for-bit untouched.
- **DJ voiceover ("hitting the post")**: When an announcer or TTS clip is queued, the planner locates a vocal-free bed ending `0.3 s` before Deck B's vocal post or `u = 0`, ducking accompaniment by `-6 dB`.

| Event | Emitted | Payload |
|---|---|---|
| `plan` | When transition is locked inside look-ahead | `starts_at`, `one_at`, `ends_at`, recipe, confidence, tempos, keys, DJ voice slot |
| `plan_dropped` | If an external render misses its deadline | Drop reason |
| `transition_started` | First fader movement | — |
| `the_one` | Downbeat `u = 0` takeover | — |
| `transition_ended` | Final fader settles | — |
| `stats` | Once per second | Drum beat lag (ms), concurrent voices/melodies/basses, level dip, CPU fraction |
