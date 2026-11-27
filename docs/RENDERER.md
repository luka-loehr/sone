# Deterministic Renderer Verification (`sone-render`)

`sone-render` renders a transition `Spec` from four stereo stems of Deck A and Deck B in pure Rust (`wasm32`-safe). It ports the Python reference renderer (`reference/python/handmix.py`) stage by stage while replacing GPL-licensed Rubber Band time-stretching with `sone-dsp`'s permissive Apache-2.0 transient-preserving phase vocoder (`sone_dsp::Stretcher`).

## Numerical Parity Against the Reference Renderer

Tested across all 13 reference transition specs in `data/versions/S7.json` (`reference/python/verify_version.sh S7` and `reference/python/compare_render.py`). Signal-to-difference ratios below are reported in dB (16-bit CD dynamic range is 96 dB):

| Spec | Deck A SNR | Deck B (Pre-Stretch) SNR | Sum (with Reference Stretched B) | Bar Level Diff (Same Stretch) | Bar Level Diff (Native Stretcher) | B Offset (Ref / Rust) | Post-Lock Drum Lag (Ref / Rust) |
|---|---:|---:|---:|---:|---:|---:|---:|
| h01 | 192 dB | 205 dB | 193 dB | 0.0 dB | <= 3.0 dB | -16.22 / -16.22 ms | 0.0 / 0.0 ms |
| h02 | 189 dB | 189 dB | 187 dB | 0.0 dB | <= 3.0 dB | 0.00 / 0.00 ms | 0.0 / 0.0 ms |
| h03 | 191 dB | 206 dB | 193 dB | 0.0 dB | <= 1.8 dB | 0.00 / 0.00 ms | -0.73 / 0.0 ms |
| h04 | 202 dB | 194 dB | 204 dB | 0.0 dB | <= 2.1 dB | 2.90 / 2.90 ms | 0.0 / 0.0 ms |
| h05 | 189 dB | 187 dB | 190 dB | 0.0 dB | <= 3.4 dB | 0.73 / 0.00 ms | 0.0 / 0.0 ms |
| h06 | 185 dB | 204 dB | 185 dB | 0.0 dB | <= 2.5 dB | 2.85 / 2.85 ms | 0.0 / 0.0 ms |
| h07 | 280 dB | 191 dB | exact | 0.0 dB | <= 3.2 dB | 1.43 / 1.43 ms | 0.0 / 0.0 ms |
| h08 | 211 dB | 209 dB | 209 dB | 0.0 dB | <= 2.7 dB | 1.44 / 0.72 ms | 0.0 / 0.0 ms |
| h09 | 191 dB | 210 dB | 190 dB | 0.0 dB | <= 2.5 dB | 3.57 / 4.29 ms | -0.73 / 0.0 ms |
| h10 | 147 dB | exact | 149 dB | 0.0 dB | <= 1.2 dB | -0.73 / -0.73 ms | 0.0 / 0.0 ms |
| h11 | 196 dB | 205 dB | 198 dB | 0.0 dB | <= 2.8 dB | -1.44 / -3.60 ms | 0.0 / 0.0 ms |
| h12 | 201 dB | 189 dB | 199 dB | 0.0 dB | <= 3.1 dB | 1.45 / 1.45 ms | 0.0 / 0.0 ms |
| h13 | 213 dB | 193 dB | 278 dB | 0.0 dB | <= 3.7 dB | 0.00 / 0.00 ms | 0.0 / 0.0 ms |

Key takeaways:

- Every DSP stage outside time-stretching (Linkwitz-Riley filters, stem gain lanes, quarter-note echo, convolution reverb, BS.1770-4 LUFS loudness normalization, and look-ahead peak limiting) matches the double-precision reference to `147` to `280 dB` SNR, well below floating-point audibility.
- With `sone-dsp`'s native transient-preserving stretcher, per-bar band differences stay within `1.2` to `3.7 dB` and occur only in quiet high-frequency passages (`-34` to `-44 dBFS` above `4 kHz`) where transient-copying preserves crisp hi-hat attacks.
- Closed-form drum onset alignment lands identically in 9 of 13 specs and within `2.2 ms` on all remaining pairs, achieving **`0.0 ms` post-alignment drum lag across all 13 transitions**.

## Time-Stretcher Verification (`sone_dsp::stretch`)

`sone_dsp::stretch` provides both a transient-preserving phase vocoder (default) and a time-domain splicing stretcher:

- At unity ratio (`1.0x`), audio passes bit-for-bit frame-accurately.
- Across continuous post-transition tempo glides from `0.985x` or `1.020x` back to `1.000x` (`crates/sone-cli/src/adapters.rs`), every impulse of a synthetic click track lands within `<= 1.5 ms` of the analytical integral of the ratio curve.

## Memory Footprint

See [MEMORY.md](MEMORY.md). Whole-buffer rendering with full per-bar diagnostic meters uses `243` to `394 MB` of linear memory; constrained 128 MB edge Workers pair `sone-live` (`< 25 MB` per station) with windowed boundary rendering via `sone-wasm` (`needs::needs`).
