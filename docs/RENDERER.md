# The renderer and how it was checked

`sone-render` renders a spec from the stems of both songs. It is the lab's Python renderer
(`reference/python/handmix.py`) moved to Rust stage by stage, with one difference: the lab stretched song B with
Rubber Band (GPL, cannot be shipped); `sone-dsp` has its own key-locked stretcher.

## Against the Python reference

The 13 accepted hand-built transitions (`data/versions/S7.json`), rendered on atlas by both, from the same stems.
`reference/python/verify_version.sh S7` runs it, `compare_render.py` compares. Numbers are the level of the signal
against the level of the difference, in dB (96 dB is the resolution of a CD).

| Spec | Deck A | Deck B before the stretch | Sum, with Python's stretched B put in | Level per bar, same | Level per bar, own stretcher | B moved by (Python / Rust) | Drums apart after that (Python / Rust) |
|---|---|---|---|---|---|---|---|
| h01 | 192 dB | 205 dB | 193 dB | 0.0 dB | up to 3.0 dB | −16.22 / −16.22 ms | 0.0 / 0.0 ms |
| h02 | 189 | 189 | 187 | 0.0 | 3.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| h03 | 191 | 206 | 193 | 0.0 | 1.8 | 0.0 / 0.0 | −0.73 / 0.0 |
| h04 | 202 | 194 | 204 | 0.0 | 2.1 | 2.9 / 2.9 | 0.0 / 0.0 |
| h05 | 189 | 187 | 190 | 0.0 | 3.4 | 0.73 / 0.0 | 0.0 / 0.0 |
| h06 | 185 | 204 | 185 | 0.0 | 2.5 | 2.85 / 2.85 | 0.0 / 0.0 |
| h07 | 280 | 191 | equal | 0.0 | 3.2 | 1.43 / 1.43 | 0.0 / 0.0 |
| h08 | 211 | 209 | 209 | 0.0 | 2.7 | 1.44 / 0.72 | 0.0 / 0.0 |
| h09 | 191 | 210 | 190 | 0.0 | 2.5 | 3.57 / 4.29 | −0.73 / 0.0 |
| h10 | 147 | equal | 149 | 0.0 | 1.2 | −0.73 / −0.73 | 0.0 / 0.0 |
| h11 | 196 | 205 | 198 | 0.0 | 2.8 | −1.44 / −3.6 | 0.0 / 0.0 |
| h12 | 201 | 189 | 199 | 0.0 | 3.1 | 1.45 / 1.45 | 0.0 / 0.0 |
| h13 | 213 | 193 | 278 | 0.0 | 3.7 | 0.0 / 0.0 | 0.0 / 0.0 |

What this says:

- Everything but the stretch is the same as the reference, far below anything audible.
- The stretched song B cannot be compared sample by sample: two stretchers give two waveforms. Per bar, the
  largest difference of a level is 1.2 to 3.7 dB, always in the band above 4 kHz where that band is quiet
  (−34 to −44 dB); there the Rust stretcher leaves more treble than Rubber Band. Whether that is heard, and as
  better or worse, only a listener can say. The planner's 13 transitions on the lab site are rendered with it.
- The automatic alignment of song B lands on the same value in 9 of 13 and within 2.2 ms in the rest; the drums
  are 0.0 ms apart afterwards in all 13.

## The stretcher

Two engines (`sone_dsp::stretch`): a phase vocoder that copies strong attacks from the input as they are (the
default), and a splicing one. Why the phase vocoder is the default is in the module's description, with the
measurements against Rubber Band behind it.

What the live engine needs of it is checked in `crates/sone-cli/src/adapters.rs`: at its own tempo it gives the song
back frame for frame, and through a glide from 0.985 or 1.02 to 1 every click of a click track comes out within
1.5 ms of where the ratios put it.

## Memory

See `MEMORY.md`. This renderer works on whole buffers and needs 243–394 MB per transition in WebAssembly. The
native binary uses it; a Worker needs the block-wise renderer (`sone-stream`).
