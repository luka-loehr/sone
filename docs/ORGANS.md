# Anatomical Neural Organs & `sone-infer/1` Wire Specification

**Sone** decomposes automatic DJ transition intelligence into four specialized neural networks named after their biological counterparts in human hearing and executive control. All four organs pack into one zero-copy `.safetensors` file (`model.safetensors`) under wire format `sone-infer/1`.

---

## 1. `Cochlea` (`cochlea.*`) — Phase-Coherent Tonotopic Stem Separator

Modeled after the frequency-place tonotopic map of the inner ear's basilar membrane, **Cochlea** separates `44.1 kHz` stereo boundary windows into four stems: `vocals`, `drums`, `bass`, and `other`.

- **High-resolution vocal-harmonic STFT (`v2-Base` & `Pro`)**: `4096`-point Hann STFT (`hop = 1024`, `10.77 Hz` bin resolution) partitioned into `64` gap-free psychoacoustic bands (`studio_bands`): `21.5 Hz` sub-bass bins (`0–172 Hz`), `43.1 Hz` vocal-fundamental and first-formant bands (`172–1,034 Hz`), `129 Hz` harmonic bands (`1.03–4.13 kHz`), sibilance bands (`4.13–11.0 kHz`), and air bands (`11.0–22.05 kHz`).
- **Dual-Axis Time+Band Transformer**: Interleaves dilated depthwise temporal convolution (`k = 5`, dilation `2^(i mod 6)`), temporal multi-head self-attention (`time_attn = true`, modeling phrase-long vocal vibrato and sustain), and cross-band multi-head self-attention (locking fundamental pitch to harmonic overtones).
- **Bounded 4-stem complex masking & Wiener conservation**: Predicts bounded complex masks `M_m = 2 * tanh(raw_m / 2)` for all four stems, projects complex spectrograms onto exact stereo mixture consistency weighted by frame-bin power `|S_m|^2`, and reconstructs `other = mix - (vocals + drums + bass)` in the time domain (`||sum_m s_m - mix||_inf < -148 dB`).
- **Zero-scramble training objective**: Combines vocal-weighted (`[2.2, 1.0, 1.1, 1.3]`) complex STFT L1, waveform L1, 5-scale multi-resolution STFT loss (`256..4096`), 128-mel perceptual vocal/accompaniment loss, active-vocal cosine phase-coherence loss, and bidirectional DJ anti-bleed penalties, distilled from `HTDemucs-ft` across Creative Commons corpora plus `MUSDB18-HQ`.

---

## 2. `Tympanum` (`tympanum.*`) — Vinyl-Equivariant 7-Head Structural Sensor

Modeled after the eardrum's transduction of acoustic pressure pulses, **Tympanum** maps mono `44.1 kHz` audio to a `50 Hz` HTK log-mel representation (`N = 2048, hop = 882`) through a dilated depthwise TCN + multi-layer BiGRU (`Base`: `128` mels, `4` conv, `2×BiGRU-64`, `304,510` params; `Pro`: `160` mels `25–16,000 Hz`, `6` conv `1..32`, `3×BiGRU-128`, `1,473,406` params).

- **Seven simultaneous outputs**:
  1. `beat` (`50 Hz` frame probability)
  2. `downbeat` (`50 Hz` bar-one probability)
  3. `voice` (`50 Hz` active singing activity)
  4. `line_end` (`50 Hz` inter-phrase singer breath boundary for clean vocal exits & echo throws)
  5. `phrase_start` (`50 Hz` 8-bar structural boundary)
  6. `key_logits` (`24` major/minor pitch classes -> Camelot wheel)
  7. `tuning_cents` (`(-50, +50)` sub-semitone concert-pitch offset)
- **Exact vinyl-speed equivariant augmentation**: Resampling audio by `alpha = 2^(delta_s / 12)` (`delta_s in [-3.5, +3.5]` semitones) transforms all five temporal targets (`t' = t / alpha`), musical key (`(k + round(delta_s)) mod 12`), and tuning cents analytically with zero phase-vocoder transient smearing or timestamp drift.
- **Neuro-symbolic comb-grid lock**: Fuses `50 Hz` neural beat/downbeat activations with physical spectral onset flux across multi-octave harmonic candidates (`{1/2, 2/3, 3/4, 1, 4/3, 3/2, 2}`) to eliminate half/double-tempo octave flips without Dynamic Bayesian Networks.

---

## 3. `Cortex` (`cortex.*`) — Executive Beat-Synchronous Planner

**Cortex** operates on a standardized `209`-token sequence (`[global; A_80; B_128]`, `24` features per beat) using a Pre-RMSNorm Beat Transformer (`Base`: `6L / 128D`, `1,221,494` params; `Pro`: `8L / 192D`, `3,604,726` params).

- **Outputs**: Transition recipe (`VoiceOverBeat`, `Sneak`, `OutroUnderIntro`, `Gap`), normalized bar offsets `(u_start, u_swap)` in `[0, 16]` bars (`0.165` bar held-out MAE in `Pro`), and `80 × 8` beat-resolution stem gain curves across both decks (`0.0170` MAE).
- **Sharp-cut & pocket supervision**: `7×` edge-weighted step supervision prevents conditional-mean fader smearing across `u = 0`, accompaniment-floor regularization eliminates pre-entry loudness holes (`+10.55 dB` blend floor improvement), and vocal-onset lock (`u_handoff`) aligns `VoiceOverBeat` entries directly to sustained singing over a `-2.5 dB` chord pocket.

---

## 4. `Cingulate` (`cingulate.*`) — 30-Defect Counterfactual Conflict Monitor

Inspired by conflict monitoring in the anterior cingulate cortex, **Cingulate** (`Base`: `4L / 96D`, `475,071` params; `Pro`: `6L / 128D`, `1,225,247` params) ingests the `209`-token beat sequence concatenated with Deck A/B candidate fader trajectories and predicts overall acceptability `q in [0, 1]` plus `30` independent DJ defect probabilities (`sone_core::Issue`).

- **On-GPU counterfactual defect synthesis**: `50%` of training batches synthesize seven realistic failure modes on the fly (premature accompaniment holes, mid-word vocal chops, double vocals, chord mush, double sub-bass, unblended hard cuts, and vocal bleed/beat drift), achieving `97.00%`–`97.28%` accuracy on strictly unseen `val.jsonl.gz` (`n = 5,707`) and `98.9%`–`100.0%` precision (`FP = 0` on `Base`, `FP = 1` on `Pro`) on human-audited mixes.
- **Best-of-`K` reranking**: Scores `K = 6` candidate plans around `Cortex`'s top recipes and bar offsets in `52–100 ms` total on 1 CPU thread.

---

## 5. `sone-infer/1` `.safetensors` Wire Format

Every unified `model.safetensors` bundle carries standard `.safetensors` JSON `__metadata__`:

```json
{
  "format": "sone-infer/1",
  "model": "sone",
  "config.cochlea": "{\"n_fft\":4096,\"hop\":1024,\"bands\":[[0,2],...],\"d\":128,\"blocks\":8,\"heads\":8,\"m\":256,\"time_attn\":true,\"four_stems\":true,\"mask_bound\":2.0}",
  "config.tympanum": "{\"n_mels\":160,\"c\":160,\"n_conv\":6,\"n_gru\":3,\"gru_h\":128,\"f_min\":25.0,\"f_max\":16000.0}",
  "config.cortex": "{\"d\":192,\"layers\":8,\"heads\":6,\"ff\":768}",
  "config.cingulate": "{\"d\":128,\"layers\":6,\"heads\":4,\"ff\":512,\"n_issues\":30}"
}
```

`sone_infer::Sone::load` validates header bounds, parses organ configs (`#[serde(default)]` backward-compatible across `v1`, `v2-Base`, and `Pro`), namespaces `cochlea.*`, `tympanum.*`, `cortex.*`, and `cingulate.*` over one shared `Arc<[u8]>` buffer with zero heap copies, and verifies bijective tensor consumption so neither missing nor unreferenced tensors are allowed.
