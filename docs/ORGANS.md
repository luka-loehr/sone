# Anatomical Neural Organs and `sone-infer/1` Wire Specification

**Sone** splits automatic DJ transition intelligence into four specialized neural networks named after their counterparts in human hearing and executive control. All four organs pack into one zero-copy `.safetensors` file (`model.safetensors`) under wire format `sone-infer/1`.

## 1. `Cochlea` (`cochlea.*`): Phase-Coherent Tonotopic Stem Separator

Modeled after the frequency-place tonotopic map of the inner ear's basilar membrane, **Cochlea** separates 44.1 kHz stereo boundary windows into four stems: `vocals`, `drums`, `bass`, and `other`.

- **High-resolution vocal-harmonic STFT (`v2-Base`, `Pro`, `Ultra`)**: `4096`-point Hann STFT (`hop = 1024`, 10.77 Hz bin resolution) partitioned into 64 gap-free psychoacoustic bands: 21.5 Hz sub-bass bins (0 to 172 Hz), 43.1 Hz vocal-fundamental and first-formant bands (172 to 1,034 Hz), 129 Hz harmonic bands (1.03 to 4.13 kHz), sibilance bands (4.13 to 11.0 kHz), and air bands (11.0 to 22.05 kHz). `Sone-6.5M` uses a smaller 2048-point STFT (`hop = 512`) with 42 bands.
- **Dual-Axis Time+Band Transformer**: Interleaves dilated depthwise temporal convolution (`k = 5`, dilation `2^(i mod 6)`), temporal multi-head self-attention (`time_attn = true`, modeling phrase-long vocal vibrato and sustain), and cross-band multi-head self-attention (locking fundamental pitch to harmonic overtones). Width and depth scale across tiers: `6B / d=80` (`v2-Base`, 11.61 M), `8B / d=128` (`Pro`, 20.07 M), and function-preserving Net2Net `10B / d=160` (`Ultra`, 29.18 M).
- **Bounded 4-stem complex masking and Wiener conservation**: Predicts bounded complex masks `M_m = 2 * tanh(raw_m / 2)` for all four stems, projects complex spectrograms onto exact stereo mixture consistency weighted by frame-bin power `|S_m|^2`, and reconstructs `other = mix - (vocals + drums + bass)` in the time domain (`||sum_m s_m - mix||_inf < -148 dB`).
- **Training objective and two-stage distillation**: Combines vocal-weighted complex STFT L1, waveform L1, 5-scale multi-resolution STFT loss (256 to 4096), 128-mel perceptual vocal/accompaniment loss, active-vocal cosine phase-coherence loss, direct time-domain SDR, and `8x` silent-vocal pre-onset suppression. Supervised on MUSDB18-HQ plus 1,101 Stage-1 `HTDemucs-ft` clips and 372 Stage-2 `BS-RoFormer-1297` (12.98 dB vocal SDR) + `HTDemucs-ft` clips from Creative Commons corpora.

## 2. `Tympanum` (`tympanum.*`): Vinyl-Equivariant 7-Head Structural Sensor

Modeled after the eardrum's transduction of acoustic pressure pulses, **Tympanum** maps mono 44.1 kHz audio to a 50 Hz HTK log-mel representation (`n_fft = 2048, hop = 882`) through a dilated depthwise TCN + multi-layer BiGRU (`Base`: 128 mels, 4 conv, 2x BiGRU-64, 304,510 params; `Pro`: 160 mels 25 to 16,000 Hz, 6 conv dilations `1..32`, 3x BiGRU-128, 1,473,406 params).

- **Seven simultaneous outputs**:
  1. `beat` (50 Hz frame probability)
  2. `downbeat` (50 Hz bar-one probability)
  3. `voice` (50 Hz active singing activity)
  4. `line_end` (50 Hz inter-phrase singer breath boundary for clean vocal exits and echo throws)
  5. `phrase_start` (50 Hz 8-bar structural boundary)
  6. `key_logits` (24 major/minor pitch classes -> Camelot wheel)
  7. `tuning_cents` (`(-50, +50)` sub-semitone concert-pitch offset)
- **Exact vinyl-speed equivariant augmentation**: Resampling audio by `alpha = 2^(delta_s / 12)` (`delta_s in [-3.5, +3.5]` semitones) transforms all five temporal targets (`t' = t / alpha`), musical key (`(k + round(delta_s)) mod 12`), and tuning cents analytically with zero phase-vocoder transient smearing or timestamp drift.
- **Neuro-symbolic comb-grid lock**: Fuses 50 Hz neural beat/downbeat activations with physical spectral onset flux across multi-octave harmonic candidates (`{1/2, 2/3, 3/4, 1, 4/3, 3/2, 2}`) to eliminate half/double-tempo octave flips without Dynamic Bayesian Networks (12/12 exact BPM locks).

## 3. `Cortex` (`cortex.*`): Executive Beat-Synchronous Planner

**Cortex** operates on a standardized 209-token sequence (`[global; A_80; B_128]`, 24 features per beat) using a Pre-RMSNorm Beat Transformer (`Base`: `6L / 128D`, 1,221,494 params; `Pro`: `8L / 192D`, 3,604,726 params).

- **Outputs**: Transition recipe (`VoiceOverBeat`, `Sneak`, `OutroUnderIntro`, `Gap`), normalized bar offsets `(u_start, u_swap)` in `[0, 16]` bars (0.127 bar held-out MAE in `Cortex v4-Pro`), and `80 x 8` beat-resolution stem gain curves across both decks (0.0157 MAE).
- **Sharp-cut, unity-lock, and pre-vocal gating**: `7x` edge-weighted step supervision prevents conditional-mean fader smearing across `u = 0`. Accompaniment-floor regularisation removes pre-entry loudness holes (`+10.55 dB` blend floor lift). Post-drop unity lock keeps Deck B full on bar 0, and pre-vocal gating holds `G_{B,vocals} = 0` until 0.05 bars before Deck B's first singing onset.

## 4. `Cingulate` (`cingulate.*`): 30-Defect Counterfactual Conflict Monitor

Inspired by conflict monitoring in the anterior cingulate cortex, **Cingulate** (`Base`: `4L / 96D`, 475,071 params; `Pro`: `6L / 128D`, 1,225,247 params) ingests the 209-token beat sequence concatenated with Deck A/B candidate fader trajectories and predicts overall acceptability `q in [0, 1]` plus up to 30 independent DJ defect probabilities (`sone_core::Issue`).

- **On-GPU counterfactual defect synthesis**: 50% of training batches synthesize 12 ear-audited failure modes on the fly (premature accompaniment holes, mid-word vocal chops, double vocals, chord mush, double sub-bass, unblended hard cuts, pre-vocal stem bleed, post-drop melody delay, and beat drift), reaching 96.61% accuracy on strictly unseen `val.jsonl.gz` (`n = 5,932`) and 100.0% precision (`FP = 0`) on 152 human-audited mixes (`Cingulate v4-Pro`).
- **Best-of-`K` reranking**: Scores `K = 6` candidate plans around `Cortex`'s top recipes and bar offsets in 52 to 100 ms total on 1 CPU thread.

## 5. `sone-infer/1` `.safetensors` Wire Format

Every unified `model.safetensors` bundle carries standard `.safetensors` JSON `__metadata__`. Example from `sone-pro-f16.safetensors` (64-element `bands` list abbreviated):

```json
{
  "format": "sone-infer/1",
  "model": "sone",
  "name": "sone-pro",
  "organs": "cochlea,tympanum,cortex,cingulate",
  "precision": "F16",
  "config.cochlea": "{\"sample_rate\":44100,\"n_fft\":4096,\"hop\":1024,\"bands\":[[0,2],...,[1536,2049]],\"dim\":128,\"blocks\":8,\"heads\":8,\"ff_mult\":2,\"kernel\":5,\"dilations\":[1,2,4,8,16,32,1,2],\"mask_hidden\":256,\"time_attn\":true,\"four_stems\":true,\"mask_bound\":2.0}",
  "config.tympanum": "{\"sample_rate\":44100,\"n_fft\":2048,\"hop\":882,\"n_mels\":160,\"fmin\":25.0,\"fmax\":16000.0,\"channels\":160,\"kernel\":5,\"dilations\":[1,2,4,8,16,32],\"gru_layers\":3,\"gru_hidden\":128}",
  "config.cortex": "{\"kind\":\"cortex\",\"a_len\":80,\"b_len\":128,\"features\":24,\"lanes\":8,\"dim\":192,\"layers\":8,\"heads\":6,\"ff\":768}",
  "config.cingulate": "{\"kind\":\"cingulate\",\"a_len\":80,\"b_len\":128,\"features\":24,\"lanes\":8,\"dim\":128,\"layers\":6,\"heads\":4,\"ff\":512}"
}
```

`sone_infer::Sone::load` validates header bounds, parses organ configs (`#[serde(default)]` backward-compatible across `6.5M`, `v2-Base`, `Pro`, and `Ultra`), namespaces `cochlea.*`, `tympanum.*`, `cortex.*`, and `cingulate.*` over one shared `Arc<[u8]>` buffer with zero heap copies, and verifies bijective tensor consumption so neither missing nor unreferenced tensors are allowed.
