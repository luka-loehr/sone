# Symbolic and Perceptual DJ Transition Rules

Every rule implemented in `sone-plan` and audited by `Cingulate` (`sone_core::Issue`) encodes a concrete acoustic constraint verified in blind listening rounds. On the continuous four-beat bar ruler `u`, **`u = 0` ("the one")** is the downbeat where Deck B assumes rhythmic and harmonic ownership from Deck A.

## 1. Tempo and Bar-Phase Pairing

1. **Common beat ratio:** Deck A and Deck B must share a 1:1 tempo ratio (or a 1:2 / 2:1 half-time / double-time lock where every second beat of the faster track aligns with one beat of the slower track).
2. **Bounded key-locked stretch:** Deck B is time-stretched with pitch preserved (`sone_dsp::Stretcher`), preferring `0` to `3.5%` stretch (hard ceiling `±6.8%` in cortical search). Varispeed pitch riding and loop-roll build-ups are never used.
3. **True bar-one phase alignment:** Zero beat-onset lag alone is insufficient if the downbeat sits on beat 2, 3, or 4. `sone-analysis::phase` verifies that backbeat snare energy (`1.5` to `6 kHz` on the drum stem) lands on beats 2 and 4 while harmonic chroma shifts land on beat 1.

## 2. Stem Coexistence Invariants

4. **At most one lead vocal:** Two singers never overlap, except when Deck A's final syllable trails off through a tempo-synced quarter-note echo throw onto `u = 0`.
5. **At most one melodic bed:** Concurrent `other` stems create harmonic mud even in adjacent keys; melodic accompaniment swaps cleanly or ducks by `-3` to `-6 dB` under an isolated incoming vocal.
6. **At most one sub-bass:** In a beat sneak, Deck A retains full sub-bass authority until `u = 0`, where bass ownership swaps sharply to Deck B.
7. **Harmonic clash isolation (`d_Camelot >= 2`):** When keys clash on the Camelot wheel, pitched stems (`vocals`, `bass`, `other`) from both decks never overlap. Only unpitched drums (`drums`, high-passed above `180` to `500 Hz`) bridge prior to `u = 0`.
8. **Half/double-tempo drum rule:** At `1:2` or `2:1` tempo locks, layering two full drum patterns produces rhythmic clutter; such pairs use `VoiceOverBeat` or `OutroUnderIntro` rather than a dual-drum sneak.
9. **Bleed-free muted stems:** Muting a vocal lane requires clean stem separation (`Cochlea` + pre-vocal silence gating) so phantom vocal artifacts never leak through `drums`, `bass`, or `other`.

## 3. Transition Recipes

- **`VoiceOverBeat`:** When Deck B opens with sustained singing before its full drums drop (and keys are compatible), Deck A finishes its vocal phrase with a quarter-note echo throw while Deck A's drums and bass hold a steady pocket. Deck B's isolated vocal enters on phrase over Deck A's groove (`u_handoff` in `[-8, -2]`), and on `u = 0` Deck A cuts cleanly as Deck B's full arrangement drops.
- **`Sneak`:** When Deck B has an instrumental beat intro, Deck B's high-passed hi-hats sneak in at `u = -8` and kicks join at `u = -4`. Deck A's accompaniment fades linearly down to `50%` (never `0%`, avoiding a pre-drop energy hole) and cuts cleanly on `u = 0` as Deck B's bass, melody, and vocal enter at unity.
- **`OutroUnderIntro`:** Deck A's outro beat carries Deck B's beatless melodic intro over 6 bars, high-pass filtering out over the final 2 bars onto `u = 0`.
- **Unblendable pairs (`Gap` rejected):** Pairs with incompatible meter or excessive drift are rejected rather than forcing an unmusical hard cut after an a cappella stem drop.

## 4. Envelope and Phrase Placement

10. **Linear fades started ahead of events:** Gain ramps are linear in amplitude and begin prior to structural boundaries so fader motion never calls attention to itself.
11. **Crisp downbeat handoffs:** Outgoing stems terminate on `u = 0` with a ~25 ms anti-click ramp and zero post-one bleed, except when Deck A's final sung phrase naturally rings across bar `0`.
12. **Zero beat holes before `u = 0`:** Accompaniment-floor regularisation prevents pre-drop loudness dips exceeding `6 dB`.
13. **Word-boundary vocal entries and exits:** Vocals never enter or cut mid-word. `Tympanum`'s `line_end` breath head and sustained-singing gate ensure Deck A exits on a breath or line end and Deck B enters at the start of a real sung phrase (skipping isolated short intro producer tags).
14. **Anchor placement (`u = 0`):** On Deck A, `u = 0` lands on the 8-bar phrase boundary where Deck A enters its outro wind-down (before any mid-song breakdown gap). On Deck B, `u = 0` lands where Deck B's full beat arrives or within `1.25` bars of its first verse line.

## 5. Post-Render Verification Gates

15. **Local window tempo measurement:** Tempo is measured specifically across the bars participating in `[u_0, u_1]` so live-drumming tempo drift outside the transition does not skew alignment.
16. **Sub-millisecond Newton drum lock & drift ceiling:** Closed-form cross-correlation aligns kicks at `u = 0` and refines stretch slope across the first and second halves of the blend (`<= 0.7 ms` rendered drift on held-out mixes). Any candidate render exceeding `12 ms` half-window drift (`sone_render::MAX_DRIFT_MS`) retries with a shorter blend window.
