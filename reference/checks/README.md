# Checks of the planner against the owner's transitions

Run on atlas in a checkout with a built `orml` (`~/djlab/orml-main`), songs in `~/djlab/stems`.

| Script | What it does |
|---|---|
| `cargo run -p mix-plan --bin plan_check -- data S7` | the planner's decisions next to the owner's 13 accepted ones: recipe, the one in A, where A's voice leaves, the one in B |
| `run13.sh` | plans and renders the 13 pairs with `orml mix` into `out/auto2` and compares them bar by bar (`compare.py`) with the owner's specs rendered by the same renderer (`out/handrust`, made with `orml render data/versions/S7.json --struct data/analysis/struct`) |
| `newpairs.py` | picks 12 pairs the owner never heard (same beat within 2.5 %, keys at most one step apart), plans and renders them, and checks each: drums within 1 ms, no hole in the beat before the one, never two voices, no dip of more than 8 dB |

State on 2026-10-01, round 2 (after the owner's notes in data/owner/verdicts_2026-10-01.json):
- 13 lab pairs: 12 planned (h10 has no common beat and gets no transition), drums within 0.7 ms, halves of the blend within 2.2 ms (h12: 7.3).
- 12 new pairs: n06 no transition (no common beat); n02, n07, n09 refused after rendering because their drums drift 15 to 36 ms across the blend (the owner had heard all three as not fitting); the other 8 pass.
- A render whose drums drift more than 12 ms between the halves of the blend (`mix_render::MAX_DRIFT_MS`) is not used: shorter blend, else no transition.
For "A's beat under B's intro" the check does not count bar -1 as a hole: A's beat is filtered up there by design.
