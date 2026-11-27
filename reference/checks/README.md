# Symbolic Planner Regression Checks

Regression utilities comparing `sone-plan` and `sone-render` against the 13 reference transitions in `data/versions/S7.json`:

| Command / Script | Purpose |
|---|---|
| `cargo run -p sone-plan --bin plan_check -- data S7` | Compares `sone-plan` recipe, handoff bar, vocal exit, and Deck B entry against the 13 accepted specs in `data/versions/S7.json` |
| `reference/checks/run13.sh` | Plans and renders the 13 reference pairs with `sone mix` and compares bar-level stem telemetry via `compare.py` |
| `reference/checks/newpairs.py` | Evaluates unseen tempo- and key-compatible pairs for <= 1 ms drum alignment, zero beat holes, zero double vocals, and <= 8 dB level dip |

Renderer safety threshold (`sone_render::MAX_DRIFT_MS = 12.0` ms): any candidate blend whose first-half vs. second-half drum offset exceeds 12 ms falls back to a shorter blend window.
