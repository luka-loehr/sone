# Repository Data Inventory

Zero copyrighted audio recordings, separated audio stems, or album artwork files are stored in this repository. All checked-in files under `data/` are compact numeric feature tables, transition `Spec` JSONs, and structured human defect annotations licensed under [Apache-2.0](../LICENSE).

## Included in `data/`

| Path | Contents | Scale |
|---|---|---|
| `data/analysis/beats/<id>.json` | 24-feature beat-resolution `BeatTable` files (bar phase, stem loudness, vocal onsets/breaths, chroma, Camelot key) | 40 tracks |
| `data/analysis/struct/<id>.json` | Bar-level structural profiles (`TrackAnalysis`) | 40 tracks |
| `data/analysis/keys.json` | Detected musical key, Camelot code, and concert-pitch tuning offset | 40 tracks |
| `data/versions/S1.json` .. `S7.json` | Reference transition `Spec` progressions across seven engineering iterations | 75 specs |
| `data/renders/S*/hNN.report.json` | Renderer verification telemetry (tempo, stretch %, beat lag, per-bar stem levels) | 75 reports |
| `data/labels.json` & `data/final_verdicts.json` | Structured defect annotations (`sone_core::Issue`) and blind listening verdicts | 41 records |
| `data/owner/` | Human preference judgements (`labelled.json`) and pair-specific invariants (`pair_rules.json`) | 152 judgements |
| `data/round0/` & `data/radio/` | Baseline engine comparison metadata (Mixxx vs. Superpowered vs. Rust) | 71 records |

## Generated Locally on Demand

Running `cargo run --release -p sone -- simulate` deterministically generates `data/sim/{train.jsonl.gz,val.jsonl.gz,owner.jsonl,stats.json}` in ~1 second from `data/analysis/` and `data/owner/` without requiring any audio files.

## External Open Datasets Used in Training

| Dataset | Upstream URL | Licence |
|---|---|---|
| **MUSDB18-HQ** | [https://sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html) | CC BY-NC-SA 4.0 |
| **MTG-Jamendo** | [https://mtg.github.io/mtg-jamendo-dataset/](https://mtg.github.io/mtg-jamendo-dataset/) | Creative Commons |
| **Song Describer Dataset** | [https://github.com/ilaria-manco/song-describer-dataset](https://github.com/ilaria-manco/song-describer-dataset) | Creative Commons |
| **Three Showcase Demo Pairs** | [assets/demos/CREDITS.md](../assets/demos/CREDITS.md) | CC BY 3.0 / CC BY-SA 3.0 |
