# Human Preference & Defect Labelling Workflow

Scripts used to turn blind listening notes into structured supervision (`data/sim/owner.jsonl`) for `Cortex` and `Cingulate`:

1. Listening verdicts and timestamped transition reports are recorded in `data/owner/`.
2. `python3 reference/learn/learn.py` maps free-text listening notes and manual annotations (`data/owner/labels_by_hand.json`) into typed `sone_core::Issue` defect labels (`data/owner/labelled.json`).
3. `cargo run --release -p sone -- simulate` combines rule-generated positive transitions, structured counterfactual perturbations, and human-audited verdicts into `data/sim/{train.jsonl.gz,val.jsonl.gz,owner.jsonl}`.
4. Pair-level constraints discovered during listening rounds are stored in `data/owner/pair_rules.json`.
