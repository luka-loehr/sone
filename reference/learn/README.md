# Learning from the owner's ear

Every transition the system makes can be judged on the lab site. The loop:

1. **He writes a note** on the lab site (`site/server.py`, run from `~/Documents/orin-app/djlab-site`).
   The server writes it at once into `data/owner/verdicts.json`, with the spec and the renderer's
   report of exactly the render he heard. A note written before a render counts for the earlier one;
   finished rounds are archived as `data/owner/verdicts_<date>.json`.
2. **`python3 reference/learn/learn.py`** turns the notes into labels (good or not, which fault) in
   `data/owner/labelled.json` and prints how the system's own checks agreed with his ear. A note it
   cannot read is listed and stays out until it is labelled in `data/owner/labels_by_hand.json`.
3. **`orml simulate`** (or `simulate`) puts every labelled verdict into the training data
   (`data/sim/owner.jsonl`) next to the generated examples. The critic and the planner train on them.
4. **Rules he states** ("never ...", "this song's voice never over another beat") go into
   `data/owner/pair_rules.json` or into the planner itself, with the note that asked for it.

For a blind test the planner's code is tagged before he listens (`git tag blind-test-N`), so nothing
can be tuned to the pairs he is about to hear.

New songs: `tools/fetch-songs` fetches his liked songs (his own account), `feed.sh` copies them to atlas,
`prep_new.sh` separates them (teachers/) and analyses them (`orml analyse`).

## Labelling by hand

Notes like "same here", "is ok" or "good but too much silence" cannot be read by keywords, and a "good" with a
complaint must keep the complaint. Every note of a round is therefore checked and, where needed, labelled in
`data/owner/labels_by_hand.json` with the reason (`why`). Faults the owner named that the list did not have are
added to `mix_core::Issue` (2026-10-01: `silence_before_one`, `b_voice_late`, `bad_pair`).

A verdict enters the training data only when both songs' beat tables are in `data/analysis/beats/`. The tables of
the 20 songs of blind test 3 are on atlas (`~/djlab/orml-main/data/analysis/beats/`) and have to be copied in.

## Between rounds

Before pairs he already rated are rendered again: archive `data/owner/verdicts.json` as
`verdicts_<date>-<round>.json` (with spec and report of what he heard) and write `{}` into it, re-key hand labels
from `current/...` to `<date>-<round>/...`, and keep the site's renders (`src/<page>_<round>`). Otherwise the server
overwrites the old spec under the same key. A verdict given in chat for a whole round is archived the same way.
