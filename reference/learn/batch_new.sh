#!/usr/bin/env bash
# All new songs at once: Demucs, BS-RoFormer, Demucs without the voice, then the analysis. Logs each step.
cd ~/djlab/orml-main
IDS="$(cat new_ids.txt)"
~/djlab/sv/bin/python teachers/separate.py $IDS 2>&1 | grep -v -i warning
~/djlab/rv/bin/python teachers/vocals.py $IDS 2>&1 | grep -v -i warning | tail -40
~/djlab/sv/bin/python teachers/separate_inst.py $IDS 2>&1 | grep -v -i warning
for id in $IDS; do ./target/release/orml analyse $id --tracks ~/djlab/tracks --stems ~/djlab/stems --out data/analysis --model models/beat_this_final0_f16.safetensors 2>&1 | tail -1; done
echo BATCH DONE
