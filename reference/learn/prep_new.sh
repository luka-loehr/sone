#!/usr/bin/env bash
# Stems and analysis of new songs: prep_new.sh <id>…  (atlas, in ~/djlab/orml-main)
# Songs must be in ~/djlab/tracks/<id>.f32. Writes ~/djlab/stems/<id>/ and data/analysis/{beats,struct_final}/<id>.json.
set -e
cd ~/djlab/orml-main
for id in "$@"; do
  [ -f data/analysis/beats/$id.json ] && { echo "$id done already"; continue; }
  t0=$(date +%s)
  ~/djlab/sv/bin/python teachers/separate.py $id > /dev/null 2>&1
  ~/djlab/rv/bin/python teachers/vocals.py $id > /dev/null 2>&1
  ~/djlab/sv/bin/python teachers/separate_inst.py $id > /dev/null 2>&1
  ./target/release/orml analyse $id --tracks ~/djlab/tracks --stems ~/djlab/stems --out data/analysis --model models/beat_this_final0_f16.safetensors 2>&1 | tail -1
  echo "$id ready in $(( $(date +%s) - t0 )) s"
done
