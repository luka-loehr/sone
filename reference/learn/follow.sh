#!/usr/bin/env bash
# Runs next to the batch: as soon as a song's voice is separated, its drums and bass without the voice
# and its analysis follow, so the GPU's second half and the CPU are used meanwhile.
cd ~/djlab/orml-main
while true; do
  left=0
  for id in $(cat new_ids.txt); do
    s=~/djlab/stems/$id
    [ -f data/analysis/beats/$id.json ] && continue
    left=1
    [ -f $s/vocals_r.flac ] || continue
    [ -f $s/bass_i.flac ] || ~/djlab/sv/bin/python teachers/separate_inst.py $id 2>&1 | grep -v -i warning | tail -1
    ./target/release/orml analyse $id --tracks ~/djlab/tracks --stems ~/djlab/stems --out data/analysis --model models/beat_this_final0_f16.safetensors 2>&1 | tail -1
  done
  [ $left = 0 ] && { echo FOLLOW DONE; break; }
  sleep 10
done
