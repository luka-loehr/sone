#!/usr/bin/env bash
# Checks the Rust renderer against handmix_dump.py for one version of the specs,
# keeping at most one spec's dumps on disk at a time.
#
# verify_version.sh <version, e.g. S7>   (run in the orml-render checkout on atlas)
# Results: /tmp/orml-verify/<version>/{same_nudge,injected}.jsonl, auto/<id>.report.json
set -euo pipefail
V=$1
PY=/tmp/orml-py/$V
OUT=/tmp/orml-verify/$V
SPECS=data/versions/$V.json
BIN=target/release/render
ARGS=(--stems "$HOME/djlab/stems" --struct data/analysis/struct)
mkdir -p "$PY" "$OUT/same" "$OUT/injected" "$OUT/auto"
ids=$(~/djlab/sv/bin/python -c "import json,sys; print(' '.join(s['id'] for s in json.load(open('$SPECS'))))")
for id in $ids; do
    if [ ! -f "$PY/$id.meta.json" ]; then
        HANDMIX_OUT=$PY ~/djlab/sv/bin/python reference/python/handmix_dump.py "$SPECS" "$id" > "$PY/$id.log" 2>&1
    fi
    # Same nudge as Python, own stretcher.
    $BIN "$SPECS" "$id" "${ARGS[@]}" --out "$OUT/same" --dump --nudge-from "$PY" > "$OUT/same/$id.log" 2>&1
    ~/djlab/sv/bin/python reference/python/compare_render.py "$PY" "$OUT/same" "$id" | grep "^{" >> "$OUT/same_nudge.jsonl"
    # Same nudge and Python's stretched deck B: everything after the stretch.
    $BIN "$SPECS" "$id" "${ARGS[@]}" --out "$OUT/injected" --dump --nudge-from "$PY" --stretched-b-from "$PY" > "$OUT/injected/$id.log" 2>&1
    ~/djlab/sv/bin/python reference/python/compare_render.py "$PY" "$OUT/injected" "$id" | grep "^{" >> "$OUT/injected.jsonl"
    # Fully automatic (own nudge search).
    $BIN "$SPECS" "$id" "${ARGS[@]}" --out "$OUT/auto" > "$OUT/auto/$id.log" 2>&1
    rm -f "$OUT"/same/*.f32 "$OUT"/injected/*.f32 "$PY"/"$id".*.f32 "$OUT"/*/"$id".wav "$PY"/"$id".wav
done
echo "done $V"
