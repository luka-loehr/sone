#!/usr/bin/env python3
"""Automatic nudge and beat lag of Rust renders against the reference, per spec.

compare_nudge.py <version> <rust dir> [<rust dir> ...]   (reference reports in /tmp/orml-py/<version>)"""
import sys, os, json

v, dirs = sys.argv[1], sys.argv[2:]
for f in sorted(os.listdir(f"/tmp/orml-py/{v}")):
    if not f.endswith(".report.json"):
        continue
    p = json.load(open(f"/tmp/orml-py/{v}/{f}"))
    row = [f"py {p['nudge_ms']:6.2f} ({p['align']['all'][0]:5.2f})"]
    for d in dirs:
        r = json.load(open(f"{d}/{f}"))
        row.append(f"{os.path.basename(d)} {r['nudge_ms']:6.2f} ({r['align']['all'][0]:5.2f}) d={r['nudge_ms'] - p['nudge_ms']:5.2f}")
    print(v, f.split(".")[0], "nudge ms (beat lag ms):", " | ".join(row))
