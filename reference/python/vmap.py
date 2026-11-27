#!/usr/bin/env python3
"""Fine vocal map: vmap.py <id> <bar0> <bar1> → per bar 16 steps (16th notes) of the vocal stem's level,
0-9 = -36..0 dB relative to the song's sung level, '·' below. Gaps show where a phrase ends or begins."""
import os, sys, json
import numpy as np, soundfile as sf
ROOT = os.path.expanduser("~/djlab"); SR = 44100
tid, b0, b1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
s = json.load(open(f"{ROOT}/hand/struct/{tid}.json"))
p = f"{ROOT}/stems/{tid}/vocals_r.flac"
if not os.path.exists(p): p = f"{ROOT}/stems/{tid}/vocals.flac"
v = sf.read(p, dtype="float32")[0].mean(1)
bars = s["bars"]
db = lambda x: 10 * np.log10(np.mean(x ** 2) + 1e-10)
ref = np.percentile([db(v[int(r["t"] * SR): int((r["t"] + r["len"]) * SR)]) for r in bars], 85)
print(f"{tid} vocals ({os.path.basename(p)}), ref {ref:.1f} dB; one char = one 16th note, | = beat")
for r in bars[b0:b1 + 1]:
    t0, L = r["t"], r["len"]
    out = ""
    for k in range(16):
        a, b = int((t0 + k * L / 16) * SR), int((t0 + (k + 1) * L / 16) * SR)
        d = db(v[a:b]) - ref
        out += ("·" if d < -30 else str(int(min(9, max(0, (d + 30) / 3.4))))) + ("|" if k % 4 == 3 else "")
    print(f"{r['bar']:4d} {r['t']:7.2f}  {out}")
