#!/usr/bin/env python3
"""Is a beatless part on the song's grid? gridcheck.py <id> <fit_lo> <fit_hi> <bar_from> <bar_to> [stem]
Fits the grid on bars fit_lo..fit_hi, extends it to bar_from..bar_to (relative to fit_lo, may be negative)
and measures where the onsets of the stem (default: all but drums) sit against the 8th-note grid."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/djlab/hand"))
import handmix as h
tid, lo, hi, b0, b1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
stem = sys.argv[6] if len(sys.argv) > 6 else "pitched"
t = h.Track(tid)
t0, T, d = t.grid(lo, fit_bars=[lo, hi])
x = t.stems[stem].mean(1) if stem != "pitched" else (t.stems["other"] + t.stems["bass"] + t.stems["vocals"]).mean(1)
a, b = int(max(t0 + (b0 - lo) * 4 * T, 0) * h.SR), int((t0 + (b1 - lo) * 4 * T) * h.SR)
env, hop = h.onset_env(x[a:b], 32)
step = T / 2
res = []
for dl in np.arange(-step / 2, step / 2, 0.001):
    pos = np.arange((t0 + (b0 - lo) * 4 * T) + dl, t0 + (b1 - lo) * 4 * T, step)
    i = np.round((pos * h.SR - a) / hop).astype(int)
    i = i[(i >= 0) & (i < len(env))]
    res.append((env[i].sum(), dl))
res.sort(reverse=True)
m = np.mean([r[0] for r in res])
print(f"{tid} tempo {60/T:.3f} anchor bar {lo} at {t0:.3f}s | bars {b0}..{b1} ({stem}): best offset {res[0][1]*1000:+.0f} ms, peak/mean {res[0][0]/m:.2f}; next {[(round(r[1]*1000), round(r[0]/m,2)) for r in res[1:4]]}")
