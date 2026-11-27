import json, sys, os
sys.path.insert(0, os.path.expanduser("~/djlab/hand"))
import handmix as h
import numpy as np
spec = [s for s in json.load(open(os.path.expanduser("~/djlab/hand/specs.json"))) if s["id"] == sys.argv[1]][0]
A, B = h.Track(spec["a"]["id"]), h.Track(spec["b"]["id"])
tA, TA, dA = A.grid(spec["a"]["anchor_bar"]); tB, TB, dB = B.grid(spec["b"]["anchor_bar"])
tA -= spec["a"].get("anchor_u", 0) * 4 * TA; tB -= spec["b"].get("anchor_u", 0) * 4 * TB
ratio = TA / TB
pa, pb = spec["align_pairs"]["a"], spec["align_pairs"]["b"]
strip = lambda d: {k: v for k, v in d.items() if k != "segments"}
ad, _, _ = h.build_deck(A, tA, TA, strip(spec["a"]), pa[0], pa[1], only=("drums",), raw=True)
for nudge in (-0.02, -0.008, 0.0, 0.008, 0.02):
    bd, _, _ = h.build_deck(B, tB + nudge, TB, strip(spec["b"]), pb[0], pb[1], only=("drums",), raw=True)
    for name, y in (("stretched", h.stretch(bd, ratio)),):
        m = min(len(ad), len(y))
        print(f"nudge {nudge*1000:+.0f} ms  {name}: lag/sharp {h.xcorr_lag(ad[:m].mean(1), y[:m].mean(1))}")
# envelope peaks against the ruler: where do the kicks sit relative to the grid, per deck
def grid_offset(x, T):
    env, hop = h.onset_env(x.mean(1), 32)
    best, bd = -1, 0
    for d in np.arange(-0.04, 0.0401, 0.0005):
        i = np.round(((np.arange(0, len(x) / h.SR - 0.05, T) + d) * h.SR) / hop).astype(int)
        i = i[(i >= 0) & (i < len(env))]
        v = env[i].sum()
        if v > best: best, bd = v, d
    return bd * 1000
bd, _, _ = h.build_deck(B, tB, TB, strip(spec["b"]), pb[0], pb[1], only=("drums",), raw=True)
print("A kicks vs ruler (ms):", grid_offset(ad, TA), " B native:", grid_offset(bd, TB), " B stretched:", grid_offset(h.stretch(bd, ratio), TA))
