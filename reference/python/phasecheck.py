#!/usr/bin/env python3
"""Is the one really the one? phasecheck.py → for every song in specs.json, on the bars used there:
per position in the bar (as the analysis counts it, in the spec's ruler beats): level of the kick band,
the clap/snare band, and how much the harmony changes on that beat (chroma change of bass + melody).
A backbeat belongs on 2 and 4, chord changes and the loudest kick on 1."""
import sys, os, json
import numpy as np
from scipy import signal
import librosa
sys.path.insert(0, os.path.expanduser("~/djlab/hand"))
import handmix as h
specs = json.load(open(os.path.expanduser("~/djlab/hand/specs.json")))
names = {k: v["name"] for k, v in json.load(open(os.path.expanduser("~/djlab/lab/data.json")))["tracks"].items()}
want = sys.argv[1:]
seen = set()
db = lambda x: 10 * np.log10(np.mean(x ** 2) + 1e-10)
for s in specs:
    for side in ("a", "b"):
        d = s[side]
        key = (d["id"], d["anchor_bar"], d.get("beat_shift", 0))
        if key in seen or (want and s["id"] not in want): continue
        seen.add(key)
        t = h._cache.get(d["id"]) or h.Track(d["id"]); h._cache[d["id"]] = t
        ta, T, _ = t.grid(d["anchor_bar"], fit_bars=d.get("fit_bars"), beat_shift=d.get("beat_shift", 0))
        bpb = d.get("beats_per_bar", 4)
        step = T * bpb / 4          # one ruler beat
        # bars looked at: 8 ruler bars with drums around the anchor (after it for B, before it for A)
        rng = range(0, 32) if side == "b" else range(-32, 0)
        dr = t.stems["drums"].mean(1)
        kick = signal.sosfilt(signal.butter(4, 120, "lowpass", fs=h.SR, output="sos"), dr)
        clap = signal.sosfilt(signal.butter(4, [1500, 6000], "bandpass", fs=h.SR, output="sos"), dr)
        pit = (t.stems["bass"] + t.stems["other"]).mean(1)
        a0, a1 = int(max(ta + rng[0] * step - 1, 0) * h.SR), int(min(ta + (rng[-1] + 2) * step, len(pit) / h.SR) * h.SR)
        y = librosa.resample(pit[a0:a1], orig_sr=h.SR, target_sr=22050)
        ch = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=512)
        def chroma_at(tt0, tt1):
            i0, i1 = int((tt0 - a0 / h.SR) * 22050 / 512), int((tt1 - a0 / h.SR) * 22050 / 512)
            if i1 <= i0 or i0 < 0 or i1 > ch.shape[1]: return None
            v = ch[:, i0:i1].mean(1); return v / (np.linalg.norm(v) + 1e-9)
        K, C, H = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
        for k in rng:
            tt = ta + k * step
            i0, i1 = int(tt * h.SR), int((tt + step * 0.5) * h.SR)
            if i0 < 0 or i1 > len(dr): continue
            K[k % 4].append(db(kick[i0:i1])); C[k % 4].append(db(clap[i0:i1]))
            c0, c1 = chroma_at(tt - step, tt), chroma_at(tt, tt + step)
            if c0 is not None and c1 is not None: H[k % 4].append(1 - float(c0 @ c1))
        f = lambda L: " ".join(f"{np.mean(x):6.1f}" if x else "   -- " for x in L)
        hc = [np.mean(x) if x else 0 for x in H]
        cl = [np.mean(x) if x else -99 for x in C]
        back = (cl[1] + cl[3]) / 2 - (cl[0] + cl[2]) / 2
        verdict = "backbeat on 2+4" if back > 1.5 else ("BACKBEAT ON 1+3 → the one is probably a beat off" if back < -1.5 else "no clear backbeat")
        print(f"{s['id']} {side.upper()} {names.get(d['id'], '')[:26]:26s} {60/T:6.1f} BPM x{bpb/4:g} | kick {f(K)} | clap {f(C)} | chords change {' '.join(f'{v:.3f}' for v in hc)} (most on beat {int(np.argmax(hc)) + 1}) | {verdict}", flush=True)
