#!/usr/bin/env python3
"""Key and tuning per track from the pitched stems (bass + other + vocals).
keys.py → hand/keys.json and a table. Camelot codes; tuning in cents against A=440."""
import os, sys, json, glob
import numpy as np, soundfile as sf, librosa

ROOT = os.path.expanduser("~/djlab")
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CAM_MAJ = {0: "8B", 7: "9B", 2: "10B", 9: "11B", 4: "12B", 11: "1B", 6: "2B", 1: "3B", 8: "4B", 3: "5B", 10: "6B", 5: "7B"}
CAM_MIN = {9: "8A", 4: "9A", 11: "10A", 6: "11A", 1: "12A", 8: "1A", 3: "2A", 10: "3A", 5: "4A", 0: "5A", 7: "6A", 2: "7A"}
names = {k: v["name"] for k, v in json.load(open(f"{ROOT}/lab/data.json"))["tracks"].items()}
old = {k: v["analysis"].get("mixxx", {}).get("key") for k, v in json.load(open(f"{ROOT}/lab/data.json"))["tracks"].items()}
path = f"{ROOT}/hand/keys.json"
res = json.load(open(path)) if os.path.exists(path) else {}
for p in sorted(glob.glob(f"{ROOT}/stems/*/other.flac")):
    tid = os.path.basename(os.path.dirname(p))
    if tid in res: continue
    y = 0
    for n in ("other", "bass", "vocals"):
        s, sr = sf.read(f"{ROOT}/stems/{tid}/{n}.flac", dtype="float32")
        y = y + s.mean(1)
    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    tuning = float(librosa.estimate_tuning(y=y, sr=22050))
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050, tuning=tuning, hop_length=2048)
    c = chroma.mean(1)
    best = []
    for k in range(12):
        best.append((float(np.corrcoef(np.roll(MAJ, k), c)[0, 1]), f"{NOTE[k]}", CAM_MAJ[k]))
        best.append((float(np.corrcoef(np.roll(MIN, k), c)[0, 1]), f"{NOTE[k]}m", CAM_MIN[k]))
    best.sort(reverse=True)
    res[tid] = {"key": best[0][1], "camelot": best[0][2], "score": round(best[0][0], 3), "second": [best[1][1], best[1][2], round(best[1][0], 3)],
                "tuning_cents": round(tuning * 100, 1), "chroma": [round(float(v), 3) for v in c]}
    json.dump(res, open(path, "w"), indent=1)
for tid, r in sorted(res.items(), key=lambda kv: kv[1]["camelot"]):
    print(f"{tid} {names.get(tid, '')[:38]:38s} {r['key']:4s} {r['camelot']:4s} {r['score']:.2f}  2nd {r['second'][0]:4s} {r['second'][1]:4s} {r['second'][2]:.2f}  tune {r['tuning_cents']:+6.1f}c   mixxx {old.get(tid)}")
