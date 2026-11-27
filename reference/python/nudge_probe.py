#!/usr/bin/env python3
"""Why does the automatic nudge differ between the reference and the Rust port?
For a spec, builds the reference's alignment signals (A's drums over align_pairs.a,
B's raw drums over align_pairs.b at Python's final nudge), stretches B's with Rubber
Band R3 and with both Rust engines, and measures for each:
  lag_vs_a     the lag handmix's xcorr_lag reports against A's drums (what the nudge loop sees)
  lag_vs_ideal the lag against B's own unstretched drums with the time axis scaled by the
               ratio (a perfect stretch gives 0)

nudge_probe.py <stretch binary> <specs.json> <id> [...]   (DJLAB as for handmix_dump.py)"""
import sys, os, json, subprocess, tempfile
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
import handmix_dump as H

def lag_vs_ideal(x, y, ratio, hop=32):
    ex, _ = H.onset_env(x.mean(1), hop)
    ey, _ = H.onset_env(y.mean(1), hop)
    exr = np.interp(np.arange(len(ey)) / ratio, np.arange(len(ex)), ex)
    n = min(len(exr), len(ey))
    a, b = exr[:n] - exr[:n].mean(), ey[:n] - ey[:n].mean()
    m = int(0.010 * H.SR / hop)
    v = np.array([np.dot(a[max(0, -l): n - max(0, l)], b[max(0, l): n - max(0, -l)]) for l in range(-m, m + 1)])
    i = int(np.argmax(v))
    frac = 0.0
    if 0 < i < len(v) - 1:
        den = v[i - 1] - 2 * v[i] + v[i + 1]
        if den < 0:
            frac = 0.5 * (v[i - 1] - v[i + 1]) / den
    return round((i + frac - m) * hop / H.SR * 1000, 3)  # positive: y's events come later than ideal

def main():
    binary, specs_path = sys.argv[1], sys.argv[2]
    specs = {s["id"]: s for s in json.load(open(specs_path))}
    for sid in sys.argv[3:]:
        spec = specs[sid]
        A, B = H.Track(spec["a"]["id"]), H.Track(spec["b"]["id"])
        tA, TA, _ = A.grid(spec["a"]["anchor_bar"], beat_shift=spec["a"].get("beat_shift", 0), fit_bars=spec["a"].get("fit_bars"))
        tB, TB, _ = B.grid(spec["b"]["anchor_bar"], beat_shift=spec["b"].get("beat_shift", 0), fit_bars=spec["b"].get("fit_bars"))
        bpbA, bpbB = spec["a"].get("beats_per_bar", 4), spec["b"].get("beats_per_bar", 4)
        tA -= spec["a"].get("anchor_u", 0) * bpbA * TA
        tB -= spec["b"].get("anchor_u", 0) * bpbB * TB
        ratio = (bpbA * TA) / (bpbB * TB)
        rep = json.load(open(os.path.join(os.environ["HANDMIX_OUT"], sid + ".meta.json")))
        nudge = rep["nudge_s"]
        pa, pb = spec["align_pairs"]["a"], spec["align_pairs"]["b"]
        strip = lambda d: {k: v for k, v in d.items() if k != "segments"}
        ad, _, _ = H.build_deck(A, tA, TA, strip(spec["a"]), pa[0], pa[1], only=("drums",), raw=True)
        bd, _, _ = H.build_deck(B, tB + nudge, TB, strip(spec["b"]), pb[0], pb[1], only=("drums",), raw=True)
        outs = {"r3": H.stretch(bd, ratio)}
        with tempfile.TemporaryDirectory() as d:
            sf.write(f"{d}/in.wav", bd, H.SR, subtype="FLOAT")
            for name, extra in (("splice", ["--splice"]), ("pv", [])):
                subprocess.run([binary, f"{d}/in.wav", f"{d}/out.wav", f"{ratio}"] + extra, check=True, capture_output=True)
                outs[name] = sf.read(f"{d}/out.wav", dtype="float32")[0]
        row = {"id": sid, "ratio": round(ratio, 6), "nudge_ms": round(nudge * 1000, 3)}
        for name, y in outs.items():
            m = min(len(ad), len(y))
            row[name] = {"lag_vs_a": round(H.xcorr_lag(ad[:m].mean(1), y[:m].mean(1))[0], 3), "lag_vs_ideal": lag_vs_ideal(bd, y, ratio)}
        print(json.dumps(row), flush=True)

if __name__ == "__main__":
    main()
