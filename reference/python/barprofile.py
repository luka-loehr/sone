#!/usr/bin/env python3
"""Where do kick and clap sit inside the bar? barprofile.py <spec id> → for both decks, on the bars that
overlap: energy per 16th note of the kick band (<120 Hz) and the clap/snare band (1.5-6 kHz) of the drum
stem, averaged over the bars; and the timing of each band against the other deck."""
import sys, os, json
import numpy as np
from scipy import signal
sys.path.insert(0, os.path.expanduser("~/djlab/hand"))
import handmix as h
spec = [s for s in json.load(open(os.path.expanduser("~/djlab/hand/specs.json"))) if s["id"] == sys.argv[1]][0]
A, B = h.Track(spec["a"]["id"]), h.Track(spec["b"]["id"])
tA, TA, _ = A.grid(spec["a"]["anchor_bar"], fit_bars=spec["a"].get("fit_bars"), beat_shift=spec["a"].get("beat_shift", 0)); tB, TB, _ = B.grid(spec["b"]["anchor_bar"], fit_bars=spec["b"].get("fit_bars"), beat_shift=spec["b"].get("beat_shift", 0))
bA, bB = spec["a"].get("beats_per_bar", 4), spec["b"].get("beats_per_bar", 4)
tA -= spec["a"].get("anchor_u", 0) * bA * TA; tB -= spec["b"].get("anchor_u", 0) * bB * TB
ratio = (bA * TA) / (bB * TB)
nudge = float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.0
pa, pb = spec["align_pairs"]["a"], spec["align_pairs"]["b"]
strip = lambda d: {k: v for k, v in d.items() if k not in ("segments",)}
ad, _, _ = h.build_deck(A, tA, TA, strip(spec["a"]), pa[0], pa[1], only=("drums",), raw=True)
bd, _, _ = h.build_deck(B, tB + nudge, TB, strip(spec["b"]), pb[0], pb[1], only=("drums",), raw=True)
bd = h.stretch(bd, ratio)
m = min(len(ad), len(bd)); a, b = ad[:m].mean(1), bd[:m].mean(1)
bar = bA * TA * h.SR
bands = {"kick <120": signal.butter(4, 120, "lowpass", fs=h.SR, output="sos"), "clap 1.5-6k": signal.butter(4, [1500, 6000], "bandpass", fs=h.SR, output="sos"), "hat >7k": signal.butter(4, 7000, "highpass", fs=h.SR, output="sos")}
def prof(x, sos):
    y = signal.sosfilt(sos, x) ** 2
    nb = int(len(y) / bar)
    p = np.zeros(16)
    for k in range(nb):
        for j in range(16):
            i0, i1 = int(k * bar + j * bar / 16), int(k * bar + (j + 1) * bar / 16)
            p[j] += y[i0:i1].mean()
    p = 10 * np.log10(p / nb + 1e-12); p -= p.max()
    return "".join(("9876543210·"[min(10, int(-v / 3))]) + ("|" if j % 4 == 3 else "") for j, v in enumerate(p))
def env(x, sos, hop=16):
    y = signal.sosfilt(sos, x)
    n = len(y) // hop
    e = (y[: n * hop].reshape(n, hop) ** 2).mean(1)
    e = np.convolve(e, np.ones(8) / 8, "same")
    d = np.diff(np.log10(e + 1e-7), prepend=0)
    return np.maximum(d, 0), hop
def lag(x, y, sos, max_ms=60):
    ea, hop = env(x, sos); eb, _ = env(y, sos)
    ea, eb = ea - ea.mean(), eb - eb.mean()
    mm = int(max_ms / 1000 * h.SR / hop); n = len(ea)
    vals = [(float((ea[l:] * eb[: n - l]).sum()) if l >= 0 else float((ea[: n + l] * eb[-l:]).sum()), l) for l in range(-mm, mm + 1)]
    best = max(vals); mean = np.mean([abs(v[0]) for v in vals])
    return best[1] * hop / h.SR * 1000, best[0] / (mean + 1e-9)
print(f"{spec['id']}: A bars u{pa} vs B bars u{pb}, tempo A {60/TA:.3f} B {60/TB:.3f}, 16 steps per bar, 9 = loudest, each step -3 dB")
for name, sos in bands.items():
    print(f"  {name:12s} A {prof(a, sos)}   B {prof(b, sos)}   B early by {lag(a, b, sos)[0]:+.1f} ms (sharp {lag(a, b, sos)[1]:.1f})")
# drift over the window, on the full attack envelope
q = m // 4
print("  lag per quarter of the window (ms):", [round(h.xcorr_lag(a[i*q:(i+1)*q], b[i*q:(i+1)*q])[0], 1) for i in range(4)])
