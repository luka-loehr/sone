#!/usr/bin/env python3
"""beatlook.py <id> <t_from> <t_to> [fit_lo fit_hi]: per beat of the grid: kick band, clap band, bass stem and
vocal level (dB), and the chroma change of the pitched stems at that beat (high = the chords change here)."""
import sys, os, json
import numpy as np
from scipy import signal
sys.path.insert(0, os.path.expanduser("~/djlab/hand"))
import handmix as h
tid, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
t = h.Track(tid)
fit = [int(sys.argv[4]), int(sys.argv[5])] if len(sys.argv) > 5 else None
lo = fit[0] if fit else 20
ta, T, _ = t.grid(lo, fit_bars=fit)
d = t.stems["drums"].mean(1)
kick = signal.sosfilt(signal.butter(4, 120, "lowpass", fs=h.SR, output="sos"), d)
clap = signal.sosfilt(signal.butter(4, [1500, 6000], "bandpass", fs=h.SR, output="sos"), d)
bass = t.stems["bass"].mean(1); voc = t.stems["vocals"].mean(1)
db = lambda x: 10 * np.log10(np.mean(x ** 2) + 1e-10)
k0 = int(np.ceil((t0 - ta) / T)); k1 = int((t1 - ta) / T)
print(f"{tid} tempo {60/T:.3f}; beat 1 of the analysis' bar is marked '1'. time, kick, clap, bass, voice (dB)")
for k in range(k0, k1):
    a, b = int((ta + k * T) * h.SR), int((ta + k * T + T * 0.5) * h.SR)
    pos = k % 4 + 1
    print(f"  {ta + k*T:7.2f}  beat {pos} {'<' if pos == 1 else ' '}  kick {db(kick[a:b]):6.1f}  clap {db(clap[a:b]):6.1f}  bass {db(bass[a:b]):6.1f}  voice {db(voc[a:b]):6.1f}")
