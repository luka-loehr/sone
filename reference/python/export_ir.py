#!/usr/bin/env python3
"""Writes the reverb impulse responses handmix.py builds (numpy default_rng(7)) as raw
little-endian float32, interleaved stereo, so the Rust renderer can embed them.

export_ir.py <out_dir>  →  <out_dir>/reverb_ir_2.4.f32, reverb_ir_2.6.f32"""
import sys, os
import numpy as np
from scipy import signal

SR = 44100

def ir(rt60):
    # the same steps as handmix.reverb()
    rng = np.random.default_rng(7)
    n = int(rt60 * 1.3 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-6.91 * t / rt60)[:, None]
    ir[: int(0.012 * SR)] *= 0.0
    sos = signal.butter(2, [300, 6500], "bandpass", fs=SR, output="sos")
    ir = signal.sosfilt(sos, ir, axis=0)
    ir /= np.sqrt((ir ** 2).sum(0)).max()
    return ir.astype(np.float32)

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for rt in (2.4, 2.6):
        x = ir(rt)
        x.astype("<f4").tofile(f"{out}/reverb_ir_{rt}.f32")
        print(rt, x.shape, float(np.abs(x).max()))
