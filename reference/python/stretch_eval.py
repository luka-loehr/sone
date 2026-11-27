#!/usr/bin/env python3
"""Measures the Rust time stretcher against Rubber Band on real music.

stretch_eval.py <stretch binary> <tracks dir> <work dir> <id> [<id> ...] [--start 60] [--secs 30]

For every song, a 30 s excerpt is stretched at 0.965, 0.985, 1.016 and 1.035 by
Rubber Band R3 (-3), Rubber Band R2 (-2) and the Rust stretcher's two engines
(rs: splicing, pv: phase vocoder). Reported:
  len     output length minus round(n * ratio)
  env     onset timing from the onset envelope (handmix.onset_env, hop 32): lag of
          the output's envelope against the input's envelope with its time axis
          scaled by the ratio, per 5 s window; the worst window, in ms
  onset   per drum attack (peaks of the onset envelope above its 97th percentile, >= 50 ms
          apart): the lag at which the attack's >1.5 kHz amplitude envelope appears in the
          output against t * ratio; median and 90th percentile of |lag| in ms (search
          +-10 ms), and the share of attacks within 1 ms
  lsd     log-spectral distance (dB) to Rubber Band R3's output, and R2 vs R3 as the yardstick
  band    level of the output minus the input's in three bands (<150 Hz, 150-4k, >4k), dB
  flux    added frame-to-frame change of the log spectrum against the input (dB; warble,
          phasiness and splice glitches raise it, 0 = as steady as the input)
  preecho median change of the >1.5 kHz energy 25-3 ms before each attack (dB)
"""
import os, sys, subprocess, json
import numpy as np, soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

SR = 44100
RATIOS = (0.965, 0.985, 1.016, 1.035)
_HP = signal.butter(4, 1500, "highpass", fs=SR, output="sos")

def onset_env(x, hop=32):
    x = signal.sosfilt(_HP, x)
    n = len(x) // hop
    e = (x[: n * hop].reshape(n, hop) ** 2).mean(1)
    e = uniform_filter1d(e, 6)
    l = np.log10(e + 1e-6)
    d = np.diff(l, prepend=l[0])
    return np.maximum(d, 0)

def parabolic(v, i):
    if 0 < i < len(v) - 1:
        a, b, c = v[i - 1], v[i], v[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return i + 0.5 * (a - c) / den
    return float(i)

def env_timing(x, y, ratio, hop=32, win_s=5.0):
    """Worst |lag| (ms) over 5 s windows between y's envelope and x's envelope scaled by the ratio."""
    ex, ey = onset_env(x.mean(1), hop), onset_env(y.mean(1), hop)
    # x's envelope on y's time grid: frame k of y is at input time k*hop/ratio
    tk = np.arange(len(ey)) / ratio
    exr = np.interp(tk, np.arange(len(ex)), ex)
    w = int(win_s * SR / hop)
    m = int(0.010 * SR / hop)
    lags = []
    for s in range(0, len(ey) - w, w):
        a, b = exr[s:s + w] - exr[s:s + w].mean(), ey[s:s + w] - ey[s:s + w].mean()
        if a.std() < 1e-6:
            continue
        v = np.array([np.dot(a[max(0, -l): w - max(0, l)], b[max(0, l): w - max(0, -l)]) for l in range(-m, m + 1)])
        i = int(np.argmax(v))
        lags.append((parabolic(v, i) - m) * hop / SR * 1000)
    lags = np.array(lags)
    return float(np.abs(lags).max()), float(np.median(lags))

def attacks(x):
    """Drum attacks of the input: peaks of the onset envelope above its 97th percentile,
    at least 50 ms apart (kicks, snares, claps and the louder hi-hats)."""
    e = onset_env(x.mean(1), 32)
    thr = np.percentile(e, 97.0)
    idx = []
    for i in np.argsort(e)[::-1]:
        if e[i] < thr:
            break
        if all(abs(i - j) * 32 > 0.05 * SR for j in idx):
            idx.append(i)
    return sorted(int(i) * 32 for i in idx)

def attack_timing(x, y, ratio, ats):
    """Lag (ms) of each attack in y against t*ratio: cross-correlation of the >1.5 kHz
    amplitude envelope (1 ms RMS), which a phase vocoder keeps even where it changes the phases."""
    def env(z):
        h = signal.sosfilt(_HP, z.mean(1))
        return np.sqrt(uniform_filter1d(h * h, 44))
    hx, hy = env(x), env(y)
    pre, post, search = int(0.010 * SR), int(0.030 * SR), int(0.010 * SR)
    lags = []
    for t in ats:
        if t - pre < 0 or t + post >= len(hx):
            continue
        seg = hx[t - pre: t + post]
        seg = seg - seg.mean()
        c = int(round(t * ratio))
        lo, hi = c - pre - search, c + post + search
        if lo < 0 or hi >= len(hy):
            continue
        ref = hy[lo:hi]
        v = np.array([np.dot(seg, ref[k:k + len(seg)] - ref[k:k + len(seg)].mean()) for k in range(2 * search + 1)])
        k = int(np.argmax(v))
        lags.append((parabolic(v, k) - search) / SR * 1000)
    lags = np.abs(np.array(lags))
    return float(np.median(lags)), float(np.percentile(lags, 90)), float((lags <= 1.0).mean())

def band_levels(x, y):
    """Level of y minus level of x in three bands (<150 Hz, 150 Hz-4 kHz, >4 kHz), dB."""
    out = []
    for sos in (signal.butter(4, 150, "lowpass", fs=SR, output="sos"), signal.butter(4, [150, 4000], "bandpass", fs=SR, output="sos"),
                signal.butter(4, 4000, "highpass", fs=SR, output="sos")):
        ex = (signal.sosfilt(sos, x.mean(1)) ** 2).mean()
        ey = (signal.sosfilt(sos, y.mean(1)) ** 2).mean()
        out.append(round(10 * np.log10(ey / ex), 2))
    return out

def flux(y):
    """Mean frame-to-frame change of the log spectrum (dB), 100 Hz - 8 kHz, loud frames only.
    Warble, phasiness and splice glitches add change a clean stretch does not have."""
    _, _, Y = signal.stft(y.mean(1), fs=SR, nperseg=2048, noverlap=1536)
    f = np.fft.rfftfreq(2048, 1 / SR)
    p = np.abs(Y[(f >= 100) & (f <= 8000)]) ** 2
    l = 10 * np.log10(p + 1e-10 * p.mean(0).max())
    d = np.abs(np.diff(l, axis=1)).mean(0)
    loud = p.mean(0)[1:] > 1e-4 * p.mean(0).max()
    return float(d[loud].mean())

def preecho(x, y, ratio, ats):
    """Median over attacks of the output's >1.5 kHz energy in the 25 ms to 3 ms before the
    attack, minus the input's in the same place (dB): what a stretcher smears ahead of a hit."""
    hx = signal.sosfilt(_HP, x.mean(1))
    hy = signal.sosfilt(_HP, y.mean(1))
    a, b = int(0.025 * SR), int(0.003 * SR)
    v = []
    for t in ats:
        c = int(round(t * ratio))
        if t - a < 0 or c - a < 0 or c >= len(hy):
            continue
        ex = (hx[t - a: t - b] ** 2).mean() + 1e-12
        ey = (hy[c - a: c - b] ** 2).mean() + 1e-12
        v.append(10 * np.log10(ey / ex))
    return float(np.median(v))

def lsd(a, b):
    """Log-spectral distance in dB, 2048-point frames, 30 Hz - 16 kHz, frames above -60 dB of the loudest."""
    n = min(len(a), len(b))
    _, _, A = signal.stft(a[:n].mean(1), fs=SR, nperseg=2048, noverlap=1536)
    _, _, B = signal.stft(b[:n].mean(1), fs=SR, nperseg=2048, noverlap=1536)
    f = np.fft.rfftfreq(2048, 1 / SR)
    band = (f >= 30) & (f <= 16000)
    pa, pb = np.abs(A[band]) ** 2, np.abs(B[band]) ** 2
    eps = 1e-10 * pa.mean(0).max()
    d = 10 * np.log10(pa + eps) - 10 * np.log10(pb + eps)
    fr = np.sqrt((d ** 2).mean(0))
    loud = pa.mean(0) > 1e-6 * pa.mean(0).max()
    return float(fr[loud].mean())

def main():
    args = sys.argv[1:]
    start = float(args[args.index("--start") + 1]) if "--start" in args else 60.0
    secs = float(args[args.index("--secs") + 1]) if "--secs" in args else 30.0
    pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--start", "--secs"))]
    binary, tracks, work, ids = pos[0], pos[1], pos[2], pos[3:]
    os.makedirs(work, exist_ok=True)
    rows = []
    for tid in ids:
        raw = np.fromfile(f"{tracks}/{tid}.f32", dtype="<f4").reshape(-1, 2)
        x = raw[int(start * SR): int((start + secs) * SR)].copy()
        inp = f"{work}/{tid}.in.wav"
        sf.write(inp, x, SR, subtype="FLOAT")
        ats = attacks(x)
        for r in RATIOS:
            outs = {}
            info = {}
            for name, cmd in (("r3", ["rubberband", "-q", "-3", "--time", f"{r:.8f}"]), ("r2", ["rubberband", "-q", "-2", "--time", f"{r:.8f}"]), ("rs", ["--splice"]), ("pv", [])):
                o = f"{work}/{tid}.{name}.{r}.wav"
                if name in ("rs", "pv"):
                    res = subprocess.run([binary, inp, o, f"{r}"] + cmd, check=True, capture_output=True, text=True)
                    info[name] = res.stdout.strip()
                else:
                    subprocess.run(cmd + [inp, o], check=True, capture_output=True)
                outs[name] = sf.read(o, dtype="float32")[0]
                os.remove(o)
            n = int(round(len(x) * r))
            row = {"id": tid[:6], "ratio": r, "attacks": len(ats)}
            for name, y in outs.items():
                row[f"len_{name}"] = len(y) - n
                em, _ = env_timing(x, y, r)
                med, p90, within = attack_timing(x, y, r, ats)
                row[f"env_{name}"] = round(em, 3)
                row[f"onset_med_{name}"] = round(med, 3)
                row[f"onset_p90_{name}"] = round(p90, 3)
                row[f"onset_1ms_{name}"] = round(within, 3)
            row["lsd_rs_r3"] = round(lsd(outs["rs"], outs["r3"]), 3)
            row["lsd_pv_r3"] = round(lsd(outs["pv"], outs["r3"]), 3)
            row["lsd_r2_r3"] = round(lsd(outs["r2"], outs["r3"]), 3)
            row["lsd_rs_pv"] = round(lsd(outs["rs"], outs["pv"]), 3)
            # Level per band against the input (dB), as a check that nothing is lost; added
            # spectral change and pre-echo.
            fx = flux(x)
            for name, y in outs.items():
                row[f"band_{name}"] = band_levels(x, y)
                row[f"flux_{name}"] = round(flux(y) - fx, 3)
                row[f"preecho_{name}"] = round(preecho(x, y, r, ats), 2)
            row["info"] = info
            rows.append(row)
            print(json.dumps(row), flush=True)
        os.remove(inp)
    json.dump(rows, open(f"{work}/stretch_eval.json", "w"), indent=1)

if __name__ == "__main__":
    main()
