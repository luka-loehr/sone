#!/usr/bin/env python3
"""Per-track structure map from the stems and the Beat This grid.
structure.py [ids…] → hand/struct/<id>.json and a readable bar map on stdout.
Per bar: level (dB relative to the track's loud level) of vocals, drums, kick,
bass, other; vocal activity per beat; phrase phase (8-bar) from the novelty at
bar lines; grid = constant-tempo fit refined on the drum stem's onsets."""
import os, sys, json, glob
import numpy as np, soundfile as sf
from scipy import signal

ROOT = os.path.expanduser("~/djlab")
SR = 44100
HOP = 128

def load_beats(tid):
    b, d = [], []
    for line in open(f"{ROOT}/tracks/{tid}.beats"):
        p = line.split()
        if not p: continue
        b.append(float(p[0])); d.append(int(p[1]) if len(p) > 1 else 0)
    return np.array(b), np.array(d)

def onset_env(x, lo=None, hi=None):
    """Onset strength of mono x at HOP resolution (half-wave rectified log-energy rise)."""
    if lo or hi:
        if lo and hi: sos = signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
        elif hi: sos = signal.butter(4, hi, "lowpass", fs=SR, output="sos")
        else: sos = signal.butter(4, lo, "highpass", fs=SR, output="sos")
        x = signal.sosfilt(sos, x)
    n = len(x) // HOP
    e = (x[: n * HOP].reshape(n, HOP) ** 2).mean(1)
    e = np.convolve(e, np.ones(4) / 4, "same")
    l = np.log10(e + 1e-7)
    d = np.diff(l, prepend=l[0])
    return np.maximum(d, 0)

def rms_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-10)

def seg_db(x, t0, t1):
    a, b = int(max(t0, 0) * SR), int(min(t1 * SR, len(x)))
    if b - a < 64: return -100.0
    return rms_db(x[a:b])

def analyse(tid):
    beats, flags = load_beats(tid)
    stems = {}
    for n in ("vocals", "drums", "bass", "other"):
        s, sr = sf.read(f"{ROOT}/stems/{tid}/{n}.flac", dtype="float32")
        stems[n] = s.mean(1)
    mix = sum(stems.values())
    dur = len(mix) / SR
    # tempo: median beat interval, then least squares on a regular grid where the track holds it
    iv = np.diff(beats)
    T = float(np.median(iv))
    # local fits over windows of 32 beats → tempo stability
    k = np.arange(len(beats))
    # beat index assuming no missing beats: cumulative rounding of intervals
    idx = np.concatenate([[0], np.cumsum(np.maximum(1, np.round(iv / T)))]).astype(int)
    # longest run of regular intervals (within 4 % of the median) → the song's steady part
    reg = np.abs(iv - T) < 0.035
    best, cur, start = (0, 0), 0, 0
    for i, ok in enumerate(reg):
        if ok:
            if cur == 0: start = i
            cur += 1
            if cur > best[1] - best[0]: best = (start, i + 1)
        else:
            cur = 0
    s0, s1 = best
    kk = np.arange(s0, s1 + 1)
    A = np.vstack([kk - s0, np.ones_like(kk)]).T
    (Tfit, t0fit), *_ = np.linalg.lstsq(A, beats[s0:s1 + 1], rcond=None)
    resid = beats[s0:s1 + 1] - (Tfit * (kk - s0) + t0fit)
    # drift inside the steady part: local tempo per 32 beats
    loc = [float(60 / np.polyfit(np.arange(32), beats[i:i + 32], 1)[0]) for i in range(s0, s1 - 32, 16)]
    # drum-onset phase refinement of the beat times: shift all beats by δ maximizing kick/snare onsets
    env = onset_env(stems["drums"])
    def score(times):
        i = np.round(times * SR / HOP).astype(int)
        i = i[(i >= 0) & (i < len(env))]
        return env[i].sum()
    shifts = np.arange(-0.05, 0.0501, 0.001)
    sc = [score(beats + s) for s in shifts]
    delta = float(shifts[int(np.argmax(sc))])
    # bar phase: mode of flagged downbeats
    ph = np.bincount(idx[flags == 1] % 4, minlength=4)
    bar_phase = int(np.argmax(ph))
    down_idx = [i for i in range(len(beats)) if idx[i] % 4 == bar_phase]
    bars = beats[down_idx]
    # levels
    ref = {}
    loud = np.percentile([seg_db(mix, bars[i], bars[i + 1]) for i in range(len(bars) - 1)], 90)
    kick = signal.sosfilt(signal.butter(4, 150, "lowpass", fs=SR, output="sos"), stems["drums"])
    hats = signal.sosfilt(signal.butter(4, 5000, "highpass", fs=SR, output="sos"), stems["drums"])
    rows = []
    for i in range(len(bars) - 1):
        t0, t1 = bars[i], bars[i + 1]
        r = {"bar": i, "t": round(float(t0), 3), "len": round(float(t1 - t0), 4)}
        for n, x in (("mix", mix), ("vocals", stems["vocals"]), ("drums", stems["drums"]), ("kick", kick), ("hats", hats), ("bass", stems["bass"]), ("other", stems["other"])):
            r[n] = round(float(seg_db(x, t0, t1) - loud), 1)
        # vocal activity per beat in the bar
        bt = np.linspace(t0, t1, 5)
        r["vbeats"] = [round(float(seg_db(stems["vocals"], bt[j], bt[j + 1]) - loud), 1) for j in range(4)]
        rows.append(r)
    # vocal threshold: relative to the track's typical sung level
    vlev = np.array([r["vocals"] for r in rows])
    vref = np.percentile(vlev, 85)
    for r in rows:
        r["sung"] = [int(v > vref - 14) for v in r["vbeats"]]
    # novelty at bar lines → 8-bar phrase phase
    feat = np.array([[max(r[n], -40) for n in ("vocals", "drums", "kick", "bass", "other", "hats")] for r in rows])
    nov = np.zeros(len(rows))
    for i in range(2, len(rows) - 2):
        nov[i] = np.abs(feat[i:i + 2].mean(0) - feat[i - 2:i].mean(0)).sum()
    ph8 = [float(nov[p::8].sum()) for p in range(8)]
    ph4 = [float(nov[p::4].sum()) for p in range(4)]
    out = {
        "id": tid, "duration": round(dur, 2), "bpm": round(60 / Tfit, 3), "beat_s": float(Tfit), "t0": float(t0fit),
        "grid_resid_ms": {"rms": round(float(np.sqrt(np.mean(resid ** 2)) * 1000), 1), "max": round(float(np.abs(resid).max() * 1000), 1)},
        "steady": [float(beats[s0]), float(beats[s1])], "local_bpm": [round(v, 2) for v in loc],
        "drum_delta_ms": round(delta * 1000, 1), "bar_phase": bar_phase, "downbeat_votes": ph.tolist(),
        "phrase8": int(np.argmax(ph8)), "phrase8_scores": [round(v, 1) for v in ph8], "phrase4": int(np.argmax(ph4)),
        "loud_db": round(float(loud), 2), "bars": rows, "novelty": [round(float(v), 1) for v in nov],
        "beats": [round(float(b), 4) for b in beats], "beat_idx": idx.tolist(),
    }
    os.makedirs(f"{ROOT}/hand/struct", exist_ok=True)
    json.dump(out, open(f"{ROOT}/hand/struct/{tid}.json", "w"))
    return out

def lvl(v):
    # 0..9 from -36..0 dB
    return "·" if v < -36 else str(int(min(9, max(0, (v + 36) / 4))))

def show(o, head=None, tail=None):
    rows = o["bars"]
    lb = o.get("local_bpm") or [0]
    print(f"== {o['id']} {NAMES.get(o['id'], '')} | bpm {o['bpm']} (local {min(lb) if lb else 0}-{max(lb) if lb else 0}) dur {o['duration']} steady {o.get('steady')} resid {o['grid_resid_ms']} drumd {o['drum_delta_ms']}ms votes {o['downbeat_votes']} phrase8 {o['phrase8']} {o['phrase8_scores']}")
    sel = rows
    if head: sel = rows[:head]
    if tail: sel = rows[-tail:]
    print("   bar    time  M  vocal  D K H B O  nov   (levels 0-9 = -36..0 dB rel. loud, per-beat vocal digits)")
    for r in sel:
        mark = "#" if r["bar"] % 8 == o["phrase8"] else ("+" if r["bar"] % 4 == o["phrase8"] % 4 else " ")
        vb = "".join(lvl(v) for v in r["vbeats"])
        print(f"{mark}{r['bar']:4d} {r['t']:7.2f}  {lvl(r['mix'])}  {vb}   {lvl(r['drums'])} {lvl(r['kick'])} {lvl(r['hats'])} {lvl(r['bass'])} {lvl(r['other'])}  {o['novelty'][r['bar']]:4.0f}")

NAMES = {}
try:
    NAMES = {k: f"{v['name']} - {', '.join(v['artists'])}" for k, v in json.load(open(f"{ROOT}/lab/data.json"))["tracks"].items()}
except Exception:
    pass

def opt(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return int(a.split("=")[1])
    return default

if __name__ == "__main__":
    ids = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not ids:
        ids = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{ROOT}/stems/*/other.flac"))
    for tid in ids:
        p = f"{ROOT}/hand/struct/{tid}.json"
        o = json.load(open(p)) if os.path.exists(p) and "--force" not in sys.argv else analyse(tid)
        if "--quiet" not in sys.argv:
            show(o, head=opt("head"), tail=opt("tail"))
        else:
            lb = o.get("local_bpm") or [0]
            print(tid, f"{NAMES.get(tid, '')[:40]:40s}", o["bpm"], f"local {min(lb)}-{max(lb)}", o["grid_resid_ms"], o["drum_delta_ms"], "votes", o["downbeat_votes"], "ph8", o["phrase8"], flush=True)
