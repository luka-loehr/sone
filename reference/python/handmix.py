#!/usr/bin/env python3
"""Hand-built DJ transitions from stems.

handmix.py <spec.json> [--check]  →  out/<id>.wav, out/<id>.json (lanes, markers, checks)

A spec places two songs on one bar ruler u (bars; u = 0 is "the one", the bar
line where the low end changes hands). Song A plays untouched at its own
tempo. Song B is cut together from its own bars (edit list), mixed from its
stems in its own time, then time-stretched as ONE signal (Rubber Band R3, key
lock) to A's tempo, so its stems stay phase-coherent. Every stem has a gain
lane; stems can be filtered (HPF/LPF sweeps) and sent to a tempo echo or a
reverb. Both decks are gain-matched by loudness; the sum runs through a
look-ahead limiter."""
import os, sys, json, subprocess, tempfile
import numpy as np, soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
import pyloudnorm as pyln

ROOT = os.path.expanduser("~/djlab")
SR = 44100
STEMS = ("vocals", "drums", "bass", "other")
_cache = {}

def smoothstep(x):
    return x * x * (3 - 2 * x)

class Track:
    def __init__(self, tid):
        self.id = tid
        self.s = json.load(open(f"{ROOT}/hand/struct/{tid}.json"))
        self.stems = {n: sf.read(f"{ROOT}/stems/{tid}/{n}.flac", dtype="float32")[0] for n in STEMS}
        # the cleaner vocal stem (BS-RoFormer) replaces Demucs' one; `other` takes the difference,
        # so the four stems still sum to the song
        vr = f"{ROOT}/stems/{tid}/vocals_r.flac"
        self.clean_vocals = os.path.exists(vr)
        self.leak_free = False
        if self.clean_vocals:
            mix = sum(self.stems.values())
            v = sf.read(vr, dtype="float32")[0]
            self.stems["vocals"] = v[: len(mix)]
            # drums and bass separated from the song without its voice: nothing of the voice can leak into them
            di, bi = f"{ROOT}/stems/{tid}/drums_i.flac", f"{ROOT}/stems/{tid}/bass_i.flac"
            self.leak_free = os.path.exists(di) and os.path.exists(bi)
            if self.leak_free:
                self.stems["drums"] = sf.read(di, dtype="float32")[0][: len(mix)]
                self.stems["bass"] = sf.read(bi, dtype="float32")[0][: len(mix)]
            self.stems["other"] = mix - self.stems["vocals"] - self.stems["drums"] - self.stems["bass"]
        self.n = len(self.stems["drums"])
        self.beats = np.array(self.s["beats"])
        self.idx = np.array(self.s["beat_idx"])
        ph = self.s["bar_phase"]
        self.down = [i for i in range(len(self.beats)) if self.idx[i] % 4 == ph]
        mix = sum(self.stems.values())
        self.lufs = pyln.Meter(SR).integrated_loudness(mix)
        self.mono_drums = self.stems["drums"].mean(1)

    def grid(self, bar, span_bars=16, beat_shift=0, fit_bars=None):
        """Local constant-tempo grid around bar → (time of the bar's downbeat, beat period).
        beat_shift moves the bar line by whole beats (when the analysis has the one on the wrong beat).
        fit_bars = [lo, hi] fits the grid on those bars only (the part with a steady beat); bars outside,
        e.g. an intro without drums, then sit on the extended grid."""
        k0 = self.idx[self.down[bar]] + beat_shift
        T0 = self.s["beat_s"]
        if fit_bars:
            sel = (self.idx >= self.idx[self.down[fit_bars[0]]]) & (self.idx <= self.idx[self.down[min(fit_bars[1], len(self.down) - 1)]])
        else:
            sel = np.abs(self.idx - k0) <= span_bars * 4
        b, k = self.beats[sel], self.idx[sel] - k0
        # drop beats off the grid (fills, breaks): two rounds of least squares
        for _ in range(3):
            T, t0 = np.polyfit(k, b, 1)
            r = b - (T * k + t0)
            ok = np.abs(r) < 0.03
            if ok.all() or ok.sum() < 8: break
            b, k = b[ok], k[ok]
        # phase on the drum stem's onsets
        lo, hi = int(max(t0 - span_bars * 4 * T, 0) * SR), int(min((t0 + span_bars * 4 * T) * SR, self.n))
        ks = np.arange(-span_bars * 4, span_bars * 4)
        if fit_bars:
            lo, hi = int(max(b.min() - 0.1, 0) * SR), int(min((b.max() + 0.5) * SR, self.n))
            ks = np.arange(int(k.min()), int(k.max()) + 1)
        env, hop = onset_env(self.mono_drums[lo:hi], 32)
        best, bd = -1, 0.0
        for d in np.arange(-0.045, 0.0451, 0.0005):
            i = np.round(((t0 + ks * T + d) * SR - lo) / hop).astype(int)
            i = i[(i >= 0) & (i < len(env))]
            v = env[i].sum()
            if v > best: best, bd = v, d
        return float(t0 + bd), float(T), float(bd)

_HP = signal.butter(4, 1500, "highpass", fs=SR, output="sos")
def onset_env(x, hop=32):
    """Attack envelope of a drum signal. Only the band above 1.5 kHz counts: there a hit is a sharp
    click, while the low end of a kick swings for 20 ms and would smear the timing."""
    x = signal.sosfilt(_HP, x)
    n = len(x) // hop
    e = (x[: n * hop].reshape(n, hop) ** 2).mean(1)
    e = uniform_filter1d(e, 6)
    l = np.log10(e + 1e-6)
    d = np.diff(l, prepend=l[0])
    return np.maximum(d, 0), hop

def lane(points, u, smooth_ms=6.0, shape="lin"):
    """Gain lane: breakpoints [(u, amplitude 0..1)] → per-sample gain on the ruler u."""
    if not points:
        return None
    pu = np.array([p[0] for p in points], dtype=float)
    pv = np.array([p[1] for p in points], dtype=float)
    g = np.interp(u, pu, pv)
    if shape == "db":
        # even to the ear: straight lines in dB between the points, silence (0) is -70 dB
        pd = 20 * np.log10(np.maximum(pv, 10 ** (-70 / 20)))
        g = 10 ** (np.interp(u, pu, pd) / 20)
        g = np.where(np.interp(u, pu, (pv > 0).astype(float)) <= 0, 0.0, g)
        g[g < 10 ** (-66 / 20)] = 0.0
    if shape == "s":
        # smooth each segment
        seg = np.searchsorted(pu, u, side="right") - 1
        seg = np.clip(seg, 0, len(pu) - 2)
        w = np.clip((u - pu[seg]) / np.maximum(pu[seg + 1] - pu[seg], 1e-9), 0, 1)
        g = pv[seg] + (pv[seg + 1] - pv[seg]) * smoothstep(w)
        g = np.where(u <= pu[0], pv[0], np.where(u >= pu[-1], pv[-1], g))
    k = max(1, int(smooth_ms * SR / 1000))
    return uniform_filter1d(g, k, mode="nearest").astype(np.float32)

def sweep(x, points, u, kind):
    """Time-varying 4th-order Butterworth (HPF/LPF) with a cutoff lane [(u, Hz)]."""
    pu = np.array([p[0] for p in points], dtype=float)
    pf = np.log(np.array([p[1] for p in points], dtype=float))
    out = np.empty_like(x)
    B = 64
    zi = np.zeros((2, x.shape[1], 2))
    for i in range(0, len(x), B):
        f = float(np.exp(np.interp(u[min(i + B // 2, len(u) - 1)], pu, pf)))
        f = min(max(f, 15.0), 20000.0)
        sos = signal.butter(4, f, "highpass" if kind == "hpf" else "lowpass", fs=SR, output="sos")
        seg = x[i:i + B]
        for ch in range(x.shape[1]):
            y, z = signal.sosfilt(sos, seg[:, ch], zi=zi[:, ch, :])
            out[i:i + B, ch] = y
            zi[:, ch, :] = z
    return out

def echo(x, delay_s, feedback=0.55, repeats=10, hp=350.0, lp=4500.0):
    """Tempo echo, wet only: each repeat is a little duller and quieter, ping-pong across the sides."""
    d = int(round(delay_s * SR))
    wet = np.zeros_like(x)
    sos = signal.butter(2, [hp, lp], "bandpass", fs=SR, output="sos")
    cur = x.copy()
    for k in range(1, repeats + 1):
        cur = signal.sosfilt(sos, cur, axis=0)
        amp = feedback ** (k - 1)
        if k * d >= len(x): break
        seg = cur[: len(x) - k * d] * amp
        if k % 2 == 0:
            seg = seg[:, ::-1] * np.array([1.0, 0.8])
        else:
            seg = seg * np.array([1.0, 0.8])
        wet[k * d:] += seg
    return wet

_ir = None
def reverb(x, rt60=2.4):
    global _ir
    if _ir is None:
        rng = np.random.default_rng(7)
        n = int(rt60 * 1.3 * SR)
        t = np.arange(n) / SR
        ir = rng.standard_normal((n, 2)) * np.exp(-6.91 * t / rt60)[:, None]
        ir[: int(0.012 * SR)] *= 0.0
        sos = signal.butter(2, [300, 6500], "bandpass", fs=SR, output="sos")
        ir = signal.sosfilt(sos, ir, axis=0)
        ir /= np.sqrt((ir ** 2).sum(0)).max()
        _ir = ir.astype(np.float32)
    return np.stack([signal.fftconvolve(x[:, c], _ir[:, c])[: len(x)] for c in range(2)], 1).astype(np.float32)

def assemble(track, stem, t_anchor, T, segments, u_from, u_to, bpb=4):
    """One stem of a deck on the ruler: [u_from, u_to) in the deck's own time, cut from its bars."""
    bar = bpb * T * SR
    n = int(round((u_to - u_from) * bar))
    src = track.stems[stem]
    out = np.zeros((n, 2), dtype=np.float32)
    XF = int(0.006 * SR)
    for si, s in enumerate(segments):
        u0 = max(s["u0"], u_from)
        u1 = min(s["u1"] if s.get("u1") is not None else u_to, u_to)
        if u1 <= u0: continue
        d0, d1 = int(round((u0 - u_from) * bar)), int(round((u1 - u_from) * bar))
        s0 = int(round((t_anchor + (s["src"] + (u0 - s["u0"])) * bpb * T) * SR))
        pre = XF if (si > 0 and d0 >= XF) else 0
        a, b = s0 - pre, s0 + (d1 - d0)
        seg = np.zeros((b - a, 2), dtype=np.float32)
        lo, hi = max(a, 0), min(b, len(src))
        if hi > lo:
            seg[lo - a: hi - a] = src[lo:hi]
        if pre:
            w = np.sin(np.linspace(0, np.pi / 2, pre))[:, None].astype(np.float32)
            out[d0 - pre: d0] = out[d0 - pre: d0] * np.cos(np.linspace(0, np.pi / 2, pre))[:, None] + seg[:pre] * w
            out[d0:d1] = seg[pre:]
        else:
            out[d0:d1] = seg
    return out

def build_deck(track, t_anchor, T, deck, u_from, u_to, only=None, raw=False):
    """A deck's signal on [u_from, u_to) in its own time. deck: segments, lanes, filters, sends.
    t_anchor is the time of ruler position 0 in the track."""
    bpb = deck.get("beats_per_bar", 4)
    bar = bpb * T * SR
    n = int(round((u_to - u_from) * bar))
    u = u_from + np.arange(n) / bar
    segments = deck.get("segments") or [{"u0": u_from, "u1": None, "src": u_from}]
    total = np.zeros((n, 2), dtype=np.float32)
    lanes_out = {}
    parts = {}
    for stem in STEMS:
        if only and stem not in only: continue
        x = assemble(track, stem, t_anchor, T, segments, u_from, u_to, bpb)
        if raw:
            total += x
            continue
        pts = deck.get("lanes", {}).get(stem, [(u_from, 1.0)])
        g = lane(pts, u, shape=deck.get("shapes", {}).get(stem, deck.get("shape", "s")))
        for kind in ("hpf", "lpf"):
            fp = deck.get(kind, {}).get(stem)
            if fp:
                # filter only where the lane moves, the rest stays bit-exact
                a = int(np.clip((fp[0][0] - u_from) * bar, 0, n)); b = int(np.clip((fp[-1][0] - u_from) * bar, 0, n))
                flat = 20.0 if kind == "hpf" else 20000.0
                if abs(fp[-1][1] - flat) > 1: b = n
                if abs(fp[0][1] - flat) > 1: a = 0
                if b > a:
                    x[a:b] = sweep(x[a:b], fp, u[a:b], kind)
        dry = x * g[:, None]
        total += dry
        parts[stem] = dry.mean(1)
        for fx in deck.get("sends", []):
            if fx["stem"] != stem: continue
            s = lane(fx["send"], u, shape="lin")
            feed = x * s[:, None] * (g[:, None] if fx.get("post", False) else 1.0)
            if fx["type"] == "echo":
                total += echo(feed, fx.get("beats", 0.75) * T, fx.get("feedback", 0.55), hp=fx.get("hp", 350.0), lp=fx.get("lp", 4500.0)) * fx.get("level", 0.7)
            else:
                total += reverb(feed, fx.get("rt60", 2.4)) * fx.get("level", 0.5)
        lanes_out[stem] = g
    if deck.get("gain_db") and not raw:
        gd = lane(deck["gain_db"], u, smooth_ms=50.0, shape="lin")
        gl = (10 ** (gd / 20)).astype(np.float32)
        total *= gl[:, None]
        parts = {k: v * gl for k, v in parts.items()}
    lanes_out["_parts"] = parts
    return total, u, lanes_out

def stretch(x, ratio):
    """Key-locked time stretch (Rubber Band R3). ratio = output length / input length."""
    if abs(ratio - 1) < 1e-5:
        return x
    with tempfile.TemporaryDirectory() as d:
        sf.write(f"{d}/in.wav", x, SR, subtype="FLOAT")
        subprocess.run(["rubberband", "-q", "-3", "--time", f"{ratio:.8f}", f"{d}/in.wav", f"{d}/out.wav"], check=True)
        y, _ = sf.read(f"{d}/out.wav", dtype="float32")
    n = int(round(len(x) * ratio))
    if len(y) < n:
        y = np.vstack([y, np.zeros((n - len(y), 2), dtype=np.float32)])
    return y[:n]

def limiter(x, ceiling=0.89, look_ms=5.0, hold_ms=60.0):
    """Look-ahead peak limiter: the gain is down before a peak arrives and held shortly after it."""
    peak = np.abs(x).max(1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    if need.min() >= 1.0:
        return x, 1.0
    k = int(look_ms * SR / 1000)
    h = int(hold_ms * SR / 1000)
    g = minimum_filter1d(need, 2 * k + 1)
    g = minimum_filter1d(g, h, origin=-(h // 2) + 1)
    g = uniform_filter1d(g, k, mode="nearest")
    g = np.minimum(g, 1.0)
    return x * g[:, None].astype(np.float32), float(g.min())

def xcorr_lag(a, b, max_ms=40.0, hop=32):
    ea, _ = onset_env(a, hop)
    eb, _ = onset_env(b, hop)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    m = int(max_ms / 1000 * SR / hop)
    best, bl = -1e9, 0
    vals = []
    for l in range(-m, m + 1):
        if l >= 0: v = float((ea[l:] * eb[: n - l]).sum())
        else: v = float((ea[: n + l] * eb[-l:]).sum())
        vals.append(v)
        if v > best: best, bl = v, l
    vals = np.array(vals)
    sharp = float(best / (np.abs(vals).mean() + 1e-9))
    return bl * hop / SR * 1000, sharp  # positive: B is early by that much (A's events come later)

def db(x):
    return 10 * np.log10(np.mean(np.square(x)) + 1e-10)

def render(spec, check=True):
    A = _cache.setdefault(spec["a"]["id"], None) or Track(spec["a"]["id"]); _cache[spec["a"]["id"]] = A
    B = _cache.setdefault(spec["b"]["id"], None) or Track(spec["b"]["id"]); _cache[spec["b"]["id"]] = B
    tA, TA, dA = A.grid(spec["a"]["anchor_bar"], beat_shift=spec["a"].get("beat_shift", 0), fit_bars=spec["a"].get("fit_bars"))
    tB, TB, dB_ = B.grid(spec["b"]["anchor_bar"], beat_shift=spec["b"].get("beat_shift", 0), fit_bars=spec["b"].get("fit_bars"))
    if spec["a"].get("tempo_bpm"): TA = 60 / spec["a"]["tempo_bpm"]
    if spec["b"].get("tempo_bpm"): TB = 60 / spec["b"]["tempo_bpm"]
    bpbA, bpbB = spec["a"].get("beats_per_bar", 4), spec["b"].get("beats_per_bar", 4)
    # the anchor bar sits at ruler position anchor_u; t* is the track time of ruler position 0
    tA -= spec["a"].get("anchor_u", 0) * bpbA * TA
    tB -= spec["b"].get("anchor_u", 0) * bpbB * TB
    ratio = (bpbA * TA) / (bpbB * TB)  # output length per B input length: one ruler bar of B lasts one ruler bar of A
    if spec.get("no_sync"):
        ratio = 1.0
    u0, u1 = spec["clip"]            # the clip on the ruler
    a_end = spec["a"]["until"]       # A is silent from here
    b_from = spec["b"]["from"]       # B's first sound
    bar = bpbA * TA * SR
    n = int(round((u1 - u0) * bar))
    # loudness match: the louder deck comes down to the quieter one
    target = min(A.lufs, B.lufs)
    gA, gB = 10 ** ((target - A.lufs) / 20), 10 ** ((target - B.lufs) / 20)
    gB *= 10 ** (spec["b"].get("trim_db", 0.0) / 20)
    gA *= 10 ** (spec["a"].get("trim_db", 0.0) / 20)
    # deck A in its own time = output time
    a_sig, ua, a_lanes = build_deck(A, tA, TA, spec["a"], u0, a_end + 4)  # + echo/reverb tail
    a_parts = a_lanes.pop("_parts")
    # deck B in its own time, then stretched
    nudge = spec["b"].get("nudge_ms", 0.0) / 1000
    if spec.get("align_pairs") and spec["b"].get("auto_nudge", True) and not spec.get("no_sync"):
        pa, pb = spec["align_pairs"]["a"], spec["align_pairs"]["b"]
        strip = lambda d: {k: v for k, v in d.items() if k != "segments"}
        for _ in range(2):
            ad, _, _ = build_deck(A, tA, TA, strip(spec["a"]), pa[0], pa[1], only=("drums",), raw=True)
            bd, _, _ = build_deck(B, tB + nudge, TB, strip(spec["b"]), pb[0], pb[1], only=("drums",), raw=True)
            bd = stretch(bd, ratio)
            m = min(len(ad), len(bd))
            lag, sharp = xcorr_lag(ad[:m].mean(1), bd[:m].mean(1))
            if abs(lag) < 0.4 or abs(lag) > 30 or sharp < 2.0:
                break
            nudge -= lag / 1000 / ratio  # B early by lag → read B that much earlier in its own time (measured: slope +1)
    # B's deck ruler: in sync it is the clip's ruler; without sync B counts its own bars from start_u on
    start_u = spec["b"].get("start_u", b_from) if spec.get("no_sync") else b_from
    native_per_bar = bar / ratio if not spec.get("no_sync") else bpbB * TB * SR
    b_to = b_from + (u1 - start_u) * bar / (native_per_bar * ratio)
    b_sig, ub, b_lanes = build_deck(B, tB + nudge, TB, spec["b"], b_from, b_to)
    b_parts = b_lanes.pop("_parts")
    b_out = stretch(b_sig, ratio)
    mix = np.zeros((n, 2), dtype=np.float32)
    na = min(len(a_sig), n)
    mix[:na] += a_sig[:na] * gA
    ob = int(round((start_u - u0) * bar))
    nb = min(len(b_out), n - ob)
    mix[ob: ob + nb] += b_out[:nb] * gB
    report = {"nudge_ms": round(nudge * 1000, 2), "clean_vocals": [A.clean_vocals, B.clean_vocals], "leak_free": [A.leak_free, B.leak_free], "tempo_a": round(60 / TA, 3), "tempo_b": round(60 / TB, 3), "stretch_pct": round((1 / ratio - 1) * 100, 2),
              "lufs_a": round(A.lufs, 2), "lufs_b": round(B.lufs, 2), "phase_a_ms": round(dA * 1000, 1), "phase_b_ms": round(dB_ * 1000, 1)}
    if check:
        # beat alignment on the raw drum stems over the bars where both decks' sources have drums
        lo_u, hi_u = spec.get("align_window", [max(b_from, -8), min(a_end, 8)])
        if spec.get("no_sync"): lo_u, hi_u = b_from, b_from + 2
        ad, _, _ = build_deck(A, tA, TA, spec["a"], lo_u, hi_u, only=("drums",), raw=True)
        bd, _, _ = build_deck(B, tB + nudge, TB, spec["b"], lo_u, hi_u, only=("drums",), raw=True)
        bd = stretch(bd, ratio)
        m = min(len(ad), len(bd))
        h = m // 2
        am, bm = ad[:m].mean(1), bd[:m].mean(1)
        if spec.get("align_pairs"):
            # compare bars of A with other bars of B (same length on the ruler), e.g. where both have their full beat
            pa, pb = spec["align_pairs"]["a"], spec["align_pairs"]["b"]
            ad, _, _ = build_deck(A, tA, TA, {k: v for k, v in spec["a"].items() if k != "segments"}, pa[0], pa[1], only=("drums",), raw=True)
            bd, _, _ = build_deck(B, tB + nudge, TB, {k: v for k, v in spec["b"].items() if k != "segments"}, pb[0], pb[1], only=("drums",), raw=True)
            bd = stretch(bd, ratio)
            m = min(len(ad), len(bd)); h = m // 2
            am, bm = ad[:m].mean(1), bd[:m].mean(1)
        report["align"] = {"all": xcorr_lag(am, bm), "first_half": xcorr_lag(am[:h], bm[:h]), "second_half": xcorr_lag(am[h:], bm[h:]),
                           "drums_a_db": round(db(am), 1), "drums_b_db": round(db(bm), 1)}
    # bus: short-term loudness, measured; optional gentle ride
    mono = mix.mean(1)
    win = int(3.0 * SR)
    sos = signal.butter(2, 100, "highpass", fs=SR, output="sos")
    kw = signal.sosfilt(sos, mono)
    p = uniform_filter1d(kw ** 2, win, mode="nearest")
    st = 10 * np.log10(p + 1e-10)
    ref_a = float(np.median(st[: int((spec["first_event"] - u0) * bar)])) if spec.get("first_event") is not None else float(np.median(st))
    report["shortterm_db"] = {"ref": round(ref_a, 2), "min": round(float(st[win // 2: -win // 2].min()), 2), "max": round(float(st[win // 2: -win // 2].max()), 2)}
    if spec.get("bus_ride", False):
        a, b = int((spec["first_event"] - u0) * bar), min(n, int((spec["last_event"] + 1.0 - u0) * bar))
        ref_b = float(np.median(st[b:])) if b < n - win else ref_a
        tgt = np.interp(np.arange(n), [a, b], [ref_a, ref_b])
        corr = np.clip(tgt - st, -spec.get("ride_max_db", 2.5), spec.get("ride_max_db", 2.5))
        corr[:a] = 0; corr[b:] = 0
        corr = uniform_filter1d(corr, int(1.0 * SR), mode="nearest")
        mix *= (10 ** (corr / 20))[:, None].astype(np.float32)
        report["bus_ride_db"] = [round(float(corr.min()), 2), round(float(corr.max()), 2)]
    mix, gr = limiter(mix)
    # meter bridge per bar: every stem after its fader (dB), the decks, the sum and its bands
    prof = []
    a_full = np.zeros((n, 2), dtype=np.float32); a_full[:na] = a_sig[:na] * gA
    b_full = np.zeros((n, 2), dtype=np.float32); b_full[ob: ob + nb] = b_out[:nb] * gB
    mono = mix.mean(1)
    lowb = signal.sosfilt(signal.butter(4, 150, "lowpass", fs=SR, output="sos"), mono)
    highb = signal.sosfilt(signal.butter(4, 4000, "highpass", fs=SR, output="sos"), mono)
    barB = bpbB * TB * SR
    for k in range(int(u1 - u0)):
        i0, i1 = int(k * bar), int((k + 1) * bar)
        uu = u0 + k
        row = {"u": int(uu), "A": round(db(a_full[i0:i1]), 1), "B": round(db(b_full[i0:i1]), 1), "mix": round(db(mix[i0:i1]), 1),
               "low": round(db(lowb[i0:i1]), 1), "high": round(db(highb[i0:i1]), 1)}
        for st in STEMS:
            pa = a_parts[st][i0:i1] * gA if i0 < len(a_parts[st]) else np.zeros(1)
            j0, j1 = int((uu - start_u) * bar / ratio), int((uu + 1 - start_u) * bar / ratio)
            pb = b_parts[st][max(j0, 0):max(j1, 0)] * gB if j1 > 0 else np.zeros(1)
            row["a_" + st[0]] = round(db(pa), 1) if len(pa) else -100.0
            row["b_" + st[0]] = round(db(pb), 1) if len(pb) else -100.0
        prof.append(row)
    report["bars"] = prof
    report["limiter_min_gain_db"] = round(20 * np.log10(max(gr, 1e-9)), 2)
    outdir = os.environ.get("HANDMIX_OUT", f"{ROOT}/hand/out")
    os.makedirs(outdir, exist_ok=True)
    out = f"{outdir}/{spec['id']}"
    sf.write(out + ".wav", mix, SR, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", out + ".wav", "-c:a", "aac", "-b:a", "256k", out + ".m4a"], check=True)
    # lanes for the site (16 points per bar)
    def sample(lanes, u):
        res = {}
        step = max(1, int(bar / 16))
        for k, g in lanes.items():
            res[k] = [round(float(v), 3) for v in g[::step]]
        return {"u0": float(u[0]), "step_bars": step / bar, "gain": res}
    step_b = max(1, int(bpbB * TB * SR / 16))
    meta = {"id": spec["id"], "clip": [u0, u1], "bar_s": bpbA * TA, "seconds": n / SR, "report": report,
            "a": sample(a_lanes, ua), "b": {"u0": float(start_u), "step_bars": step_b * ratio / bar, "gain": {k: [round(float(v), 3) for v in g[::step_b]] for k, g in b_lanes.items()}}}
    json.dump(meta, open(out + ".json", "w"), default=float)
    return report

if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    specs = spec if isinstance(spec, list) else [spec]
    want = [a for a in sys.argv[2:] if not a.startswith("--")]
    for s in specs:
        if want and s["id"] not in want: continue
        r = render(s)
        json.dump(r, open(os.path.join(os.environ.get("HANDMIX_OUT", f"{ROOT}/hand/out"), s["id"] + ".report.json"), "w"), default=float)
        bars = r.pop("bars")
        print(s["id"], json.dumps(r, default=float), flush=True)
        f = lambda v: "  -- " if v < -70 else f"{v:5.0f}"
        print("   u |  A: voc  drm  bas  oth  sum |  B: voc  drm  bas  oth  sum |  mix   low  high")
        for b in bars:
            print(f"{b['u']:4d} |    {f(b['a_v'])}{f(b['a_d'])}{f(b['a_b'])}{f(b['a_o'])}{f(b['A'])} |    {f(b['b_v'])}{f(b['b_d'])}{f(b['b_b'])}{f(b['b_o'])}{f(b['B'])} |{f(b['mix'])} {f(b['low'])} {f(b['high'])}")
