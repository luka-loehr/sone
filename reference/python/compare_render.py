#!/usr/bin/env python3
"""Compares the Rust renderer's dumps with handmix_dump.py's, spec by spec.

compare_render.py <python dump dir> <rust dump dir> [<rust auto-nudge dir>] [ids...]

Per spec: signal-to-error ratio (dB) of every stage (Rust against Python, over the
common length; the lengths are printed when they differ), the report's numbers
(tempo, stretch, nudge, loudness, phase, beat lag, per-bar levels) side by side,
and with a third directory the automatically found nudge of a Rust run without
--nudge-from. Prints one JSON line per spec and a summary.
"""
import sys, os, json
import numpy as np

STAGES = ("deck_a", "deck_b", "deck_b_stretched", "mix_unlimited", "mix")

def load(path):
    return np.fromfile(path, dtype="<f4").reshape(-1, 2).astype(np.float64)

def ser(ref, test):
    n = min(len(ref), len(test))
    r, t = ref[:n], test[:n]
    e = ((r - t) ** 2).sum()
    s = (r ** 2).sum()
    if e == 0:
        return float("inf")
    return float(10 * np.log10(s / e))

def main():
    args = sys.argv[1:]
    py, rs = args[0], args[1]
    auto = args[2] if len(args) > 2 and os.path.isdir(args[2]) else None
    ids = [a for a in args[(3 if auto else 2):]]
    if not ids:
        ids = sorted(f[:-len(".meta.json")] for f in os.listdir(py) if f.endswith(".meta.json"))
    rows = []
    for sid in ids:
        row = {"id": sid}
        for st in STAGES:
            a, b = load(f"{py}/{sid}.{st}.f32"), load(f"{rs}/{sid}.{st}.f32")
            row[st] = round(ser(a, b), 1)
            if len(a) != len(b):
                row[st + "_len"] = [len(a), len(b)]
        rp = json.load(open(f"{py}/{sid}.report.json"))
        rr = json.load(open(f"{rs}/{sid}.report.json"))
        mp = json.load(open(f"{py}/{sid}.meta.json"))
        mr = json.load(open(f"{rs}/{sid}.meta.json"))
        row["tempo_diff"] = [round(rr["tempo_a"] - rp["tempo_a"], 4), round(rr["tempo_b"] - rp["tempo_b"], 4)]
        row["grid_diff_us"] = [round((mr["grid_a"][0] - mp["grid_a"][0]) * 1e6, 3), round((mr["grid_b"][0] - mp["grid_b"][0]) * 1e6, 3)]
        row["beat_diff_ns"] = [round((mr["grid_a"][1] - mp["grid_a"][1]) * 1e9, 3), round((mr["grid_b"][1] - mp["grid_b"][1]) * 1e9, 3)]
        row["stretch_pct"] = [rp["stretch_pct"], rr["stretch_pct"]]
        row["lufs_diff"] = [round(rr["lufs_a"] - rp["lufs_a"], 3), round(rr["lufs_b"] - rp["lufs_b"], 3)]
        row["lufs_exact_diff"] = [round(mr["lufs"][0] - mp["lufs"][0], 4), round(mr["lufs"][1] - mp["lufs"][1], 4)]
        row["phase_ms"] = [[rp["phase_a_ms"], rr["phase_a_ms"]], [rp["phase_b_ms"], rr["phase_b_ms"]]]
        row["lag_ms"] = [rp["align"]["all"][0], rr["align"]["all"][0]]
        row["limiter_db"] = [rp["limiter_min_gain_db"], rr["limiter_min_gain_db"]]
        row["shortterm_diff"] = [round(rr["shortterm_db"][k] - rp["shortterm_db"][k], 2) for k in ("ref", "min", "max")]
        worst, where = 0.0, None
        for bp, br in zip(rp["bars"], rr["bars"]):
            for k, v in bp.items():
                if k == "u":
                    continue
                d = abs(br[k] - v)
                if d > worst:
                    worst, where = d, (bp["u"], k, v, br[k])
        row["bars_max_diff_db"] = round(worst, 2)
        row["bars_worst"] = where
        row["bars_len"] = [len(rp["bars"]), len(rr["bars"])]
        row["nudge_ms_py"] = rp["nudge_ms"]
        if auto:
            ra = json.load(open(f"{auto}/{sid}.report.json"))
            row["nudge_ms_rs_auto"] = ra["nudge_ms"]
            row["nudge_diff_ms"] = round(ra["nudge_ms"] - rp["nudge_ms"], 2)
            row["lag_ms_rs_auto"] = ra["align"]["all"][0]
        rows.append(row)
        print(json.dumps(row), flush=True)
    print("min SER (dB):", {st: min(r[st] for r in rows) for st in STAGES})
    print("max bar diff (dB):", max(r["bars_max_diff_db"] for r in rows))

if __name__ == "__main__":
    main()
