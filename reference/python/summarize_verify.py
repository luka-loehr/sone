#!/usr/bin/env python3
"""One line per spec from verify_version.sh's results.

summarize_verify.py <version> [<version> ...]"""
import sys, json

for v in sys.argv[1:]:
    d = f"/tmp/orml-verify/{v}"
    same = [json.loads(l) for l in open(f"{d}/same_nudge.jsonl")]
    inj = {r["id"]: r for r in map(json.loads, open(f"{d}/injected.jsonl"))}
    for r in same:
        sid = r["id"]
        i = inj[sid]
        a = json.load(open(f"{d}/auto/{sid}.report.json"))
        p = json.load(open(f"/tmp/orml-py/{v}/{sid}.report.json"))
        print(v, sid, json.dumps({
            "deck_a": r["deck_a"], "deck_b": r["deck_b"], "b_stretched": r["deck_b_stretched"], "mix": r["mix"],
            "mix_injected": i["mix"], "bars_injected": i["bars_max_diff_db"], "shortterm_injected": i["shortterm_diff"],
            "bars_same_nudge": r["bars_max_diff_db"], "bars_worst": r["bars_worst"],
            "tempo": r["tempo_diff"], "lufs": r["lufs_exact_diff"], "phase": r["phase_ms"], "stretch_pct": r["stretch_pct"],
            "nudge": [p["nudge_ms"], a["nudge_ms"]], "lag": [round(p["align"]["all"][0], 2), round(a["align"]["all"][0], 2)],
            "lag_same_nudge": r["lag_ms"]}))
