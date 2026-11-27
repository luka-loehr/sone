#!/usr/bin/env python3
"""Gemini listens to every transition on the site (2 runs each by default) → listen.json and a table."""
import json, os, sys, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import listen
site = json.load(open(f"{HERE}/../djlab/lab/hand.json"))["transitions"]
specs = {s["id"]: s for s in json.load(open(f"{HERE}/specs.json"))}
want = [a for a in sys.argv[1:] if not a.isdigit()]
runs = int([a for a in sys.argv[1:] if a.isdigit()][0]) if [a for a in sys.argv[1:] if a.isdigit()] else 2
def one(t):
    if want and t["id"] not in want: return None
    c0 = t["clip"][0]
    sw = specs[t["id"]].get("swap_u", 0)
    start, mid, end = (t["first_event"] - c0) * t["bar_s"], (sw - c0) * t["bar_s"], (t["last_event"] - c0) * t["bar_s"]
    r = listen.listen(f"{HERE}/out/{t['id']}.m4a", start, mid, max(end, mid + 1), runs)
    r["id"] = t["id"]; r["times"] = [round(start, 1), round(mid, 1), round(end, 1)]
    return r
with cf.ThreadPoolExecutor(5) as ex:
    res = [r for r in ex.map(one, site) if r]
path = f"{HERE}/listen.json"
old = json.load(open(path)) if os.path.exists(path) else {}
for r in res: old[r["id"]] = r
json.dump(old, open(path, "w"), indent=1, ensure_ascii=False)
flags = ["kicks_flam", "two_vocals", "key_clash", "level_problem", "abrupt", "artifacts"]
print("id    flam 2voc key  lvl  abr  art | energy entry pro")
for r in res:
    print(f"{r['id']}   " + "  ".join(f"{r[f]}/{runs}" for f in flags) + f" |  {r['energy']:4.1f}  {r['b_entry']:4.1f} {r['pro']:4.1f}   {r['times']}")
for r in res:
    print("\n" + r["id"])
    for n, w, f in zip(r["notes"], r["what"], r["fix"]):
        print("  flags:", n); print("  hears:", w); print("  fix:  ", f)
