"""Renders the pairs of an earlier blind round again with the current code, keeping their ids, into out/<round>/:
the x pairs of out/blindx/done.json and the b pairs of out/blind_result.json. Usage: round_rerender.py r4"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
os.chdir(os.path.expanduser("~/djlab/orml-main"))
out = f"out/{sys.argv[1]}"
os.makedirs(out, exist_ok=True)
pairs = [(d["id"], d["a"], d["b"]) for d in json.load(open("out/blindx/done.json"))]
pairs += [(d["id"], d["a"], d["b"]) for d in json.load(open("out/blind_result.json"))["made"]]

def one(p):
    pid, a, b = p
    r = subprocess.run(["./target/release/orml", "mix", a, b, "--stems", os.path.expanduser("~/djlab/stems"), "--beats", "data/analysis/beats", "--out", out, "--id", pid], capture_output=True, text=True)
    if r.returncode != 0:
        return {"id": pid, "a": a, "b": b, "refused": ([l for l in r.stderr.splitlines() if "Error" in l] or r.stderr.splitlines()[-1:])[-1]}
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out}/{pid}.wav", "-c:a", "aac", "-b:a", "256k", f"{out}/{pid}.m4a"], check=True)
    os.remove(f"{out}/{pid}.wav")
    rep = json.load(open(f"{out}/{pid}.report.json")); spec = json.load(open(f"{out}/{pid}.spec.json"))[0]
    al = rep["align"]
    return {"id": pid, "a": a, "b": b, "recipe": spec.get("recipe"), "stretch_pct": rep["stretch_pct"], "drift_ms": al["second_half"][0] - al["first_half"][0]}

with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(one, pairs))
json.dump(res, open(f"{out}/done.json", "w"), indent=1)
for d in res:
    print(d["id"], d.get("refused") or f"{d['recipe']:<18} stretch {d['stretch_pct']:+.2f} %  drift {d['drift_ms']:+.1f} ms")
