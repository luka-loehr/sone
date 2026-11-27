"""Renders the blind pairs already chosen (out/blindx/done.json) again with the current code, keeping their
ids, and checks each for bars before the one in which nothing but drums and bass plays."""
import json, os, subprocess
os.chdir(os.path.expanduser("~/djlab/orml-main"))
done = json.load(open("out/blindx/done.json"))
keep = []
for d in done:
    pid, a, b = d["id"], d["a"], d["b"]
    r = subprocess.run(["./target/release/orml", "mix", a, b, "--stems", os.path.expanduser("~/djlab/stems"), "--beats", "data/analysis/beats", "--out", "out/blindx", "--id", pid], capture_output=True, text=True)
    if r.returncode != 0:
        print(pid, "now refused:", [l for l in r.stderr.splitlines() if "Error" in l or "drift" in l][-1:]); continue
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"out/blindx/{pid}.wav", "-c:a", "aac", "-b:a", "256k", f"out/blindx/{pid}.m4a"], check=True)
    os.remove(f"out/blindx/{pid}.wav")
    rep = json.load(open(f"out/blindx/{pid}.report.json")); spec = json.load(open(f"out/blindx/{pid}.spec.json"))[0]
    bars = {x["u"]: x for x in rep["bars"]}
    bare = [u for u in range(int(spec["first_event"]), 0) if u in bars and max(bars[u]["a_v"], bars[u]["a_o"], bars[u]["b_v"], bars[u]["b_o"]) < -40]
    al = rep["align"]
    d.update({"recipe": spec.get("recipe"), "stretch_pct": rep["stretch_pct"], "drift_ms": al["second_half"][0] - al["first_half"][0], "bare_bars": bare})
    keep.append(d)
    print(pid, spec.get("recipe"), f"drift {d['drift_ms']:+.1f} ms", "bars of only drums and bass:", bare or "none")
json.dump(keep, open("out/blindx/done.json", "w"), indent=1)
