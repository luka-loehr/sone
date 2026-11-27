"""Blind test: transitions between songs nobody has built or heard in the lab, made by the frozen code.

Every ordered pair of the new songs is planned; pairs the planner refuses (no common beat, ...) are
counted; of the rest, in a random order, each song at most once as A and once as B, up to 12 are
rendered. A render that fails the system's own checks is refused and counted, not shown."""
import json, os, random, subprocess, sys
os.chdir(os.path.expanduser("~/djlab/orml-main"))
new = json.load(open("blind_songs.json"))
names = {m["id"]: m["name"] for m in new}
ids = [m["id"] for m in new if os.path.exists(f"data/analysis/beats/{m['id']}.json")]
os.makedirs("out/blind", exist_ok=True)
for f in os.listdir("out/blind"): os.remove(os.path.join("out/blind", f))
pairs = [(a, b) for a in ids for b in ids if a != b]
random.seed(11); random.shuffle(pairs)
plannable, refused_plan = [], []
for a, b in pairs:
    r = subprocess.run(["./target/release/orml", "plan", a, b, "--beats", "data/analysis/beats", "--out", "/dev/null"], capture_output=True, text=True)
    (plannable if r.returncode == 0 else refused_plan).append((a, b))
print(f"{len(ids)} new songs, {len(pairs)} ordered pairs: {len(plannable)} plannable, {len(refused_plan)} refused by the planner (mostly no common beat)")
made, refused_render, used_a, used_b = [], [], set(), set()
for a, b in plannable:
    if len(made) >= 12: break
    if a in used_a or b in used_b: continue
    pid = f"b{len(made) + 1:02d}"
    r = subprocess.run(["./target/release/orml", "mix", a, b, "--stems", os.path.expanduser("~/djlab/stems"), "--beats", "data/analysis/beats", "--out", "out/blind", "--id", pid], capture_output=True, text=True)
    if r.returncode != 0:
        why = [l for l in r.stderr.splitlines() if "drift" in l or "Error" in l]
        refused_render.append((a, b, why[-1] if why else r.stderr[-200:]))
        continue
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"out/blind/{pid}.wav", "-c:a", "aac", "-b:a", "256k", f"out/blind/{pid}.m4a"], check=True)
    rep = json.load(open(f"out/blind/{pid}.report.json")); spec = json.load(open(f"out/blind/{pid}.spec.json"))[0]
    al = rep["align"]
    made.append({"id": pid, "a": a, "b": b, "recipe": spec.get("recipe"), "stretch_pct": rep["stretch_pct"], "drums_ms": al["all"][0], "drift_ms": al["second_half"][0] - al["first_half"][0]})
    used_a.add(a); used_b.add(b)
    print(f"{pid} {names[a][:24]:24} → {names[b][:24]:24} {spec.get('recipe'):18} stretch {rep['stretch_pct']:+.2f} %, drums {al['all'][0]:+.1f} ms, drift {al['second_half'][0]-al['first_half'][0]:+.1f} ms")
for a, b, why in refused_render:
    print(f"refused after rendering: {names[a][:24]} → {names[b][:24]}: {why}")
json.dump({"made": made, "refused_render": refused_render, "refused_plan": len(refused_plan), "plannable": len(plannable)}, open("out/blind_result.json", "w"), indent=1)
