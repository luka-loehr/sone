"""Picks new pairs of songs that share a beat (no stretch beyond 2.5 %, keys at most one step apart),
none of them a pair the owner has heard; plans and renders them with the binary; checks every render."""
import json, glob, os, random, subprocess, statistics, sys
os.chdir(os.path.expanduser("~/djlab/orml-main"))
tables = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob("data/analysis/beats/*.json")}
heard = {(s["a"]["id"], s["b"]["id"]) for v in glob.glob("data/versions/*.json") for s in json.load(open(v))}
def camelot(c):
    return (int(c[:-1]), c[-1]) if c and c[:-1].isdigit() else None
def key_dist(a, b):
    ca, cb = camelot(a["camelot"]), camelot(b["camelot"])
    if not ca or not cb: return 9
    d = min((ca[0] - cb[0]) % 12, (cb[0] - ca[0]) % 12)
    return d + (0 if ca[1] == cb[1] else (0 if d == 0 else 1))
def stretch(a, b):
    best = 99
    for r in (1, 2, 0.5):
        best = min(best, abs(a["bpm"] / (b["bpm"] * r) - 1) * 100)
    return best
# The pairs are fixed once chosen (the owner's notes refer to them): newpairs.json, else choose 12.
if os.path.exists("newpairs.json"):
    pick = [(p["a"], p["b"]) for p in json.load(open("newpairs.json"))]
else:
    cands = [(x, y) for x in tables for y in tables if x != y and (x, y) not in heard
             and stretch(tables[x], tables[y]) <= 2.5 and key_dist(tables[x], tables[y]) <= 1 and len(tables[x]["bars"]) > 40 and len(tables[y]["bars"]) > 40]
    random.seed(7)
    random.shuffle(cands)
    pick, used_a, used_b = [], set(), set()
    for x, y in cands:
        if x in used_a or y in used_b: continue
        pick.append((x, y)); used_a.add(x); used_b.add(y)
        if len(pick) == 12: break
names = json.load(open("data/round0/data.json"))["tracks"] if os.path.exists("data/round0/data.json") else {}
os.makedirs("out/new", exist_ok=True)
for f in glob.glob("out/new/*"): os.remove(f)
rows = []
for i, (x, y) in enumerate(pick, 1):
    pid = f"n{i:02d}"
    r = subprocess.run(["./target/release/orml", "mix", x, y, "--stems", os.path.expanduser("~/djlab/stems"), "--beats", "data/analysis/beats", "--out", "out/new", "--id", pid], capture_output=True, text=True)
    if r.returncode != 0:
        why = [l for l in r.stderr.splitlines() if "Error" in l or "no transition" in l]
        rows.append((pid, x, y, "FAILED", (why[-1] if why else r.stderr[-300:]))); continue
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"out/new/{pid}.wav", "-c:a", "aac", "-b:a", "256k", f"out/new/{pid}.m4a"], check=True)
    spec = json.load(open(f"out/new/{pid}.spec.json"))[0]
    rep = json.load(open(f"out/new/{pid}.report.json"))
    bars = {b["u"]: b for b in rep["bars"]}
    first = int(spec["first_event"])
    problems = []
    al = rep.get("align") or {}
    lag = al.get("all", [0])[0]
    if abs(lag) > 1.0: problems.append(f"drums {lag:+.1f} ms apart")
    drift = al.get("second_half", [0])[0] - al.get("first_half", [0])[0]
    if abs(drift) > 12.0: problems.append(f"drums drift {drift:+.1f} ms across the blend")
    # By design A's beat is filtered up (A's beat under B's intro) in the bar before the one.
    design_bar = -1 if spec.get("recipe") == "outro_under_intro" else None
    beat = [max(bars[u]["a_d"], bars[u]["b_d"]) for u in range(first, 0) if u in bars]
    if beat:
        med = statistics.median(beat)
        holes = [u for u in range(first, 0) if u in bars and u != design_bar and max(bars[u]["a_d"], bars[u]["b_d"]) < med - 10]
        if holes: problems.append(f"beat hole in bar(s) {holes}")
    voices = [u for u in range(first, int(spec["last_event"]) + 2) if u in bars and bars[u]["a_v"] > -30 and bars[u]["b_v"] > -30]
    if len(voices) > 1: problems.append(f"two voices in bars {voices}")
    melodies = [u for u in range(first, int(spec["last_event"]) + 2) if u in bars and bars[u]["a_o"] > -30 and bars[u]["b_o"] > -30 and bars[u]["a_v"] > -30]
    dip = rep["shortterm_db"]["min"] - rep["shortterm_db"]["ref"]
    if dip < -8: problems.append(f"level dips {dip:.1f} dB")
    if rep["limiter_min_gain_db"] < -3: problems.append(f"limiter {rep['limiter_min_gain_db']} dB")
    rows.append((pid, x, y, spec.get("recipe"), problems, rep["stretch_pct"], lag, round(dip, 1)))
for row in rows:
    pid, x, y = row[:3]
    nm = lambda i: names.get(i, {}).get("name", i)[:24]
    if row[3] == "FAILED":
        print(f"{pid} {nm(x)} → {nm(y)}: FAILED {row[4]}"); continue
    rec, problems, st, lag, dip = row[3:]
    al = json.load(open(f"out/new/{pid}.report.json")).get("align") or {}
    halves = f"{al.get('first_half',[0])[0]:+.1f}/{al.get('second_half',[0])[0]:+.1f}"
    print(f"{pid} {nm(x):24} → {nm(y):24} {rec:18} stretch {st:+.2f} %, drums {lag:+.1f} ms (halves {halves}), dip {dip} dB  {'OK' if not problems else '; '.join(problems)}")
json.dump([{"id": r[0], "a": r[1], "b": r[2], "result": r[3] if r[3] == "FAILED" else "ok", "problems": (r[4] if r[3] != "FAILED" else [r[4]])} for r in rows], open("out/new_checks.json", "w"), indent=1)
