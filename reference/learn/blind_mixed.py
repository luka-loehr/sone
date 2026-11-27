"""Blind test, mixed: each new song that is ready, once as song A and once as song B, with a song of the
lab's first 40 that shares its beat (chosen at random among those the planner accepts). Frozen code."""
import json, os, random, subprocess, glob
os.chdir(os.path.expanduser("~/djlab/orml-main"))
new = {m["id"]: m["name"] for m in json.load(open("blind_songs.json"))}
old_names = {k: v["name"] for k, v in json.load(open("data/round0/data.json"))["tracks"].items()}
ready = [i for i in new if os.path.exists(f"data/analysis/beats/{i}.json")]
old = [os.path.basename(p)[:-5] for p in glob.glob("data/analysis/beats/*.json") if os.path.basename(p)[:-5] not in new]
os.makedirs("out/blindx", exist_ok=True)
done = json.load(open("out/blindx/done.json")) if os.path.exists("out/blindx/done.json") else []
have = {(d["a"], d["b"]) for d in done} | {(d["new"], d["side"]) for d in done}
random.seed(13)
def name(i): return new.get(i) or old_names.get(i, i)
for n in ready:
    for side in ("a", "b"):
        if (n, side) in have:
            continue
        partners = old[:]; random.shuffle(partners)
        for o in partners:
            a, b = (n, o) if side == "a" else (o, n)
            if subprocess.run(["./target/release/orml", "plan", a, b, "--beats", "data/analysis/beats", "--out", "/dev/null"], capture_output=True).returncode != 0:
                continue
            pid = f"x{len(done) + 1:02d}"
            r = subprocess.run(["./target/release/orml", "mix", a, b, "--stems", os.path.expanduser("~/djlab/stems"), "--beats", "data/analysis/beats", "--out", "out/blindx", "--id", pid], capture_output=True, text=True)
            if r.returncode != 0:
                why = [l for l in r.stderr.splitlines() if "drift" in l or "Error" in l]
                print(f"refused after rendering: {name(a)[:28]} → {name(b)[:28]}: {why[-1] if why else ''}", flush=True)
                continue
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"out/blindx/{pid}.wav", "-c:a", "aac", "-b:a", "256k", f"out/blindx/{pid}.m4a"], check=True)
            os.remove(f"out/blindx/{pid}.wav")
            rep = json.load(open(f"out/blindx/{pid}.report.json")); spec = json.load(open(f"out/blindx/{pid}.spec.json"))[0]
            al = rep["align"]
            done.append({"id": pid, "a": a, "b": b, "new": n, "side": side, "recipe": spec.get("recipe"), "stretch_pct": rep["stretch_pct"], "drift_ms": al["second_half"][0] - al["first_half"][0]})
            json.dump(done, open("out/blindx/done.json", "w"), indent=1)
            print(f"{pid} {name(a)[:28]:28} → {name(b)[:28]:28} {spec.get('recipe')}, stretch {rep['stretch_pct']:+.2f} %, drift {done[-1]['drift_ms']:+.1f} ms", flush=True)
            break
        else:
            print(f"no partner for {name(n)[:28]} as song {side.upper()}", flush=True)
