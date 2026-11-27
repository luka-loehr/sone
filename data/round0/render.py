#!/usr/bin/env python3
"""Analyse every track with both engines, chain the songs like a DJ set
(each song -> its best-matching next song), render 10 styles x 2 engines per
transition, encode to AAC (m4a) and write lab/data.json."""
import json, os, random, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(ROOT, "tracks")
LAB = os.path.join(ROOT, "lab")
CLIPS = os.path.join(LAB, "clips")
COVERS = os.path.join(LAB, "covers")
ENGINES = ["mixxx", "sp"]
# only what club/radio DJs actually do
STYLES = ["bass_swap", "filter_blend", "echo_out", "loop_roll", "reverb_wash"]
N_TRANSITIONS = 20
os.makedirs(CLIPS, exist_ok=True)
os.makedirs(COVERS, exist_ok=True)
JOBS = int(os.environ.get("JOBS", "8"))

meta = json.load(open(os.path.join(T, "meta.json")))
tracks = []
for m in meta:
    tid = m["uri"].split(":")[-1]
    if os.path.exists(os.path.join(T, tid + ".f32")):
        tracks.append({"id": tid, "name": m["name"], "artists": m["artists"], "album": m.get("album"), "image": m.get("image"), "year": m.get("year")})
print(len(tracks), "tracks")

def run(args):
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(" ".join(args) + "\n" + out.stderr[-500:])
    return json.loads(out.stdout.strip().splitlines()[-1])

# 1. analysis (cached)
cache_path = os.path.join(ROOT, "analysis.json")
analysis = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
def analyse(job):
    tid, eng = job
    key = f"{eng}:{tid}"
    if key in analysis:
        return
    analysis[key] = run([os.path.join(ROOT, "build", f"djmix-{eng}"), "analyze", os.path.join(T, tid + ".f32")])["track"]
with ThreadPoolExecutor(JOBS) as ex:
    list(ex.map(analyse, [(t["id"], e) for t in tracks for e in ENGINES]))
json.dump(analysis, open(cache_path, "w"), indent=1)
for t in tracks:
    t["analysis"] = {e: analysis[f"{e}:{t['id']}"] for e in ENGINES}

# 2. covers
def cover(t):
    path = os.path.join(COVERS, t["id"] + ".jpg")
    if t["image"] and not os.path.exists(path):
        try:
            urllib.request.urlretrieve(t["image"].replace("ab67616d0000b273", "ab67616d00001e02"), path)
        except Exception as e:
            print("cover", t["name"], e)
with ThreadPoolExecutor(8) as ex:
    list(ex.map(cover, tracks))

# 3. the set: greedy chain on tempo (half/double aware) and Camelot distance,
#    using both engines' readings (their mean). Every song gets one next song.
CAM = ["11B","6B","1B","8B","3B","10B","5B","12B","7B","2B","9B","4B","8A","3A","10A","5A","12A","7A","2A","9A","4A","11A","6A","1A"]
def camelot(t):
    k = t["analysis"]["sp"]["key"].split(" / ")[-1]
    return int(k[:-1]), k[-1]
def bpm(t):
    return (t["analysis"]["sp"]["bpm"] + t["analysis"]["mixxx"]["bpm"]) / 2
def cost(a, b):
    ba, bb = bpm(a), bpm(b)
    d = min(abs(bb * m - ba) / ba for m in (0.5, 1, 2))
    (na, la), (nb, lb) = camelot(a), camelot(b)
    step = min((na - nb) % 12, (nb - na) % 12)
    kd = 0 if (step == 0 and la == lb) else 1 if (step == 1 and la == lb) or (step == 0) else 2 + step
    return d * 100 + kd * 1.5
def bt_bpm(t):  # Beat This! tempo (the one the mix uses)
    ts = [float(l.split()[0]) for l in open(os.path.join(T, t["id"] + ".beats"))]
    d = sorted(y - x for x, y in zip(ts, ts[1:]))
    return 60 / d[len(d) // 2]
for t in tracks:
    t["bt_bpm"] = bt_bpm(t)
def tempo_gap(a, b):
    ba, bb = a["bt_bpm"], b["bt_bpm"]
    return min(abs(bb * m - ba) / ba for m in (0.5, 1, 2))
def cost2(a, b):
    (na, la), (nb, lb) = camelot(a), camelot(b)
    step = min((na - nb) % 12, (nb - na) % 12)
    kd = 0 if (step == 0 and la == lb) else 1 if (step == 1 and la == lb) or (step == 0) else 2 + step
    return tempo_gap(a, b) * 100 + kd * 1.5
# the N best pairs a DJ would actually mix: tempo within 6 % (after x2 / /2),
# compatible keys first; every song leads at most once, no A->B and B->A twice
cands = sorted(((cost2(a, b), a, b) for a in tracks for b in tracks if a["id"] != b["id"] and tempo_gap(a, b) <= 0.06), key=lambda c: c[0])
# probe the mix itself (bass swap, Superpowered): the local tempo at the mix point
# decides; keep only pairs that sync with at most 6 % stretch
def probe(c):
    _, a, b = c
    tmp = os.path.join(ROOT, "out", f"probe-{a['id'][:8]}-{b['id'][:8]}.f32")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    try:
        p = run([os.path.join(ROOT, "build", "djmix-sp"), "mix", os.path.join(T, a["id"] + ".f32"), os.path.join(T, b["id"] + ".f32"), "bass_swap", tmp])
    except Exception:
        return False
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return p["synced"] and abs(p["stretch_pct"]) <= 6.0
cands = cands[:300]
with ThreadPoolExecutor(JOBS) as ex:
    ok = list(ex.map(probe, cands))
pairs, used_a, used_pair = [], set(), set()
for (c, a, b), good in zip(cands, ok):
    key = frozenset((a["id"], b["id"]))
    if not good or a["id"] in used_a or key in used_pair:
        continue
    pairs.append((a, b)); used_a.add(a["id"]); used_pair.add(key)
    if len(pairs) == N_TRANSITIONS:
        break
print(len(pairs), "pairs pass the probe")

# 4. render
jobs = [(i, a, b, s, e) for i, (a, b) in enumerate(pairs) for s in STYLES for e in ENGINES]
results = {}
def render(job):
    i, a, b, s, e = job
    name = f"{a['id'][:8]}-{b['id'][:8]}-{s}-{e}"
    m4a = os.path.join(CLIPS, name + ".m4a")
    js = os.path.join(CLIPS, name + ".json")
    if os.path.exists(m4a) and os.path.exists(js):
        results[name] = json.load(open(js)); return
    raw = os.path.join(ROOT, "out", name + ".f32")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    plan = run([os.path.join(ROOT, "build", f"djmix-{e}"), "mix", os.path.join(T, a["id"] + ".f32"), os.path.join(T, b["id"] + ".f32"), s, raw])
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", "44100", "-ac", "2", "-i", raw, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", m4a], check=True)
    os.remove(raw)
    json.dump(plan, open(js, "w"))
    results[name] = plan
done = 0
with ThreadPoolExecutor(JOBS) as ex:
    for _ in ex.map(render, jobs):
        done += 1
        if done % 40 == 0:
            print(f"{done}/{len(jobs)}", flush=True)

data = {
    "engines": {"mixxx": "Mixxx engine", "sp": "Superpowered SDK"},
    "styles": STYLES,
    "tracks": {t["id"]: {k: t[k] for k in ("name", "artists", "album", "year", "analysis")} for t in tracks},
    "transitions": [
        {"index": i, "a": a["id"], "b": b["id"],
         "clips": {s: {e: {"file": f"clips/{a['id'][:8]}-{b['id'][:8]}-{s}-{e}.m4a", **results[f"{a['id'][:8]}-{b['id'][:8]}-{s}-{e}"]} for e in ENGINES} for s in STYLES}}
        for i, (a, b) in enumerate(pairs)
    ],
}
json.dump(data, open(os.path.join(LAB, "data.json"), "w"))
print("wrote lab/data.json with", len(pairs), "transitions,", len(jobs), "clips")
