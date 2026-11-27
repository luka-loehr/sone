#!/usr/bin/env python3
"""Gemini compares several versions of one transition. compare10.py <rounds> <id>... → ranking.
Each round sends all clips (cut to the transition, order shuffled, neutral labels) in one request."""
import base64, json, os, random, subprocess, sys, tempfile, time, urllib.request, collections
MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")
KEY = os.environ["GEMINI_API_KEY"]
HERE = os.path.dirname(os.path.abspath(__file__))
ONE = 12 * 2.7698   # the one in the full clip (s)
CUT0, LEN = ONE - 20, 36
PROMPT = """You are an experienced open-format club and radio DJ with a very good ear. You hear {n} versions of the SAME transition, labelled {labels}. Song A is a slow hip-hop ballad with a sung hook (about 87 BPM). Song B is a house track (130 BPM). The two songs are in clashing keys and their tempos are in a 2:3 ratio, so they cannot simply be layered. In every clip song B takes over at about {one:.0f} seconds.
Listen to each version completely and judge it as a listener of a live radio mix would: does it feel like one deliberate, smooth, musical DJ transition (good), or like a cut, a train wreck, a muddy or dissonant overlap, an awkward hole, or an energy collapse (bad)?
For each version give: what you hear between {a:.0f} and {b:.0f} s (one sentence), the worst defect (or "none"), and scores 1-10 for smooth (flows without abrupt events), musical (rhythm and harmony of what overlaps fit), energy (holds and lands), overall.
Then rank all versions from best to worst.
Answer ONLY JSON: {{"versions": {{"<label>": {{"hears": "...", "defect": "...", "smooth": n, "musical": n, "energy": n, "overall": n}}}}, "ranking": ["<best label>", "..."], "why_best": "...", "how_to_improve_best": "..."}}"""
def cut(i):
    out = tempfile.mktemp(suffix=".mp3")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{CUT0:.2f}", "-t", str(LEN), "-i", f"{HERE}/out/{i}.m4a", "-ac", "2", "-ar", "44100", "-b:a", "128k", out], check=True)
    return base64.b64encode(open(out, "rb").read()).decode()
def ask(parts):
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
    err = None
    for k in range(3):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(json.load(r)["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            err = e; time.sleep(3 + 4 * k)
    raise err
rounds = int(sys.argv[1]); ids = sys.argv[2:]
audio = {i: cut(i) for i in ids}
score = collections.defaultdict(list); ranks = collections.defaultdict(list); notes = collections.defaultdict(list); tips = []
for r in range(rounds):
    order = ids[:]; random.Random(100 + r).shuffle(order)
    labels = [chr(ord("P") + k) for k in range(len(order))]
    parts = []
    for lab, i in zip(labels, order):
        parts += [{"text": f"Version {lab}:"}, {"inline_data": {"mime_type": "audio/mp3", "data": audio[i]}}]
    parts.append({"text": PROMPT.format(n=len(order), labels=", ".join(labels), one=ONE - CUT0, a=ONE - CUT0 - 12, b=ONE - CUT0 + 8)})
    ans = ask(parts)
    m = dict(zip(labels, order))
    for lab, v in ans.get("versions", {}).items():
        if lab in m:
            score[m[lab]].append([v.get(k, 0) for k in ("smooth", "musical", "energy", "overall")])
            notes[m[lab]].append((v.get("hears"), v.get("defect")))
    for pos, lab in enumerate(ans.get("ranking", [])):
        if lab in m: ranks[m[lab]].append(pos + 1)
    tips.append((m.get((ans.get("ranking") or ["?"])[0]), ans.get("why_best"), ans.get("how_to_improve_best")))
    print(f"round {r+1}: ranking", [m.get(l, l) for l in ans.get("ranking", [])], flush=True)
print("\nid     smooth musical energy overall | mean rank (ranks)")
res = {}
for i in ids:
    s = score[i]; mean = [sum(x[k] for x in s) / max(len(s), 1) for k in range(4)]
    mr = sum(ranks[i]) / max(len(ranks[i]), 1)
    res[i] = {"scores": mean, "mean_rank": mr, "ranks": ranks[i], "notes": notes[i]}
    print(f"{i:6s} {mean[0]:5.1f} {mean[1]:7.1f} {mean[2]:6.1f} {mean[3]:7.1f} | {mr:.2f} {ranks[i]}")
for i in ids:
    print("\n" + i)
    for h, d in notes[i]: print("   hears:", h, "| defect:", d)
print()
for t in tips: print("best:", t[0], "| why:", t[1], "| improve:", t[2])
json.dump({"results": res, "tips": tips}, open(f"{HERE}/compare10_{'_'.join(ids)[:40]}.json", "w"), indent=1, ensure_ascii=False)
