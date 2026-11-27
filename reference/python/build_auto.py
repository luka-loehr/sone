#!/usr/bin/env python3
"""The planner's own transitions on the lab site: lab/auto.json and lab/auto/<id>.m4a,
from what `orml mix` wrote (<id>.m4a, <id>.spec.json, <id>.report.json, <id>.lanes.json)."""
import json, os, shutil, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "djlab", "lab")
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/auto_out"
tracks = json.load(open(f"{LAB}/data.json"))["tracks"]
keys = json.load(open(f"{HERE}/keys.json"))
os.makedirs(f"{LAB}/auto", exist_ok=True)
out = []
for p in sorted(glob.glob(f"{SRC}/p*.spec.json")):
    s = json.load(open(p))[0]
    i = s["id"]
    m = json.load(open(f"{SRC}/{i}.lanes.json"))
    r = json.load(open(f"{SRC}/{i}.report.json"))
    shutil.copy(f"{SRC}/{i}.m4a", f"{LAB}/auto/{i}.m4a")
    def song(d, bpm, lanes):
        t = tracks[d["id"]]; k = keys.get(d["id"], {})
        return {"id": d["id"], "name": t["name"], "artists": ", ".join(t["artists"]), "bpm": round(bpm, 1), "key": f"{k.get('key', '?')} ({k.get('camelot', '?')})",
                "u0": lanes["u0"], "step_bars": lanes["step_bars"], "gain": lanes["gain"]}
    lag = (r.get("align") or {}).get("all", [None])[0]
    st = r["stretch_pct"]
    out.append({
        "id": i, "answer": "", "technique": s.get("technique", ""), "why": s.get("why", ""), "steps": s.get("steps", []), "fx": s.get("fx", []), "phases": [],
        "a": song(s["a"], r["tempo_a"], m["a"]), "b": song(s["b"], r["tempo_b"], m["b"]),
        "clip": m["clip"], "seconds": m["seconds"], "bar_s": m["bar_s"],
        "first_event": s["first_event"], "last_event": s["last_event"],
        "overlap_bars": round(s["last_event"] - s["first_event"], 2), "overlap_s": (s["last_event"] - s["first_event"]) * m["bar_s"],
        "stretch_txt": f"{st:+.1f} % (Key-Lock, Tonhöhe bleibt)" if abs(st) > 0.05 else "unverändert",
        "align_txt": "—" if lag is None else f"{abs(lag):.1f} ms",
        "key_txt": s.get("key_txt", ""),
        "audio": f"auto/{i}.m4a?v={int(os.path.getmtime(p))}",
    })
intro = """<b>Dieselben 13 Paare, aber nicht von Hand gebaut:</b> geplant vom Regel-Planer und gerendert vom Rust-Renderer, ohne dass ich etwas nachjustiert habe.
<ul><li>Der Planer kennt nur deine Regeln und die Analyse der Songs. Er wählt die Bauweise, die Eins in beiden Songs und alle Regler selbst.</li>
<li>Bei 5 von 13 Paaren setzt er die Eins in Song A an dieselbe Stelle wie die Hand-Fassung, bei 8 woanders. Ob das besser oder schlechter ist, kannst nur du hören.</li>
<li>Jede Notiz hier wird ein Trainingsbeispiel. Bewertung und Tasten wie auf der anderen Seite.</li></ul>
<p><a href="index.html" style="color:#ffd23d">→ zu den Hand-Fassungen</a></p>"""
json.dump({"intro": intro, "transitions": out}, open(f"{LAB}/auto.json", "w"))
html = open(f"{LAB}/hand.html").read()
assert html.count('"hand.json?"') == 1 and html.count('"hand:"') == 2, (html.count('"hand.json?"'), html.count('"hand:"'))
html = html.replace('"hand.json?"', '"auto.json?"').replace('"hand:"', '"auto:"')
html = html.replace("<title>Orin DJ Lab · Handgebaut</title>", "<title>Orin DJ Lab · Planer</title>").replace("· 10 handgebaute Übergänge", "· 13 Übergänge vom Planer")
open(f"{LAB}/auto.html", "w").write(html)
print(len(out), "transitions on the site")
