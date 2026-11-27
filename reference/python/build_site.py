#!/usr/bin/env python3
"""Collects the rendered transitions into the lab site: lab/hand.json and lab/hand/<id>.m4a."""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "djlab", "lab")
specs = json.load(open(f"{HERE}/specs.json"))
tracks = json.load(open(f"{LAB}/data.json"))["tracks"]
keys = json.load(open(f"{HERE}/keys.json"))
os.makedirs(f"{LAB}/hand", exist_ok=True)
subprocess.run(["rsync", "-aq", "--include=h*.m4a", "--include=h*.json", "--exclude=*", "atlas:djlab/hand/out/", f"{HERE}/out/"], check=True)
out = []
for s in specs:
    p = f"{HERE}/out/{s['id']}.json"
    if not os.path.exists(p) or not s.get("technique"):
        continue
    m = json.load(open(p))
    shutil.copy(f"{HERE}/out/{s['id']}.m4a", f"{LAB}/hand/{s['id']}.m4a")
    r = m["report"]
    def song(d, bpm):
        t = tracks[d["id"]]; k = keys.get(d["id"], {})
        return {"id": d["id"], "name": t["name"], "artists": ", ".join(t["artists"]), "bpm": round(bpm * (2 if d.get("double") else 1), 1), "key": f"{k.get('key', '?')} ({k.get('camelot', '?')})"}
    lag = r.get("align", {}).get("all", [None])[0]
    st = r["stretch_pct"]
    out.append({
        "id": s["id"], "answer": s.get("answer", ""), "technique": s["technique"], "why": s["why"], "steps": s["steps"], "fx": s.get("fx", []), "phases": s.get("phases", []),
        "a": song(s["a"], r["tempo_a"]), "b": song(s["b"], r["tempo_b"]),
        "clip": f"hand/{s['id']}.m4a?v={int(os.path.getmtime(p))}", "seconds": m["seconds"], "bar_s": m["bar_s"], "clip_u": m["clip"],
        "first_event": s["first_event"], "last_event": s["last_event"],
        "overlap_bars": s["last_event"] - s["first_event"], "overlap_s": (s["last_event"] - s["first_event"]) * m["bar_s"],
        "stretch_txt": "eigenes Tempo (kein Angleichen)" if s.get("no_sync") else (f"{st:+.1f} % (Key-Lock, Tonhöhe bleibt)" if abs(st) > 0.05 else "unverändert"),
        "align_txt": "—" if s.get("no_sync") or lag is None else f"{abs(lag):.1f} ms",
        "key_txt": s.get("key_txt", ""),
        "gain_a": m["a"], "gain_b": m["b"],
    })
for t in out:
    t["clip_range"] = t.pop("clip_u")
    t["a"].update({"u0": t["gain_a"]["u0"], "step_bars": t["gain_a"]["step_bars"], "gain": t["gain_a"]["gain"]})
    t["b"].update({"u0": t["gain_b"]["u0"], "step_bars": t["gain_b"]["step_bars"], "gain": t["gain_b"]["gain"]})
    del t["gain_a"], t["gain_b"]
    t["audio"] = t["clip"]; t["clip"] = t.pop("clip_range")
intro = """<b>So sind die Übergänge gebaut:</b> jeder einzeln von Hand, aus den getrennten Spuren beider Songs (Gesang, Melodie, Bass, Drums).
<ul><li><b>Deine Regeln bisher:</b> B's Melodie blendet langsam ein, Stimme und Bass von B dürfen direkt einsetzen. Drums und Bass von A werden nie hart abgeschnitten, sie blenden über 2–3 Sekunden aus. Es läuft immer nur eine Melodie.</li>
<li>Gleicher Beat: die Paare haben das gleiche Tempo oder halbe/doppelte Zählzeit. Angeglichen wird mit Key-Lock, die Tonhöhe bleibt. Keine Roll-outs, kein Pitch-Riding.</li>
<li>Es singt immer nur eine Stimme.</li>
<li>Die Timeline zeigt, was jede Spur macht. Klick in die Timeline springt dorthin. Tasten: <kbd>Leertaste</kbd> Play/Pause, <kbd>1</kbd>–<kbd>5</kbd> Sterne, <kbd>J</kbd>/<kbd>K</kbd> nächster/voriger, <kbd>←</kbd>/<kbd>→</kbd> 5 s.</li></ul>"""
status = sys.argv[1] if len(sys.argv) > 1 else ""
json.dump({"intro": intro + (f"<p style='color:#ffd23d;margin:8px 0 0'>{status}</p>" if status else ""), "transitions": out}, open(f"{LAB}/hand.json", "w"))
print(len(out), "transitions on the site")
