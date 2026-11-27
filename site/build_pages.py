#!/usr/bin/env python3
"""Builds the lab pages auto.html (the planner's transitions) and rust.html (the hand specs through the Rust
renderer) from what `orml mix` / `orml render` wrote into src/auto and src/rust."""
import json, os, glob, shutil
L = os.path.dirname(os.path.abspath(__file__))
tracks = json.load(open(f"{L}/data.json"))["tracks"]
if os.path.exists(f"{L}/src/new_songs.json"):
    tracks.update(json.load(open(f"{L}/src/new_songs.json")))
keys = json.load(open(f"{L}/src/keys.json"))

def page(name, prefix, title, h1, intro):
    os.makedirs(f"{L}/{name}", exist_ok=True)
    out = []
    for p in sorted(glob.glob(f"{L}/src/{name}/{prefix}*.spec.json")):
        s = json.load(open(p))[0]; i = s["id"]
        m = json.load(open(f"{L}/src/{name}/{i}.lanes.json")); r = json.load(open(f"{L}/src/{name}/{i}.report.json"))
        shutil.copy(f"{L}/src/{name}/{i}.m4a", f"{L}/{name}/{i}.m4a")
        def song(d, bpm, lanes):
            t = tracks[d["id"]]; k = keys.get(d["id"], {})
            return {"id": d["id"], "name": t["name"], "artists": ", ".join(t["artists"]), "bpm": round(bpm, 1), "key": f"{k.get('key', '?')} ({k.get('camelot', '?')})", "u0": lanes["u0"], "step_bars": lanes["step_bars"], "gain": lanes["gain"]}
        lag = (r.get("align") or {}).get("all", [None])[0]; st = r["stretch_pct"]
        out.append({"id": i, "answer": "", "technique": s.get("technique", ""), "why": s.get("why", ""), "steps": s.get("steps", []), "fx": s.get("fx", []), "phases": [],
                    "a": song(s["a"], r["tempo_a"], m["a"]), "b": song(s["b"], r["tempo_b"], m["b"]), "clip": m["clip"], "seconds": m["seconds"], "bar_s": m["bar_s"],
                    "first_event": s["first_event"], "last_event": s["last_event"], "overlap_bars": round(s["last_event"] - s["first_event"], 2), "overlap_s": (s["last_event"] - s["first_event"]) * m["bar_s"],
                    "stretch_txt": f"{st:+.1f} % (Key-Lock)" if abs(st) > 0.05 else "unverändert", "align_txt": "—" if lag is None else f"{abs(lag):.1f} ms", "key_txt": s.get("key_txt", ""),
                    "audio": f"{name}/{i}.m4a?v={int(os.path.getmtime(p))}"})
    json.dump({"intro": intro, "transitions": out}, open(f"{L}/{name}.json", "w"))
    html = open(f"{L}/hand.html").read()
    html = html.replace('"hand.json?"', f'"{name}.json?"').replace('"hand:"', f'"{name}:"')
    html = html.replace("<title>Orin DJ Lab · Handgebaut</title>", f"<title>{title}</title>").replace("· 10 handgebaute Übergänge", h1)
    open(f"{L}/{name}.html", "w").write(html)
    print(name, len(out))

nav = ('<p><a href="index.html" style="color:#ffd23d">Hand-Fassungen</a> · <a href="rust.html" style="color:#ffd23d">Hand-Fassungen im Rust-Renderer</a> · '
       '<a href="auto.html" style="color:#ffd23d">Planer: deine 13 Paare</a> · <a href="new.html" style="color:#ffd23d">Planer: 12 neue Paare</a> · <a href="blindnew.html" style="color:#ffd23d">Blindtest: neu × neu</a> · <a href="blind.html" style="color:#ffd23d">Blindtest: neu × bekannt</a></p>')
page("auto", "p", "Orin DJ Lab · Planer", "· deine 13 Paare, vom Planer (Runde 2)",
     "<b>Runde 2: deine 13 Paare, vom Planer allein geplant und gerendert.</b> Geändert nach deinen Notizen: "
     "<b>11</b> ist jetzt genau wie deine Fassung (A's Stimme geht an der Atempause, die Eins auf Takt 123). "
     "<b>12</b> schneidet A's Stimme nicht mehr mitten im Refrain: A singt zu Ende, B's Stimme kommt über A's letzte zwei Takte. "
     "<b>10</b> fehlt: Tempo-Sprünge gibt es nicht mehr, die beiden Songs haben keinen gemeinsamen Beat. "
     "Deine alten Notizen stehen noch dran; schreib einfach drüber." + nav)
page("rust", "h", "Orin DJ Lab · Rust-Renderer", "· deine 13 Fassungen im Rust-Renderer",
     "<b>Deine 13 abgenommenen Fassungen, gerendert vom Rust-Renderer.</b>" + nav)
page("new", "n", "Orin DJ Lab · Neue Paare", "· dieselben 12 neuen Paare, Runde 2",
     "<b>Runde 2: dieselben 12 Paare.</b> Was sich nach deinen Notizen geändert hat: "
     "<b>n06</b> gibt es nicht mehr: kein gemeinsamer Beat, also kein Übergang, der Sender muss einen anderen Song wählen. "
     "<b>n02, n07, n09</b> lehnt das System jetzt selbst ab: nach dem Rendern prüft es, ob die Drums über die Überblendung auseinanderlaufen. "
     "Bei deinen 13 Paaren bleiben sie innerhalb von ±2 ms, bei diesen dreien laufen sie 15 bis 36 ms auseinander (Gimme! Gimme! Gimme! ist mit schwankendem Tempo gespielt, 115 bis 122 BPM). "
     "Auch eine kürzere Überblendung lief auseinander, also kein Übergang. "
     "Good For The Soul bekommt als Song B nie mehr eine Stimme über fremdem Beat (deine Notizen zu 09 und n02). "
     "Die übrigen 8 sind im Aufbau dieselben wie in Runde 1; das Tempo misst das System jetzt nur noch auf den 8 Takten, die in der Überblendung wirklich klingen, deshalb können sie minimal anders klingen." + nav)

import os as _os
if _os.path.isdir(f"{L}/src/blind"):
    page("blind", "x", "Orin DJ Lab · Blindtest", "· Blindtest 1: neue Songs, eingefrorene Regeln",
         open(f"{L}/src/blind/intro.html").read() + nav)

if _os.path.isdir(f"{L}/src/blindnew"):
    page("blindnew", "b", "Orin DJ Lab · Blindtest neu", "· Blindtest: zwei neue Songs",
         open(f"{L}/src/blindnew/intro.html").read() + nav)
