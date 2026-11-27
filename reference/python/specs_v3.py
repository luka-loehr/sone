#!/usr/bin/env python3
"""The hand-built transitions, version 3. Every number was set by hand from the songs' bar maps
(hand/maps, vmap.py) and the renderer's meter bridge, then changed after the owner's notes.

The owner's rules (2026-09-29):
  - same beat: equal tempo or half/double time, key-locked, only a little stretch
  - A's drums, bass and melody go down together, slowly, and reach zero exactly on the one
  - before the one B adds a single layer: its voice, or its melody when it has no voice there
  - B's drums, bass and melody start on the one (in the "sneak" recipe B's beat may come earlier)
  - one voice at a time, never a hard stop on a beat
specs.py → specs.json"""
import json

S = []
UNIT = {"drums": "db", "bass": "db", "other": "db"}          # A's instrumental fades as one, even to the ear

def unit_fade(start, u_from, u_to=0.0, level=1.0):
    """A stem of A: full until u_from, then down slowly, zero exactly at u_to."""
    return [[start, level], [u_from, level], [u_to - 0.12, 0.125 * level], [u_to, 0.0]]

def at_one(u_first, u=0.0, to=2.0):
    """B's melody: silent before the one, there at once but soft, full two bars later."""
    return [[u_first, 0.0], [u - 0.02, 0.0], [u, 0.4], [u + to, 1.0]]

def cut_in(u_first, u):
    """A stem of B that starts on a bar line (kick, bass, voice)."""
    return [[u_first, 0.0], [u - 0.03, 0.0], [u, 1.0]]

def echo(stem, u0, u1, beats=1.0, level=0.7):
    return {"type": "echo", "stem": stem, "beats": beats, "feedback": 0.45, "level": level,
            "send": [[u0 - 0.1, 0], [u0, 1], [u1 - 0.05, 1], [u1, 0]]}

# 01 ─ Where Them Girls At → Where Have You Been ───────────────────────────
S.append({
 "id": "h01", "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, -1], "b": [0, 7]},
 "key_txt": "Es-Dur → c-Moll (verwandt, passt)",
 "technique": "B's Stimme über A's Instrumental, B übernimmt auf der Eins",
 "why": "A singt seine Refrain-Zeile zu Ende, dann läuft A's komplettes Instrumental weiter und B's Stimme legt sich darüber. A geht als Ganzes langsam runter und ist genau auf der Eins bei null. Dort setzt B ein: Kick, Bass und – erst jetzt – B's Melodie.",
 "steps": [
  {"u": -8, "text": "A's Gesang endet mit einem Echo. A's Drums, Bass und Melodie laufen voll weiter. B's Stimme setzt direkt ein – sonst kommt von B noch nichts."},
  {"u": -4, "text": "A's Drums, Bass und Melodie blenden gemeinsam langsam aus – über vier Takte, bis genau zur Eins."},
  {"u": 0, "text": "Die Eins: A ist bei null. B's Kick setzt ein, B's Melodie beginnt erst hier."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -8.5, "u1": -7, "label": "Echo"}],
 "a": {"id": "7ks6AZmFcm3Y6PGGxGSmlB", "anchor_bar": 96, "anchor_u": -8, "until": 0,
  "segments": [{"u0": -22, "u1": -1, "src": -22}, {"u0": -1, "u1": 1.5, "src": -5}],
  "lanes": {"vocals": [[-22,1],[-8.12,1],[-8.0,0]],
            "drums": unit_fade(-22, -4), "bass": unit_fade(-22, -4), "other": unit_fade(-22, -4)},
  "shape": "lin", "shapes": UNIT,
  "sends": [echo("vocals", -8.5, -8.0)]},
 "b": {"id": "3tyPOhuVnt5zd5kGfxbCyL", "anchor_bar": 8, "anchor_u": 0, "from": -8,
  "lanes": {"vocals": [[-8,1.0]], "drums": cut_in(-8, 0), "bass": cut_in(-8, 0), "other": at_one(-8)},
  "shape": "lin"}
})

# 02 ─ Mr. Saxobeat → Don't Stop The Music ─────────────────────────────────
S.append({
 "id": "h02", "swap_u": -4, "clip": [-24, 12], "first_event": -12, "last_event": 0, "align_pairs": {"a": [-12, -4], "b": [-8, 0]},
 "key_txt": "h-Moll → fis-Moll (Nachbarn, passt)",
 "technique": "B's Beat schleicht sich unter A's letzten Refrain",
 "why": "B hat ein langes Beat-Intro. Erst nur die Höhen von B's Drums unter A, vier Takte vor der Eins kommen B's Kick und Bass dazu. Von da an geht A's Instrumental langsam runter und ist genau auf der Eins bei null. A's Stimme singt bis dahin zu Ende, dann kommen B's Stimme und B's Melodie.",
 "steps": [
  {"u": -12, "text": "B's Drums kommen ganz leise dazu – nur die Höhen, ohne Kick."},
  {"u": -4, "text": "B's Kick und Bass setzen ein. A's Drums, Bass und Melodie blenden gemeinsam langsam aus – vier Takte lang, bis genau zur Eins."},
  {"u": 0, "text": "Die Eins: A ist bei null, A's letzte Zeile klingt im Echo aus. B's Stimme setzt ein, B's Melodie beginnt erst hier."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -0.8, "u1": 1, "label": "Echo"}, {"deck": "b", "stem": "drums", "u0": -12, "u1": -4, "label": "nur Höhen"}],
 "a": {"id": "3LWDPEXEaLBRGNLbdcbwBR", "anchor_bar": 96, "anchor_u": -4, "until": 0,
  "lanes": {"vocals": [[-24,1],[-0.2,1],[-0.08,0]],
            "drums": unit_fade(-24, -4), "bass": unit_fade(-24, -4), "other": unit_fade(-24, -4)},
  "shape": "lin", "shapes": UNIT,
  "sends": [echo("vocals", -0.75, -0.1)]},
 "b": {"id": "0ByMNEPAPpOR5H69DVrTNy", "anchor_bar": 8, "anchor_u": 0, "from": -12,
  "segments": [{"u0": -12, "u1": -8, "src": -8}, {"u0": -8, "u1": None, "src": -8}],
  "lanes": {"vocals": cut_in(-12, 0),
            "drums": [[-12,0],[-11,0.9],[-8,1.6],[-4.03,2.2],[-4,1]],
            "bass": cut_in(-12, -4),
            "other": at_one(-12)},
  "shape": "lin",
  "hpf": {"drums": [[-12,500],[-8,260],[-4.05,200],[-4,20]]}}
})

# 03 ─ Sexy Bitch → Blessings ──────────────────────────────────────────────
# Rebuilt as the "sneak" recipe. B's first eight bars with beat (21-28) run under A with B's voice and
# melody muted; on the one B goes back to bar 21 and starts for real.
S.append({
 "id": "h03", "swap_u": -4, "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, -2], "b": [0, 6]},
 "key_txt": "h-Moll → h-Moll (gleich)",
 "technique": "B's Beat schleicht sich unter A's letzten Refrain",
 "why": "Neu gebaut: jetzt die andere Bauweise. Gleiche Tonart, gleiches Tempo. B's Beat kommt unter A's Hook dazu – erst nur die Höhen, vier Takte vor der Eins mit Kick und Bass. A's Instrumental geht langsam bis zur Eins runter. B's Gesang und Melodie starten erst auf der Eins.",
 "steps": [
  {"u": -8, "text": "B's Drums kommen leise dazu – nur die Höhen, ohne Kick. A läuft komplett weiter."},
  {"u": -4, "text": "B's Kick und Bass setzen ein. A's Drums, Bass und Melodie blenden gemeinsam langsam aus – vier Takte lang, bis genau zur Eins."},
  {"u": 0, "text": "Die Eins: A ist bei null, A's Hook klingt im Echo aus. B's Strophe beginnt – Gesang und Melodie von B erst hier."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -0.6, "u1": 1, "label": "Echo"}, {"deck": "b", "stem": "drums", "u0": -8, "u1": -4, "label": "nur Höhen"}],
 "a": {"id": "2dGxjfhFXDIUX9h5ImyKkE", "anchor_bar": 96, "anchor_u": -8, "until": 0,
  "lanes": {"vocals": [[-22,1],[-0.15,1],[0,0]],
            "drums": unit_fade(-22, -4), "bass": unit_fade(-22, -4), "other": unit_fade(-22, -4)},
  "shape": "lin", "shapes": UNIT,
  "sends": [echo("vocals", -0.55, -0.05)]},
 "b": {"id": "78nx0HDJIFD5xDq2L5420Z", "anchor_bar": 21, "anchor_u": 0, "from": -8, "fit_bars": [21, 34],
  "segments": [{"u0": -8, "u1": 0, "src": 0}, {"u0": 0, "u1": None, "src": 0}],
  "lanes": {"vocals": cut_in(-8, 0),
            "drums": [[-8,0],[-7,0.9],[-4.03,1.6],[-4,1]],
            "bass": cut_in(-8, -4),
            "other": at_one(-8)},
  "shape": "lin",
  "hpf": {"drums": [[-8,500],[-4.05,220],[-4,20]]}}
})

# 04 ─ Firework → That's When You Know ─────────────────────────────────────
S.append({
 "id": "h04", "clip": [-22, 12], "first_event": -9, "last_event": 0, "align_pairs": {"a": [-8, 0], "b": [0, 8]},
 "key_txt": "As-Dur → As-Dur (gleich)",
 "technique": "B's Stimme über A's Instrumental, B übernimmt auf der Eins",
 "why": "Gleiche Tonart, gleiches Tempo. A's Refrain-Zeile endet, A's komplettes Instrumental läuft weiter – die Melodie bleibt jetzt länger – und B singt darüber. A geht als Ganzes langsam bis zur Eins runter. Dort setzt B's eigener Beat ein, mit Bass und Melodie.",
 "steps": [
  {"u": -9, "text": "A's letzte Refrain-Zeile läuft aus, Echo auf dem letzten Wort."},
  {"u": -8, "text": "B's Stimme setzt direkt ein. A's Drums, Bass und Melodie laufen voll weiter."},
  {"u": -4, "text": "A's Drums, Bass und Melodie blenden gemeinsam langsam aus – bis genau zur Eins."},
  {"u": 0, "text": "Die Eins: A ist bei null. B's Beat und Bass setzen ein, B's Melodie beginnt erst hier."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -9, "u1": -7.5, "label": "Echo"}],
 "a": {"id": "3lXbHd0EoryoQh3vpevNEP", "anchor_bar": 112, "anchor_u": -8, "until": 0,
  "segments": [{"u0": -22, "u1": -8, "src": -22}, {"u0": -8, "u1": -4, "src": -12}, {"u0": -4, "u1": 1.5, "src": -12}],
  "lanes": {"vocals": [[-22,1],[-8.5,1],[-8.38,0]],
            "drums": unit_fade(-22, -4), "bass": unit_fade(-22, -4), "other": unit_fade(-22, -4)},
  "shape": "lin", "shapes": UNIT,
  "sends": [echo("vocals", -8.95, -8.45)]},
 "b": {"id": "1XhItJSBxL5yKh0V0ymg6D", "anchor_bar": 9, "anchor_u": 0, "from": -8,
  "gain_db": [[-8,2.0],[8,2.0],[12,1.0]],
  "lanes": {"vocals": [[-8,1.0]], "drums": [[-8,1.0]], "bass": cut_in(-8, 0), "other": at_one(-8)},
  "shape": "lin"}
})

# 05 ─ Talking Body → Yo No Lo Sé ──────────────────────────────────────────
S.append({
 "id": "h05", "swap_u": -4, "clip": [-22, 12], "first_event": -8, "last_event": 1, "align_pairs": {"a": [-8, -4], "b": [-8, -4]},
 "key_txt": "e-Moll → a-Moll (Nachbarn, passt)",
 "technique": "B's Beat schleicht sich unter A's letzten Refrain",
 "why": "Gleiches Tempo (120). B's House-Beat kommt erst nur mit den Höhen dazu, vier Takte vor der Eins mit Kick und Bass. Neu: A's Drums und Bass gehen nicht mehr vor der Melodie raus – A's ganzes Instrumental blendet gemeinsam langsam bis zur Eins aus. A's letzter Ton klingt aus, dann kommt B's Stimme.",
 "steps": [
  {"u": -8, "text": "B's Drums kommen leise dazu (nur Höhen)."},
  {"u": -4, "text": "B's Kick und Bass setzen ein. A's Drums, Bass und Melodie blenden gemeinsam langsam aus – vier Takte lang, bis genau zur Eins."},
  {"u": 0, "text": "Die Eins: A's Instrumental ist bei null, A's letzter Ton klingt mit Echo aus. B's Melodie beginnt hier."},
  {"u": 1, "text": "B's Stimme setzt ein, ab hier nur noch B."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": 0, "u1": 1.5, "label": "Echo"}, {"deck": "b", "stem": "drums", "u0": -8, "u1": -4, "label": "nur Höhen"}],
 "a": {"id": "5vXnH99Tn0qAh84SWU5sfc", "anchor_bar": 112, "anchor_u": -4, "until": 1,
  "lanes": {"vocals": [[-22,1],[0.5,1],[0.8,0]],
            "drums": unit_fade(-22, -4), "bass": unit_fade(-22, -4), "other": unit_fade(-22, -4)},
  "shape": "lin", "shapes": UNIT,
  "sends": [echo("vocals", -0.2, 0.7, level=0.6)]},
 "b": {"id": "5ihq4qbjaO2r5Ga9aGlsMj", "anchor_bar": 9, "anchor_u": 0, "from": -8, "fit_bars": [1, 15],
  "gain_db": [[-8,2.5],[6,2.5],[12,1.5]],
  "lanes": {"vocals": cut_in(-8, 1),
            "drums": [[-8,0],[-7,1.0],[-4.03,1.8],[-4,1]],
            "bass": cut_in(-8, -4),
            "other": at_one(-8)},
  "shape": "lin",
  "hpf": {"drums": [[-8,450],[-4.05,200],[-4,20]]}}
})

# 06 ─ Obsessed → Genie In a Bottle ────────────────────────────────────────
S.append({
 "id": "h06", "swap_u": -4, "clip": [-16, 9], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-6, -4], "b": [-2, 0]},
 "key_txt": "c-Moll → f-Moll (Nachbarn, passt)",
 "technique": "Intro aus B's eigenem Beat gebaut",
 "why": "B hat nur zwei Takte Beat vor der Strophe. Die loope ich zu einem achttaktigen Intro: erst nur die Höhen unter A, vier Takte vor der Eins mit Kick und Bass. A's Instrumental geht gemeinsam langsam bis zur Eins runter. B's Melodie und Strophe beginnen erst auf der Eins.",
 "steps": [
  {"u": -8, "text": "B's Drums kommen leise dazu (nur Höhen)."},
  {"u": -4, "text": "B's Kick und Bass setzen ein. A's Drums, Bass und Melodie blenden gemeinsam langsam aus – bis genau zur Eins."},
  {"u": -0.5, "text": "A's Stimme endet mit Echo, B's Auftakt kommt herein."},
  {"u": 0, "text": "Die Eins: B's Strophe beginnt, B's Melodie beginnt erst hier."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -1, "u1": 0.5, "label": "Echo"}, {"deck": "b", "stem": "drums", "u0": -8, "u1": -4, "label": "nur Höhen"}],
 "a": {"id": "1lEIT08AUU67qef6MkKAKx", "anchor_bar": 86, "anchor_u": 0, "until": 0,
  "lanes": {"vocals": [[-16,1],[-0.62,1],[-0.5,0]],
            "drums": unit_fade(-16, -4), "bass": unit_fade(-16, -4), "other": unit_fade(-16, -4)},
  "shape": "lin", "shapes": UNIT,
  "sends": [echo("vocals", -0.95, -0.5, level=0.6)]},
 "b": {"id": "11mwFrKvLXCbcVGNxffGyP", "anchor_bar": 3, "anchor_u": 0, "from": -8,
  "segments": [{"u0": -8, "u1": -6, "src": -2}, {"u0": -6, "u1": -4, "src": -2}, {"u0": -4, "u1": -2, "src": -2}, {"u0": -2, "u1": None, "src": -2}],
  "lanes": {"vocals": [[-8,0],[-0.55,0],[-0.5,1]],
            "drums": [[-8,0],[-7,0.9],[-4.03,1.6],[-4,1]],
            "bass": cut_in(-8, -4),
            "other": at_one(-8)},
  "shape": "lin",
  "hpf": {"drums": [[-8,600],[-4.05,220],[-4,20]]}}
})

# 07 ─ Make Me → Heads Will Roll (owner: "matches fine") – unchanged ───────
S.append({
 "id": "h07", "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, 0], "b": [0, 8]},
 "key_txt": "es-Moll → gis-Moll (Nachbarn, passt)",
 "technique": "Outro-Beat von A unter dem Synth-Intro von B",
 "why": "Der klassische Club-Übergang: A endet mit acht Takten reinem Beat, B beginnt mit acht Takten Synthesizer ohne Beat. Beides liegt genau übereinander. B's Synth blendet langsam ein, auf der Eins kommen B's Beat und Stimme. (Unverändert – du fandest ihn passend.)",
 "steps": [
  {"u": -8, "text": "A's Gesang ist vorbei, A's Beat läuft weiter. B's Synth beginnt langsam einzublenden."},
  {"u": -4, "text": "A's Melodie ist fast weg, B's Synth ist deutlich zu hören."},
  {"u": -2, "text": "A's Drums werden hochgefiltert und gehen zurück."},
  {"u": 0, "text": "Die Eins: B's Beat, Bass und Stimme setzen ein."}
 ],
 "fx": [{"deck": "a", "stem": "drums", "u0": -2, "u1": 0.75, "label": "Filter ↑"}],
 "a": {"id": "57508shL0obX0KUOUL8CJk", "anchor_bar": 96, "anchor_u": -8, "until": 0,
  "lanes": {"vocals": [[-22,1],[-8,1],[-7.9,0]],
            "drums": [[-22,1],[-1.5,1],[-0.25,0.6],[0.75,0]],
            "bass": [[-22,1],[-1.25,1],[0.1,0]],
            "other": [[-22,1],[-8,1],[-3,0.3],[0.5,0]]},
  "segments": [{"u0": -22, "u1": 0, "src": -22}, {"u0": 0, "u1": 1.5, "src": -4}],
  "hpf": {"drums": [[-2,20],[-1,140],[-0.05,800]]}},
 "b": {"id": "5qGIfWDRdyO67LCTqA2sEn", "anchor_bar": 11, "anchor_u": 0, "from": -8, "fit_bars": [11, 40],
  "gain_db": [[-8,2.5],[8,2.5],[12,1.5]],
  "lanes": {"vocals": [[-8,0],[-0.1,0],[-0.07,1]],
            "drums": [[-8,1]], "bass": [[-8,1]],
            "other": [[-8,0],[-2,1]]},
  "shape": "lin"}
})

# 08 ─ Sprinter → Harleys In Hawaii (owner: "sehr sehr clean") – unchanged ─
S.append({
 "id": "h08", "clip": [-24, 14], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, -4], "b": [0, 4]},
 "key_txt": "b-Moll → b-Moll (gleich)",
 "technique": "A's Beat trägt das Gitarren-Intro von B",
 "why": "Gleiche Tonart, gleicher Beat über halbe/doppelte Zählzeit (139 / 70). B's Gitarre blendet schon unter den letzten Rap-Zeilen langsam ein. Wenn der Rap fertig ist, läuft A's Beat allein mit B's Gitarre weiter – auf der Eins beginnt B's Strophe mit eigenem Beat. (Unverändert – du fandest ihn sehr sauber.)",
 "steps": [
  {"u": -8, "text": "B's Gitarre beginnt ganz leise unter A's letzten Rap-Zeilen einzublenden."},
  {"u": -4, "text": "A's Rap ist zu Ende. A's Beat und Bass laufen weiter, B's Gitarre steht jetzt im Vordergrund."},
  {"u": -2, "text": "A's Drums werden hochgefiltert und gehen zurück."},
  {"u": 0, "text": "Die Eins: B's Strophe beginnt mit eigenem Beat und Bass."}
 ],
 "fx": [{"deck": "a", "stem": "drums", "u0": -2, "u1": 0.75, "label": "Filter ↑"}],
 "a": {"id": "6NYbDTmstd7JSUWl2hVBqR", "anchor_bar": 127, "anchor_u": -8, "until": 0,
  "segments": [{"u0": -24, "u1": -4, "src": -24}, {"u0": -4, "u1": 0, "src": -12}, {"u0": 0, "u1": 1.5, "src": -12}],
  "lanes": {"vocals": [[-24,1],[-4.55,1],[-4.4,0]],
            "drums": [[-24,1],[-1.5,1],[-0.25,0.6],[0.75,0]],
            "bass": [[-24,1],[-1.25,1],[0.1,0]],
            "other": [[-24,1],[-8,1],[-4,0.4],[-1,0.1],[0.5,0]]},
  "hpf": {"drums": [[-2,20],[-1,140],[-0.05,800]], "bass": [[-1,20],[-0.05,300]]}},
 "b": {"id": "3B4etxytImrRKQgV3XHjnR", "anchor_bar": 2, "anchor_u": 0, "from": -8, "beats_per_bar": 2, "double": True,
  "gain_db": [[-8,1.5],[10,1.5],[14,1.0]],
  "segments": [{"u0": -8, "u1": -4, "src": -4}, {"u0": -4, "u1": None, "src": -4}],
  "lanes": {"vocals": [[-8,0],[-0.3,0],[-0.26,1]],
            "drums": [[-8,1]], "bass": [[-8,1]],
            "other": [[-8,0],[-2,1]]},
  "shape": "lin"}
})

# 09 ─ Roman d'Amour → Good For The Soul ───────────────────────────────────
# Now by the same rule as 01 and 04: B's voice over A's whole instrumental, B's bass and melody on the one.
S.append({
 "id": "h09", "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, 0], "b": [0, 8]},
 "key_txt": "c-Moll → f-Moll (Nachbarn)",
 "technique": "B's Stimme über A's Instrumental, B übernimmt auf der Eins",
 "why": "Neu gebaut nach der gleichen Regel wie 01 und 04: von B kommt vor der Eins nur die Stimme. A's Instrumental läuft komplett darunter weiter und geht als Ganzes langsam bis zur Eins runter. B's Beat, Bass und Melodie setzen erst auf der Eins ein – so überlagern sich keine zwei Melodien mehr.",
 "steps": [
  {"u": -8, "text": "B's Stimme setzt direkt ein – sonst kommt von B noch nichts. A's Instrumental läuft voll weiter."},
  {"u": -4, "text": "A's Drums, Bass und Melodie blenden gemeinsam langsam aus – bis genau zur Eins."},
  {"u": 0, "text": "Die Eins: A ist bei null. B's Beat und Bass setzen ein, B's Melodie beginnt erst hier."}
 ],
 "fx": [],
 "a": {"id": "1fVQmMgrOUnrh8KgndKGzP", "anchor_bar": 71, "anchor_u": -8, "until": 0, "fit_bars": [55, 78],
  "segments": [{"u0": -22, "u1": 0, "src": -22}, {"u0": 0, "u1": 1.5, "src": -4}],
  "lanes": {"vocals": [[-22,1],[-8,1],[-7.9,0]],
            "drums": unit_fade(-22, -4), "bass": unit_fade(-22, -4), "other": unit_fade(-22, -4)},
  "shape": "lin", "shapes": UNIT},
 "b": {"id": "03D2TnaZuwlddRTh7P6Mq5", "anchor_bar": 15, "anchor_u": 0, "from": -8, "fit_bars": [15, 38],
  "lanes": {"vocals": [[-8,1]], "drums": cut_in(-8, 0), "bass": cut_in(-8, 0), "other": at_one(-8)},
  "shape": "lin"}
})

# 10 ─ Empire State Of Mind → Sexy Bitch ───────────────────────────────────
# 86.65 and 130 BPM are 2:3, B is locked to that (one ruler bar = 4 beats of A = 6 beats of B).
# Owner's idea: before the one only B's synth, B's drums start on the one.
S.append({
 "id": "h10", "clip": [-12, 9], "first_event": -4, "last_event": 2, "align_pairs": {"a": [-8, -4], "b": [0, 4]},
 "key_txt": "Fis-Dur → h-Moll (Dominante → Grundton, löst sich auf)",
 "technique": "Tempo-Sprung: erst nur B's Synth unter A, B's Drums ab der Eins",
 "why": "Deine Idee umgesetzt: Vor der Eins kommt von B nur der Synth, er blendet unter A's Hook ein. A's Drums, Bass und Klavier gehen gemeinsam langsam bis zur Eins runter. B's Drums und Bass starten erst auf der Eins. B ist 3 gegen 2 auf A eingerastet, kein Song ändert sein Tempo. A's Hook klingt danach im Hall aus.",
 "steps": [
  {"u": -4, "text": "B's Synth beginnt einzublenden – sonst kommt von B nichts. A's Drums, Bass und Klavier blenden gemeinsam langsam aus, bis genau zur Eins. Der Hall auf A's Stimme wächst."},
  {"u": 0, "text": "Die Eins: B's Drums und Bass setzen ein. A's Hook singt weiter und blendet über gut 5 s im Hall aus."},
  {"u": 2, "text": "A ist ausgeklungen, ab hier nur noch B."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -4, "u1": 2, "label": "Hall wächst"}],
 "a": {"id": "2igwFfvr1OAGX9SKDCPBwO", "anchor_bar": 94, "anchor_u": 0, "until": 2,
  "lanes": {"vocals": [[-12,1],[0,1],[0.5,0.8],[2,0]],
            "drums": unit_fade(-12, -4), "bass": unit_fade(-12, -4), "other": unit_fade(-12, -4)},
  "shape": "lin", "shapes": UNIT,
  "sends": [{"type": "reverb", "stem": "vocals", "rt60": 2.6, "level": 0.5, "send": [[-4,0],[-1,0.5],[0,0.8],[1.2,0.8],[2,0]]}]},
 "b": {"id": "2dGxjfhFXDIUX9h5ImyKkE", "anchor_bar": 0, "anchor_u": 0, "from": -4, "beats_per_bar": 6, "fit_bars": [0, 23],
  "segments": [{"u0": -4, "u1": 0, "src": 0}, {"u0": 0, "u1": None, "src": 0}],
  "lanes": {"vocals": [[-4,1]],
            "drums": cut_in(-4, 0),
            "bass": cut_in(-4, 0),
            "other": [[-4,0],[-2.5,2.0],[-0.5,3.0],[0,1.6],[4,1.0]]},
  "shape": "lin"}
})

# 11 ─ Legendary Lovers → Mr. Saxobeat ─────────────────────────────────────
# B enters 8 bars before its first chorus: two bars of vocal chops, then the build with the sax riff
# (no voice there, so the riff is B's one layer). B's drums and bass start on the one with the chorus.
S.append({
 "id": "h11", "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-16, -8], "b": [0, 8]},
 "key_txt": "e-Moll → h-Moll (Nachbarn, passt)",
 "technique": "A's Instrumental trägt B's Aufbau – B's Refrain landet auf der Eins",
 "why": "Fast gleiches Tempo (126 / 127). A singt die Refrain-Zeile zu Ende, A's Instrumental läuft weiter. Von B kommt erst die Stimme, danach das Saxofon-Riff (dort singt B nicht). A geht langsam bis zur Eins runter – und genau dort kommt B's Refrain mit Beat und Bass.",
 "steps": [
  {"u": -8, "text": "A's Gesang endet am Zeilenende (Echo). A's Instrumental läuft weiter, A's Melodie geht zurück. B's Stimme setzt direkt ein."},
  {"u": -6, "text": "B's Stimme ist fertig, B's Saxofon-Riff blendet langsam ein."},
  {"u": -4, "text": "A's Drums und Bass blenden langsam aus – bis genau zur Eins."},
  {"u": 0, "text": "Die Eins: B's Refrain mit Beat und Bass."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -8.5, "u1": -7, "label": "Echo"}],
 "a": {"id": "5Uc37Y6FijZcM538PqqkWd", "anchor_bar": 115, "anchor_u": -8, "until": 0,
  "segments": [{"u0": -22, "u1": -8, "src": -22}, {"u0": -8, "u1": 1.5, "src": -16}],
  "lanes": {"vocals": [[-22,1],[-8.25,1],[-8.0,0]],
            "drums": unit_fade(-22, -4, level=1.0), "bass": unit_fade(-22, -4),
            "other": [[-22,1],[-8,1],[-5.5,0.06],[-5.3,0]]},
  "gain_db": [[-22,0],[-8,0],[-7,2.0],[0,2.0]],
  "shape": "lin", "shapes": UNIT,
  "sends": [echo("vocals", -8.5, -8.0)]},
 "b": {"id": "3LWDPEXEaLBRGNLbdcbwBR", "anchor_bar": 13, "anchor_u": 0, "from": -8, "fit_bars": [13, 40],
  "gain_db": [[-8,3.0],[-1,3.0],[0,0.5]],
  "lanes": {"vocals": [[-8,1.0]],
            "drums": cut_in(-8, 0),
            "bass": cut_in(-8, 0),
            "other": [[-8,0.0],[-6.2,0.0],[-3,1.0]]},
  "shape": "lin"}
})

# 12 ─ Girl You Loud → Where Them Girls At ─────────────────────────────────
S.append({
 "id": "h12", "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, 0], "b": [0, 8]},
 "key_txt": "As-Dur → Es-Dur (Nachbarn, passt)",
 "technique": "Halbe gegen doppelte Zählzeit: B's Stimme über A's Instrumental",
 "why": "A läuft mit 65 BPM, B mit 130 – jeder zweite Schlag von B ist ein Schlag von A. Beide bleiben in ihrem Tempo (B nur 0,4 % langsamer). B's Stimme rappt über A's komplettes Instrumental, A geht langsam bis zur Eins runter, dort kommt B's Refrain mit dem geraden Dance-Beat.",
 "steps": [
  {"u": -8, "text": "A's Gesang endet am Zeilenende (Echo). A's Drums, Bass und Melodie laufen voll weiter. B's Stimme setzt direkt ein – sonst kommt von B noch nichts."},
  {"u": -4, "text": "A's Drums, Bass und Melodie blenden gemeinsam langsam aus – bis genau zur Eins."},
  {"u": 0, "text": "Die Eins: A ist bei null. B's Refrain mit Beat und Bass, B's Melodie beginnt erst hier."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -8.5, "u1": -7, "label": "Echo"}],
 "a": {"id": "6dRVOzSxLqRdqUjc9mctGg", "anchor_bar": 49, "anchor_u": -8, "until": 0, "beats_per_bar": 2, "fit_bars": [37, 58], "double": True,
  "lanes": {"vocals": [[-22,1],[-8.2,1],[-8.0,0]],
            "drums": unit_fade(-22, -4, level=0.85), "bass": unit_fade(-22, -4), "other": unit_fade(-22, -4)},
  "shape": "lin", "shapes": UNIT,
  "sends": [echo("vocals", -8.5, -8.0, beats=0.5)]},
 "b": {"id": "7ks6AZmFcm3Y6PGGxGSmlB", "anchor_bar": 16, "anchor_u": 0, "from": -8,
  "gain_db": [[-8,1.0],[0,2.0],[8,2.0],[12,1.0]],
  "lanes": {"vocals": [[-8,1.0]], "drums": cut_in(-8, 0), "bass": cut_in(-8, 0), "other": at_one(-8)},
  "shape": "lin"}
})

# 13 ─ Hollaback Girl (Slowed) → Kavkaz ────────────────────────────────────
S.append({
 "id": "h13", "swap_u": -4, "clip": [-16, 10], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-14, -10], "b": [-4, 0]},
 "key_txt": "g-Moll → g-Moll (gleich)",
 "technique": "B's Beat schleicht sich unter A's Schluss",
 "why": "Fast gleiches Tempo (91 / 92), gleiche Tonart. B's Drums kommen erst nur mit den Höhen dazu, vier Takte vor der Eins mit Kick und Bass. A's Instrumental geht gemeinsam langsam bis zur Eins runter, A's Stimme läuft bis zu ihrem Ende. B's Melodie beginnt erst auf der Eins.",
 "steps": [
  {"u": -8, "text": "B's Drums kommen leise dazu (nur Höhen)."},
  {"u": -4, "text": "B's Kick und Bass setzen ein. A's Drums, Bass und Melodie blenden gemeinsam langsam aus – bis genau zur Eins."},
  {"u": 0, "text": "Die Eins: A ist bei null, A's Stimme klingt im Echo aus. B's Melodie beginnt erst hier."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -0.6, "u1": 1, "label": "Echo"}, {"deck": "b", "stem": "drums", "u0": -8, "u1": -4, "label": "nur Höhen"}],
 "a": {"id": "6N185ACZ5M6TYngUuOPpoW", "anchor_bar": 38, "anchor_u": -4, "until": 0.5,
  "lanes": {"vocals": [[-16,1],[-0.1,1],[0.4,0]],
            "drums": unit_fade(-16, -4), "bass": unit_fade(-16, -4), "other": unit_fade(-16, -4)},
  "shape": "lin", "shapes": UNIT,
  "sends": [echo("vocals", -0.5, 0.4, level=0.6)]},
 "b": {"id": "7yoqSltRmGVlJF34OdLyZG", "anchor_bar": 2, "anchor_u": -4, "from": -8,
  "segments": [{"u0": -8, "u1": -6, "src": -6}, {"u0": -6, "u1": None, "src": -6}],
  "lanes": {"vocals": cut_in(-8, 0),
            "drums": [[-8,0],[-7,1.0],[-4.03,1.6],[-4,1]],
            "bass": cut_in(-8, -4),
            "other": at_one(-8)},
  "shape": "lin",
  "hpf": {"drums": [[-8,500],[-4.05,220],[-4,20]]}}
})

json.dump(S, open("specs.json", "w"), indent=1)
print(len(S), "specs")
