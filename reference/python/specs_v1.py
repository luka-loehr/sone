#!/usr/bin/env python3
"""The ten hand-built transitions. Every number here was set by hand from the
songs' bar maps (hand/maps) and the meter bridge of the renderer.
specs.py → specs.json"""
import json

S = []

# 01 ─ Where Them Girls At → Where Have You Been ───────────────────────────
# A: last chorus 80-103 (song ends after bar 103). Its vocal sings bars 80-95; bars 96-103 go on
#    as the bed (drums, bass, a little melody). Bar 103 has no drums in the song, so bar 99 plays there.
# B: bars 0-7 "Where have you been…" over a pad, bar 8 = the kick comes in.
S.append({
 "id": "h01", "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, -1], "b": [0, 7]},
 "key_txt": "Es-Dur → c-Moll (verwandt, passt)",
 "technique": "A's Beat trägt das Intro von B",
 "why": "B beginnt ohne Beat, nur Stimme und Fläche. Statt A auslaufen zu lassen, singt A seinen Refrain zu Ende, dann läuft nur noch A's Beat weiter und B's Intro-Gesang legt sich darüber. Auf der Eins übernimmt B's eigener Kick. Beide Songs stehen in verwandten Tonarten (Es-Dur / c-Moll).",
 "steps": [
  {"u": -8, "text": "A's Gesang endet am Ende der Refrain-Zeile mit einem Echo. Drums und Bass von A laufen weiter. B's Stimme setzt direkt ein; B's Melodie blendet langsam über sieben Takte ein, während die Melodie von A im gleichen Zug zurückgeht."},
  {"u": -2, "text": "A's Drums werden über zwei Takte hochgefiltert (der Bass wird dünner) – Spannung vor dem Einsatz."},
  {"u": 0, "text": "Die Eins: B's Kick setzt ein, A ist komplett weg. Ab hier läuft nur noch B."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -8.5, "u1": -7, "label": "Echo"}, {"deck": "a", "stem": "drums", "u0": -2, "u1": 0, "label": "Filter ↑"}],
 "a": {"id": "7ks6AZmFcm3Y6PGGxGSmlB", "anchor_bar": 96, "anchor_u": -8, "until": 0,
  "segments": [{"u0": -22, "u1": -1, "src": -22}, {"u0": -1, "u1": 0.5, "src": -5}],
  "lanes": {"vocals": [[-22,1],[-8.12,1],[-8.0,0]],
            "drums": [[-22,1],[-8,1],[-7,0.85],[-0.02,0.85],[0,0]],
            "bass": [[-22,1],[-8,1],[-0.02,1],[0,0]],
            "other": [[-22,1],[-8,1],[-5,0.6],[-2,0.3],[-0.02,0.2],[0,0]]},
  "hpf": {"drums": [[-2,20],[-1,120],[-0.05,700]], "bass": [[-1,20],[-0.05,300]]},
  "sends": [{"type": "echo", "stem": "vocals", "beats": 1.0, "feedback": 0.45, "level": 0.7, "send": [[-8.6,0],[-8.5,1],[-8.0,1],[-7.95,0]]}]},
 "b": {"id": "3tyPOhuVnt5zd5kGfxbCyL", "anchor_bar": 8, "anchor_u": 0, "from": -8,
  "lanes": {"vocals": [[-8,1.0]], "drums": [[-8,0.0],[-0.02,0.0],[0,1]], "bass": [[-8,1]], "other": [[-8,0.0],[-1,1.0]]},
  "shape": "lin", "trim_db": 0.0}
})


# 02 ─ Mr. Saxobeat → Don't Stop The Music ─────────────────────────────────
# A: last chorus 84-99, the song ends on bar 100. B: 16 bars of beat intro, whispered hook from bar 0,
#    bass from bar 4, verse at 16. B's first four bars play twice (tease), then B runs through.
S.append({
 "id": "h02", "swap_u": -4, "clip": [-24, 12], "first_event": -12, "last_event": 0, "align_pairs": {"a": [-12, -4], "b": [-8, 0]},
 "key_txt": "h-Moll → fis-Moll (Nachbarn, passt)",
 "technique": "B's Beat schleicht sich unter A's letzten Refrain",
 "why": "B hat ein langes Beat-Intro. Erst hört man nur die Hi-Hats von B ganz leise unter A, dann wechseln Kick und Bass auf der Eins zu B – A's Stimme und das Saxofon singen ihren Refrain über B's Groove zu Ende. Wenn A fertig ist, kommt B's geflüsterte Hook.",
 "steps": [
  {"u": -12, "text": "B's Drums kommen ganz leise dazu – nur die Höhen (Hi-Hats, Claps), ohne Kick."},
  {"u": -8, "text": "B's Drums werden deutlicher, B's Melodie beginnt langsam einzublenden."},
  {"u": -4, "text": "Die Eins: Kick und Bass wechseln zu B. Von A bleiben nur Stimme und Saxofon – sie laufen über B's Groove weiter."},
  {"u": 0, "text": "A's letzte Zeile endet mit einem Echo. B's Stimme setzt ein, ab hier nur noch B."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -0.8, "u1": 1, "label": "Echo"}, {"deck": "b", "stem": "drums", "u0": -12, "u1": -4, "label": "nur Höhen"}],
 "a": {"id": "3LWDPEXEaLBRGNLbdcbwBR", "anchor_bar": 96, "anchor_u": -4, "until": 0,
  "lanes": {"vocals": [[-24,1],[-0.2,1],[-0.08,0]],
            "drums": [[-24,1],[-8,1],[-4.04,0.9],[-4,0]],
            "bass": [[-24,1],[-4.04,1],[-4,0]],
            "other": [[-24,1],[-4,1],[-1,0.6],[-0.05,0.35],[0,0]]},
  "sends": [{"type": "echo", "stem": "vocals", "beats": 1.0, "feedback": 0.5, "level": 0.7, "send": [[-0.85,0],[-0.75,1],[-0.2,1],[-0.1,0]]}]},
 "b": {"id": "0ByMNEPAPpOR5H69DVrTNy", "anchor_bar": 8, "anchor_u": 0, "from": -12,
  "segments": [{"u0": -12, "u1": -8, "src": -8}, {"u0": -8, "u1": None, "src": -8}],
  "lanes": {"vocals": [[-12,0],[-0.03,0],[0,1]],
            "drums": [[-12,0],[-11,0.9],[-8,1.6],[-4.03,2.2],[-4,1]],
            "bass": [[-12,0],[-4.03,0],[-4,1]],
            "other": [[-12,0],[-8,0],[0,0.9],[2,1]]},
  "shape": "lin",
  "hpf": {"drums": [[-12,500],[-8,260],[-4.05,200],[-4,20]]}}
})

# 03 ─ Sexy Bitch → Blessings ──────────────────────────────────────────────
# A: the hook chant runs in two-bar lines, each new line starts after a gap on the one of an odd bar.
#    It stops in the gap before bar 97; bars 97-102 go on as the bed, then bars 101-102 once more
#    (bar 103 is the song's closing bar). B: second half of the first verse (bars 13-20, no beat), beat at 21.
S.append({
 "id": "h03", "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, -2], "b": [0, 6]},
 "key_txt": "h-Moll → h-Moll (gleich)",
 "technique": "A's Beat trägt die Strophe von B",
 "why": "Gleiche Tonart, gleiches Tempo. A's Hook hört in ihrer natürlichen Atempause auf, Kick und Bass von A laufen weiter und B's Stimme singt darüber. B's Klavier blendet langsam ein. Genau dort, wo in B der eigene Beat einsetzt, übernimmt B komplett.",
 "steps": [
  {"u": -8, "text": "A's Hook endet in der Pause zwischen zwei Zeilen (Echo). Drums und Bass von A laufen weiter, B's Stimme setzt direkt ein."},
  {"u": -6, "text": "B's Melodie blendet langsam ein, A's Melodie geht im gleichen Zug zurück."},
  {"u": -2, "text": "A's Drums werden hochgefiltert, B's eigener Trommelwirbel baut sich darunter auf."},
  {"u": 0, "text": "Die Eins: B's Kick und Bass setzen ein, A ist weg."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -8.5, "u1": -7, "label": "Echo"}, {"deck": "a", "stem": "drums", "u0": -2, "u1": 0, "label": "Filter ↑"}],
 "a": {"id": "2dGxjfhFXDIUX9h5ImyKkE", "anchor_bar": 97, "anchor_u": -8, "until": 0,
  "segments": [{"u0": -22, "u1": -2, "src": -22}, {"u0": -2, "u1": 0.5, "src": -4}],
  "lanes": {"vocals": [[-22,1],[-8.06,1],[-8.0,0]],
            "drums": [[-22,1],[-8,1],[-7,0.9],[-0.02,0.9],[0,0]],
            "bass": [[-22,1],[-0.02,1],[0,0]],
            "other": [[-22,1],[-8,1],[-4,0.5],[-0.02,0.15],[0,0]]},
  "hpf": {"drums": [[-2,20],[-1,140],[-0.05,800]], "bass": [[-1,20],[-0.05,300]]},
  "sends": [{"type": "echo", "stem": "vocals", "beats": 1.0, "feedback": 0.45, "level": 0.7, "send": [[-8.6,0],[-8.5,1],[-8.0,1],[-7.95,0]]}]},
 "b": {"id": "78nx0HDJIFD5xDq2L5420Z", "anchor_bar": 21, "anchor_u": 0, "from": -8, "fit_bars": [21, 34],
  "lanes": {"vocals": [[-8,1.0]],
            "drums": [[-8,0.0],[-1.05,0.0],[-1,1.0]],
            "bass": [[-8,0],[-0.02,0],[0,1]],
            "other": [[-8,0.0],[-1,1.0]]},
  "shape": "lin"}
})

# 04 ─ Firework → That's When You Know ─────────────────────────────────────
# A: last chorus 100-115. Its line ends on beat 2 of bar 111. From there bars 108-111 play twice more
#    as the bed, drums only. B: starts singing at once (bar 1), no beat until bar 9.
S.append({
 "id": "h04", "clip": [-22, 12], "first_event": -9, "last_event": 0, "align_pairs": {"a": [-8, 0], "b": [0, 8]},
 "key_txt": "As-Dur → As-Dur (gleich)",
 "technique": "A's Beat trägt die Strophe von B",
 "why": "Gleiche Tonart, gleiches Tempo. B fängt sofort mit Gesang an und hat acht Takte lang keinen Beat. A's Refrain-Zeile endet, dann laufen Drums und Bass von A weiter und B singt darüber. B's Gitarre blendet langsam ein. Beim ersten eigenen Beat von B ist A weg.",
 "steps": [
  {"u": -9, "text": "A's letzte Refrain-Zeile läuft aus, Echo auf dem letzten Wort."},
  {"u": -8, "text": "B's Stimme setzt direkt ein. Drums und Bass von A laufen weiter, A's Melodie geht zurück."},
  {"u": -6, "text": "B's Melodie blendet langsam ein."},
  {"u": -2, "text": "A's Drums werden hochgefiltert."},
  {"u": 0, "text": "Die Eins: B's eigener Beat und Bass setzen ein, A ist weg."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -9, "u1": -7.5, "label": "Echo"}, {"deck": "a", "stem": "drums", "u0": -2, "u1": 0, "label": "Filter ↑"}],
 "a": {"id": "3lXbHd0EoryoQh3vpevNEP", "anchor_bar": 112, "anchor_u": -8, "until": 0,
  "segments": [{"u0": -22, "u1": -8, "src": -22}, {"u0": -8, "u1": -4, "src": -12}, {"u0": -4, "u1": 0.5, "src": -12}],
  "lanes": {"vocals": [[-22,1],[-8.5,1],[-8.38,0]],
            "drums": [[-22,1],[-8,1],[-7,0.9],[-0.02,0.9],[0,0]],
            "bass": [[-22,1],[-0.02,1],[0,0]],
            "other": [[-22,1],[-8,1],[-5,0.4],[-2,0.15],[-0.02,0.1],[0,0]]},
  "hpf": {"drums": [[-2,20],[-1,140],[-0.05,800]], "bass": [[-1,20],[-0.05,300]]},
  "sends": [{"type": "echo", "stem": "vocals", "beats": 1.0, "feedback": 0.45, "level": 0.7, "send": [[-9.0,0],[-8.95,1],[-8.5,1],[-8.45,0]]}]},
 "b": {"id": "1XhItJSBxL5yKh0V0ymg6D", "anchor_bar": 9, "anchor_u": 0, "from": -8,
  "gain_db": [[-8,2.0],[8,2.0],[12,1.0]],
  "lanes": {"vocals": [[-8,1.0]], "drums": [[-8,1.0]], "bass": [[-8,0],[-0.02,0],[0,1.0]], "other": [[-8,0.0],[-1,1.0]]},
  "shape": "lin"}
})

# 05 ─ Talking Body → Yo No Lo Sé ──────────────────────────────────────────
# A: last chorus to the end (bar 116 is the held last note). B: beat from the start, its voice fades in
#    during bars 4-7; here it stays out until bar 10.
S.append({
 "id": "h05", "swap_u": -4, "clip": [-22, 12], "first_event": -8, "last_event": 1, "align_pairs": {"a": [-8, -4], "b": [-8, -4]},
 "key_txt": "e-Moll → a-Moll (Nachbarn, passt)",
 "technique": "B's Beat schleicht sich unter A's letzten Refrain",
 "why": "Gleiches Tempo (120). B's House-Beat kommt erst nur mit den Höhen dazu, auf der Eins wechseln Kick und Bass. A singt seinen letzten Refrain über B's Groove zu Ende, der letzte gehaltene Ton klingt aus – dann kommt B's Stimme.",
 "steps": [
  {"u": -8, "text": "B's Drums kommen leise dazu (nur Höhen), B's Melodie beginnt ganz langsam einzublenden."},
  {"u": -4, "text": "Die Eins: Kick und Bass wechseln zu B. A's Stimme singt über B's Groove weiter."},
  {"u": 0, "text": "A's letzter Ton klingt mit Echo aus."},
  {"u": 1, "text": "B's Stimme setzt ein, ab hier nur noch B."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": 0, "u1": 1.5, "label": "Echo"}, {"deck": "b", "stem": "drums", "u0": -8, "u1": -4, "label": "nur Höhen"}],
 "a": {"id": "5vXnH99Tn0qAh84SWU5sfc", "anchor_bar": 112, "anchor_u": -4, "until": 1,
  "lanes": {"vocals": [[-22,1],[0.5,1],[0.8,0]],
            "drums": [[-22,1],[-8,1],[-4.04,0.9],[-4,0]],
            "bass": [[-22,1],[-4.04,1],[-4,0]],
            "other": [[-22,1],[-4,1],[0,0.5],[0.8,0]]},
  "sends": [{"type": "echo", "stem": "vocals", "beats": 1.0, "feedback": 0.5, "level": 0.6, "send": [[-0.3,0],[-0.2,1],[0.6,1],[0.7,0]]}]},
 "b": {"id": "5ihq4qbjaO2r5Ga9aGlsMj", "anchor_bar": 9, "anchor_u": 0, "from": -8, "fit_bars": [1, 15],
  "gain_db": [[-8,2.5],[6,2.5],[12,1.5]],
  "lanes": {"vocals": [[-8,0],[0.97,0],[1,1]],
            "drums": [[-8,0],[-7,1.0],[-4.03,1.8],[-4,1]],
            "bass": [[-8,0],[-4.03,0],[-4,1]],
            "other": [[-8,0],[0,0.9],[2,1]]},
  "shape": "lin",
  "hpf": {"drums": [[-8,450],[-4.05,200],[-4,20]]}}
})

# 06 ─ Obsessed → Genie In a Bottle ────────────────────────────────────────
# A: eight-bar sections; the one starting at bar 86 opens with two thin bars, A leaves there.
# B: only two bars of beat before the verse (bars 1-2). They are looped as B's intro, the verse starts on bar 3.
S.append({
 "id": "h06", "swap_u": -4, "clip": [-16, 9], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-6, -4], "b": [-2, 0]},
 "key_txt": "c-Moll → f-Moll (Nachbarn, passt)",
 "technique": "Intro aus B's eigenem Beat gebaut",
 "why": "B hat nur zwei Takte Beat vor der Strophe. Die loope ich zu einem achttaktigen Intro: erst nur die Höhen unter A, dann wechseln Kick und Bass auf der Eins. A's Stimme läuft über B's Beat aus, B's Strophe beginnt genau am Anfang.",
 "steps": [
  {"u": -8, "text": "B's Drums kommen leise dazu (nur Höhen)."},
  {"u": -6, "text": "B's Melodie beginnt langsam einzublenden."},
  {"u": -4, "text": "Die Eins: Kick und Bass wechseln zu B. A's Stimme singt über B's Beat weiter."},
  {"u": -1, "text": "A's Stimme endet mit Echo, B's Auftakt („oh-oh“) kommt herein."},
  {"u": 0, "text": "B's Strophe beginnt, ab hier nur noch B."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -1, "u1": 0.5, "label": "Echo"}, {"deck": "b", "stem": "drums", "u0": -8, "u1": -4, "label": "nur Höhen"}],
 "a": {"id": "1lEIT08AUU67qef6MkKAKx", "anchor_bar": 86, "anchor_u": 0, "until": 0,
  "lanes": {"vocals": [[-16,1],[-0.62,1],[-0.5,0]],
            "drums": [[-16,1],[-8,1],[-4.03,0.9],[-4,0]],
            "bass": [[-16,1],[-4.03,1],[-4,0]],
            "other": [[-16,1],[-4,1],[-1,0.4],[0,0]]},
  "sends": [{"type": "echo", "stem": "vocals", "beats": 1.0, "feedback": 0.45, "level": 0.6, "send": [[-1.0,0],[-0.95,1],[-0.55,1],[-0.5,0]]}]},
 "b": {"id": "11mwFrKvLXCbcVGNxffGyP", "anchor_bar": 3, "anchor_u": 0, "from": -8,
  "segments": [{"u0": -8, "u1": -6, "src": -2}, {"u0": -6, "u1": -4, "src": -2}, {"u0": -4, "u1": -2, "src": -2}, {"u0": -2, "u1": None, "src": -2}],
  "lanes": {"vocals": [[-8,0],[-0.55,0],[-0.5,1]],
            "drums": [[-8,0],[-7,0.9],[-4.03,1.6],[-4,1]],
            "bass": [[-8,0],[-4.03,0],[-4,1]],
            "other": [[-8,0],[-6,0],[0,1]]},
  "shape": "lin",
  "hpf": {"drums": [[-8,600],[-4.05,220],[-4,20]]}}
})

# 07 ─ Make Me → Heads Will Roll ───────────────────────────────────────────
# A: eight bars of drum outro (96-103), no voice, no bass. B: eight bars of synth intro, beat and voice at bar 11.
S.append({
 "id": "h07", "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, 0], "b": [0, 8]},
 "key_txt": "es-Moll → gis-Moll (Nachbarn, passt)",
 "technique": "Outro-Beat von A unter dem Synth-Intro von B",
 "why": "Der klassische Club-Übergang: A endet mit acht Takten reinem Beat, B beginnt mit acht Takten Synthesizer ohne Beat. Beides liegt genau übereinander. B's Synth blendet langsam ein, auf der Eins kommen B's Beat und Stimme.",
 "steps": [
  {"u": -8, "text": "A's Gesang ist vorbei, A's Beat läuft weiter. B's Synth beginnt langsam einzublenden."},
  {"u": -4, "text": "A's Melodie ist fast weg, B's Synth ist deutlich zu hören."},
  {"u": -2, "text": "A's Drums werden hochgefiltert – Spannung vor dem Einsatz."},
  {"u": 0, "text": "Die Eins: B's Beat, Bass und Stimme setzen ein, A ist weg."}
 ],
 "fx": [{"deck": "a", "stem": "drums", "u0": -2, "u1": 0, "label": "Filter ↑"}],
 "a": {"id": "57508shL0obX0KUOUL8CJk", "anchor_bar": 96, "anchor_u": -8, "until": 0,
  "lanes": {"vocals": [[-22,1],[-8,1],[-7.9,0]],
            "drums": [[-22,1],[-0.02,1],[0,0]],
            "bass": [[-22,1],[-0.02,1],[0,0]],
            "other": [[-22,1],[-8,1],[-3,0.3],[-0.02,0.1],[0,0]]},
  "hpf": {"drums": [[-2,20],[-1,140],[-0.05,800]]}},
 "b": {"id": "5qGIfWDRdyO67LCTqA2sEn", "anchor_bar": 11, "anchor_u": 0, "from": -8, "fit_bars": [11, 40],
  "gain_db": [[-8,2.5],[8,2.5],[12,1.5]],
  "lanes": {"vocals": [[-8,0],[-0.1,0],[-0.07,1]],
            "drums": [[-8,1]], "bass": [[-8,1]],
            "other": [[-8,0],[-2,1]]},
  "shape": "lin"}
})

# 08 ─ Sprinter → Harleys In Hawaii ────────────────────────────────────────
# A: the rap ends on beat 2 of bar 130. After it bars 123-126 go on as the bed (voice out).
# B: counted in half time (70 BPM), so one ruler bar is two of B's beats. Two bars of guitar intro
#    (= 4 ruler bars), played twice, the verse starts on the one.
S.append({
 "id": "h08", "clip": [-24, 14], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, -4], "b": [0, 4]},
 "key_txt": "b-Moll → b-Moll (gleich)",
 "technique": "A's Beat trägt das Gitarren-Intro von B",
 "why": "Gleiche Tonart, fast gleiches Tempo (139 / 140). B's Gitarre blendet schon unter den letzten Rap-Zeilen langsam ein. Wenn der Rap fertig ist, läuft A's Beat allein mit B's Gitarre weiter – auf der Eins beginnt B's Strophe mit eigenem Beat.",
 "steps": [
  {"u": -8, "text": "B's Gitarre beginnt ganz leise unter A's letzten Rap-Zeilen einzublenden."},
  {"u": -4, "text": "A's Rap ist zu Ende. A's Beat und Bass laufen weiter, B's Gitarre steht jetzt im Vordergrund."},
  {"u": -2, "text": "A's Drums werden hochgefiltert."},
  {"u": 0, "text": "Die Eins: B's Strophe beginnt mit eigenem Beat und Bass, A ist weg."}
 ],
 "fx": [{"deck": "a", "stem": "drums", "u0": -2, "u1": 0, "label": "Filter ↑"}],
 "a": {"id": "6NYbDTmstd7JSUWl2hVBqR", "anchor_bar": 127, "anchor_u": -8, "until": 0,
  "segments": [{"u0": -24, "u1": -4, "src": -24}, {"u0": -4, "u1": 0.5, "src": -12}],
  "lanes": {"vocals": [[-24,1],[-4.55,1],[-4.4,0]],
            "drums": [[-24,1],[-0.02,1],[0,0]],
            "bass": [[-24,1],[-0.02,1],[0,0]],
            "other": [[-24,1],[-8,1],[-4,0.4],[-1,0.1],[0,0]]},
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
# A: 24 bars of instrumental at the end (55-78). B: verse without beat (bars 7-14), its beat starts at bar 15.
S.append({
 "id": "h09", "clip": [-22, 12], "first_event": -8, "last_event": 0, "align_pairs": {"a": [-8, 0], "b": [0, 8]},
 "key_txt": "c-Moll → f-Moll (Nachbarn, passt)",
 "technique": "A's Instrumental trägt die Strophe von B",
 "why": "A endet mit einem langen Instrumental, B hat eine Strophe ohne Beat. B's Stimme und Bass setzen direkt über A's Drums ein, B's Melodie blendet langsam ein. Genau beim ersten eigenen Beat von B ist A weg.",
 "steps": [
  {"u": -8, "text": "Bass-Wechsel: A's Bass geht raus, B's Stimme und Bass setzen direkt ein. A's Drums laufen weiter."},
  {"u": -6, "text": "B's Melodie blendet langsam ein, A's Melodie geht zurück."},
  {"u": -2, "text": "A's Drums werden hochgefiltert."},
  {"u": 0, "text": "Die Eins: B's eigener Beat setzt ein, A ist weg."}
 ],
 "fx": [{"deck": "a", "stem": "drums", "u0": -2, "u1": 0, "label": "Filter ↑"}],
 "a": {"id": "1fVQmMgrOUnrh8KgndKGzP", "anchor_bar": 71, "anchor_u": -8, "until": 0, "fit_bars": [55, 78],
  "lanes": {"vocals": [[-22,1],[-8,1],[-7.9,0]],
            "drums": [[-22,1],[-8,1],[-7,0.85],[-0.02,0.85],[0,0]],
            "bass": [[-22,1],[-8.03,1],[-8,0]],
            "other": [[-22,1],[-8,1],[-5,0.4],[-2,0]]},
  "hpf": {"drums": [[-2,20],[-1,140],[-0.05,800]]}},
 "b": {"id": "03D2TnaZuwlddRTh7P6Mq5", "anchor_bar": 15, "anchor_u": 0, "from": -8, "fit_bars": [15, 38],
  "lanes": {"vocals": [[-8,1]], "drums": [[-8,0],[-0.02,0],[0,1]], "bass": [[-8,1]], "other": [[-8,0],[-2,1]]},
  "shape": "lin"}
})

# 10 ─ Empire State Of Mind → Sexy Bitch ───────────────────────────────────
# Tempo gap (86.7 → 130). No beat overlap at all: A's beat drops out, the hook hangs in a growing reverb,
# then B starts on the one at its own tempo.
S.append({
 "id": "h10", "clip": [-12, 8], "first_event": -4, "last_event": 1, "no_sync": True,
 "key_txt": "Fis-Dur → h-Moll (Dominante → Grundton, löst sich auf)",
 "technique": "Großer Tempo-Sprung: Beat raus, Hook im Hall, dann B auf der Eins",
 "why": "Die beiden Songs sind 43 BPM auseinander, da lässt sich nichts übereinanderlegen. Also wie ein DJ am Pult: A's Beat und Bass gehen raus, die Hook steht vier Takte frei im wachsenden Hall – und genau auf der nächsten Eins startet B mit vollem Beat im eigenen Tempo. Kein Roll-out, keine Tonhöhenänderung.",
 "steps": [
  {"u": -4, "text": "A's Drums und Bass blenden über einen Takt aus. Stimme und Klavier bleiben, der Hall wächst."},
  {"u": -1, "text": "Letzter Takt der Hook, der Hall ist jetzt groß."},
  {"u": 0, "text": "Die Eins: B startet mit vollem Beat im eigenen Tempo (130). A's Hallfahne klingt darunter aus. B's Melodie blendet langsam ein."}
 ],
 "fx": [{"deck": "a", "stem": "vocals", "u0": -4, "u1": 1, "label": "Hall wächst"}],
 "a": {"id": "2igwFfvr1OAGX9SKDCPBwO", "anchor_bar": 94, "anchor_u": 0, "until": 0,
  "lanes": {"vocals": [[-12,1],[-0.12,1],[0,0]],
            "drums": [[-12,1],[-4,1],[-3,0]],
            "bass": [[-12,1],[-4,1],[-3,0]],
            "other": [[-12,1],[-4,1],[-1,0.8],[-0.05,0.7],[0,0]]},
  "sends": [{"type": "reverb", "stem": "vocals", "rt60": 2.6, "level": 0.5, "send": [[-4,0],[-1,0.6],[-0.05,1.0],[0,0]]},
            {"type": "reverb", "stem": "other", "rt60": 2.6, "level": 0.4, "send": [[-2,0],[-0.05,0.8],[0,0]]}]},
 "b": {"id": "2dGxjfhFXDIUX9h5ImyKkE", "anchor_bar": 0, "anchor_u": 0, "from": 0, "start_u": 0,
  "lanes": {"vocals": [[0,1]], "drums": [[0,1]], "bass": [[0,1]], "other": [[0,0.2],[6,1]]},
  "shape": "lin"}
})

json.dump(S, open("specs.json", "w"), indent=1)
print(len(S), "specs")
