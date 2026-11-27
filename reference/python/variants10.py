#!/usr/bin/env python3
"""Five ways to get from Empire State Of Mind (86.65 BPM, F sharp major) to Sexy Bitch (130 BPM, B minor).
The keys clash (A sharp / D sharp against A / D), so nothing pitched of both may sound together for long;
the tempos are 2:3, so B is locked to A's bar lines (one ruler bar = 4 beats of A = 6 beats of B).
variants10.py → variants10.json"""
import json
from specs import lin_fade, hold_tail, at_one, cut_in, echo

A_ID, B_ID = "2igwFfvr1OAGX9SKDCPBwO", "2dGxjfhFXDIUX9h5ImyKkE"
def base(i, a, b, first, last):
    a = dict({"id": A_ID, "anchor_bar": 94, "anchor_u": 0, "shape": "lin"}, **a)
    b = dict({"id": B_ID, "anchor_bar": 0, "anchor_u": 0, "beats_per_bar": 6, "fit_bars": [0, 23], "shape": "lin"}, **b)
    return {"id": i, "clip": [-12, 9], "first_event": first, "last_event": last, "align_pairs": {"a": [-8, -4], "b": [0, 4]}, "a": a, "b": b}
REV = lambda stem, pts, level=0.5: {"type": "reverb", "stem": stem, "rt60": 2.6, "level": level, "send": pts}
V = []
# a: echo-out, B drops on the one
V.append(base("h10a",
  {"until": 0, "lanes": {"vocals": hold_tail(-12), "drums": hold_tail(-12), "bass": hold_tail(-12), "other": hold_tail(-12)},
   "sends": [echo("vocals", -0.5, 0.0, beats=1.0, level=0.9), REV("other", [[-0.5,0],[-0.25,1],[0,1],[0.02,0]], 0.5)]},
  {"from": 0, "lanes": {"vocals": [[0,1]], "drums": [[0,1]], "bass": [[0,1]], "other": [[0,0.4],[2,1]]}}, -0.5, 1))
# b: breakdown, B's beat rises from under water, A's voice rides two bars, B's synth only after it
V.append(base("h10b",
  {"until": 2, "lanes": {"vocals": [[-12,1],[0.5,1],[2,0]], "drums": lin_fade(-12, -4, -2), "bass": lin_fade(-12, -4, -2), "other": lin_fade(-12, -2, 0)},
   "sends": [REV("vocals", [[-4,0],[-1,0.5],[0,0.8],[1.2,0.8],[2,0]])]},
  {"from": -2, "segments": [{"u0": -2, "u1": 0, "src": 0}, {"u0": 0, "u1": None, "src": 0}],
   "lanes": {"vocals": [[-2,1]], "drums": [[-2,0],[-1.5,1.0],[0,1.0]], "bass": cut_in(-2, 0), "other": at_one(-2, 2.667)},
   "lpf": {"drums": [[-2,250],[-1,700],[-0.25,4000],[0,20000]]}}, -4, 2.667))
# c: A fades to the one, B's beat starts hard on it, A's voice rides two bars
V.append(base("h10c",
  {"until": 2, "lanes": {"vocals": [[-12,1],[0.5,1],[2,0]], "drums": lin_fade(-12, -5), "bass": lin_fade(-12, -5), "other": lin_fade(-12, -5)},
   "sends": [REV("vocals", [[-4,0],[-1,0.5],[0,0.8],[1.2,0.8],[2,0]])]},
  {"from": 0, "lanes": {"vocals": [[0,1]], "drums": [[0,1]], "bass": [[0,1]], "other": at_one(0, 2.667)}}, -5, 2.667))
# d: A stays whole until the one and is cut there, B's beat rises from under water before it
V.append(base("h10d",
  {"until": 0, "lanes": {"vocals": hold_tail(-12), "drums": hold_tail(-12), "bass": hold_tail(-12), "other": hold_tail(-12)},
   "sends": [echo("vocals", -0.5, 0.0, beats=1.0, level=0.9)]},
  {"from": -2, "segments": [{"u0": -2, "u1": 0, "src": 0}, {"u0": 0, "u1": None, "src": 0}],
   "lanes": {"vocals": [[-2,1]], "drums": [[-2,0],[-1.5,0.8],[0,1.0]], "bass": cut_in(-2, 0), "other": at_one(-2, 0)},
   "lpf": {"drums": [[-2,200],[-1,500],[-0.25,3000],[0,20000]]}}, -2, 1))
# e: the hook alone: A's band leaves two bars before the one, the voice stands free in a big room, B on the one,
#    the voice's last note rings out over B's beat, B's synth after it
V.append(base("h10e",
  {"until": 1.5, "lanes": {"vocals": [[-12,1],[0,1],[1.5,0]], "drums": lin_fade(-12, -3, -2), "bass": lin_fade(-12, -3, -2), "other": lin_fade(-12, -3, -1)},
   "sends": [REV("vocals", [[-3,0],[-2,0.6],[0,1.0],[1.0,0.6],[1.5,0]], 0.6), echo("vocals", -0.25, 0.5, beats=1.0, level=0.6)]},
  {"from": 0, "lanes": {"vocals": [[0,1]], "drums": [[0,1]], "bass": [[0,1]], "other": at_one(0, 2.667)}}, -3, 2.667))
json.dump(V, open("variants10.json", "w"), indent=1)
print(len(V), "variants")
