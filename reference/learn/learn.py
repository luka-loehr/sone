#!/usr/bin/env python3
"""One round of learning from the owner's ear.

Reads every verdict he gave on a transition the system made (data/owner/verdicts.json, written by the
lab server as he types, and the archived rounds data/owner/verdicts_*.json), turns each note into a
label (good or not, and which fault), writes data/owner/labelled.json for the training data generator,
and prints how well the system's own checks agreed with his ear.

    python3 reference/learn/learn.py [repo dir]

A note it cannot read is listed as such and stays out of the training data until it is labelled by hand
(an entry in data/owner/labels_by_hand.json: {"<round>/<key>": {"good": 0|1, "issues": [...]}}).
"""
import glob, json, os, re, sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..")
OWNER = os.path.join(REPO, "data", "owner")

# What his words mean. Checked in this order; the first that matches decides.
BAD = [
    (r"trash|ass\b|shit|wtf|matsch|mush", []),
    (r"(not|nicht|aint|isn'?t)\s+(\w+\s+)?(perfect|eprfekt)?\s*(beat\s*)?sync|beat.*(doesn'?t|dont|not).*fit|not perfectly beat", ["beats_offset"]),
    (r"doesn'?t fit|dont fit|passt nicht|odd|nope|gefällt mir nicht", ["wrong_recipe"]),
    (r"hard cut|harter? schnitt", ["cut_not_transition"]),
]
GOOD = r"\b(good|nice|fits|perfect|amazing|clean|usable|gut|passt)\b"

def label(note):
    n = note.lower()
    for pattern, issues in BAD:
        if re.search(pattern, n):
            return 0, issues
    if re.search(GOOD, n):
        return 1, []
    return None, []

entries = []
for path in sorted(glob.glob(os.path.join(OWNER, "verdicts_*.json"))):
    rnd = os.path.basename(path)[len("verdicts_"):-len(".json")]
    # Round 1 was archived as a list; later rounds are the server's file as it was (with the renderer's reports).
    vs = json.load(open(path))
    for v in (vs.values() if isinstance(vs, dict) else vs):
        entries.append((rnd, f"{v['page']}:{v['id']}", v["note"], v.get("spec"), v.get("report")))
current = os.path.join(OWNER, "verdicts.json")
if os.path.exists(current):
    for key, v in json.load(open(current)).items():
        entries.append(("current", key, v.get("note", ""), v.get("spec"), v.get("report")))
by_hand = {}
if os.path.exists(os.path.join(OWNER, "labels_by_hand.json")):
    by_hand = json.load(open(os.path.join(OWNER, "labels_by_hand.json")))

out, unread = [], []
for rnd, key, note, spec, report in entries:
    hand = by_hand.get(f"{rnd}/{key}")
    good, issues = (hand["good"], hand.get("issues", [])) if hand else label(note)
    if good is None or not spec:
        unread.append((rnd, key, note))
        continue
    out.append({"id": f"{rnd}/{key}", "note": note, "good": good, "issues": issues, "spec": spec, "report": report})
json.dump(out, open(os.path.join(OWNER, "labelled.json"), "w"), indent=1, ensure_ascii=False)

print(f"{len(out)} verdicts labelled ({sum(e['good'] for e in out)} good), {len(unread)} not readable")
for rnd, key, note in unread:
    print(f"  not readable: {rnd}/{key}: {note!r}")
# How the system's own check (drums drifting apart across the blend) agreed with his ear.
agree = [e for e in out if e["report"] and e["report"].get("align")]
if agree:
    def drift(r):
        a = r["align"]
        return a["second_half"][0] - a["first_half"][0]
    rows = sorted(((abs(drift(e["report"])), e["good"], e["id"]) for e in agree))
    print("drift check against his ear (|drift| ms, his verdict):")
    for d, g, i in rows:
        print(f"  {d:5.1f} ms  {'good' if g else 'bad '}  {i}")
