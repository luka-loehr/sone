"""Planner render (out/auto2/pNN) against the owner's version through the same renderer (out/handrust/hNN):
per ruler bar around the one, the level of the mix and of every stem of both decks."""
import json, sys
D = sys.argv[1] if len(sys.argv) > 1 else "out"
def rows(path):
    r = json.load(open(path))
    return {row["u"]: row for row in r["bars"]}, r
STEMS = ["v", "d", "b", "o"]
for i in range(1, 14):
    h, hr = rows(f"{D}/handrust/h{i:02d}.report.json")
    p, pr = rows(f"{D}/auto2/p{i:02d}.report.json")
    worst = []
    for u in range(-10, 5):
        if u not in h or u not in p:
            continue
        d = p[u]["mix"] - h[u]["mix"]
        stems = []
        for side in "ab":
            for st in STEMS:
                k = f"{side}_{st}"
                if k in h[u] and k in p[u]:
                    hv, pv = h[u][k], p[u][k]
                    on_h, on_p = hv > -30, pv > -30
                    if on_h != on_p or (on_h and abs(pv - hv) > 6):
                        stems.append(f"{k} {hv:.0f}/{pv:.0f}")
        worst.append((u, round(d, 1), stems))
    big = [w for w in worst if abs(w[1]) > 3 or w[2]]
    mx = max((abs(w[1]) for w in worst), default=0)
    print(f"p{i:02d}: mix differs by up to {mx:.1f} dB in bars -10..+4; bars that differ: {len(big)}")
    for u, d, st in big[:6]:
        print(f"     u {u:+3d}: mix {d:+5.1f} dB  {'; '.join(st)}")
