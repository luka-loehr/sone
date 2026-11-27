#!/usr/bin/env bash
# Plans and renders the 13 lab pairs with the binary into out/auto2 and compares them with the owner's versions.
set -e
cd ~/djlab/orml-main
source ~/.cargo/env
cargo build --release -q -p orml 2>&1 | grep -E "^error" -A5 || true
rm -rf out/auto2 && mkdir -p out/auto2
python3 - <<PY
import json,subprocess
for s in json.load(open("data/versions/S7.json")):
    pid="p"+s["id"][1:]
    r=subprocess.run(["./target/release/orml","mix",s["a"]["id"],s["b"]["id"],"--stems","/home/luka/djlab/stems","--beats","data/analysis/beats","--out","out/auto2","--id",pid],capture_output=True,text=True)
    last=[l for l in r.stderr.splitlines() if l.startswith(pid+": ") and "audio in" in l]
    print(pid, r.returncode, (last[-1][40:150] if last else r.stderr[-400:]))
    if r.returncode==0:
        al=json.load(open(f"out/auto2/{pid}.report.json")).get("align") or {}
        print("    halves", round(al.get("first_half",[0])[0],1), round(al.get("second_half",[0])[0],1))
    if r.returncode==0:
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"out/auto2/{pid}.wav","-c:a","aac","-b:a","256k",f"out/auto2/{pid}.m4a"],check=True)
PY
python3 compare.py out
