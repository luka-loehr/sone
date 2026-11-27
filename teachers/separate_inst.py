#!/usr/bin/env python3
"""Drums and bass from the song WITHOUT its voice: Demucs runs on (mix − BS-RoFormer vocals), so no
vocal can leak into the drum or bass stem. stems/<id>/drums_i.flac, bass_i.flac. `other` stays the rest."""
import os, sys, time
import numpy as np, torch, soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
ROOT = os.path.expanduser("~/djlab")
model = get_model("htdemucs_ft"); model.eval()
names = model.sources
for i, tid in enumerate(sys.argv[1:]):
    d = f"{ROOT}/stems/{tid}"
    if os.path.exists(f"{d}/bass_i.flac") or not os.path.exists(f"{d}/vocals_r.flac"):
        print("skip", tid, flush=True); continue
    t0 = time.time()
    mix = np.fromfile(f"{ROOT}/tracks/{tid}.f32", dtype=np.float32).reshape(-1, 2)
    v = sf.read(f"{d}/vocals_r.flac", dtype="float32")[0][: len(mix)]
    inst = (mix[: len(v)] - v).T.copy()
    wav = torch.from_numpy(inst)
    ref = wav.mean(0); mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        est = apply_model(model, ((wav - mean) / std)[None], device="cuda", shifts=2, split=True, overlap=0.25, progress=False)[0]
    est = (est * std + mean).cpu().numpy()
    for k, n in enumerate(names):
        if n in ("drums", "bass"):
            sf.write(f"{d}/{n}_i.flac", est[k].T, 44100, subtype="PCM_24")
    print(f"{i+1} {tid} in {time.time()-t0:.1f}s", flush=True)
print("done")
