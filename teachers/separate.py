#!/usr/bin/env python3
"""Stem separation for the DJ lab: tracks/<id>.f32 (44.1 kHz stereo f32) →
stems/<id>/{vocals,drums,bass,other}.flac with htdemucs_ft. `other` is the mix
minus the three others, so the four stems sum to the song exactly."""
import os, sys, time, glob
import numpy as np, torch, soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model

ROOT = os.path.expanduser("~/djlab")
model = get_model("htdemucs_ft")
model.eval()
dev = "cuda"
names = model.sources  # drums, bass, other, vocals
ids = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/tracks/*.f32"))
for i, tid in enumerate(ids):
    out = f"{ROOT}/stems/{tid}"
    if os.path.exists(f"{out}/other.flac"):
        continue
    t0 = time.time()
    x = np.fromfile(f"{ROOT}/tracks/{tid}.f32", dtype=np.float32).reshape(-1, 2).T.copy()
    wav = torch.from_numpy(x)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        est = apply_model(model, ((wav - mean) / std)[None], device=dev, shifts=2, split=True, overlap=0.25, progress=False)[0]
    est = (est * std + mean).cpu().numpy()
    stems = {n: est[k] for k, n in enumerate(names)}
    stems["other"] = x - stems["vocals"] - stems["drums"] - stems["bass"]
    os.makedirs(out, exist_ok=True)
    for n, s in stems.items():
        sf.write(f"{out}/{n}.flac", s.T, 44100, subtype="PCM_24")
    print(f"{i+1}/{len(ids)} {tid} {x.shape[1]/44100:.0f}s in {time.time()-t0:.1f}s", flush=True)
print("done")
