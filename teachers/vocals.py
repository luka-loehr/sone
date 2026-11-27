#!/usr/bin/env python3
"""Cleaner vocal stem with BS-RoFormer (viperx 1297): stems/<id>/vocals_r.flac.
The other three stems stay from htdemucs_ft; `other` is recomputed by the mixer as
mix − vocals_r − drums − bass, so the four still sum to the song exactly."""
import os, sys, glob, time, shutil, tempfile
import numpy as np, soundfile as sf
from audio_separator.separator import Separator

ROOT = os.path.expanduser("~/djlab")
first = sys.argv[1:]
ids = first + [i for i in sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/tracks/*.f32")) if i not in first]
tmp = tempfile.mkdtemp(dir=f"{ROOT}/hand")
sep = Separator(output_dir=tmp, output_format="FLAC", model_file_dir=f"{ROOT}/hand/models", log_level=30)
sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
for i, tid in enumerate(ids):
    out = f"{ROOT}/stems/{tid}/vocals_r.flac"
    if os.path.exists(out):
        continue
    t0 = time.time()
    x = np.fromfile(f"{ROOT}/tracks/{tid}.f32", dtype=np.float32).reshape(-1, 2)
    wav = f"{tmp}/{tid}.wav"
    sf.write(wav, x, 44100, subtype="FLOAT")
    files = sep.separate(wav)
    voc = [f for f in files if "Vocals" in f or "vocals" in f][0]
    v, sr = sf.read(os.path.join(tmp, voc), dtype="float32")
    n = len(x)
    if len(v) < n:
        v = np.vstack([v, np.zeros((n - len(v), 2), dtype=np.float32)])
    sf.write(out, v[:n], 44100, subtype="PCM_24")
    for f in files:
        os.remove(os.path.join(tmp, f))
    os.remove(wav)
    print(f"{i+1}/{len(ids)} {tid} in {time.time()-t0:.1f}s", flush=True)
shutil.rmtree(tmp, ignore_errors=True)
print("done")
