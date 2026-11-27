#!/usr/bin/env python3
"""Writes a song's final stems as raw 16-bit PCM, the way a Worker's cache would hold them:
<out>/<id>/{vocals,drums,bass,other}.s16 (interleaved stereo, 44.1 kHz) and meta.json (frames, lufs).

prepare.py <stems dir> <out dir> <id>…     needs numpy, soundfile, pyloudnorm"""
import json, os, sys
import numpy as np, soundfile as sf, pyloudnorm

stems_dir, out_dir, ids = sys.argv[1], sys.argv[2], sys.argv[3:]
for i in ids:
    d = os.path.join(stems_dir, i)
    rd = lambda name: sf.read(os.path.join(d, name + ".flac"), dtype="float32")[0]
    v, dr, b, o = rd("vocals"), rd("drums"), rd("bass"), rd("other")
    n = min(len(v), len(dr), len(b), len(o))
    mix = v[:n] + dr[:n] + b[:n] + o[:n]
    if os.path.exists(os.path.join(d, "vocals_r.flac")):
        v = rd("vocals_r")[:n]
        if os.path.exists(os.path.join(d, "drums_i.flac")) and os.path.exists(os.path.join(d, "bass_i.flac")):
            dr, b = rd("drums_i")[:n], rd("bass_i")[:n]
        o = mix - v[:n] - dr[:n] - b[:n]
    os.makedirs(os.path.join(out_dir, i), exist_ok=True)
    for name, x in (("vocals", v), ("drums", dr), ("bass", b), ("other", o)):
        pcm = np.clip(np.round(x[:n] * 32767.0), -32768, 32767).astype("<i2")
        pcm.tofile(os.path.join(out_dir, i, name + ".s16"))
    lufs = float(pyloudnorm.Meter(44100).integrated_loudness(mix.astype(np.float64)))
    json.dump({"frames": int(n), "lufs": lufs}, open(os.path.join(out_dir, i, "meta.json"), "w"))
    print(i, n, round(lufs, 2))
