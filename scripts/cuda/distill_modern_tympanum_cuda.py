#!/usr/bin/env python3
"""GPU Stem-Isolated Silver Label Distiller & MUSDB18-HQ Annotator for Tympanum v2.
Separates modern Creative Commons full songs into clean `vocals`, `drums`, `bass`, `other`
on GPU via HTDemucs/HDemucs-Plus, then extracts stem-isolated ground-truth labels:
  - `beats` & `downbeats`: tracked on isolated `drums + bass` (unmasked kick/snare transients + kick-vs-snare bar phase)
  - `vocals` & `line_ends` (singer breaths): measured on isolated `vocals` RMS envelope (-30 dB gate)
  - `phrase_starts`: 8-bar structural novelty boundaries
  - `key` (24 major/minor) & `tuning_cents`: Krumhansl-Temperley chroma + tuning estimation on isolated `bass + other`.

IMPORTANT: Uses `multiprocessing.get_context("spawn")` with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
isolated per-PID Numba cache directories, zero top-level `torch` imports in CPU workers, and per-future
exception isolation so workers never segfault or deadlock on Python 3.13."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_CACHE_DIR"] = f"/tmp/numba_tympanum_{os.getpid()}"

import argparse
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["NUMBA_CACHE_DIR"] = f"/tmp/numba_tympanum_{os.getpid()}"
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)


def estimate_key_and_tuning(harm_mono: np.ndarray, sr: int = 44100):
    tuning_frac = float(librosa.estimate_tuning(y=harm_mono, sr=sr))
    tuning_cents = float(np.clip(tuning_frac * 100.0, -49.0, 49.0))
    chroma = librosa.feature.chroma_cqt(y=harm_mono, sr=sr, tuning=tuning_frac)
    c_vec = np.mean(chroma, axis=1).astype(np.float64)
    if np.std(c_vec) < 1e-6:
        return None, tuning_cents
    c_norm = (c_vec - np.mean(c_vec)) / (np.std(c_vec) + 1e-8)
    best_corr = -1e9
    best_key = "C major"
    for root in range(12):
        maj_p = np.roll(MAJOR_PROFILE, root)
        maj_n = (maj_p - np.mean(maj_p)) / np.std(maj_p)
        cm = float(np.dot(c_norm, maj_n))
        if cm > best_corr:
            best_corr = cm
            best_key = f"{NOTE_NAMES[root]} major"
        min_p = np.roll(MINOR_PROFILE, root)
        min_n = (min_p - np.mean(min_p)) / np.std(min_p)
        cmin = float(np.dot(c_norm, min_n))
        if cmin > best_corr:
            best_corr = cmin
            best_key = f"{NOTE_NAMES[root]} minor"
    return best_key, tuning_cents


def annotate_one_song(args_tuple):
    sid, mix_mono, voc_mono, drm_mono, bas_mono, oth_mono, out_dir = args_tuple
    out_dir = Path(out_dir)
    sr = 44100
    try:
        perc = drm_mono + 0.6 * bas_mono
        if float(np.sqrt(np.mean(perc ** 2))) < 0.008:
            return False
        _tempo, beat_frames = librosa.beat.beat_track(y=perc, sr=sr, hop_length=512, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512).tolist()
        if len(beat_times) < 24:
            return False

        # Determine 4/4 downbeat phase via low-band kick energy (30..140 Hz) vs mid-band snare on beats
        S_drm = np.abs(librosa.stft(drm_mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        kick_env = np.mean(S_drm[(freqs >= 30) & (freqs <= 140), :], axis=0)
        snare_env = np.mean(S_drm[(freqs >= 180) & (freqs <= 2500), :], axis=0)
        votes = np.zeros(4, dtype=np.float64)
        for bi, bf in enumerate(beat_frames):
            lo, hi = max(0, bf - 2), min(S_drm.shape[1], bf + 3)
            k_e = float(np.max(kick_env[lo:hi])) if hi > lo else 0.0
            s_e = float(np.max(snare_env[lo:hi])) if hi > lo else 0.0
            votes[bi % 4] += 1.5 * k_e - 0.35 * s_e
            votes[(bi - 1) % 4] += 0.45 * s_e
        down_phase = int(np.argmax(votes))
        downbeats = [beat_times[i] for i in range(down_phase, len(beat_times), 4)]
        phrase_starts = [downbeats[i] for i in range(0, len(downbeats), 8)]

        # Vocal intervals & breath/line_ends from clean isolated vocal stem (50 Hz frames)
        hop_v = 882
        n_vf = len(voc_mono) // hop_v
        voc_rms = np.array([
            np.sqrt(np.mean(voc_mono[i * hop_v:(i + 1) * hop_v] ** 2) + 1e-12)
            for i in range(n_vf)
        ])
        peak_v = float(np.percentile(voc_rms, 95)) if n_vf > 0 else 0.0
        voc_segs = []
        line_ends = []
        if peak_v > 0.012:
            thr = max(0.005, peak_v * 0.12)
            active = voc_rms >= thr
            for i in range(1, n_vf - 6):
                if active[i - 1] and not active[i]:
                    if np.any(active[i + 1:i + 7]):
                        active[i:i + 6] = True
            in_seg = False
            s_idx = 0
            for i in range(n_vf):
                if active[i] and not in_seg:
                    in_seg = True
                    s_idx = i
                elif not active[i] and in_seg:
                    in_seg = False
                    dur = (i - s_idx) / 50.0
                    if dur >= 0.25:
                        voc_segs.append([round(s_idx / 50.0, 3), round(i / 50.0, 3)])
            if in_seg and (n_vf - s_idx) / 50.0 >= 0.25:
                voc_segs.append([round(s_idx / 50.0, 3), round(n_vf / 50.0, 3)])

            for j, (sv, ev) in enumerate(voc_segs):
                next_s = voc_segs[j + 1][0] if j + 1 < len(voc_segs) else 9999.0
                if (ev - sv) >= 0.65 and (next_s - ev) >= 0.28:
                    line_ends.append(ev)

        harm_mono = bas_mono + oth_mono + 0.35 * voc_mono
        key_str, tuning_cents = estimate_key_and_tuning(harm_mono, sr)

        wav_path = out_dir / f"{sid}.flac"
        sf.write(str(wav_path), np.clip(mix_mono, -1.0, 1.0), sr, subtype="PCM_16")
        meta = {
            "audio": f"{sid}.flac",
            "beats": [round(t, 4) for t in beat_times],
            "downbeats": [round(t, 4) for t in downbeats],
            "vocals": voc_segs,
            "line_ends": line_ends,
            "phrase_starts": [round(t, 4) for t in phrase_starts],
            "key": key_str,
            "tuning_cents": round(tuning_cents, 2),
        }
        with open(out_dir / f"{sid}.json", "w") as f:
            json.dump(meta, f)
        return True
    except Exception as e:
        print(f"[AnnotateWarn] {sid}: {e}", flush=True)
        return False


def prep_musdb_dir(args_t):
    idx, sdir_str, out_d_str = args_t
    sdir = Path(sdir_str)
    out_d = Path(out_d_str)
    clean = re.sub(r"[^A-Za-z0-9]+", "_", sdir.name)[:24].strip("_")
    sid = f"mus_{idx:03d}_{clean}"
    if (out_d / f"{sid}.json").exists() and (out_d / f"{sid}.flac").exists():
        return True
    try:
        stems_m = {}
        for nm in ("vocals", "drums", "bass", "other"):
            x, _ = sf.read(str(sdir / f"{nm}.wav"), start=15 * 44100, frames=60 * 44100, dtype="float32", always_2d=True)
            stems_m[nm] = x.mean(axis=1)
        mix_m = stems_m["vocals"] + stems_m["drums"] + stems_m["bass"] + stems_m["other"]
        return annotate_one_song((sid, mix_m, stems_m["vocals"], stems_m["drums"], stems_m["bass"], stems_m["other"], str(out_d)))
    except Exception as e:
        print(f"[MUSDBWarn] {sid}: {e}", flush=True)
        return False


def make_spawn_pool(max_workers: int = 4):
    ctx = mp.get_context("spawn")
    return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init)


def run_tasks_isolated(pool_holder: list, fn, tasks: list, max_workers: int = 4, label: str = "Pool"):
    """Submit tasks to a spawn ProcessPoolExecutor with per-future try/except and automatic pool recovery."""
    results = []
    # Process in chunks of 16 so memory stays bounded and progress prints steadily
    chunk_sz = 16
    for start in range(0, len(tasks), chunk_sz):
        sub = tasks[start:start + chunk_sz]
        if pool_holder[0] is None:
            pool_holder[0] = make_spawn_pool(max_workers)
        ex = pool_holder[0]
        try:
            futs = [(t, ex.submit(fn, t)) for t in sub]
            for t, fut in futs:
                try:
                    results.append(bool(fut.result(timeout=180)))
                except BrokenProcessPool as bp:
                    print(f"[{label}] BrokenProcessPool ({bp}); recreating spawn pool and retrying item inline...", flush=True)
                    try:
                        ex.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    pool_holder[0] = None
                    try:
                        results.append(bool(fn(t)))
                    except Exception:
                        results.append(False)
                    break
                except Exception as e:
                    print(f"[{label}] Task warning: {e}", flush=True)
                    results.append(False)
            # If pool broke mid-chunk, finish remaining items of `sub` inline then fresh pool next chunk
            done_n = len(results) - start
            if done_n < len(sub):
                for rem_t in sub[done_n:]:
                    try:
                        results.append(bool(fn(rem_t)))
                    except Exception:
                        results.append(False)
        except Exception as e:
            print(f"[{label}] Sub-batch fallback ({e})", flush=True)
            pool_holder[0] = None
            for rem_t in sub[len(results) - start:]:
                try:
                    results.append(bool(fn(rem_t)))
                except Exception:
                    results.append(False)
        if len(tasks) > chunk_sz:
            print(f"[{label}] Progress: {sum(1 for r in results if r)}/{len(results)} ok (of {len(tasks)} total)", flush=True)
    return results


def run_prep_musdb(musdb_raw: Path, musdb_train: Path, musdb_valid: Path, workers: int = 4):
    musdb_train.mkdir(parents=True, exist_ok=True)
    musdb_valid.mkdir(parents=True, exist_ok=True)
    tasks = []
    idx = 0
    train_dir = musdb_raw / "train"
    test_dir = musdb_raw / "test"
    if train_dir.exists():
        for p in sorted(train_dir.iterdir()):
            if p.is_dir():
                tasks.append((idx, str(p), str(musdb_train)))
                idx += 1
    if test_dir.exists():
        for i, p in enumerate(sorted(test_dir.iterdir())):
            if p.is_dir():
                tasks.append((idx, str(p), str(musdb_valid if i < 16 else musdb_train)))
                idx += 1
    t0 = time.perf_counter()
    pool_holder = [make_spawn_pool(workers)]
    try:
        res = run_tasks_isolated(pool_holder, prep_musdb_dir, tasks, max_workers=workers, label="MUSDB")
    finally:
        if pool_holder[0] is not None:
            pool_holder[0].shutdown(wait=True)
    ok = sum(1 for r in res if r)
    print(f"[MUSDB] Annotated {ok}/{len(tasks)} MUSDB18-HQ studio tracks with full 7-head labels in {time.perf_counter()-t0:.1f}s!", flush=True)


def run_distill(args):
    import torch
    from distill_modern_stems_cuda import load_teacher, stream_raw_songs

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.json")))
    if existing >= min(args.max_songs, 430):
        print(f"[TympanumDistill] Already have {existing} songs in {out_dir}, skipping!", flush=True)
        return

    pool_holder = [make_spawn_pool(args.workers)]
    try:
        device = torch.device("cuda")
        teacher_fn, t_name = load_teacher(device)
        t0 = time.perf_counter()
        saved = existing
        buf_ids, buf_wavs = [], []

        def flush():
            nonlocal saved
            if not buf_wavs:
                return
            min_l = (min(w.shape[1] for w in buf_wavs) // 1024) * 1024
            batch = torch.stack([w[:, :min_l] for w in buf_wavs], dim=0)
            stems = teacher_fn(batch)
            tasks = []
            for bi, sid in enumerate(buf_ids):
                mix_m = batch[bi].mean(dim=0).numpy()
                voc_m = stems["vocals"][bi].mean(dim=0).numpy()
                drm_m = stems["drums"][bi].mean(dim=0).numpy()
                bas_m = stems["bass"][bi].mean(dim=0).numpy()
                oth_m = stems["other"][bi].mean(dim=0).numpy()
                tasks.append((sid, mix_m, voc_m, drm_m, bas_m, oth_m, str(out_dir)))
            results = run_tasks_isolated(pool_holder, annotate_one_song, tasks, max_workers=args.workers, label="DistillCPU")
            saved += sum(1 for r in results if r)
            buf_ids.clear()
            buf_wavs.clear()
            print(f"[TympanumDistill:{t_name}] Annotated {saved}/{args.max_songs} stem-isolated tracks ({time.perf_counter()-t0:.1f}s)", flush=True)

        cache_dir = Path("/content/hf_ear_cache")
        for sid, wav in stream_raw_songs(args.shard_offset, args.max_songs - existing + 80, cache_dir, args.clip_seconds):
            if saved >= args.max_songs:
                break
            if (out_dir / f"{sid}.json").exists():
                continue
            buf_ids.append(sid)
            buf_wavs.append(wav)
            if len(buf_wavs) >= 8:
                flush()
        flush()
        print(f"[TympanumDistill] DONE! {saved} tracks ready in {out_dir}", flush=True)
    finally:
        if pool_holder[0] is not None:
            pool_holder[0].shutdown(wait=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-musdb", default=None, help="Path to unpacked MUSDB18-HQ root to annotate in standalone mode")
    ap.add_argument("--musdb-train", default="data/musdb/train")
    ap.add_argument("--musdb-valid", default="data/musdb/valid")
    ap.add_argument("--out", default="data/tympanum_distilled")
    ap.add_argument("--max-songs", type=int, default=450)
    ap.add_argument("--shard-offset", type=int, default=3)
    ap.add_argument("--clip-seconds", type=float, default=45.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    if args.prep_musdb:
        run_prep_musdb(Path(args.prep_musdb), Path(args.musdb_train), Path(args.musdb_valid), workers=args.workers)
    else:
        run_distill(args)


if __name__ == "__main__":
    main()
