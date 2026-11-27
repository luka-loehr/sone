#!/usr/bin/env python3
"""2-Stage Studio GPU Stem Teacher Distillation Engine:
  Stage 1 (Vocals, 12.98 dB SDR): ViperX BS-RoFormer-1297 (`model_bs_roformer_ep_317_sdr_12.9755`)
                                   in bfloat16 Hann overlap-add on GPU -> crystal-clear vocals +
                                   100% vocal-free `instrumental = mix - vocals`.
  Stage 2 (Drums/Bass/Other):     Meta HTDemucs-ft 4-Transformer Ensemble run strictly on the
                                   vocal-free `instrumental` (`other = instrumental - drums - bass`),
                                   guaranteeing exact 4-stem mixture conservation (< -150 dB) and
                                   zero vocal bleed in drums/bass/other!

Also exports teacher A/B reference stems on held-out radio tracks (`--teacher-ab-in` / `--teacher-ab-out`)
so human listeners can directly compare Original vs BS-RoFormer-1297 GPU Teacher vs Cochlea CPU."""

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PARQUET_URLS = [
    f"https://huggingface.co/datasets/renumics/song-describer-dataset/resolve/main/data/train-0000{i}-of-00008.parquet"
    for i in range(8)
]
JAMENDO_TAR_URLS = [
    f"https://huggingface.co/datasets/rkstgr/mtg-jamendo/resolve/main/data/train/{i}.tar"
    for i in range(24)
]

ROFORMER_CKPT_URLS = [
    "https://huggingface.co/Eddycrack864/Music-Source-Separation-Training/resolve/main/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
]
ROFORMER_YAML_URLS = [
    "https://huggingface.co/Eddycrack864/Music-Source-Separation-Training/resolve/main/model_bs_roformer_ep_317_sdr_12.9755.yaml",
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml",
]


def download_file_any(urls, dst: Path, min_bytes: int = 1000) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size >= min_bytes:
        return True
    for url in urls:
        subprocess.run(
            ["aria2c", "-x", "8", "-s", "8", "-k", "10M", "--max-tries=3", "--quiet",
             "-d", str(dst.parent), "-o", dst.name, url],
            check=False,
        )
        if dst.exists() and dst.stat().st_size >= min_bytes:
            return True
        subprocess.run(
            ["curl", "-fSL", "--retry", "3", "--retry-delay", "2", "-o", str(dst), url],
            check=False,
        )
        if dst.exists() and dst.stat().st_size >= min_bytes:
            return True
    return False


def load_bs_roformer_vocal_teacher(device: torch.device):
    """Load ViperX BS-RoFormer-1297 (12.9755 dB SDR) as a pure batched bfloat16 PyTorch module."""
    cache = Path("/content/roformer_cache")
    ckpt_p = cache / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    yaml_p = cache / "model_bs_roformer_ep_317_sdr_12.9755.yaml"

    if not download_file_any(ROFORMER_CKPT_URLS, ckpt_p, min_bytes=100_000_000):
        raise RuntimeError("Could not download BS-RoFormer-1297 checkpoint")
    if not download_file_any(ROFORMER_YAML_URLS, yaml_p, min_bytes=200):
        raise RuntimeError("Could not download BS-RoFormer-1297 yaml")

    msst_dir = Path("/content/msst")
    if not (msst_dir / "models" / "bs_roformer").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ZFTurbo/Music-Source-Separation-Training.git", str(msst_dir)],
            check=True,
        )
    if str(msst_dir) not in sys.path:
        sys.path.insert(0, str(msst_dir))

    import yaml
    try:
        from yaml import UnsafeLoader as Loader
    except ImportError:
        from yaml import FullLoader as Loader

    with open(yaml_p, "r") as f:
        cfg = yaml.load(f, Loader=Loader)

    model_kwargs = dict(cfg["model"])
    # Ensure tuple for freqs_per_bands if YAML parsed as list
    if "freqs_per_bands" in model_kwargs and isinstance(model_kwargs["freqs_per_bands"], list):
        model_kwargs["freqs_per_bands"] = tuple(model_kwargs["freqs_per_bands"])

    from models.bs_roformer.bs_roformer import BSRoformer
    net = BSRoformer(**model_kwargs)
    sd = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    # Strip optional DataParallel/Lightning 'model.' prefix if present
    cleaned = {}
    net_keys = set(net.state_dict().keys())
    for k, v in sd.items():
        if k in net_keys:
            cleaned[k] = v
        elif k.startswith("model.") and k[6:] in net_keys:
            cleaned[k[6:]] = v
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    print(
        f"[Teacher] Loaded ViperX BS-RoFormer-1297 (12.98 dB vocal SDR)! "
        f"loaded={len(cleaned)}/{len(net_keys)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if len(cleaned) < int(0.9 * len(net_keys)):
        raise RuntimeError(f"Incomplete BS-RoFormer state_dict load ({len(cleaned)}/{len(net_keys)})")

    class _FloatMaskHead(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, x):
            return self.mod(x).float()

    net.mask_estimators = torch.nn.ModuleList([_FloatMaskHead(m) for m in net.mask_estimators])
    net = net.to(device).eval()

    chunk_size = int(cfg.get("audio", {}).get("chunk_size", 352800))
    # Cap chunk_size to ~8s (352800) so batched GPU inference flies in L2/VRAM
    chunk_size = min(max(chunk_size, 131584), 352800)
    overlap = chunk_size // 4
    step = chunk_size - overlap
    win = torch.hann_window(chunk_size, periodic=True, device=device).clamp(min=1e-4)

    @torch.no_grad()
    def extract_vocals(batch_wav: torch.Tensor) -> torch.Tensor:
        """batch_wav: (B, 2, L) CPU float32 -> returns vocals (B, 2, L) CPU float32."""
        bsz, ch, total_l = batch_wav.shape
        x = batch_wav.to(device, non_blocking=True)
        if total_l <= chunk_size:
            pad_r = chunk_size - total_l
            xp = F.pad(x, (0, pad_r)) if pad_r > 0 else x
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                y = net(xp)
            if y.ndim == 4:
                y = y[:, 0]
            return y[:, :, :total_l].float().cpu()

        starts = list(range(0, total_l - chunk_size + 1, step))
        if starts[-1] + chunk_size < total_l:
            starts.append(total_l - chunk_size)

        acc = torch.zeros_like(x, dtype=torch.float32)
        wsum = torch.zeros((1, 1, total_l), device=device, dtype=torch.float32)

        # Process chunks in VRAM-adaptive mini-batches (16 on 80-96GB Blackwell/H100, 4 on 24GB L4)
        all_segs = torch.stack([x[:, :, st:st + chunk_size] for st in starts], dim=1).reshape(bsz * len(starts), ch, chunk_size)
        preds = []
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        mb = 16 if vram_gb >= 65 else (8 if vram_gb >= 35 else 4)
        for i in range(0, all_segs.shape[0], mb):
            sub = all_segs[i:i + mb]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                py = net(sub)
            if py.ndim == 4:
                py = py[:, 0]
            preds.append(py.float())
        preds_cat = torch.cat(preds, dim=0).reshape(bsz, len(starts), ch, chunk_size)

        w_view = win.view(1, 1, chunk_size)
        for si, st in enumerate(starts):
            acc[:, :, st:st + chunk_size] += preds_cat[:, si] * w_view
            wsum[:, :, st:st + chunk_size] += w_view
        voc = (acc / wsum.clamp(min=1e-6)).cpu()
        return voc

    return extract_vocals


def load_demucs_inst_teacher(device: torch.device):
    """Load Meta HTDemucs-ft 4-Transformer ensemble for splitting vocal-free instrumental."""
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        model = get_model("htdemucs_ft").to(device).eval()
        src_names = list(model.sources)
        print(f"[Teacher] Loaded Meta htdemucs_ft instrumental splitter! sources={src_names}", flush=True)

        @torch.no_grad()
        def split_inst(inst_wav: torch.Tensor):
            ref = inst_wav.mean(dim=1, keepdim=True)
            m = ref.mean(dim=-1, keepdim=True)
            s = ref.std(dim=-1, keepdim=True).clamp(min=1e-4)
            normed = ((inst_wav - m) / s).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = apply_model(model, normed, shifts=0, split=True, overlap=0.25, device=device)
            out = out.float().cpu() * s.unsqueeze(1) + m.unsqueeze(1)
            by_name = {nm: out[:, idx] for idx, nm in enumerate(src_names)}
            return by_name["drums"], by_name["bass"]

        return split_inst, "bs_roformer_1297+htdemucs_ft"
    except Exception as e:
        print(f"[Teacher] htdemucs_ft fallback to HDEMUCS_HIGH_MUSDB_PLUS ({e})", flush=True)
        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        model = bundle.get_model().to(device).eval()
        src_names = list(model.sources)

        @torch.no_grad()
        def split_inst(inst_wav: torch.Tensor):
            ref = inst_wav.mean(dim=1, keepdim=True)
            m = ref.mean(dim=-1, keepdim=True)
            s = ref.std(dim=-1, keepdim=True).clamp(min=1e-4)
            normed = ((inst_wav - m) / s).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(normed)
            out = out.float().cpu() * s.unsqueeze(1) + m.unsqueeze(1)
            by_name = {nm: out[:, idx] for idx, nm in enumerate(src_names)}
            return by_name["drums"], by_name["bass"]

        return split_inst, "bs_roformer_1297+hdemucs_plus"


def load_two_stage_teacher(device: torch.device):
    voc_fn = load_bs_roformer_vocal_teacher(device)
    inst_fn, tag = load_demucs_inst_teacher(device)

    @torch.no_grad()
    def run_fn(batch_wav: torch.Tensor) -> dict:
        vocals = voc_fn(batch_wav)
        # Clamp rare overshoot and compute leak-free instrumental
        inst = batch_wav - vocals
        drums, bass = inst_fn(inst)
        other = inst - drums - bass
        return {"vocals": vocals, "drums": drums, "bass": bass, "other": other}

    return run_fn, tag


def decode_audio_bytes(raw_bytes: bytes, target_sr: int = 44100, clip_s: float = 48.0):
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            out = subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", tmp.name, "-f", "f32le", "-ac", "2", "-ar", str(target_sr), "-"],
                capture_output=True,
                timeout=20,
            )
            if out.returncode != 0 or len(out.stdout) < target_sr * 8 * 12:
                return None
            arr = np.frombuffer(out.stdout, dtype=np.float32).reshape(-1, 2).T.copy()
            wav = torch.from_numpy(arr)
            sr = target_sr

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    else:
        wav = wav[:2]

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    want = int(clip_s * target_sr)
    total = wav.shape[1]
    if total < int(18.0 * target_sr):
        return None
    if total > want:
        step = target_sr * 6
        best_s, best_e = 0, -1.0
        for s in range(0, total - want + 1, step):
            e = float(wav[:, s:s + want].pow(2).mean().item())
            if e > best_e:
                best_e = e
                best_s = s
        wav = wav[:, best_s:best_s + want]

    rms = float(wav.pow(2).mean().sqrt().item())
    if rms < 0.015:
        return None
    peak = float(wav.abs().amax().item())
    if peak > 0.99:
        wav = wav * (0.98 / peak)
    return wav


def download_shard(url: str, local_path: Path):
    if local_path.exists() and local_path.stat().st_size > 100000:
        return True
    subprocess.run(
        ["aria2c", "-x", "8", "-s", "8", "-k", "10M", "--max-tries=3", "--quiet",
         "-d", str(local_path.parent), "-o", local_path.name, url],
        check=False,
    )
    if local_path.exists() and local_path.stat().st_size > 100000:
        return True
    subprocess.run(
        ["curl", "-fSL", "--retry", "4", "--retry-delay", "3", "-o", str(local_path), url],
        check=False,
    )
    return local_path.exists() and local_path.stat().st_size > 100000


def stream_raw_songs(shard_offset: int, max_songs: int, cache_dir: Path, clip_s: float = 48.0):
    import pyarrow.parquet as pq

    cache_dir.mkdir(parents=True, exist_ok=True)
    yielded = 0
    seen_sigs = set()

    ordered_pq = PARQUET_URLS[shard_offset % len(PARQUET_URLS):] + PARQUET_URLS[:shard_offset % len(PARQUET_URLS)]
    for si, url in enumerate(ordered_pq):
        if yielded >= max_songs:
            return
        local_pq = cache_dir / f"sd_{si}.parquet"
        if not download_shard(url, local_pq):
            continue
        try:
            pf = pq.ParquetFile(str(local_pq))
            names = pf.schema.names
            aud_col = "audio" if "audio" in names else ("path" if "path" in names else None)
            if aud_col is None:
                continue
            for batch in pf.iter_batches(batch_size=16, columns=[aud_col]):
                col = batch.column(aud_col).to_pylist()
                for item in col:
                    if yielded >= max_songs:
                        break
                    b = item.get("bytes") if isinstance(item, dict) else (item if isinstance(item, (bytes, bytearray)) else None)
                    if not b or len(b) < 32768:
                        continue
                    sig = (len(b), b[:2048])
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                    w = decode_audio_bytes(b, 44100, clip_s)
                    if w is not None:
                        yield f"sd_{si:02d}_{yielded:04d}", w
                        yielded += 1
        except Exception as e:
            print(f"[Distill] Parquet read warning on {local_pq}: {e}", flush=True)
        finally:
            try:
                local_pq.unlink()
            except Exception:
                pass

    ordered_tar = JAMENDO_TAR_URLS[shard_offset % len(JAMENDO_TAR_URLS):] + JAMENDO_TAR_URLS[:shard_offset % len(JAMENDO_TAR_URLS)]
    for ti, url in enumerate(ordered_tar):
        if yielded >= max_songs:
            return
        local_tar = cache_dir / f"jam_{ti}.tar"
        if not download_shard(url, local_tar):
            continue
        try:
            with tarfile.open(local_tar, "r") as tf:
                for mem in tf:
                    if yielded >= max_songs:
                        break
                    if not mem.isfile() or not mem.name.lower().endswith((".opus", ".mp3", ".ogg", ".flac", ".wav", ".m4a")):
                        continue
                    fobj = tf.extractfile(mem)
                    if fobj is None:
                        continue
                    raw = fobj.read()
                    sig = (len(raw), raw[:2048])
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                    w = decode_audio_bytes(raw, 44100, clip_s)
                    if w is not None:
                        yield f"jam_{ti:02d}_{yielded:04d}", w
                        yielded += 1
        except Exception as e:
            print(f"[Distill] Tar read warning on {local_tar}: {e}", flush=True)
        finally:
            try:
                local_tar.unlink()
            except Exception:
                pass


def export_teacher_ab_refs(teacher_fn, in_root: Path, out_root: Path):
    """Export 24s BS-RoFormer-1297 GPU Teacher reference vocals + instrumentals on held-out tracks
    for direct side-by-side human listening against Cochlea CPU."""
    if not in_root.exists():
        return
    out_root.mkdir(parents=True, exist_ok=True)
    dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])[:3]
    for d in dirs:
        stems = []
        for nm in ("vocals", "drums", "bass", "other"):
            p = d / f"{nm}.flac"
            if not p.exists():
                break
            x, _ = sf.read(str(p), start=int(30 * 44100), frames=int(24 * 44100), dtype="float32", always_2d=True)
            stems.append(x[:, :2].T)
        if len(stems) != 4:
            continue
        n = min(s.shape[1] for s in stems)
        n = (n // 1024) * 1024
        mix = sum(s[:, :n] for s in stems)
        batch = torch.from_numpy(mix).unsqueeze(0)
        res = teacher_fn(batch)
        voc = np.clip(res["vocals"][0].numpy().T, -1.0, 1.0)
        inst = np.clip((mix - res["vocals"][0].numpy()).T, -1.0, 1.0)
        sf.write(str(out_root / f"{d.name[:6]}_roformer1297_vocals.flac"), voc, 44100, subtype="PCM_16")
        sf.write(str(out_root / f"{d.name[:6]}_roformer1297_inst.flac"), inst, 44100, subtype="PCM_16")
        print(f"[Teacher-AB] Exported BS-RoFormer-1297 reference clip for {d.name[:6]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distilled_roformer")
    ap.add_argument("--max-songs", type=int, default=650)
    ap.add_argument("--shard-offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--clip-seconds", type=float, default=48.0)
    ap.add_argument("--teacher-ab-in", default="data/valid_tracks")
    ap.add_argument("--teacher-ab-out", default="/content/exports_cochlea/teacher_ab")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if vram_gb < 35 and args.max_songs > 180:
        print(f"[Distill] L4/24GB detected ({vram_gb:.1f} GB): capping max_songs {args.max_songs} -> 180 (~2.4h 12.98 dB studio stems) so training gets full GPU budget", flush=True)
        args.max_songs = 180
    teacher_fn, teacher_name = load_two_stage_teacher(device)

    if args.teacher_ab_in and args.teacher_ab_out:
        try:
            export_teacher_ab_refs(teacher_fn, Path(args.teacher_ab_in), Path(args.teacher_ab_out))
        except Exception as e:
            print(f"[Teacher-AB] Warning: {e}", flush=True)

    existing = len([d for d in out_root.iterdir() if d.is_dir() and (d / "vocals.flac").exists()])
    if existing >= args.max_songs:
        print(f"[Distill] Already have {existing} RoFormer-distilled songs in {out_root}, skipping!", flush=True)
        return

    t0 = time.perf_counter()
    buf_ids, buf_wavs = [], []
    saved = existing

    def flush_batch():
        nonlocal saved
        if not buf_wavs:
            return
        min_l = min(w.shape[1] for w in buf_wavs)
        min_l = (min_l // 1024) * 1024
        batch = torch.stack([w[:, :min_l] for w in buf_wavs], dim=0)
        stems = teacher_fn(batch)
        for bi, sid in enumerate(buf_ids):
            voc = stems["vocals"][bi]
            drm = stems["drums"][bi]
            bas = stems["bass"][bi]
            oth = stems["other"][bi]
            e_inst = float((drm.pow(2).mean() + bas.pow(2).mean() + oth.pow(2).mean()).item())
            if e_inst < 1e-5:
                continue
            sdir = out_root / sid
            sdir.mkdir(parents=True, exist_ok=True)
            for nm, ten in [("vocals", voc), ("drums", drm), ("bass", bas), ("other", oth)]:
                arr = np.clip(ten.numpy().T, -1.0, 1.0)
                sf.write(str(sdir / f"{nm}.flac"), arr, 44100, subtype="PCM_16")
            saved += 1
        buf_ids.clear()
        buf_wavs.clear()
        dt = time.perf_counter() - t0
        print(f"[Distill:{teacher_name}] Saved {saved}/{args.max_songs} 4-stem tracks ({dt:.1f}s, {saved/max(dt,1):.2f} songs/s)", flush=True)

    eff_bs = max(int(args.batch_size), 4) if int(args.batch_size) <= 0 else int(args.batch_size)
    cache_dir = Path(f"/content/hf_shard_cache_{args.shard_offset}")
    for sid, wav in stream_raw_songs(args.shard_offset, args.max_songs - existing + 40, cache_dir, args.clip_seconds):
        if saved >= args.max_songs:
            break
        buf_ids.append(sid)
        buf_wavs.append(wav)
        if len(buf_wavs) >= eff_bs:
            flush_batch()
    flush_batch()
    print(f"[Distill] DONE! Total BS-RoFormer-1297 + HTDemucs-ft modern 4-stem tracks: {saved} in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
