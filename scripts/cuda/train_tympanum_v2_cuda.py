#!/usr/bin/env python3
"""Studio-Grade NVIDIA CUDA Trainer for Sone's Tympanum v2 Organ (160-Mel 6-Block Dilated Conv + 3-Layer BiGRU).
Features exact Vinyl-Speed Key/Pitch/Tuning/Tempo Equivariant Augmentation + SpecAugment
so beat, downbeat, active singing, breath/line-end, phrase, 24-key, and tuning generalize across
all modern genres without overfitting. Exports exact sone-infer/1 safetensors (f32 + f16)."""

import argparse
import concurrent.futures
import glob
import json
import math
import os
import queue
import random
import threading
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

EPS = 1e-5
TUNING_RANGE = 50.0


def tympanum_config(size: str = "v2") -> dict:
    if size == "base":
        return dict(
            sample_rate=44100,
            n_fft=2048,
            hop=882,
            n_mels=128,
            fmin=30.0,
            fmax=11000.0,
            channels=96,
            kernel=5,
            dilations=[1, 2, 4, 8],
            gru_layers=2,
            gru_hidden=64,
        )
    return dict(
        sample_rate=44100,
        n_fft=2048,
        hop=882,
        n_mels=160,
        fmin=25.0,
        fmax=16000.0,
        channels=160,
        kernel=5,
        dilations=[1, 2, 4, 8, 16, 32],
        gru_layers=3,
        gru_hidden=128,
    )


def mel_filters_htk(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
    bins = n_fft // 2 + 1
    mel = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)
    hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m0, m1 = mel(fmin), mel(fmax)
    pts = [hz(m0 + (m1 - m0) * i / (n_mels + 1)) for i in range(n_mels + 2)]
    out = np.zeros((n_mels, bins), dtype=np.float32)
    freqs = np.arange(bins, dtype=np.float64) * sr / n_fft
    for m in range(n_mels):
        lo, mid, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (freqs - lo) / (mid - lo)
        dn = (hi - freqs) / (hi - mid)
        v = np.where((freqs > lo) & (freqs < mid), up, np.where((freqs >= mid) & (freqs < hi), dn, 0.0))
        out[m] = v.astype(np.float32)
    return torch.from_numpy(out)


def parse_key(s: str):
    if not s:
        return None
    t = s.strip()
    roots = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    if not t or t[0].upper() not in roots:
        return None
    root = roots[t[0].upper()]
    rest = t[1:]
    shift = 0
    if rest.startswith("#") or rest.startswith("♯"):
        shift, rest = 1, rest[1:]
    elif rest.startswith("b") or rest.startswith("♭"):
        shift, rest = 11, rest[1:]
    mode = rest.lstrip(": -_").lower()
    minor = mode.startswith("min") or mode == "m"
    if not (minor or mode == "" or mode.startswith("maj")):
        return None
    return (root + shift) % 12 + (12 if minor else 0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = EPS):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.gain


class TimeBlock1D(nn.Module):
    def __init__(self, dim: int, inner: int, kernel: int, dilation: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        pad = (kernel // 2) * dilation
        self.conv = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation, groups=dim)
        self.l1 = nn.Linear(dim, inner)
        self.l2 = nn.Linear(inner, dim)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h = self.conv(h).transpose(1, 2)
        return x + self.l2(F.gelu(self.l1(h), approximate="none"))


class TympanumNet(nn.Module):
    def __init__(self, cfg: dict, dropout: float = 0.1):
        super().__init__()
        self.cfg = cfg
        c, h, mels = cfg["channels"], cfg["gru_hidden"], cfg["n_mels"]
        self.in_scale = nn.Parameter(torch.ones(mels))
        self.in_shift = nn.Parameter(torch.zeros(mels))
        self.proj = nn.Linear(mels, c)
        self.drop = nn.Dropout(dropout)
        self.convs = nn.ModuleList([TimeBlock1D(c, 2 * c, cfg["kernel"], d) for d in cfg["dilations"]])
        self.grus = nn.ModuleList([
            nn.GRU(c if l == 0 else 2 * h, h, num_layers=1, batch_first=True, bidirectional=True)
            for l in range(cfg["gru_layers"])
        ])
        self.frame_head = nn.Linear(2 * h, 5)
        self.key_head = nn.Linear(2 * h, 24)
        self.tuning_head = nn.Linear(2 * h, 1)

    def forward(self, logmel):
        x = logmel * self.in_scale + self.in_shift
        x = F.gelu(self.proj(x), approximate="none")
        for conv in self.convs:
            x = conv(x)
            if self.training:
                x = self.drop(x)
        for gru in self.grus:
            x, _ = gru(x)
            if self.training:
                x = self.drop(x)
        frames = self.frame_head(x)
        pooled = x.mean(dim=1)
        key = self.key_head(pooled)
        tuning = self.tuning_head(pooled).squeeze(-1)
        return frames, key, tuning

    def export_safetensors(self, out_dir: str, tag: str):
        os.makedirs(out_dir, exist_ok=True)
        tensors = {
            "in_scale": self.in_scale.detach().cpu().float().contiguous(),
            "in_shift": self.in_shift.detach().cpu().float().contiguous(),
            "proj.w": self.proj.weight.detach().cpu().float().contiguous(),
            "proj.b": self.proj.bias.detach().cpu().float().contiguous(),
        }
        for i, conv in enumerate(self.convs):
            tensors[f"conv{i}.norm"] = conv.norm.gain.detach().cpu().float().contiguous()
            tensors[f"conv{i}.conv.w"] = conv.conv.weight.detach().cpu().float().squeeze(1).contiguous()
            tensors[f"conv{i}.conv.b"] = conv.conv.bias.detach().cpu().float().contiguous()
            tensors[f"conv{i}.l1.w"] = conv.l1.weight.detach().cpu().float().contiguous()
            tensors[f"conv{i}.l1.b"] = conv.l1.bias.detach().cpu().float().contiguous()
            tensors[f"conv{i}.l2.w"] = conv.l2.weight.detach().cpu().float().contiguous()
            tensors[f"conv{i}.l2.b"] = conv.l2.bias.detach().cpu().float().contiguous()
        for l, gru in enumerate(self.grus):
            tensors[f"gru{l}.fwd.w_ih"] = gru.weight_ih_l0.detach().cpu().float().contiguous()
            tensors[f"gru{l}.fwd.w_hh"] = gru.weight_hh_l0.detach().cpu().float().contiguous()
            tensors[f"gru{l}.fwd.b_ih"] = gru.bias_ih_l0.detach().cpu().float().contiguous()
            tensors[f"gru{l}.fwd.b_hh"] = gru.bias_hh_l0.detach().cpu().float().contiguous()
            tensors[f"gru{l}.bwd.w_ih"] = gru.weight_ih_l0_reverse.detach().cpu().float().contiguous()
            tensors[f"gru{l}.bwd.w_hh"] = gru.weight_hh_l0_reverse.detach().cpu().float().contiguous()
            tensors[f"gru{l}.bwd.b_ih"] = gru.bias_ih_l0_reverse.detach().cpu().float().contiguous()
            tensors[f"gru{l}.bwd.b_hh"] = gru.bias_hh_l0_reverse.detach().cpu().float().contiguous()
        for nm, mod in [("frame_head", self.frame_head), ("key_head", self.key_head), ("tuning_head", self.tuning_head)]:
            tensors[f"{nm}.w"] = mod.weight.detach().cpu().float().contiguous()
            tensors[f"{nm}.b"] = mod.bias.detach().cpu().float().contiguous()

        meta = {
            "model": "tympanum",
            "config": json.dumps(self.cfg, separators=(",", ":")),
            "format": "sone-infer/1",
        }
        p32 = os.path.join(out_dir, f"sone-tympanum-{tag}-f32.safetensors")
        p16 = os.path.join(out_dir, f"sone-tympanum-{tag}-f16.safetensors")
        save_file(tensors, p32, metadata=meta)
        save_file({k: v.half().contiguous() for k, v in tensors.items()}, p16, metadata=meta)
        return p32, p16


def compute_logmel(mono: torch.Tensor, cfg: dict, hann: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    spec = torch.stft(
        mono.float(),
        cfg["n_fft"],
        hop_length=cfg["hop"],
        window=hann,
        center=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-12).transpose(1, 2)
    mel = torch.matmul(mag, filters.T)
    return torch.log1p(1000.0 * mel)


def spec_augment(mel: torch.Tensor) -> torch.Tensor:
    # Mild SpecAugment + loudness offset jitter (never masks beat grid globally)
    b, t, m = mel.shape
    out = mel * torch.empty((b, 1, 1), device=mel.device).uniform_(0.88, 1.12)
    for i in range(b):
        if random.random() < 0.55:
            f0 = random.randint(0, m - 18)
            fw = random.randint(4, 16)
            out[i, :, f0:f0 + fw] *= 0.15
    return out


def _load_one_song(jf: str):
    with open(jf, "r", encoding="utf-8") as f:
        lab = json.load(f)
    audio = lab["audio"]
    if not os.path.isabs(audio):
        audio = os.path.join(os.path.dirname(jf), audio)
    if not os.path.exists(audio):
        return None
    pcm, _sr = sf.read(audio, dtype="int16", always_2d=True)
    if pcm.shape[1] == 1:
        mono_i16 = np.ascontiguousarray(pcm[:, 0])
    else:
        mono_i16 = ((pcm[:, 0].astype(np.int32) + pcm[:, 1].astype(np.int32)) >> 1).astype(np.int16)

    ev_np = {}
    for h, k in [(0, "beats"), (1, "downbeats"), (3, "line_ends"), (4, "phrase_starts")]:
        ev = lab.get(k)
        ev_np[h] = np.sort(np.asarray(ev, dtype=np.float64)) if ev is not None else None

    segs = lab.get("vocals")
    voc_np = np.asarray(segs, dtype=np.float64).reshape(-1, 2) if segs is not None else None

    head_mask = np.array([
        1.0 if lab.get(k) is not None else 0.0
        for k in ["beats", "downbeats", "vocals", "line_ends", "phrase_starts"]
    ], dtype=np.float32)

    parsed_key = parse_key(lab.get("key")) if lab.get("key") else None
    tuning = float(lab["tuning_cents"]) if lab.get("tuning_cents") is not None else 0.0
    tuning_mask = 1.0 if lab.get("tuning_cents") is not None else 0.0

    return dict(
        label=lab,
        audio=audio,
        mono_i16=mono_i16,
        frames=int(mono_i16.shape[0]),
        key=parsed_key,
        ev_np=ev_np,
        voc_np=voc_np,
        head_mask=head_mask,
        tuning=tuning,
        tuning_mask=tuning_mask,
    )


def index_ear_songs(roots):
    jfiles = []
    for root in roots:
        if os.path.isdir(root):
            jfiles.extend(sorted(glob.glob(os.path.join(root, "*.json"))))
    if not jfiles:
        return []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        loaded = list(ex.map(_load_one_song, jfiles))
    songs = [s for s in loaded if s is not None]
    ram_mb = sum(s["mono_i16"].nbytes for s in songs) / 1e6
    print(f"Preloaded {len(songs)} Tympanum songs into RAM ({ram_mb:.1f} MB PCM16 mono) in {time.perf_counter()-t0:.1f}s", flush=True)
    return songs


class EquivariantInMemorySampler:
    """Samples 10s chunks with exact Vinyl-Platter Resample Augmentation:
    resampling by `rate = 2^(delta_semitones / 12)` shifts pitch/key/tuning AND tempo/beat-times
    simultaneously with zero phase-vocoder artifacts and 0.0 ms label drift.
    Uses a multi-threaded background prefetcher so CPU resampling overlaps 100% with GPU compute."""

    def __init__(self, songs, cfg, batch_size: int, chunk_seconds: float = 10.0, augment: bool = True, workers: int = 6, prefetch: int = 3):
        self.songs = songs
        self.full_songs = [s for s in songs if s["label"].get("beats") is not None]
        self.voc_songs = [s for s in songs if s["label"].get("beats") is None]
        self.cfg = cfg
        self.bs = batch_size
        self.augment = augment
        self.chunk_len = int(round(chunk_seconds * cfg["sample_rate"]))
        self.n_frames = self.chunk_len // cfg["hop"] + 1
        fps = cfg["sample_rate"] / cfg["hop"]
        self.rel_frames = np.arange(self.n_frames, dtype=np.float64) / fps
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self._q = queue.Queue(maxsize=prefetch)
        self._stop = False
        self._bg = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._bg.start()

    def _fill_item(self, args_t):
        b, mono_np, tg_np, hm_np, key_np, tun_np, tm_np = args_t
        rng = random.Random()
        sr = self.cfg["sample_rate"]
        fps = sr / self.cfg["hop"]
        both = bool(self.full_songs and self.voc_songs)
        if both:
            s = rng.choice(self.full_songs if rng.random() < 0.65 else self.voc_songs)
        else:
            s = rng.choice(self.songs)

        # Vinyl pitch/tempo shift: delta_semi in [-4.5, +4.5] semitones (72% of samples)
        if self.augment and rng.random() < 0.72:
            delta_semi = rng.uniform(-4.4, 4.4)
            rate = 2.0 ** (delta_semi / 12.0)
        else:
            delta_semi = 0.0
            rate = 1.0

        src_len = max(2, int(round(self.chunk_len * rate)))
        max_start = max(0, s["frames"] - src_len - 1)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        raw = s["mono_i16"][start:start + src_len].astype(np.float32) * (1.0 / 32768.0)
        if raw.shape[0] < src_len:
            raw = np.pad(raw, (0, src_len - raw.shape[0]))

        if raw.shape[0] == self.chunk_len:
            mono_np[b] = raw
        else:
            xs = np.linspace(0.0, raw.shape[0] - 1, self.chunk_len, dtype=np.float32)
            mono_np[b] = np.interp(xs, np.arange(raw.shape[0], dtype=np.float32), raw)

        if self.augment:
            gain = 10.0 ** (rng.uniform(-6.0, 2.5) / 20.0)
            mono_np[b] *= gain

        # Equivariant beat/downbeat/vocal/phrase timestamps: t_out = (t_src - t0_src) / rate
        t0_src = start / sr
        t_end_src = t0_src + ((self.n_frames - 1) / fps) * rate
        ev_np = s["ev_np"]
        for h in (0, 1, 3, 4):
            arr = ev_np[h]
            if arr is not None and arr.size > 0:
                i0 = int(np.searchsorted(arr, t0_src - 0.06, side="left"))
                i1 = int(np.searchsorted(arr, t_end_src + 0.06, side="right"))
                sub = arr[i0:i1]
                if sub.size > 0:
                    t_rel = (sub - t0_src) / rate
                    idx = np.rint(t_rel * fps).astype(np.int64)
                    m1, p1 = idx - 1, idx + 1
                    tg_np[b, m1[(m1 >= 0) & (m1 < self.n_frames)], h] = 0.5
                    tg_np[b, p1[(p1 >= 0) & (p1 < self.n_frames)], h] = 0.5
                    tg_np[b, idx[(idx >= 0) & (idx < self.n_frames)], h] = 1.0

        voc = s["voc_np"]
        if voc is not None and voc.size > 0:
            ov = voc[(voc[:, 1] >= t0_src) & (voc[:, 0] <= t_end_src)]
            if ov.size > 0:
                times_src = t0_src + self.rel_frames * rate
                i_s = np.searchsorted(times_src, ov[:, 0], side="left")
                i_e = np.searchsorted(times_src, ov[:, 1], side="right")
                for s_i, e_i in zip(i_s, i_e):
                    if e_i > s_i:
                        tg_np[b, s_i:e_i, 2] = 1.0

        hm_np[b] = s["head_mask"]

        # Equivariant key & tuning_cents shift
        if s["key"] is not None:
            total_semi = (s["tuning"] / 100.0) + delta_semi
            k_shift = int(round(total_semi))
            new_cents = (total_semi - k_shift) * 100.0
            base_k = s["key"]
            new_k = ((base_k + k_shift) % 12) if base_k < 12 else (12 + ((base_k - 12 + k_shift) % 12))
            key_np[b, new_k] = 1.0
            tun_np[b] = float(np.clip(new_cents, -49.5, 49.5))
            tm_np[b] = 1.0
        else:
            tun_np[b] = s["tuning"]
            tm_np[b] = s["tuning_mask"]

    def _build_cpu_batch(self):
        mono_np = np.zeros((self.bs, self.chunk_len), dtype=np.float32)
        tg_np = np.zeros((self.bs, self.n_frames, 5), dtype=np.float32)
        hm_np = np.zeros((self.bs, 5), dtype=np.float32)
        key_np = np.zeros((self.bs, 24), dtype=np.float32)
        tun_np = np.zeros((self.bs,), dtype=np.float32)
        tm_np = np.zeros((self.bs,), dtype=np.float32)
        list(self._pool.map(
            self._fill_item,
            [(b, mono_np, tg_np, hm_np, key_np, tun_np, tm_np) for b in range(self.bs)],
        ))
        return (
            torch.from_numpy(mono_np),
            torch.from_numpy(tg_np),
            torch.from_numpy(hm_np),
            torch.from_numpy(key_np),
            torch.from_numpy(tun_np),
            torch.from_numpy(tm_np),
        )

    def close(self):
        self._stop = True
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass
        self._pool.shutdown(wait=False)

    def _prefetch_loop(self):
        while not self._stop:
            try:
                batch = self._build_cpu_batch()
                self._q.put(batch)
            except Exception as e:
                print(f"[SamplerWarn] {e}", flush=True)
                time.sleep(0.05)

    def next_batch(self, device: torch.device):
        mono_t, tg_t, hm_t, key_t, tun_t, tm_t = self._q.get()
        return (
            mono_t.to(device, non_blocking=True),
            tg_t.to(device, non_blocking=True),
            hm_t.to(device, non_blocking=True),
            key_t.to(device, non_blocking=True),
            tun_t.to(device, non_blocking=True),
            tm_t.to(device, non_blocking=True),
        )


def ear_loss(frames_logits, key_logits, tun_raw, tg, hm, key_oh, tun, tmask, pos_w, head_w, w_key=0.65, w_tun=0.5):
    total = 0.0
    for h in range(5):
        per_song = F.binary_cross_entropy_with_logits(
            frames_logits[:, :, h], tg[:, :, h], pos_weight=pos_w[h], reduction="none"
        ).mean(dim=1)
        m = hm[:, h]
        l_h = (per_song * m).sum() / (m.sum() + 1e-6)
        total = total + l_h * head_w[h]
    known = key_oh.sum(dim=1)
    log_p = F.log_softmax(key_logits, dim=-1)
    l_key = -(key_oh * log_p).sum(dim=-1).sum() / (known.sum() + 1e-6)
    cents = torch.tanh(tun_raw) * TUNING_RANGE
    l_tun = ((cents - tun).abs() / TUNING_RANGE * tmask).sum() / (tmask.sum() + 1e-6)
    return total + l_key * w_key + l_tun * w_tun


def peaks(p: np.ndarray, thr=0.5):
    out = []
    n = len(p)
    for i in range(n):
        if p[i] < thr:
            continue
        lo, hi = max(0, i - 2), min(n, i + 3)
        if all(not (p[j] > p[i] or (j < i and p[j] >= p[i])) for j in range(lo, hi)):
            out.append(i)
    return out


@torch.no_grad()
def validate(model, songs, cfg, hann, filters, device, amp_dtype, pos_w, head_w, chunk_seconds=10.0, valid_seconds=60.0):
    model.eval()
    chunk_len = int(round(chunk_seconds * cfg["sample_rate"]))
    n_frames = chunk_len // cfg["hop"] + 1
    fps = cfg["sample_rate"] / cfg["hop"]
    rel_frames = np.arange(n_frames, dtype=np.float64) / fps
    tol = int(round(0.07 * fps))
    tot_loss, n_chunks = 0.0, 0
    f_stats = [[0, 0, 0] for _ in range(4)]
    v_ok, v_n, k_ok, k_n, t_err, t_n = 0, 0, 0, 0, 0.0, 0

    for s in songs:
        limit = min(int(round(valid_seconds * cfg["sample_rate"])), s["frames"])
        hm_np = s["head_mask"]
        hm = torch.from_numpy(hm_np).unsqueeze(0).to(device)
        key_oh_np = np.zeros((1, 24), dtype=np.float32)
        if s["key"] is not None:
            key_oh_np[0, s["key"]] = 1.0
        key_oh = torch.from_numpy(key_oh_np).to(device)
        tun = torch.tensor([float(s["tuning"])], device=device)
        tm = torch.tensor([float(s["tuning_mask"])], device=device)
        start = 0
        while start + chunk_len // 2 <= max(1, limit):
            raw = s["mono_i16"][start:start + chunk_len].astype(np.float32) * (1.0 / 32768.0)
            if raw.shape[0] < chunk_len:
                raw = np.pad(raw, (0, chunk_len - raw.shape[0]))
            tg_np = np.zeros((n_frames, 5), dtype=np.float32)
            t0 = start / cfg["sample_rate"]
            t_end = t0 + (n_frames - 1) / fps
            for h in (0, 1, 3, 4):
                arr = s["ev_np"][h]
                if arr is not None and arr.size > 0:
                    sub = arr[(arr >= t0 - 0.05) & (arr <= t_end + 0.05)]
                    if sub.size > 0:
                        idx = np.rint((sub - t0) * fps).astype(np.int64)
                        m1, p1 = idx - 1, idx + 1
                        tg_np[m1[(m1 >= 0) & (m1 < n_frames)], h] = 0.5
                        tg_np[p1[(p1 >= 0) & (p1 < n_frames)], h] = 0.5
                        tg_np[idx[(idx >= 0) & (idx < n_frames)], h] = 1.0
            if s["voc_np"] is not None and s["voc_np"].size > 0:
                ov = s["voc_np"][(s["voc_np"][:, 1] >= t0) & (s["voc_np"][:, 0] <= t_end)]
                times = t0 + rel_frames
                for sv, ev in ov:
                    tg_np[(times >= sv) & (times <= ev), 2] = 1.0

            mono = torch.from_numpy(raw).unsqueeze(0).to(device)
            tg = torch.from_numpy(tg_np).unsqueeze(0).to(device)
            mel = compute_logmel(mono, cfg, hann, filters)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                fl, kl, tr = model(mel)
                loss = ear_loss(fl, kl, tr, tg, hm, key_oh, tun, tm, pos_w, head_w)
            tot_loss += loss.item()
            n_chunks += 1
            probs = torch.sigmoid(fl.float())[0].cpu().numpy()
            for slot, h, thr in [(0, 0, 0.5), (1, 1, 0.5), (2, 3, 0.35), (3, 4, 0.35)]:
                if hm_np[h] == 0:
                    continue
                pred = peaks(probs[:, h], thr)
                truth = np.where(tg_np[:, h] == 1.0)[0]
                hits = sum(1 for g in truth if any(abs(q - g) <= tol for q in pred))
                f_stats[slot][0] += hits
                f_stats[slot][1] += len(pred)
                f_stats[slot][2] += len(truth)
            if hm_np[2] > 0:
                v_ok += int(((probs[:, 2] > 0.5) == (tg_np[:, 2] > 0.5)).sum())
                v_n += n_frames
            if s["key"] is not None:
                k_ok += int(int(kl[0].argmax().item()) == s["key"])
                k_n += 1
            if s["tuning_mask"] > 0:
                t_err += abs(TUNING_RANGE * math.tanh(float(tr[0].item())) - float(s["tuning"]))
                t_n += 1
            start += chunk_len
    model.train()
    fm = lambda x: (2.0 * x[0] / (x[1] + x[2])) if (x[1] + x[2]) > 0 else 0.0
    beat_f = fm(f_stats[0])
    down_f = fm(f_stats[1])
    voice_acc = v_ok / max(1, v_n)
    key_acc = k_ok / max(1, k_n)
    composite = 0.35 * beat_f + 0.25 * down_f + 0.20 * voice_acc + 0.20 * key_acc
    return dict(
        loss=tot_loss / max(1, n_chunks),
        beat_f=beat_f,
        downbeat_f=down_f,
        line_end_f=fm(f_stats[2]),
        phrase_f=fm(f_stats[3]),
        voice_acc=voice_acc,
        key_acc=key_acc,
        tuning_mae=t_err / max(1, t_n),
        composite=composite,
    )


def lr_at(step, total, base_lr=8e-4, warmup=1000, min_ratio=0.03):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--valid", nargs="+", required=True)
    ap.add_argument("--out", default="/content/exports_tympanum")
    ap.add_argument("--size", default="v2", choices=["base", "v2"])
    ap.add_argument("--steps", type=int, default=32000)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=8e-4)
    args = ap.parse_args()

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(0)
    # Use float16 AMP + GradScaler so cuDNN fused Tensor-Core BiGRU (_cudnn_rnn) engages (~15x faster than unfused bf16 GRU)
    use_bf16 = False
    amp_dtype = torch.float16
    cfg = tympanum_config(args.size)

    train_songs = index_ear_songs(args.train)
    valid_songs = index_ear_songs(args.valid)
    print(f"Tympanum {args.size} on {torch.cuda.get_device_name(0)}: train={len(train_songs)}, valid={len(valid_songs)}", flush=True)
    assert train_songs, "No Tympanum training songs found"

    hann = torch.hann_window(cfg["n_fft"], periodic=True, device=device)
    filters = mel_filters_htk(cfg["sample_rate"], cfg["n_fft"], cfg["n_mels"], cfg["fmin"], cfg["fmax"]).to(device)
    pos_w = [torch.tensor(v, device=device) for v in [4.0, 12.0, 1.0, 30.0, 60.0]]
    head_w = [1.2, 1.2, 0.6, 1.0, 1.0]

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    model = TympanumNet(cfg).to(device)
    exact_params = sum(p.numel() for p in model.parameters())
    hw_meta = {
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "amp_dtype": str(amp_dtype),
        "size": args.size,
        "batch_size": args.batch_size,
        "exact_params": exact_params,
        "train_songs": len(train_songs),
        "valid_songs": len(valid_songs),
    }
    print(f"Tympanum params: {exact_params:,} ({exact_params/1e6:.3f}M) | HW: {hw_meta}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    sampler = EquivariantInMemorySampler(train_songs, cfg, batch_size=args.batch_size, chunk_seconds=10.0, augment=True)
    best_score = -1e9
    best_val = None
    best_step = 0
    history = []
    train_log = []
    start_step = 0
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        best_score = float(ckpt.get("best_score", -1e9))
        best_val = ckpt.get("best_val")
        best_step = int(ckpt.get("best_step", 0))
        history = ckpt.get("history", [])
        train_log = ckpt.get("train_log", [])
        print(f"Resumed Tympanum {args.size} from {ckpt_path} at step {start_step}/{args.steps} (best_score={best_score:.4f})", flush=True)

    t0 = time.perf_counter()
    val_every = max(1500, args.steps // 16)

    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps, base_lr=args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        mono, tg, hm, key_oh, tun, tm = sampler.next_batch(device)
        mel = spec_augment(compute_logmel(mono, cfg, hann, filters))
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            fl, kl, tr = model(mel)
        loss = ear_loss(fl.float(), kl.float(), tr.float(), tg, hm, key_oh, tun, tm, pos_w, head_w)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        scaler.step(opt)
        scaler.update()

        done = step + 1
        if done % 250 == 0 or done == args.steps:
            dt = time.perf_counter() - t0
            sps = max(1, done - start_step) / max(dt, 1e-6)
            print(f"step {done:6d}/{args.steps} | {sps:5.1f} st/s | loss={loss.item():.4f}", flush=True)
            train_log.append({"step": done, "elapsed_s": dt, "steps_per_s": sps, "lr": lr, "loss": float(loss.item())})
        if done % val_every == 0 or done == args.steps:
            vm = validate(model, valid_songs, cfg, hann, filters, device, amp_dtype, pos_w, head_w)
            print(f"  VALID step {done}: {vm}", flush=True)
            history.append({"step": done, "elapsed_s": time.perf_counter() - t0, "all": vm})
            if vm["composite"] > best_score:
                best_score = vm["composite"]
                best_val = vm
                best_step = done
                model.export_safetensors(args.out, "best")
                print(f"  -> NEW BEST TYMPANUM @ step {done} (composite={best_score:.4f}, beat_f={vm['beat_f']:.3f}, key={vm['key_acc']:.3f})", flush=True)
            model.export_safetensors(args.out, "latest")
            tmp_ckpt = ckpt_path + ".tmp"
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": done,
                "best_score": best_score,
                "best_val": best_val,
                "best_step": best_step,
                "history": history,
                "train_log": train_log,
            }, tmp_ckpt)
            os.replace(tmp_ckpt, ckpt_path)
            hw_meta["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(0) / 1e6, 1)
            with open(os.path.join(args.out, "tympanum_results.json"), "w") as f:
                json.dump({
                    "hardware": hw_meta,
                    "best_step": best_step,
                    "best_score": best_score,
                    "best_val": best_val,
                    "elapsed_s": time.perf_counter() - t0,
                    "history": history,
                    "train_log": train_log,
                }, f, indent=2)
    sampler.close()
    print("TYMPANUM V2 TRAINING DONE!", flush=True)


if __name__ == "__main__":
    main()
