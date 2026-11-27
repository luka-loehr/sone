#!/usr/bin/env python3
"""Native NVIDIA CUDA trainer for Sone-6.5M's Ear network (Conv1d + BiGRU).
Uses in-RAM PCM16 mono waveforms (zero disk I/O during training), GPU cuFFT log-mel,
cuDNN depthwise Conv1d, fused cuDNN BiGRU, BF16/FP16, and fused AdamW.
Exports exact sone-infer/1 safetensors (f32 + f16)."""

import argparse
import concurrent.futures
import glob
import json
import math
import os
import random
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


def ear_config() -> dict:
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


class EarNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        c, h, mels = cfg["channels"], cfg["gru_hidden"], cfg["n_mels"]
        self.in_scale = nn.Parameter(torch.ones(mels))
        self.in_shift = nn.Parameter(torch.zeros(mels))
        self.proj = nn.Linear(mels, c)
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
        for gru in self.grus:
            x, _ = gru(x)
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
    key_oh = np.zeros(24, dtype=np.float32)
    if parsed_key is not None:
        key_oh[parsed_key] = 1.0
    tuning = np.float32(float(lab["tuning_cents"]) if lab.get("tuning_cents") is not None else 0.0)
    tuning_mask = np.float32(1.0 if lab.get("tuning_cents") is not None else 0.0)

    return dict(
        label=lab,
        audio=audio,
        mono_i16=mono_i16,
        frames=int(mono_i16.shape[0]),
        key=parsed_key,
        ev_np=ev_np,
        voc_np=voc_np,
        head_mask=head_mask,
        key_oh=key_oh,
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
    print(f"Preloaded {len(songs)} songs into RAM ({ram_mb:.1f} MB PCM16 mono) in {time.perf_counter()-t0:.1f}s", flush=True)
    return songs


def read_mono_chunk_i16(song: dict, start: int, length: int, out_buf: np.ndarray):
    arr = song["mono_i16"]
    n = arr.shape[0]
    s = min(start, max(0, n - 1)) if n > 0 else 0
    end = min(n, s + length)
    got = end - s
    if got > 0:
        out_buf[:got] = arr[s:end]
    if got < length:
        out_buf[got:length] = 0


def build_targets_into(song: dict, start: int, n_frames: int, cfg: dict, tg_out: np.ndarray, rel_frames: np.ndarray):
    fps = cfg["sample_rate"] / cfg["hop"]
    t0 = start / cfg["sample_rate"]
    t_end = t0 + (n_frames - 1) / fps
    tg_out.fill(0.0)
    ev_np = song["ev_np"]
    for h in (0, 1, 3, 4):
        arr = ev_np[h]
        if arr is not None and arr.size > 0:
            i0 = int(np.searchsorted(arr, t0 - 0.05, side="left"))
            i1 = int(np.searchsorted(arr, t_end + 0.05, side="right"))
            sub = arr[i0:i1]
            if sub.size > 0:
                idx = np.rint((sub - t0) * fps).astype(np.int64)
                m1 = idx - 1
                p1 = idx + 1
                tg_out[m1[(m1 >= 0) & (m1 < n_frames)], h] = 0.5
                tg_out[p1[(p1 >= 0) & (p1 < n_frames)], h] = 0.5
                tg_out[idx[(idx >= 0) & (idx < n_frames)], h] = 1.0
    voc = song["voc_np"]
    if voc is not None and voc.size > 0:
        ov = voc[(voc[:, 1] >= t0) & (voc[:, 0] <= t_end)]
        if ov.size > 0:
            times = t0 + rel_frames
            for s_v, e_v in ov:
                tg_out[(times >= s_v) & (times <= e_v), 2] = 1.0


class InMemoryBatchSampler:
    def __init__(self, songs, cfg, batch_size: int, chunk_seconds: float = 10.0, pin: bool = True):
        self.songs = songs
        self.full_songs = [s for s in songs if s["label"].get("beats") is not None]
        self.voc_songs = [s for s in songs if s["label"].get("beats") is None]
        self.cfg = cfg
        self.bs = batch_size
        self.chunk_len = int(round(chunk_seconds * cfg["sample_rate"]))
        self.n_frames = self.chunk_len // cfg["hop"] + 1
        fps = cfg["sample_rate"] / cfg["hop"]
        self.rel_frames = np.arange(self.n_frames, dtype=np.float64) / fps

        self.mono_t = torch.empty((batch_size, self.chunk_len), dtype=torch.int16, pin_memory=pin)
        self.tg_t = torch.empty((batch_size, self.n_frames, 5), dtype=torch.float32, pin_memory=pin)
        self.hm_t = torch.empty((batch_size, 5), dtype=torch.float32, pin_memory=pin)
        self.key_t = torch.empty((batch_size, 24), dtype=torch.float32, pin_memory=pin)
        self.tun_t = torch.empty((batch_size,), dtype=torch.float32, pin_memory=pin)
        self.tm_t = torch.empty((batch_size,), dtype=torch.float32, pin_memory=pin)

    def next_batch(self, device: torch.device):
        mono_np = self.mono_t.numpy()
        tg_np = self.tg_t.numpy()
        hm_np = self.hm_t.numpy()
        key_np = self.key_t.numpy()
        tun_np = self.tun_t.numpy()
        tm_np = self.tm_t.numpy()

        both = bool(self.full_songs and self.voc_songs)
        for b in range(self.bs):
            if both:
                s = random.choice(self.full_songs if random.random() < 0.55 else self.voc_songs)
            else:
                s = random.choice(self.songs)
            start = random.randint(0, s["frames"] - self.chunk_len) if s["frames"] > self.chunk_len else 0
            read_mono_chunk_i16(s, start, self.chunk_len, mono_np[b])
            build_targets_into(s, start, self.n_frames, self.cfg, tg_np[b], self.rel_frames)
            hm_np[b] = s["head_mask"]
            key_np[b] = s["key_oh"]
            tun_np[b] = s["tuning"]
            tm_np[b] = s["tuning_mask"]

        mono = self.mono_t.to(device, non_blocking=True).float().mul_(1.0 / 32768.0)
        tg = self.tg_t.to(device, non_blocking=True)
        hm = self.hm_t.to(device, non_blocking=True)
        key_oh = self.key_t.to(device, non_blocking=True)
        tun = self.tun_t.to(device, non_blocking=True)
        tm = self.tm_t.to(device, non_blocking=True)
        return mono, tg, hm, key_oh, tun, tm


def ear_loss(frames_logits, key_logits, tun_raw, tg, hm, key_oh, tun, tmask, pos_w, head_w, w_key=0.5, w_tun=0.5):
    total = 0.0
    terms = []
    for h in range(5):
        per_song = F.binary_cross_entropy_with_logits(
            frames_logits[:, :, h], tg[:, :, h], pos_weight=pos_w[h], reduction="none"
        ).mean(dim=1)
        m = hm[:, h]
        l_h = (per_song * m).sum() / (m.sum() + 1e-6)
        terms.append(l_h)
        total = total + l_h * head_w[h]
    known = key_oh.sum(dim=1)
    log_p = F.log_softmax(key_logits, dim=-1)
    l_key = -(key_oh * log_p).sum(dim=-1).sum() / (known.sum() + 1e-6)
    cents = torch.tanh(tun_raw) * TUNING_RANGE
    l_tun = ((cents - tun).abs() / TUNING_RANGE * tmask).sum() / (tmask.sum() + 1e-6)
    total = total + l_key * w_key + l_tun * w_tun
    return total, terms, l_key, l_tun


def peaks(p: np.ndarray, thr=0.5):
    out = []
    n = len(p)
    for i in range(n):
        if p[i] < thr:
            continue
        lo, hi = max(0, i - 2), min(n, i + 3)
        ok = True
        for j in range(lo, hi):
            if p[j] > p[i] or (j < i and p[j] >= p[i]):
                ok = False
                break
        if ok:
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
    f_stats = [[0, 0, 0] for _ in range(4)]  # beat(0), downbeat(1), line_end(3), phrase(4)
    v_ok, v_n, k_ok, k_n, t_err, t_n = 0, 0, 0, 0, 0.0, 0

    mono_buf = np.empty(chunk_len, dtype=np.int16)
    tg_np = np.empty((n_frames, 5), dtype=np.float32)

    for s in songs:
        limit = min(int(round(valid_seconds * cfg["sample_rate"])), s["frames"])
        hm_np = s["head_mask"]
        hm = torch.from_numpy(hm_np).unsqueeze(0).to(device)
        key_oh = torch.from_numpy(s["key_oh"]).unsqueeze(0).to(device)
        tun = torch.tensor([float(s["tuning"])], device=device)
        tm = torch.tensor([float(s["tuning_mask"])], device=device)
        start = 0
        while start + chunk_len // 2 <= max(1, limit):
            read_mono_chunk_i16(s, start, chunk_len, mono_buf)
            build_targets_into(s, start, n_frames, cfg, tg_np, rel_frames)
            mono = torch.from_numpy(mono_buf).unsqueeze(0).to(device).float().mul_(1.0 / 32768.0)
            tg = torch.from_numpy(tg_np).unsqueeze(0).to(device)

            mel = compute_logmel(mono, cfg, hann, filters)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                fl, kl, tr = model(mel)
                loss, _, _, _ = ear_loss(fl, kl, tr, tg, hm, key_oh, tun, tm, pos_w, head_w)
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
    fm = lambda x: (2.0 * x[0] / (x[1] + x[2])) if (x[1] + x[2]) > 0 else float("nan")
    return dict(
        loss=tot_loss / max(1, n_chunks),
        beat_f=fm(f_stats[0]),
        downbeat_f=fm(f_stats[1]),
        line_end_f=fm(f_stats[2]),
        phrase_f=fm(f_stats[3]),
        voice_acc=v_ok / max(1, v_n),
        key_acc=k_ok / max(1, k_n),
        tuning_mae=t_err / max(1, t_n),
    )


def lr_at(step, total, base_lr=1e-3, warmup=1000, min_ratio=0.02):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--valid", nargs="+", required=True)
    ap.add_argument("--out", default="/content/exports_ear")
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(0)
    use_bf16 = major >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    cfg = ear_config()

    train_songs = index_ear_songs(args.train)
    valid_songs = index_ear_songs(args.valid)
    print(f"Ear CUDA on {torch.cuda.get_device_name(0)} (sm_{major}{minor}): train={len(train_songs)} songs, valid={len(valid_songs)} songs", flush=True)
    assert train_songs, "No Ear training songs found"

    hann = torch.hann_window(cfg["n_fft"], periodic=True, device=device)
    filters = mel_filters_htk(cfg["sample_rate"], cfg["n_fft"], cfg["n_mels"], cfg["fmin"], cfg["fmax"]).to(device)
    pos_w = [torch.tensor(v, device=device) for v in [4.0, 12.0, 1.0, 30.0, 60.0]]
    head_w = [1.0, 1.0, 0.5, 1.0, 1.0]

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    model = EarNet(cfg).to(device)
    exact_params = sum(p.numel() for p in model.parameters())
    hw_meta = {
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "amp_dtype": str(amp_dtype),
        "batch_size": args.batch_size,
        "exact_params": exact_params,
        "train_songs": len(train_songs),
        "valid_songs": len(valid_songs),
    }
    print(f"Ear params: {exact_params:,} | HW: {hw_meta}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    sampler = InMemoryBatchSampler(train_songs, cfg, batch_size=args.batch_size, chunk_seconds=10.0, pin=True)

    best_loss = float("inf")
    best_val = None
    best_step = 0
    history = []
    train_log = []
    radio_valid_songs = [s for s in valid_songs if s["key"] is not None]
    musdb_valid_songs = [s for s in valid_songs if s["key"] is None]
    t0 = time.perf_counter()
    val_every = max(1000, args.steps // 12)
    os.makedirs(args.out, exist_ok=True)

    for step in range(args.steps):
        lr = lr_at(step, args.steps, base_lr=args.lr, warmup=min(1000, args.steps // 10))
        for pg in opt.param_groups:
            pg["lr"] = lr
        mono, tg, hm, key_oh, tun, tm = sampler.next_batch(device)

        mel = compute_logmel(mono, cfg, hann, filters)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            fl, kl, tr = model(mel)
            loss, _, _, _ = ear_loss(fl, kl, tr, tg, hm, key_oh, tun, tm, pos_w, head_w)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        done = step + 1
        if done % 250 == 0 or done == args.steps:
            dt = time.perf_counter() - t0
            sps = done / max(dt, 1e-6)
            print(f"step {done:6d}/{args.steps} | {sps:5.1f} st/s | loss={loss.item():.4f}", flush=True)
            train_log.append({"step": done, "elapsed_s": dt, "steps_per_s": sps, "lr": lr, "loss": float(loss.item())})
        if done % val_every == 0 or done == args.steps:
            vm = validate(model, valid_songs, cfg, hann, filters, device, amp_dtype, pos_w, head_w)
            vo = validate(model, radio_valid_songs, cfg, hann, filters, device, amp_dtype, pos_w, head_w) if radio_valid_songs else None
            vmu = validate(model, musdb_valid_songs, cfg, hann, filters, device, amp_dtype, pos_w, head_w) if musdb_valid_songs else None
            print(f"  VALID step {done}: all={vm} | radio={vo}", flush=True)
            history.append({"step": done, "elapsed_s": time.perf_counter() - t0, "all": vm, "orin": vo, "musdb": vmu})
            if vm["loss"] < best_loss:
                best_loss = vm["loss"]
                best_val = vm
                best_step = done
                model.export_safetensors(args.out, "best")
            hw_meta["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(0) / 1e6, 1)
            with open(os.path.join(args.out, "ear_results.json"), "w") as f:
                json.dump({
                    "hardware": hw_meta,
                    "best_step": best_step,
                    "best_val": best_val,
                    "elapsed_s": time.perf_counter() - t0,
                    "history": history,
                    "train_log": train_log,
                }, f, indent=2)
    print("EAR TRAINING DONE!", flush=True)


if __name__ == "__main__":
    main()
