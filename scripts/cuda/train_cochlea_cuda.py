#!/usr/bin/env python3
"""Native NVIDIA CUDA trainer for Sone-6.5M's Cochlea network (Band-Split Transformer).
Uses cuFFT STFT/iSTFT, cuDNN depthwise Conv1d, FlashAttention SDPA, batched band-group
Tensor Core projections, BF16+TF32 (or FP16+GradScaler on T4), fused AdamW, and
auto-checkpointing. Exports exact sone-infer/1 safetensors (f32 + f16)."""

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

STEMS = ["vocals", "drums", "bass"]
STEM_NAMES = ["vocals", "drums", "bass", "other"]
ALIASES = [
    ("vocals_r", "vocals"),
    ("drums_i", "drums"),
    ("bass_i", "bass"),
    ("other",),
]
EPS = 1e-5
BAND_EPS = 1e-10


def default_bands(n_fft: int = 2048):
    bins = n_fft // 2 + 1
    scale = n_fft / 2048.0
    table = [(48, 4), (192, 12), (384, 24), (768, 48), (bins, 128)]
    out = []
    at = 0
    for end, width in table:
        end = min(int(end * scale), bins)
        width = max(int(round(width * scale)), 1)
        while at < end:
            stop = min(at + width, end)
            if stop - at < (width + 1) // 2 and out:
                out[-1][1] = stop
            else:
                out.append([at, stop])
            at = stop
    return out


def stems_config(size: str = "base") -> dict:
    dims = {"small": (48, 4, 4, 128), "base": (64, 6, 4, 192), "large": (96, 8, 6, 256)}
    dim, blocks, heads, mask_hidden = dims[size]
    cycle = [1, 2, 4, 8, 16, 32]
    return dict(
        sample_rate=44100,
        n_fft=2048,
        hop=512,
        bands=default_bands(2048),
        dim=dim,
        blocks=blocks,
        heads=heads,
        ff_mult=2,
        kernel=5,
        dilations=[cycle[i % len(cycle)] for i in range(blocks)],
        mask_hidden=mask_hidden,
    )


def band_groups(bands):
    groups = []
    for k, (lo, hi) in enumerate(bands):
        w = hi - lo
        if groups and groups[-1][2] == w:
            groups[-1] = (groups[-1][0], groups[-1][1] + 1, w)
        else:
            groups.append((k, 1, w))
    return groups


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = EPS):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.gain


class TimeBlock(nn.Module):
    def __init__(self, dim: int, inner: int, kernel: int, dilation: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        pad = (kernel // 2) * dilation
        self.conv = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation, groups=dim)
        self.l1 = nn.Linear(dim, inner)
        self.l2 = nn.Linear(inner, dim)

    def forward(self, x):
        # x: (B, T, K, D)
        b, t, k, d = x.shape
        h = self.norm(x).permute(0, 2, 3, 1).reshape(b * k, d, t)
        h = self.conv(h).reshape(b, k, d, t).permute(0, 3, 1, 2)
        return x + self.l2(F.gelu(self.l1(h), approximate="none"))


class BandAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.heads = heads
        self.hd = dim // heads

    def forward(self, x):
        # x: (B, T, K, D) -> attend over K
        b, t, k, d = x.shape
        h = self.norm(x).reshape(b * t, k, d)
        q, kk, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b * t, k, self.heads, self.hd).transpose(1, 2)
        kk = kk.view(b * t, k, self.heads, self.hd).transpose(1, 2)
        v = v.view(b * t, k, self.heads, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, kk, v)
        o = self.out(o.transpose(1, 2).contiguous().view(b, t, k, d))
        return x + o


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.l1 = nn.Linear(dim, inner)
        self.l2 = nn.Linear(inner, dim)

    def forward(self, x):
        return x + self.l2(F.gelu(self.l1(self.norm(x)), approximate="none"))


class StemsNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.n_fft = cfg["n_fft"]
        self.hop = cfg["hop"]
        self.bands = cfg["bands"]
        self.groups = band_groups(self.bands)
        d = cfg["dim"]
        k = len(self.bands)
        ff = d * cfg["ff_mult"]
        m = cfg["mask_hidden"]

        self.register_buffer("hann", torch.hann_window(self.n_fft, periodic=True))

        # Batched per-group split weights for fast Tensor Core matmuls
        self.split_norm = nn.ParameterList()
        self.split_w = nn.ParameterList()
        self.split_b = nn.ParameterList()
        for _, cnt, w in self.groups:
            w4 = 4 * w
            self.split_norm.append(nn.Parameter(torch.ones(cnt, w4)))
            self.split_w.append(nn.Parameter(torch.empty(cnt, d, w4)))
            self.split_b.append(nn.Parameter(torch.zeros(cnt, d)))
            nn.init.kaiming_uniform_(self.split_w[-1], a=math.sqrt(5))

        self.band_emb = nn.Parameter(torch.randn(k, d) * 0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                TimeBlock(d, ff, cfg["kernel"], cfg["dilations"][i]),
                BandAttention(d, cfg["heads"]),
                FeedForward(d, ff),
            ])
            for i in range(cfg["blocks"])
        ])
        self.norm = RMSNorm(d)

        # Batched per-stem, per-group mask heads
        self.mask_l1_w = nn.Parameter(torch.empty(3, k, m, d))
        self.mask_l1_b = nn.Parameter(torch.zeros(3, k, m))
        nn.init.kaiming_uniform_(self.mask_l1_w, a=math.sqrt(5))

        self.mask_l2_w = nn.ParameterList()
        self.mask_l2_b = nn.ParameterList()
        for _, cnt, w in self.groups:
            w4 = 4 * w
            p = nn.Parameter(torch.empty(3, cnt, w4, m))
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.mask_l2_w.append(p)
            self.mask_l2_b.append(nn.Parameter(torch.zeros(3, cnt, w4)))

    def stft_stereo(self, wave):
        # wave: (B, 2, L) -> spec: (B, T, F, 4) [re_l, im_l, re_r, im_r]
        b, ch, length = wave.shape
        x = wave.float().reshape(b * ch, length)
        c = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop,
            window=self.hann,
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (B*2, F, T)
        ri = torch.view_as_real(c)  # (B*2, F, T, 2)
        ri = ri.reshape(b, ch, ri.shape[1], ri.shape[2], 2).permute(0, 3, 2, 1, 4).reshape(b, ri.shape[2], ri.shape[1], 4)
        return ri

    def istft_stereo(self, spec, length: int):
        # spec: (N, T, F, 4) -> wave: (N, 2, length)
        n, t, f, _ = spec.shape
        ri = spec.float().reshape(n, t, f, 2, 2).permute(0, 3, 2, 1, 4).contiguous().reshape(n * 2, f, t, 2)
        # Zero imaginary parts of DC and Nyquist to match dsp.rs
        ri = ri.clone()
        ri[:, 0, :, 1] = 0.0
        ri[:, -1, :, 1] = 0.0
        c = torch.view_as_complex(ri)
        w = torch.istft(
            c,
            self.n_fft,
            hop_length=self.hop,
            window=self.hann,
            center=True,
            normalized=False,
            onesided=True,
            length=length,
        )
        return w.reshape(n, 2, length)

    def _trunk(self, mix_spec):
        b, t_n = mix_spec.shape[0], mix_spec.shape[1]
        # Band split
        parts = []
        for gi, (k0, cnt, w) in enumerate(self.groups):
            f0 = self.bands[k0][0]
            f1 = f0 + cnt * w
            seg = mix_spec[:, :, f0:f1, :].reshape(b, t_n, cnt, 4 * w)
            normed = seg * torch.rsqrt(seg.pow(2).mean(dim=-1, keepdim=True) + BAND_EPS) * self.split_norm[gi]
            proj = torch.einsum("btkw,kdw->btkd", normed, self.split_w[gi]) + self.split_b[gi]
            parts.append(proj)
        x = torch.cat(parts, dim=2) + self.band_emb

        for tb, ba, ff in self.blocks:
            x = ff(ba(tb(x)))
        x = self.norm(x)

        # Mask heads: l1 across all 3 stems and K bands at once
        h = F.gelu(torch.einsum("btkd,skmd->sbtkm", x, self.mask_l1_w) + self.mask_l1_b[:, None, None, :, :], approximate="none")
        mask_parts = []
        for gi, (k0, cnt, w) in enumerate(self.groups):
            hg = h[:, :, :, k0:k0 + cnt, :]  # (3, B, T, cnt, m)
            mg = torch.einsum("sbtkm,skwm->sbtkw", hg, self.mask_l2_w[gi]) + self.mask_l2_b[gi][:, None, None, :, :]
            mask_parts.append(mg.reshape(3, b, t_n, cnt * w, 4))
        return torch.cat(mask_parts, dim=3).permute(1, 0, 2, 3, 4)

    def forward(self, mix):
        # mix: (B, 2, L)
        b, _, length = mix.shape
        mix_spec = self.stft_stereo(mix)  # (B, T, F, 4) fp32
        t_n = mix_spec.shape[1]

        masks = self._trunk(mix_spec).float()  # (B, 3, T, F, 4)

        # Complex multiply mask * mix_spec
        mx = mix_spec.unsqueeze(1)  # (B, 1, T, F, 4)
        mr_l, mi_l, mr_r, mi_r = masks[..., 0], masks[..., 1], masks[..., 2], masks[..., 3]
        xr_l, xi_l, xr_r, xi_r = mx[..., 0], mx[..., 1], mx[..., 2], mx[..., 3]
        est3_spec = torch.stack([
            mr_l * xr_l - mi_l * xi_l,
            mr_l * xi_l + mi_l * xr_l,
            mr_r * xr_r - mi_r * xi_r,
            mr_r * xi_r + mi_r * xr_r,
        ], dim=-1)  # (B, 3, T, F, 4)
        other_spec = mix_spec.unsqueeze(1) - est3_spec.sum(dim=1, keepdim=True)
        est_spec = torch.cat([est3_spec, other_spec], dim=1)  # (B, 4, T, F, 4)

        est3_wave = self.istft_stereo(est3_spec.reshape(b * 3, t_n, mix_spec.shape[2], 4), length).reshape(b, 3, 2, length)
        other_wave = mix.unsqueeze(1) - est3_wave.sum(dim=1, keepdim=True)
        est_wave = torch.cat([est3_wave, other_wave], dim=1)  # (B, 4, 2, L)
        return est_spec, est_wave, mix_spec

    def export_safetensors(self, out_dir: str, tag: str):
        os.makedirs(out_dir, exist_ok=True)
        tensors = {}
        for gi, (k0, cnt, _) in enumerate(self.groups):
            sn = self.split_norm[gi].detach().cpu().float()
            sw = self.split_w[gi].detach().cpu().float()
            sb = self.split_b[gi].detach().cpu().float()
            for j in range(cnt):
                b = k0 + j
                tensors[f"split.{b}.norm"] = sn[j].contiguous()
                tensors[f"split.{b}.w"] = sw[j].contiguous()
                tensors[f"split.{b}.b"] = sb[j].contiguous()
        tensors["band_emb"] = self.band_emb.detach().cpu().float().contiguous()
        for i, (tb, ba, ff) in enumerate(self.blocks):
            tensors[f"block{i}.time.norm"] = tb.norm.gain.detach().cpu().float().contiguous()
            tensors[f"block{i}.time.conv.w"] = tb.conv.weight.detach().cpu().float().squeeze(1).contiguous()
            tensors[f"block{i}.time.conv.b"] = tb.conv.bias.detach().cpu().float().contiguous()
            tensors[f"block{i}.time.l1.w"] = tb.l1.weight.detach().cpu().float().contiguous()
            tensors[f"block{i}.time.l1.b"] = tb.l1.bias.detach().cpu().float().contiguous()
            tensors[f"block{i}.time.l2.w"] = tb.l2.weight.detach().cpu().float().contiguous()
            tensors[f"block{i}.time.l2.b"] = tb.l2.bias.detach().cpu().float().contiguous()
            tensors[f"block{i}.band.norm"] = ba.norm.gain.detach().cpu().float().contiguous()
            tensors[f"block{i}.band.qkv"] = ba.qkv.weight.detach().cpu().float().contiguous()
            tensors[f"block{i}.band.out.w"] = ba.out.weight.detach().cpu().float().contiguous()
            tensors[f"block{i}.band.out.b"] = ba.out.bias.detach().cpu().float().contiguous()
            tensors[f"block{i}.ff.norm"] = ff.norm.gain.detach().cpu().float().contiguous()
            tensors[f"block{i}.ff.l1.w"] = ff.l1.weight.detach().cpu().float().contiguous()
            tensors[f"block{i}.ff.l1.b"] = ff.l1.bias.detach().cpu().float().contiguous()
            tensors[f"block{i}.ff.l2.w"] = ff.l2.weight.detach().cpu().float().contiguous()
            tensors[f"block{i}.ff.l2.b"] = ff.l2.bias.detach().cpu().float().contiguous()
        tensors["norm"] = self.norm.gain.detach().cpu().float().contiguous()
        l1w = self.mask_l1_w.detach().cpu().float()
        l1b = self.mask_l1_b.detach().cpu().float()
        for si, stem in enumerate(STEMS):
            for b in range(len(self.bands)):
                tensors[f"mask.{stem}.{b}.l1.w"] = l1w[si, b].contiguous()
                tensors[f"mask.{stem}.{b}.l1.b"] = l1b[si, b].contiguous()
            for gi, (k0, cnt, _) in enumerate(self.groups):
                l2w = self.mask_l2_w[gi].detach().cpu().float()
                l2b = self.mask_l2_b[gi].detach().cpu().float()
                for j in range(cnt):
                    b = k0 + j
                    tensors[f"mask.{stem}.{b}.l2.w"] = l2w[si, j].contiguous()
                    tensors[f"mask.{stem}.{b}.l2.b"] = l2b[si, j].contiguous()

        meta = {
            "model": "cochlea",
            "config": json.dumps(self.cfg, separators=(",", ":")),
            "format": "sone-infer/1",
        }
        p32 = os.path.join(out_dir, f"sone-cochlea-{tag}-f32.safetensors")
        p16 = os.path.join(out_dir, f"sone-cochlea-{tag}-f16.safetensors")
        save_file(tensors, p32, metadata=meta)
        save_file({k: v.half().contiguous() for k, v in tensors.items()}, p16, metadata=meta)
        return p32, p16


def stems_loss(est_spec, est_wave, mix_spec, target_wave, model: StemsNet, mr_wins: dict):
    # target_wave: (B, 4, 2, L)
    b, s, ch, length = target_wave.shape
    tgt_spec = model.stft_stereo(target_wave.reshape(b * s, ch, length)).reshape(b, s, est_spec.shape[2], est_spec.shape[3], 4)

    # 1. Complex spec L1
    spec_l1 = (est_spec - tgt_spec).abs().mean() / math.sqrt(model.n_fft)
    # 2. Wave L1
    wave_l1 = (est_wave - target_wave).abs().mean()
    # 3. Multi-res magnitude L1
    est_flat = est_wave.float().reshape(b * s * ch, length)
    tgt_flat = target_wave.float().reshape(b * s * ch, length)
    mr = 0.0
    for n_fft, win in mr_wins.items():
        hop = n_fft // 4
        ce = torch.stft(est_flat, n_fft, hop_length=hop, window=win, center=True, pad_mode="constant", return_complex=True)
        ct = torch.stft(tgt_flat, n_fft, hop_length=hop, window=win, center=True, pad_mode="constant", return_complex=True)
        me = torch.sqrt(ce.real.pow(2) + ce.imag.pow(2) + 1e-8)
        mt = torch.sqrt(ct.real.pow(2) + ct.imag.pow(2) + 1e-8)
        mr = mr + (me - mt).abs().mean() / math.sqrt(n_fft)
    multires = mr / len(mr_wins)

    # 4. Vocal-weighted leak loss (exact match to crates/mix-train/src/losses.rs)
    e_est = est_spec.pow(2).sum(dim=-1)          # (B, 4, T, F)
    e_tgt = tgt_spec.detach().pow(2).sum(dim=-1) # (B, 4, T, F)
    e_mix = mix_spec.detach().pow(2).sum(dim=-1) # (B, T, F)
    mean_mix = e_mix.mean(dim=(1, 2), keepdim=True)
    tau = 10.0 ** (-30.0 / 10.0)
    thresh = (e_mix * tau + mean_mix * 1e-6).unsqueeze(1)
    silent = (e_tgt < thresh).float()
    voice = (e_tgt[:, :1] >= thresh).float()
    is_instr = torch.tensor([0.0, 1.0, 1.0, 1.0], device=est_spec.device).view(1, 4, 1, 1)
    weight = voice * is_instr * (4.0 - 1.0) + 1.0
    leaked = (e_est * silent * weight).sum(dim=(1, 2, 3))
    total_e = e_mix.sum(dim=(1, 2)) + 1e-8
    leak = (leaked / total_e).mean()

    total = 10.0 * spec_l1 + 1.0 * wave_l1 + 5.0 * multires + 0.5 * leak
    return total, dict(spec=spec_l1.item(), wave=wave_l1.item(), mr=multires.item(), leak=leak.item())


def find_stem(song_dir: Path, aliases):
    for name in aliases:
        for ext in ("flac", "wav"):
            p = song_dir / f"{name}.{ext}"
            if p.exists():
                return p
    return None


def read_audio_file(p: Path, start: int, length: int) -> np.ndarray:
    x, _ = sf.read(str(p), start=start, frames=length, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2].T  # (2, L)


def index_songs(roots):
    songs = []
    for root in roots:
        root = Path(os.path.expanduser(root))
        if not root.exists():
            continue
        candidates = [root] + sorted([p for p in root.iterdir() if p.is_dir()])
        for d in candidates:
            st = [find_stem(d, a) for a in ALIASES]
            mix = find_stem(d, ("mixture", "mix"))
            if any(x is None for x in st[:3]) or (st[3] is None and mix is None):
                continue
            orig = [find_stem(d, (nm,)) for nm in STEM_NAMES]
            use_residual_other = (
                st[0].stem == "vocals_r"
                and (mix is not None or all(x is not None for x in orig))
            )
            infos = [sf.info(str(p)) for p in st if p is not None] + ([sf.info(str(mix))] if mix else [])
            n = min(i.frames for i in infos)
            songs.append(dict(name=d.name, stems=st, orig=orig, mix=mix, residual_other=use_residual_other, frames=n))
    songs.sort(key=lambda s: s["name"])
    return songs


def read_stem_chunk(song, s_idx: int, start: int, length: int) -> np.ndarray:
    if s_idx == 3 and (song["stems"][3] is None or song.get("residual_other")):
        if song["mix"] is not None:
            out = read_audio_file(song["mix"], start, length).copy()
        else:
            out = sum(read_audio_file(p, start, length) for p in song["orig"])
        for i in range(3):
            out -= read_audio_file(song["stems"][i], start, length)
        return out
    return read_audio_file(song["stems"][s_idx], start, length)


class StemsDataset(torch.utils.data.IterableDataset):
    def __init__(self, songs, chunk_len: int, remix_prob: float = 0.5):
        super().__init__()
        self.songs = [s for s in songs if s["frames"] > chunk_len]
        self.chunk_len = chunk_len
        self.remix_prob = remix_prob

    def __iter__(self):
        rng = random.Random(int.from_bytes(os.urandom(8), "little"))
        while True:
            base_s = rng.choice(self.songs)
            base_off = rng.randint(0, base_s["frames"] - self.chunk_len - 1)
            remix = rng.random() < self.remix_prob
            stems = np.zeros((4, 2, self.chunk_len), dtype=np.float32)
            for i in range(4):
                if remix:
                    s = rng.choice(self.songs)
                    off = rng.randint(0, s["frames"] - self.chunk_len - 1)
                else:
                    s, off = base_s, base_off
                x = read_stem_chunk(s, i, off, self.chunk_len)
                if rng.random() < 0.5:
                    x = x[::-1].copy()
                gain = 10.0 ** (rng.uniform(-6.0, 3.0) / 20.0)
                stems[i] = x * gain
            mix = stems.sum(axis=0)
            yield torch.from_numpy(mix), torch.from_numpy(stems)


def sdr_db(target: np.ndarray, est: np.ndarray) -> float:
    num = float(np.sum(target.astype(np.float64) ** 2))
    den = float(np.sum((target.astype(np.float64) - est.astype(np.float64)) ** 2)) + 1e-12
    return 10.0 * math.log10((num + 1e-12) / den)


@torch.no_grad()
def validate(model: StemsNet, songs, device, amp_dtype, valid_seconds=60.0, chunk_s=8.0, overlap_s=1.0):
    if not songs:
        return None
    model.eval()
    hop = model.hop
    chunk = (int(chunk_s * 44100) // hop) * hop
    overlap = (int(overlap_s * 44100) // hop) * hop
    step = chunk - overlap
    sdr_sum = np.zeros(4, dtype=np.float64)
    leak_sum = 0.0
    for song in songs:
        start = int(30.0 * 44100)
        if start + chunk >= song["frames"]:
            start = 0
        n = min(int(valid_seconds * 44100), song["frames"] - start)
        tgts = np.stack([read_stem_chunk(song, i, start, n) for i in range(4)], axis=0)  # (4, 2, n)
        mix = tgts.sum(axis=0)  # (2, n)
        out = np.zeros((4, 2, n), dtype=np.float32)
        wsum = np.zeros(n, dtype=np.float32)
        fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        starts = list(range(0, max(1, n - chunk), step))
        if not starts or starts[-1] + chunk < n:
            starts.append(max(0, n - chunk))
        for idx, st in enumerate(starts):
            seg = np.zeros((2, chunk), dtype=np.float32)
            avail = min(chunk, n - st)
            seg[:, :avail] = mix[:, st:st + avail]
            inp = torch.from_numpy(seg).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _, est_w, _ = model(inp)
            est_np = est_w[0, :3].float().cpu().numpy()
            w = np.ones(chunk, dtype=np.float32)
            if idx > 0:
                w[:overlap] = fade
            if idx + 1 < len(starts):
                w[-overlap:] = 1.0 - fade
            out[:3, :, st:st + avail] += est_np[:, :, :avail] * w[:avail]
            wsum[st:st + avail] += w[:avail]
        out[:3] /= np.maximum(wsum, 1e-8)
        out[3] = mix - out[:3].sum(axis=0)
        for i in range(4):
            sdr_sum[i] += sdr_db(tgts[i], out[i])
        # Vocal leak metric (20ms windows where vocals > instrumental)
        win = 882
        err, voc = 1e-12, 1e-12
        inst_t = tgts[1:].sum(axis=0)
        inst_e = out[1:].sum(axis=0)
        for w0 in range(0, n, win):
            v = tgts[0, :, w0:w0 + win].astype(np.float64)
            it = inst_t[:, w0:w0 + win].astype(np.float64)
            ie = inst_e[:, w0:w0 + win].astype(np.float64)
            ev, ei = float(np.sum(v * v)), float(np.sum(it * it))
            if ev > ei:
                voc += ev
                err += float(np.sum((ie - it) ** 2))
        leak_sum += 10.0 * math.log10(err / voc)
    model.train()
    k = len(songs)
    return dict(
        sdr_vocals=sdr_sum[0] / k,
        sdr_drums=sdr_sum[1] / k,
        sdr_bass=sdr_sum[2] / k,
        sdr_other=sdr_sum[3] / k,
        sdr_mean=float(sdr_sum.mean() / k),
        leak_db=leak_sum / k,
    )


def lr_at(step, total, base_lr=5e-4, warmup=2000, min_ratio=0.02):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--valid", nargs="+", default=[])
    ap.add_argument("--valid-atlas", nargs="+", default=[])
    ap.add_argument("--out", default="/content/stems_run")
    ap.add_argument("--size", default="base", choices=["small", "base", "large"])
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto based on GPU VRAM")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--chunk-seconds", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--valid-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    use_bf16 = major >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if args.batch_size <= 0:
        if vram_gb >= 70:
            args.batch_size = 32
        elif vram_gb >= 35:
            args.batch_size = 24
        elif vram_gb >= 20:
            args.batch_size = 16
        else:
            args.batch_size = 8
    cfg = stems_config(args.size)
    chunk_len = (int(args.chunk_seconds * cfg["sample_rate"]) // cfg["hop"]) * cfg["hop"]

    train_songs = index_songs(args.train)
    valid_songs = index_songs(args.valid)[:12] if args.valid else []
    valid_atlas = index_songs(args.valid_atlas)[:12] if args.valid_atlas else []
    print(
        f"Stems {args.size} on {torch.cuda.get_device_name(0)} (sm_{major}{minor}, {vram_gb:.1f}GB): "
        f"bs={args.batch_size} amp={amp_dtype} | {len(train_songs)} train, {len(valid_songs)} valid, "
        f"{len(valid_atlas)} valid_atlas, chunk={chunk_len/44100:.2f}s",
        flush=True,
    )

    torch.manual_seed(1)
    model = StemsNet(cfg).to(device)
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
        "chunk_seconds": chunk_len / 44100.0,
        "exact_params": exact_params,
        "train_songs": len(train_songs),
        "valid_musdb_songs": len(valid_songs),
        "valid_radio_songs": len(valid_atlas),
    }
    print(f"Params: {exact_params:,} ({exact_params/1e6:.3f}M) | HW: {hw_meta}", flush=True)
    if use_bf16 and not args.no_compile:
        try:
            model._trunk = torch.compile(model._trunk, mode="default")
            print("Compiled _trunk with torch.compile", flush=True)
        except Exception as e:
            print(f"torch.compile skipped: {e}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    start_step = 0
    best_sdr = -1e9
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = args.resume or os.path.join(args.out, "latest.pt")
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        best_sdr = ck.get("best_sdr", -1e9)
        print(f"Resumed from {ckpt_path} at step {start_step} (best_sdr={best_sdr:.2f})", flush=True)

    ds = StemsDataset(train_songs, chunk_len)
    loader = iter(torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, prefetch_factor=4, persistent_workers=True))
    mr_wins = {n: torch.hann_window(n, periodic=True, device=device) for n in (512, 1024, 2048)}

    t0 = time.perf_counter()
    acc = {"loss": 0.0, "spec": 0.0, "wave": 0.0, "mr": 0.0, "leak": 0.0, "n": 0}
    history = []
    train_log = []
    for step in range(start_step, args.steps):
        cur_lr = lr_at(step, args.steps, base_lr=args.lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        mix, tgt = next(loader)
        mix = mix.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            est_spec, est_wave, mix_spec = model(mix)
            loss, parts = stems_loss(est_spec, est_wave, mix_spec, tgt, model, mr_wins)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        acc["loss"] += loss.item()
        for k, v in parts.items():
            acc[k] += v
        acc["n"] += 1

        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            dt = time.perf_counter() - t0
            sps = (done - start_step) / max(dt, 1e-6)
            n = acc["n"]
            print(
                f"step {done:6d}/{args.steps} | {sps:5.2f} st/s ({sps*args.batch_size:5.1f} ex/s) | "
                f"lr={cur_lr:.2e} | loss={acc['loss']/n:.4f} (spec={acc['spec']/n:.4f} wave={acc['wave']/n:.4f} mr={acc['mr']/n:.4f} leak={acc['leak']/n:.4f})",
                flush=True,
            )
            train_log.append({
                "step": done,
                "elapsed_s": dt,
                "steps_per_s": sps,
                "lr": cur_lr,
                "loss": acc["loss"] / n,
                "spec": acc["spec"] / n,
                "wave": acc["wave"] / n,
                "mr": acc["mr"] / n,
                "leak": acc["leak"] / n,
            })
            acc = {"loss": 0.0, "spec": 0.0, "wave": 0.0, "mr": 0.0, "leak": 0.0, "n": 0}

        if done % args.valid_every == 0 or done == args.steps:
            tmp_ck = os.path.join(args.out, "latest.pt.tmp")
            torch.save({"step": done, "model": model.state_dict(), "opt": opt.state_dict(), "best_sdr": best_sdr}, tmp_ck)
            os.replace(tmp_ck, os.path.join(args.out, "latest.pt"))
            rec = {"step": done, "elapsed_s": time.perf_counter() - t0}
            v = validate(model, valid_songs, device, amp_dtype)
            if v:
                rec["valid"] = v
                print(
                    f"VALID step {done:6d} | SDR voc={v['sdr_vocals']:.2f} drm={v['sdr_drums']:.2f} bas={v['sdr_bass']:.2f} oth={v['sdr_other']:.2f} mean={v['sdr_mean']:.2f} dB | leak={v['leak_db']:.2f} dB",
                    flush=True,
                )
            va = validate(model, valid_atlas, device, amp_dtype)
            if va:
                rec["valid_atlas"] = va
                print(
                    f"ATLAS step {done:6d} | SDR voc={va['sdr_vocals']:.2f} drm={va['sdr_drums']:.2f} bas={va['sdr_bass']:.2f} oth={va['sdr_other']:.2f} mean={va['sdr_mean']:.2f} dB | leak={va['leak_db']:.2f} dB",
                    flush=True,
                )
            score = va["sdr_mean"] if va else (v["sdr_mean"] if v else None)
            if va and v:
                score = 0.5 * (va["sdr_mean"] + v["sdr_mean"])
            if score is not None and score > best_sdr:
                best_sdr = score
                model.export_safetensors(args.out, "best")
                print(f"NEW BEST EXPORTED (score {best_sdr:.2f} dB)", flush=True)
            elif score is None:
                model.export_safetensors(args.out, "best")
            history.append(rec)
            hw_meta["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(0) / 1e6, 1)
            with open(os.path.join(args.out, "results.json"), "w") as f:
                json.dump({"hardware": hw_meta, "best_sdr": best_sdr, "history": history, "train_log": train_log}, f, indent=2)
            model.export_safetensors(args.out, "latest")


if __name__ == "__main__":
    main()
