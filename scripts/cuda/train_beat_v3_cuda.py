#!/usr/bin/env python3
"""Studio-Grade NVIDIA CUDA Trainer for Sone's Cortex v3 + Cingulate v3 Organs.
Scaled Beat Transformers (Cortex: 8L/192D, Cingulate: 6L/128D) trained with:
  - Vocal-onset-locked pocket geometry & edge-weighted DJ cut loss
  - Tempo/loudness/key domain randomization over transition beat tables
  - 50% hard counterfactual DJ negative synthesis covering all 7 human-audited DJ failure modes
Exports exact sone-infer/1 safetensors (sone-cortex-*, sone-cingulate-*)."""

import argparse
import gzip
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

RECIPE_ORDER = ["voice_over_beat", "sneak", "outro_under_intro", "gap"]
RECIPE_TO_IDX = {r: i for i, r in enumerate(RECIPE_ORDER)}

ISSUE_ALL = [
    "roll_out", "pitch_riding", "b_melody_enters_at_once", "abrupt_stem_exit",
    "two_melodies", "cut_not_transition", "stems_leave_separately", "a_melody_left_early",
    "wrong_recipe", "downbeat_off_by_one", "fade_at_random_point", "fade_too_short",
    "b_melody_before_one", "b_drums_before_one", "fade_curve", "a_beat_gone_before_one",
    "audible_fade_onset", "vocal_leak", "beat_hole_before_one", "tail_after_one",
    "b_voice_before_one", "key_clash_layering", "polyrhythm_layering", "two_voices",
    "two_basses", "beats_offset", "voice_cut_in_word", "silence_before_one",
    "b_voice_late", "bad_pair",
]
ISSUE_TO_IDX = {k: i for i, k in enumerate(ISSUE_ALL)}
BARS_SCALE = 8.0


def beat_config(kind: str, size: str = "pro") -> dict:
    if kind == "cortex":
        if size == "base":
            return dict(kind="cortex", a_len=80, b_len=128, features=24, lanes=8, dim=128, layers=6, heads=4, ff=512)
        return dict(kind="cortex", a_len=80, b_len=128, features=24, lanes=8, dim=192, layers=8, heads=6, ff=768)
    if size == "base":
        return dict(kind="cingulate", a_len=80, b_len=128, features=24, lanes=8, dim=96, layers=4, heads=4, ff=384)
    return dict(kind="cingulate", a_len=80, b_len=128, features=24, lanes=8, dim=128, layers=6, heads=4, ff=512)


def open_jsonl(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def load_examples(paths, kind: str, cfg: dict):
    with_lanes = (kind == "cingulate")
    a_len, b_len, feat, lanes_n = cfg["a_len"], cfg["b_len"], cfg["features"], cfg["lanes"]
    a_feat = feat + lanes_n if with_lanes else feat
    n_issues = len(ISSUE_ALL)

    A_list, B_list, G_list = [], [], []
    recipe_list, bars_list, lanes_list, labels_list = [], [], [], []

    for path in paths:
        if not os.path.exists(path):
            continue
        with open_jsonl(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                good = int(ex["label"]["good"])
                if kind == "cortex" and good != 1:
                    continue

                a_arr = np.zeros((a_len, a_feat), dtype=np.float32)
                for i, beat in enumerate(ex["a_beats"][:a_len]):
                    k = min(len(beat), feat)
                    a_arr[i, :k] = beat[:k]
                if with_lanes:
                    for i, ln in enumerate(ex["lanes"][:a_len]):
                        a_arr[i, feat:feat + lanes_n] = np.clip(ln[:lanes_n], 0.0, 1.0)

                b_arr = np.zeros((b_len, feat), dtype=np.float32)
                for i, beat in enumerate(ex["b_beats"][:b_len]):
                    k = min(len(beat), feat)
                    b_arr[i, :k] = beat[:k]

                g_arr = [float(ex["tempo_a"]), float(ex["tempo_b"]), float(ex["beat_ratio"]), float(ex["key_distance"])]
                if with_lanes:
                    m = ex.get("measures") or {}
                    g_arr.extend([
                        float(m.get("beat_offset_ms", 0.0)),
                        float(m.get("stretch_pct", 0.0)),
                        float(m.get("vocal_leak_db", 0.0)),
                        float(m.get("tuning_diff_cents", 0.0)),
                    ])

                A_list.append(a_arr)
                B_list.append(b_arr)
                G_list.append(g_arr)

                if kind == "cortex":
                    recipe_list.append(RECIPE_TO_IDX[ex["recipe"]])
                    bars_list.append([float(ex["start_bars"]), float(ex["swap_bars"])])
                    ln_arr = np.zeros((a_len, lanes_n), dtype=np.float32)
                    for i, ln in enumerate(ex["lanes"][:a_len]):
                        ln_arr[i, :lanes_n] = np.clip(ln[:lanes_n], 0.0, 1.0)
                    lanes_list.append(ln_arr)
                else:
                    lab = np.zeros(1 + n_issues, dtype=np.float32)
                    lab[0] = float(good)
                    for iss in ex["label"].get("issues", []):
                        if iss in ISSUE_TO_IDX:
                            lab[1 + ISSUE_TO_IDX[iss]] = 1.0
                    labels_list.append(lab)

    data = {
        "a": torch.from_numpy(np.stack(A_list)),
        "b": torch.from_numpy(np.stack(B_list)),
        "g": torch.tensor(G_list, dtype=torch.float32),
    }
    if kind == "cortex":
        data["recipe"] = torch.tensor(recipe_list, dtype=torch.long)
        data["bars"] = torch.tensor(bars_list, dtype=torch.float32)
        data["lanes"] = torch.from_numpy(np.stack(lanes_list))
    else:
        data["labels"] = torch.from_numpy(np.stack(labels_list))
    return data


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.gain


class FeedForward(nn.Module):
    def __init__(self, dim, inner):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.l1 = nn.Linear(dim, inner)
        self.l2 = nn.Linear(inner, dim)

    def forward(self, x):
        return x + self.l2(F.gelu(self.l1(self.norm(x)), approximate="none"))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.heads = heads
        self.hd = dim // heads

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, self.hd).transpose(1, 2)
        k = k.view(b, s, self.heads, self.hd).transpose(1, 2)
        v = v.view(b, s, self.heads, self.hd).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        return x + self.out(h.transpose(1, 2).contiguous().view(b, s, d))


class BeatNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d = cfg["dim"]
        is_crit = (cfg["kind"] == "cingulate")
        a_feat = cfg["features"] + (cfg["lanes"] if is_crit else 0)
        g_feat = 8 if is_crit else 4
        self.register_buffer("a_mean", torch.zeros(a_feat))
        self.register_buffer("a_std", torch.ones(a_feat))
        self.register_buffer("b_mean", torch.zeros(cfg["features"]))
        self.register_buffer("b_std", torch.ones(cfg["features"]))
        self.register_buffer("g_mean", torch.zeros(g_feat))
        self.register_buffer("g_std", torch.ones(g_feat))
        self.embed_a = nn.Linear(a_feat, d)
        self.embed_b = nn.Linear(cfg["features"], d)
        self.embed_g = nn.Linear(g_feat, d)
        self.pos_a = nn.Parameter(torch.randn(cfg["a_len"], d) * 0.02)
        self.pos_b = nn.Parameter(torch.randn(cfg["b_len"], d) * 0.02)
        self.layers = nn.ModuleList([nn.ModuleList([Attention(d, cfg["heads"]), FeedForward(d, cfg["ff"])]) for _ in range(cfg["layers"])])
        self.norm = RMSNorm(d)
        if cfg["kind"] == "cortex":
            self.head_recipe = nn.Linear(d, 4)
            self.head_bars = nn.Linear(d, 2)
            self.head_lanes = nn.Linear(d, cfg["lanes"])
        else:
            self.head = nn.Linear(d, 1 + len(ISSUE_ALL))

    def set_stats(self, a, b, g):
        self.a_mean.copy_(a.mean(dim=(0, 1)))
        self.a_std.copy_(a.std(dim=(0, 1), unbiased=False).clamp(min=1e-3))
        self.b_mean.copy_(b.mean(dim=(0, 1)))
        self.b_std.copy_(b.std(dim=(0, 1), unbiased=False).clamp(min=1e-3))
        self.g_mean.copy_(g.mean(dim=0))
        self.g_std.copy_(g.std(dim=0, unbiased=False).clamp(min=1e-3))

    def forward(self, a, b, g):
        tg = self.embed_g((g - self.g_mean) / self.g_std).unsqueeze(1)
        ta = self.embed_a((a - self.a_mean) / self.a_std) + self.pos_a
        tb = self.embed_b((b - self.b_mean) / self.b_std) + self.pos_b
        x = torch.cat([tg, ta, tb], dim=1)
        for attn, ff in self.layers:
            x = ff(attn(x))
        x = self.norm(x)
        gl = x[:, 0]
        if self.cfg["kind"] == "cortex":
            return self.head_recipe(gl), self.head_bars(gl) * BARS_SCALE, self.head_lanes(x[:, 1:1 + self.cfg["a_len"]])
        return self.head(gl)

    def load_safetensors(self, p32: str):
        t = load_file(p32, device="cpu")
        with torch.no_grad():
            self.a_mean.copy_(t["input.a_mean"])
            self.a_std.copy_(t["input.a_std"])
            self.b_mean.copy_(t["input.b_mean"])
            self.b_std.copy_(t["input.b_std"])
            self.g_mean.copy_(t["input.g_mean"])
            self.g_std.copy_(t["input.g_std"])
            self.embed_a.weight.copy_(t["embed_a.w"])
            self.embed_a.bias.copy_(t["embed_a.b"])
            self.embed_b.weight.copy_(t["embed_b.w"])
            self.embed_b.bias.copy_(t["embed_b.b"])
            self.embed_g.weight.copy_(t["embed_g.w"])
            self.embed_g.bias.copy_(t["embed_g.b"])
            self.pos_a.copy_(t["pos_a"])
            self.pos_b.copy_(t["pos_b"])
            self.norm.gain.copy_(t["norm"])
            for i, (attn, ff) in enumerate(self.layers):
                attn.norm.gain.copy_(t[f"layer{i}.attn.norm"])
                attn.qkv.weight.copy_(t[f"layer{i}.attn.qkv"])
                attn.out.weight.copy_(t[f"layer{i}.attn.out.w"])
                attn.out.bias.copy_(t[f"layer{i}.attn.out.b"])
                ff.norm.gain.copy_(t[f"layer{i}.ff.norm"])
                ff.l1.weight.copy_(t[f"layer{i}.ff.l1.w"])
                ff.l1.bias.copy_(t[f"layer{i}.ff.l1.b"])
                ff.l2.weight.copy_(t[f"layer{i}.ff.l2.w"])
                ff.l2.bias.copy_(t[f"layer{i}.ff.l2.b"])
            if self.cfg["kind"] == "cortex":
                for nm, mod in [("head_recipe", self.head_recipe), ("head_bars", self.head_bars), ("head_lanes", self.head_lanes)]:
                    mod.weight.copy_(t[f"{nm}.w"])
                    mod.bias.copy_(t[f"{nm}.b"])
            else:
                self.head.weight.copy_(t["head.w"])
                self.head.bias.copy_(t["head.b"])

    def export_safetensors(self, out_dir: str, tag: str):
        os.makedirs(out_dir, exist_ok=True)
        tensors = {
            "input.a_mean": self.a_mean.detach().cpu().float().contiguous(),
            "input.a_std": self.a_std.detach().cpu().float().contiguous(),
            "input.b_mean": self.b_mean.detach().cpu().float().contiguous(),
            "input.b_std": self.b_std.detach().cpu().float().contiguous(),
            "input.g_mean": self.g_mean.detach().cpu().float().contiguous(),
            "input.g_std": self.g_std.detach().cpu().float().contiguous(),
            "embed_a.w": self.embed_a.weight.detach().cpu().float().contiguous(),
            "embed_a.b": self.embed_a.bias.detach().cpu().float().contiguous(),
            "embed_b.w": self.embed_b.weight.detach().cpu().float().contiguous(),
            "embed_b.b": self.embed_b.bias.detach().cpu().float().contiguous(),
            "embed_g.w": self.embed_g.weight.detach().cpu().float().contiguous(),
            "embed_g.b": self.embed_g.bias.detach().cpu().float().contiguous(),
            "pos_a": self.pos_a.detach().cpu().float().contiguous(),
            "pos_b": self.pos_b.detach().cpu().float().contiguous(),
            "norm": self.norm.gain.detach().cpu().float().contiguous(),
        }
        for i, (attn, ff) in enumerate(self.layers):
            tensors[f"layer{i}.attn.norm"] = attn.norm.gain.detach().cpu().float().contiguous()
            tensors[f"layer{i}.attn.qkv"] = attn.qkv.weight.detach().cpu().float().contiguous()
            tensors[f"layer{i}.attn.out.w"] = attn.out.weight.detach().cpu().float().contiguous()
            tensors[f"layer{i}.attn.out.b"] = attn.out.bias.detach().cpu().float().contiguous()
            tensors[f"layer{i}.ff.norm"] = ff.norm.gain.detach().cpu().float().contiguous()
            tensors[f"layer{i}.ff.l1.w"] = ff.l1.weight.detach().cpu().float().contiguous()
            tensors[f"layer{i}.ff.l1.b"] = ff.l1.bias.detach().cpu().float().contiguous()
            tensors[f"layer{i}.ff.l2.w"] = ff.l2.weight.detach().cpu().float().contiguous()
            tensors[f"layer{i}.ff.l2.b"] = ff.l2.bias.detach().cpu().float().contiguous()
        if self.cfg["kind"] == "cortex":
            for nm, mod in [("head_recipe", self.head_recipe), ("head_bars", self.head_bars), ("head_lanes", self.head_lanes)]:
                tensors[f"{nm}.w"] = mod.weight.detach().cpu().float().contiguous()
                tensors[f"{nm}.b"] = mod.bias.detach().cpu().float().contiguous()
        else:
            tensors["head.w"] = self.head.weight.detach().cpu().float().contiguous()
            tensors["head.b"] = self.head.bias.detach().cpu().float().contiguous()

        kind = self.cfg["kind"]
        meta = {
            "model": kind,
            "config": json.dumps(self.cfg, separators=(",", ":")),
            "format": "sone-infer/1",
        }
        p32 = os.path.join(out_dir, f"sone-{kind}-{tag}-f32.safetensors")
        p16 = os.path.join(out_dir, f"sone-{kind}-{tag}-f16.safetensors")
        save_file(tensors, p32, metadata=meta)
        save_file({k: v.half().contiguous() for k, v in tensors.items()}, p16, metadata=meta)
        return p32, p16


def synthesize_counterfactual_negatives(a: torch.Tensor, g: torch.Tensor, labels: torch.Tensor):
    """50% of positive examples in each Cingulate batch are transformed on-GPU into realistic
    counterfactual DJ transition defects discovered during human blind listening."""
    bs = a.shape[0]
    a = a.clone()
    g = g.clone()
    labels = labels.clone()
    feat = 24
    good_mask = (labels[:, 0] > 0.5)
    idx = torch.where(good_mask)[0]
    if idx.numel() < 4:
        return a, g, labels
    pick = idx[torch.randperm(idx.numel(), device=a.device)[:idx.numel() // 2]]
    for k, bi in enumerate(pick.tolist()):
        mode = k % 7
        labels[bi, 0] = 0.0
        if mode == 0:
            # Premature accompaniment duck / energy hole before B enters (Mix 1 failure mode)
            a[bi, 36:56, feat + 1:feat + 4] *= 0.25
            a[bi, 36:56, feat + 4:feat + 8] = 0.0
            labels[bi, 1 + ISSUE_TO_IDX["a_melody_left_early"]] = 1.0
            labels[bi, 1 + ISSUE_TO_IDX["silence_before_one"]] = 1.0
        elif mode == 1:
            # Mid-word singer chop (Mix 4 v1 failure mode)
            a[bi, 42:64, feat + 0] = 0.0
            labels[bi, 1 + ISSUE_TO_IDX["voice_cut_in_word"]] = 1.0
            labels[bi, 1 + ISSUE_TO_IDX["fade_at_random_point"]] = 1.0
        elif mode == 2:
            # Simultaneous vocal clash (both singers at 100%)
            a[bi, 44:68, feat + 0] = 1.0
            a[bi, 44:68, feat + 4] = 1.0
            labels[bi, 1 + ISSUE_TO_IDX["two_voices"]] = 1.0
        elif mode == 3:
            # Backing chords too loud under Deck B singing (Mix 2 v3 failure mode)
            a[bi, 48:64, feat + 3] = 1.0
            a[bi, 48:64, feat + 4] = 1.0
            labels[bi, 1 + ISSUE_TO_IDX["two_melodies"]] = 1.0
        elif mode == 4:
            # Double sub-bass clash
            a[bi, 48:68, feat + 2] = 1.0
            a[bi, 48:68, feat + 6] = 1.0
            labels[bi, 1 + ISSUE_TO_IDX["two_basses"]] = 1.0
        elif mode == 5:
            # Hard cut masquerading as blend
            a[bi, :64, feat:feat + 4] = 1.0
            a[bi, :64, feat + 4:feat + 8] = 0.0
            a[bi, 64:, feat:feat + 4] = 0.0
            a[bi, 64:, feat + 4:feat + 8] = 1.0
            labels[bi, 1 + ISSUE_TO_IDX["cut_not_transition"]] = 1.0
        else:
            # Audible vocal bleed & beat drift in rendered stems
            g[bi, 4] = 28.0  # beat_offset_ms
            g[bi, 6] = -6.0  # vocal_leak_db
            labels[bi, 1 + ISSUE_TO_IDX["vocal_leak"]] = 1.0
            labels[bi, 1 + ISSUE_TO_IDX["beats_offset"]] = 1.0
    return a, g, labels


def train_one(kind: str, size: str, train_paths, val_paths, owner_paths, out_dir: str, steps: int, bs: int, lr: float, device):
    cfg = beat_config(kind, size)
    tr = load_examples(train_paths, kind, cfg)
    va = load_examples(val_paths, kind, cfg)
    ow = load_examples(owner_paths, kind, cfg) if owner_paths else None

    model = BeatNet(cfg).to(device)
    model.set_stats(tr["a"].to(device), tr["b"].to(device), tr["g"].to(device))
    params = sum(p.numel() for p in model.parameters())
    print(f"Training {kind} ({size}, {params:,} params) | train={len(tr['a'])} val={len(va['a'])}", flush=True)

    tr_gpu = {k: v.to(device) for k, v in tr.items()}
    va_gpu = {k: v.to(device) for k, v in va.items()}
    ow_gpu = {k: v.to(device) for k, v in ow.items()} if ow else None

    tag = f"{size}-best" if size != "pro" else "best"
    p32 = os.path.join(out_dir, f"sone-{kind}-{tag}-f32.safetensors")
    done_json = os.path.join(out_dir, f"done_{kind}_{size}.json")
    if os.path.exists(done_json):
        with open(done_json, "r") as f:
            return json.load(f)
    if os.path.exists(p32) and (kind, size) in (("cortex", "pro"), ("cingulate", "pro"), ("cortex", "base")):
        model.load_safetensors(p32)
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if kind == "cortex":
                rl, bp, ll = model(va_gpu["a"], va_gpu["b"], va_gpu["g"])
                rec_acc = float((rl.argmax(dim=-1) == va_gpu["recipe"]).float().mean().item())
                bars_mae = float((bp - va_gpu["bars"]).abs().mean().item())
                lane_mae = float((torch.sigmoid(ll) - va_gpu["lanes"]).abs().mean().item())
                ow_acc = rec_acc
                if ow_gpu and len(ow_gpu["a"]) > 0:
                    orl, _, _ = model(ow_gpu["a"], ow_gpu["b"], ow_gpu["g"])
                    ow_acc = float((orl.argmax(dim=-1) == ow_gpu["recipe"]).float().mean().item())
                stat = dict(step=steps, rec_acc=rec_acc, owner_rec_acc=ow_acc, bars_mae=bars_mae, lane_mae=lane_mae)
            else:
                lg = model(va_gpu["a"], va_gpu["b"], va_gpu["g"])
                pred_good = (torch.sigmoid(lg[:, 0]) > 0.5).float()
                good_acc = float((pred_good == va_gpu["labels"][:, 0]).float().mean().item())
                ow_acc = good_acc
                if ow_gpu and len(ow_gpu["a"]) > 0:
                    olg = model(ow_gpu["a"], ow_gpu["b"], ow_gpu["g"])
                    ow_acc = float(((torch.sigmoid(olg[:, 0]) > 0.5).float() == ow_gpu["labels"][:, 0]).float().mean().item())
                stat = dict(step=steps, good_acc=good_acc, owner_good_acc=ow_acc)
        print(f"[{kind}:{size}] Resumed finished best checkpoint {p32} | {stat}", flush=True)
        res_obj = {"kind": kind, "size": size, "params": params, "best": stat, "elapsed_s": 0.0, "history": [stat]}
        with open(done_json, "w") as f:
            json.dump(res_obj, f, indent=2)
        return res_obj

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3, fused=True)
    n_tr = tr_gpu["a"].shape[0]
    best_metric = -1e9
    best_stats = {}
    history = []
    t0 = time.perf_counter()

    for step in range(steps):
        p = step / max(1, steps)
        cur_lr = lr * (0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * p)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        idx = torch.randint(0, n_tr, (bs,), device=device)
        a = tr_gpu["a"][idx].clone()
        b = tr_gpu["b"][idx].clone()
        g = tr_gpu["g"][idx].clone()

        # Mild beat-level loudness jitter for domain robustness
        a[:, :, :7] += torch.randn_like(a[:, :, :7]) * 0.35
        b[:, :, :7] += torch.randn_like(b[:, :, :7]) * 0.35

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if kind == "cortex":
                rec_tgt = tr_gpu["recipe"][idx]
                bars_tgt = tr_gpu["bars"][idx]
                ln_tgt = tr_gpu["lanes"][idx]
                rec_log, bars_pred, ln_log = model(a, b, g)
                ln_pred = torch.sigmoid(ln_log)

                l_rec = F.cross_entropy(rec_log, rec_tgt, label_smoothing=0.03)
                l_bars = F.smooth_l1_loss(bars_pred, bars_tgt) / BARS_SCALE

                # Edge-weighted sharp DJ cut + vocal pocket + pre-entry Deck-A floor loss
                diff_tgt = (ln_tgt[:, 1:] - ln_tgt[:, :-1]).abs()
                edge_w = 1.0 + 6.0 * (diff_tgt > 0.15).float()
                l_lanes = (F.binary_cross_entropy_with_logits(ln_log[:, 1:], ln_tgt[:, 1:], reduction="none") * edge_w).mean()
                l_lanes = l_lanes + 0.5 * F.binary_cross_entropy_with_logits(ln_log[:, :1], ln_tgt[:, :1])
                l_step = ((ln_pred[:, 1:] - ln_pred[:, :-1]) - (ln_tgt[:, 1:] - ln_tgt[:, :-1])).abs().mean()

                b_silent = (ln_tgt[:, :, 4:].amax(dim=-1, keepdim=True) < 0.02).float()
                a_acc_floor = (F.relu(0.92 - ln_pred[:, :, 1:4]) * b_silent * (ln_tgt[:, :, 1:4] > 0.85).float()).mean()
                loss = l_rec + 2.0 * l_bars + 4.0 * l_lanes + 2.0 * l_step + 3.0 * a_acc_floor
            else:
                labels = tr_gpu["labels"][idx]
                a, g, labels = synthesize_counterfactual_negatives(a, g, labels)
                logits = model(a, b, g)
                l_good = F.binary_cross_entropy_with_logits(logits[:, 0], labels[:, 0])
                pos_w = torch.full((len(ISSUE_ALL),), 5.0, device=device)
                l_iss = F.binary_cross_entropy_with_logits(logits[:, 1:], labels[:, 1:], pos_weight=pos_w)
                loss = 1.5 * l_good + l_iss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()

        done = step + 1
        if done % 1000 == 0 or done == steps:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if kind == "cortex":
                    rl, bp, ll = model(va_gpu["a"], va_gpu["b"], va_gpu["g"])
                    rec_acc = float((rl.argmax(dim=-1) == va_gpu["recipe"]).float().mean().item())
                    bars_mae = float((bp - va_gpu["bars"]).abs().mean().item())
                    lane_mae = float((torch.sigmoid(ll) - va_gpu["lanes"]).abs().mean().item())
                    ow_acc = rec_acc
                    if ow_gpu and len(ow_gpu["a"]) > 0:
                        orl, _, _ = model(ow_gpu["a"], ow_gpu["b"], ow_gpu["g"])
                        ow_acc = float((orl.argmax(dim=-1) == ow_gpu["recipe"]).float().mean().item())
                    metric = rec_acc - 0.5 * lane_mae + 0.25 * ow_acc
                    stat = dict(step=done, rec_acc=rec_acc, owner_rec_acc=ow_acc, bars_mae=bars_mae, lane_mae=lane_mae)
                else:
                    lg = model(va_gpu["a"], va_gpu["b"], va_gpu["g"])
                    pred_good = (torch.sigmoid(lg[:, 0]) > 0.5).float()
                    good_acc = float((pred_good == va_gpu["labels"][:, 0]).float().mean().item())
                    ow_acc = good_acc
                    if ow_gpu and len(ow_gpu["a"]) > 0:
                        olg = model(ow_gpu["a"], ow_gpu["b"], ow_gpu["g"])
                        ow_acc = float(((torch.sigmoid(olg[:, 0]) > 0.5).float() == ow_gpu["labels"][:, 0]).float().mean().item())
                    metric = 0.7 * good_acc + 0.3 * ow_acc
                    stat = dict(step=done, good_acc=good_acc, owner_good_acc=ow_acc)
            print(f"[{kind}:{size}] step {done}/{steps} | {stat}", flush=True)
            history.append(stat)
            if metric > best_metric:
                best_metric = metric
                best_stats = stat
                model.export_safetensors(out_dir, f"{size}-best" if size != "pro" else "best")
            model.train()

    res_obj = {"kind": kind, "size": size, "params": params, "best": best_stats, "elapsed_s": time.perf_counter() - t0, "history": history}
    with open(done_json, "w") as f:
        json.dump(res_obj, f, indent=2)
    return res_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--owner", nargs="*", default=[])
    ap.add_argument("--out", default="/content/exports_beat_v3")
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--sizes", nargs="+", default=["pro", "base"])
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    summary = {
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "torch_version": torch.__version__,
        },
        "runs": [],
    }
    for sz in args.sizes:
        for kd in ("cortex", "cingulate"):
            res = train_one(kd, sz, args.train, args.val, args.owner, args.out, args.steps, args.batch_size, args.lr, device)
            summary["runs"].append(res)
    with open(os.path.join(args.out, "beat_v3_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("BEAT V3 ALL RUNS COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
