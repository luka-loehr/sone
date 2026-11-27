#!/usr/bin/env python3
"""Function-preserving Net2Wider + Net2Deeper expansion (Chen et al., ICLR 2016)
adapted to RMSNorm + Scaled Dot-Product Multi-Head Attention + Bounded Wiener Masking:
Expands 7.84 dB Cochlea-Pro (d=128, B=8, H=8, m=256; 20,069,012 params) into
Cochlea-Ultra (d=160, B=10, H=8, m=320; 30,836,180 params) with <1e-4 waveform
deviation at Step 0 and full-rank gradient flow across all new channels/blocks."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/content")
sys.path.insert(0, "/home/user/bot/sone/scripts/cuda")
from train_cochlea_v2_cuda import CochleaNet, cochlea_config


def expand_pro_to_ultra(pro_st_path: str, out_st_path: str) -> None:
    torch.manual_seed(42)
    pro_cfg = cochlea_config("pro")
    ult_cfg = cochlea_config("ultra")

    pro_net = CochleaNet(pro_cfg).eval()
    pro_net.load_safetensors(pro_st_path)
    ult_net = CochleaNet(ult_cfg).eval()

    d0, d1 = 128, 160
    h = 8
    hd0, hd1 = d0 // h, d1 // h  # 16 -> 20
    ff0, ff1 = 2 * d0, 2 * d1    # 256 -> 320
    m0, m1 = 256, 320
    rms_scale = math.sqrt(d0 / d1)
    qk_scale = (hd1 / hd0) ** 0.25

    with torch.no_grad():
        for gi in range(len(pro_net.groups)):
            ult_net.split_norm[gi].copy_(pro_net.split_norm[gi])
            ult_net.split_w[gi].zero_()
            ult_net.split_b[gi].zero_()
            ult_net.split_w[gi][:, :d0, :].copy_(pro_net.split_w[gi])
            ult_net.split_b[gi][:, :d0].copy_(pro_net.split_b[gi])

        ult_net.band_emb.zero_()
        ult_net.band_emb[:, :d0].copy_(pro_net.band_emb)

        def copy_mha(src_attn, dst_attn):
            dst_attn.norm.gain.zero_()
            dst_attn.norm.gain[:d0].copy_(src_attn.norm.gain * rms_scale)
            dst_attn.norm.gain[d0:].fill_(1.0)

            dst_attn.qkv.weight.zero_()
            sq, sk, sv = src_attn.qkv.weight.chunk(3, dim=0)
            dq, dk, dv = dst_attn.qkv.weight.chunk(3, dim=0)
            for hi in range(h):
                dq[hi * hd1 : hi * hd1 + hd0, :d0].copy_(sq[hi * hd0 : (hi + 1) * hd0] * qk_scale)
                dk[hi * hd1 : hi * hd1 + hd0, :d0].copy_(sk[hi * hd0 : (hi + 1) * hd0] * qk_scale)
                dv[hi * hd1 : hi * hd1 + hd0, :d0].copy_(sv[hi * hd0 : (hi + 1) * hd0])
                dq[hi * hd1 + hd0 : (hi + 1) * hd1, d0:].normal_(0.0, 0.01)
                dk[hi * hd1 + hd0 : (hi + 1) * hd1, d0:].normal_(0.0, 0.01)
                dv[hi * hd1 + hd0 : (hi + 1) * hd1, d0:].normal_(0.0, 0.01)

            dst_attn.out.weight.zero_()
            dst_attn.out.bias.zero_()
            for hi in range(h):
                dst_attn.out.weight[:d0, hi * hd1 : hi * hd1 + hd0].copy_(
                    src_attn.out.weight[:, hi * hd0 : (hi + 1) * hd0]
                )
                dst_attn.out.weight[d0:, hi * hd1 + hd0 : (hi + 1) * hd1].normal_(0.0, 0.01)
            dst_attn.out.bias[:d0].copy_(src_attn.out.bias)

        for i in range(8):
            sb, db = pro_net.blocks[i], ult_net.blocks[i]
            st, dt = sb["time"], db["time"]
            dt.norm.gain.zero_()
            dt.norm.gain[:d0].copy_(st.norm.gain * rms_scale)
            dt.norm.gain[d0:].fill_(1.0)

            dt.conv.weight.zero_()
            dt.conv.bias.zero_()
            dt.conv.weight[:d0].copy_(st.conv.weight)
            dt.conv.bias[:d0].copy_(st.conv.bias)
            dt.conv.weight[d0:].normal_(0.0, 0.02)

            dt.l1.weight.zero_()
            dt.l1.bias.zero_()
            dt.l1.weight[:ff0, :d0].copy_(st.l1.weight)
            dt.l1.bias[:ff0].copy_(st.l1.bias)
            dt.l1.weight[ff0:, :d0].normal_(0.0, 0.015)
            dt.l1.weight[ff0:, d0:].normal_(0.0, 0.015)

            dt.l2.weight.zero_()
            dt.l2.bias.zero_()
            dt.l2.weight[:d0, :ff0].copy_(st.l2.weight)
            dt.l2.bias[:d0].copy_(st.l2.bias)

            copy_mha(sb["t_attn"], db["t_attn"])
            copy_mha(sb["band"], db["band"])

            sff, dff = sb["ff"], db["ff"]
            dff.norm.gain.zero_()
            dff.norm.gain[:d0].copy_(sff.norm.gain * rms_scale)
            dff.norm.gain[d0:].fill_(1.0)

            dff.l1.weight.zero_()
            dff.l1.bias.zero_()
            dff.l1.weight[:ff0, :d0].copy_(sff.l1.weight)
            dff.l1.bias[:ff0].copy_(sff.l1.bias)
            dff.l1.weight[ff0:, :d0].normal_(0.0, 0.015)
            dff.l1.weight[ff0:, d0:].normal_(0.0, 0.015)

            dff.l2.weight.zero_()
            dff.l2.bias.zero_()
            dff.l2.weight[:d0, :ff0].copy_(sff.l2.weight)
            dff.l2.bias[:d0].copy_(sff.l2.bias)

        for i in (8, 9):
            db = ult_net.blocks[i]
            db["time"].l2.weight.zero_()
            db["time"].l2.bias.zero_()
            db["t_attn"].out.weight.zero_()
            db["t_attn"].out.bias.zero_()
            db["band"].out.weight.zero_()
            db["band"].out.bias.zero_()
            db["ff"].l2.weight.zero_()
            db["ff"].l2.bias.zero_()

        ult_net.norm.gain.zero_()
        ult_net.norm.gain[:d0].copy_(pro_net.norm.gain * rms_scale)
        ult_net.norm.gain[d0:].fill_(1.0)

        ult_net.mask_l1_w.zero_()
        ult_net.mask_l1_b.zero_()
        ult_net.mask_l1_w[:, :, :m0, :d0].copy_(pro_net.mask_l1_w)
        ult_net.mask_l1_b[:, :, :m0].copy_(pro_net.mask_l1_b)
        ult_net.mask_l1_w[:, :, m0:, :d0].normal_(0.0, 0.015)
        ult_net.mask_l1_w[:, :, m0:, d0:].normal_(0.0, 0.015)

        for gi in range(len(pro_net.groups)):
            ult_net.mask_l2_w[gi].zero_()
            ult_net.mask_l2_w[gi][:, :, :, :m0].copy_(pro_net.mask_l2_w[gi])
            ult_net.mask_l2_b[gi].copy_(pro_net.mask_l2_b[gi])

        x = torch.randn(1, 2, 44100 * 2) * 0.25
        _, w_pro, _ = pro_net(x)
        _, w_ult, _ = ult_net(x)
        max_diff = float((w_pro - w_ult).abs().max().item())
        print(
            f"[Net2Net] Verified Pro ({sum(p.numel() for p in pro_net.parameters()):,}) -> "
            f"Ultra ({sum(p.numel() for p in ult_net.parameters()):,}) | max_waveform_diff = {max_diff:.2e}",
            flush=True,
        )
        assert max_diff < 1e-4, f"Expected <1e-4 step-0 diff, got {max_diff}"

    out_dir = Path(out_st_path).parent
    p32, p16 = ult_net.export_safetensors(str(out_dir), "init-ultra")
    Path(p32).rename(out_st_path)
    print(f"[Net2Net] Saved 7.84 dB Step-0 Cochlea-Ultra init -> {out_st_path} (and {p16})", flush=True)


if __name__ == "__main__":
    in_p = sys.argv[1] if len(sys.argv) > 1 else "/content/init_cochlea_pro.safetensors"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "/content/init_cochlea_ultra.safetensors"
    expand_pro_to_ultra(in_p, out_p)
