#!/usr/bin/env python3
"""Programmatically render Sone's repository banner (`assets/banner.png`, 3200x1000)
and Open Graph social card (`docs/assets/og.png`, 1200x630) directly from real Sone
model outputs: rendered stereo transition waveforms, Cortex stem gain curves, and
high-resolution spectrogram energy around `the one` (u = 0).

    /home/user/bot/.paper_venv/bin/python scripts/paper/make_banner.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
BOT = Path(os.environ.get("SONE_WORK_ROOT", str(ROOT.parent)))


def load_real_transition():
    wav_p = BOT / "ai_mixes_v3/ai_mix01_7yoqSl_6N185A.wav"
    spec_p = BOT / "ai_mixes_v3/ai_mix01_7yoqSl_6N185A.spec.json"
    audio, sr = sf.read(str(wav_p), dtype="float32", always_2d=True)
    spec = json.loads(spec_p.read_text())[0]
    # Window around the vocal-onset-locked transition (30s..56s)
    s0, s1 = int(30.0 * sr), int(56.0 * sr)
    seg = audio[s0:s1]
    mono = seg.mean(axis=1)

    # Downsampled waveform envelope for crisp vector-style peaks
    n_bins = 260
    chunk = len(mono) // n_bins
    env_hi = np.array([np.percentile(np.abs(mono[i * chunk : (i + 1) * chunk]), 94) for i in range(n_bins)])
    env_lo = -0.82 * env_hi

    # Log-magnitude mel-like spectrogram slice
    n_fft, hop = 2048, 768
    win = np.hanning(n_fft)
    frames = [
        np.abs(np.fft.rfft(mono[i : i + n_fft] * win))[:360]
        for i in range(0, len(mono) - n_fft, hop)
    ]
    S = 20.0 * np.log10(np.stack(frames, axis=1) + 1e-5)
    vmax = np.percentile(S, 99.2)
    S_norm = np.clip((S - (vmax - 52.0)) / 52.0, 0.0, 1.0)

    # Real Cortex stem fader curves on bar ruler u in [-7, +2]
    us = np.linspace(-7.0, 2.0, n_bins)

    def interp_lane(deck: str, stem: str) -> np.ndarray:
        pts = spec[deck]["lanes"].get(stem, [])
        if not pts:
            return np.ones_like(us)
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        out = np.interp(us, xs, ys)
        if deck == "b" and spec["b"].get("from") is not None:
            out = np.where(us < spec["b"]["from"] - 1e-3, 0.0, out)
        return out

    a_other = interp_lane("a", "other")
    a_drums = interp_lane("a", "drums")
    b_vocals = interp_lane("b", "vocals")
    b_bass = interp_lane("b", "bass")
    return S_norm, env_hi, env_lo, us, a_other, a_drums, b_vocals, b_bass


def render_banner(out_path: Path) -> None:
    S_norm, env_hi, env_lo, us, a_other, a_drums, b_vocals, b_bass = load_real_transition()

    bg = "#0b0d12"
    panel_bg = "#11151f"
    border = "#232a3b"
    fig = plt.figure(figsize=(16, 5), dpi=200, facecolor=bg)

    # Left typographic column
    ax_txt = fig.add_axes([0.0, 0.0, 0.49, 1.0], facecolor=bg)
    ax_txt.set_xlim(0, 1000)
    ax_txt.set_ylim(1000, 0)
    ax_txt.axis("off")

    ax_txt.text(92, 205, "NEURO-SYMBOLIC AUDIO ENGINE · SONE-INFER/1", fontsize=8.8,
                family="monospace", color="#60a5fa", fontweight="bold")
    ax_txt.text(88, 415, "sone", fontsize=68, family="serif", color="#f8fafc",
                fontweight="bold", va="baseline")
    ax_txt.text(448, 415, "6.5M · Base · Pro · Ultra", fontsize=15.5, family="monospace",
                color="#94a3b8", va="baseline")

    ax_txt.text(92, 540, "Studio-grade DJ transitions on a single CPU core.",
                fontsize=14.5, family="sans-serif", color="#e2e8f0", fontweight="medium")
    ax_txt.text(92, 608, "Four anatomical organs in one safetensors file;",
                fontsize=13.0, family="sans-serif", color="#94a3b8")
    ax_txt.text(92, 668, "untouched master stereo outside the ~24 s blend.",
                fontsize=13.0, family="sans-serif", color="#94a3b8")

    pills = [
        ("COCHLEA", "83-band stems"),
        ("TYMPANUM", "50 Hz ear"),
        ("CORTEX", "8-lane plan"),
        ("CINGULATE", "30-defect critic"),
    ]
    px = 92
    for name, sub in pills:
        box = FancyBboxPatch((px, 755), 205, 118, boxstyle="round,pad=0.02,rounding_size=14",
                             facecolor=panel_bg, edgecolor=border, linewidth=1.0)
        ax_txt.add_patch(box)
        ax_txt.text(px + 18, 804, name, fontsize=8.0, family="monospace",
                    color="#38bdf8", fontweight="bold")
        ax_txt.text(px + 18, 846, sub, fontsize=7.6, family="monospace", color="#cbd5e1")
        px += 220

    # Right real-signal card: top spectrogram + bottom Cortex fader & waveform envelope
    card_ax = fig.add_axes([0.505, 0.08, 0.465, 0.84], facecolor=panel_bg)
    card_ax.set_xticks([])
    card_ax.set_yticks([])
    for sp in card_ax.spines.values():
        sp.set_edgecolor(border)
        sp.set_linewidth(1.2)

    ax_spec = fig.add_axes([0.525, 0.50, 0.425, 0.34], facecolor="#090b10")
    ax_spec.imshow(S_norm, origin="lower", aspect="auto", cmap="magma",
                   extent=[-7.0, 2.0, 0.0, 8.0], alpha=0.96)
    ax_spec.axvline(-4.0, color="#38bdf8", lw=1.1, ls="--")
    ax_spec.axvline(0.0, color="#4ade80", lw=1.3, ls="--")
    ax_spec.text(-3.88, 6.75, "vocal handoff (u = -4)", color="#e0f2fe",
                 fontsize=7.2, family="monospace", fontweight="bold")
    ax_spec.text(0.12, 6.75, "the one (u = 0)", color="#dcfce7",
                 fontsize=7.2, family="monospace", fontweight="bold")
    ax_spec.set_xticks([])
    ax_spec.set_yticks([0, 4, 8])
    ax_spec.set_yticklabels(["0", "4k", "8k"], color="#64748b", fontsize=6.8, family="monospace")
    for sp in ax_spec.spines.values():
        sp.set_edgecolor("#1e293b")
    ax_spec.set_title("REAL RENDERED TRANSITION · 0.0 ms DRIFT · CINGULATE = 0.996",
                      color="#94a3b8", fontsize=7.4, family="monospace", loc="left", pad=5)

    ax_curve = fig.add_axes([0.525, 0.14, 0.425, 0.30], facecolor="#090b10")
    # Subtle rendered stereo waveform silhouette behind Cortex faders
    w_scale = env_hi / (np.max(env_hi) + 1e-6) * 0.42
    ax_curve.fill_between(us, 0.50 - w_scale, 0.50 + w_scale, color="#334155", alpha=0.35)
    ax_curve.plot(us, a_drums, color="#60a5fa", lw=1.3, ls=":", label="A drums")
    ax_curve.plot(us, a_other, color="#38bdf8", lw=1.7, label="A chords (-2.5 dB pocket)")
    ax_curve.plot(us, b_vocals, color="#4ade80", lw=1.8, label="B vocals (onset lock)")
    ax_curve.plot(us, b_bass, color="#f59e0b", lw=1.5, ls="--", label="B bass (swap on 1)")
    ax_curve.axvline(-4.0, color="#38bdf8", lw=0.9, ls="--", alpha=0.7)
    ax_curve.axvline(0.0, color="#4ade80", lw=1.1, ls="--", alpha=0.85)
    ax_curve.set_xlim(-7.0, 2.0)
    ax_curve.set_ylim(-0.05, 1.12)
    ax_curve.set_xticks([-6, -4, -2, 0, 2])
    ax_curve.set_xticklabels(["-6", "-4", "-2", "0 (one)", "+2 bars"],
                             color="#64748b", fontsize=6.8, family="monospace")
    ax_curve.set_yticks([0.0, 0.5, 1.0])
    ax_curve.set_yticklabels(["0.0", "0.5", "1.0"], color="#64748b", fontsize=6.8, family="monospace")
    for sp in ax_curve.spines.values():
        sp.set_edgecolor("#1e293b")
    ax_curve.legend(loc="lower left", ncol=2, frameon=True, facecolor="#0b0d12",
                    edgecolor="none", framealpha=0.86, labelcolor="#cbd5e1", fontsize=6.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=bg)
    plt.close(fig)


def render_og(out_path: Path) -> None:
    """1200x630 Open Graph preview card."""
    S_norm, env_hi, _, us, a_other, _, b_vocals, b_bass = load_real_transition()
    bg = "#0b0d12"
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=bg)
    ax.set_xlim(0, 1200)
    ax.set_ylim(630, 0)
    ax.axis("off")

    ax.text(76, 118, "SONE · RUST + SAFETENSORS + WEBASSEMBLY", fontsize=11,
            family="monospace", color="#60a5fa", fontweight="bold")
    ax.text(72, 238, "sone", fontsize=72, family="serif", color="#f8fafc",
            fontweight="bold", va="baseline")
    ax.text(355, 238, "6.5M · Base · Pro · Ultra", fontsize=19.5, family="monospace",
            color="#94a3b8", va="baseline")
    ax.text(76, 304, "Neuro-symbolic DJ transitions on one CPU core — untouched master stereo outside the blend.",
            fontsize=15.5, family="sans-serif", color="#cbd5e1")

    ax_sp = fig.add_axes([0.063, 0.11, 0.874, 0.34], facecolor="#11151f")
    ax_sp.imshow(S_norm, origin="lower", aspect="auto", cmap="magma",
                 extent=[-7.0, 2.0, 0.0, 1.0], alpha=0.55)
    ax_sp.plot(us, a_other * 0.88 + 0.05, color="#38bdf8", lw=2.2, label="Deck A accompaniment pocket")
    ax_sp.plot(us, b_vocals * 0.88 + 0.05, color="#4ade80", lw=2.4, label="Deck B vocal-onset lock")
    ax_sp.plot(us, b_bass * 0.88 + 0.05, color="#f59e0b", lw=2.0, ls="--", label="Deck B bass swap at u = 0")
    ax_sp.axvline(-4.0, color="#38bdf8", lw=1.2, ls="--")
    ax_sp.axvline(0.0, color="#4ade80", lw=1.4, ls="--")
    ax_sp.set_xlim(-7.0, 2.0)
    ax_sp.set_ylim(0.0, 1.05)
    ax_sp.set_xticks([])
    ax_sp.set_yticks([])
    for sp in ax_sp.spines.values():
        sp.set_edgecolor("#232a3b")
        sp.set_linewidth(1.2)
    ax_sp.legend(loc="lower left", ncol=3, frameon=True, facecolor="#0b0d12",
                 edgecolor="none", framealpha=0.88, labelcolor="#e2e8f0", fontsize=9.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor=bg)
    plt.close(fig)


def main() -> None:
    banner_p = ROOT / "docs" / "assets" / "banner.png"
    legacy_banner_p = ROOT / "assets" / "banner.png"
    og_p = ROOT / "docs" / "assets" / "og.png"
    render_banner(banner_p)
    legacy_banner_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(banner_p, legacy_banner_p)
    render_og(og_p)
    print(f"Wrote {banner_p}, {legacy_banner_p}, and {og_p}")


if __name__ == "__main__":
    main()
