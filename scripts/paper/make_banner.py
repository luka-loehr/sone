#!/usr/bin/env python3
"""Render Sone's repository banner (`assets/banner.png`, 3200x1000) and Open Graph
card (`docs/assets/og.png`, 1200x630) from a real rendered Sone transition.

    /home/user/bot/.paper_venv/bin/python scripts/paper/make_banner.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
BOT = Path(os.environ.get("SONE_WORK_ROOT", str(ROOT.parent)))

BG = "#000000"

# Scientific perceptual colormap anchored at pure #000000 so low-energy regions
# merge invisibly into the black canvas.
SPEC_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "sone_spec",
    [
        (0.00, "#000000"),
        (0.14, "#080714"),
        (0.30, "#1c133d"),
        (0.48, "#561d5e"),
        (0.66, "#a43259"),
        (0.82, "#df6f46"),
        (0.93, "#f4b96c"),
        (1.00, "#fbf5e6"),
    ],
)


def load_real_spectrogram(
    u_min: float = -7.0,
    u_max: float = 3.0,
    n_freq_bins: int = 520,
    f_min: float = 36.0,
    f_max: float = 14500.0,
) -> tuple[np.ndarray, float]:
    """Compute a high-resolution log-frequency STFT directly from a rendered Sone
    transition (`ai_mix01_7yoqSl_6N185A.wav`) and return the edge-feathered RGBA
    image together with the exact normalized horizontal position of `u = 0`."""
    wav_p = BOT / "ai_mixes_v3/ai_mix01_7yoqSl_6N185A.wav"
    spec_p = BOT / "ai_mixes_v3/ai_mix01_7yoqSl_6N185A.spec.json"

    audio, sr = sf.read(str(wav_p), dtype="float32", always_2d=True)
    spec = json.loads(spec_p.read_text())[0]
    clip_u0, clip_u1 = spec["clip"]  # [-18.0, 14.0] bars
    dur_s = len(audio) / float(sr)
    sec_per_bar = dur_s / (clip_u1 - clip_u0)

    t0 = (u_min - clip_u0) * sec_per_bar
    t1 = (u_max - clip_u0) * sec_per_bar
    s0 = max(0, int(round(t0 * sr)))
    s1 = min(len(audio), int(round(t1 * sr)))
    mono = audio[s0:s1].mean(axis=1)

    n_fft = 4096
    hop = 256
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(mono) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + (np.arange(n_frames) * hop)[:, None]
    frames = mono[idx] * win[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1)).T  # (n_fft//2 + 1, n_frames)

    # Interpolate linearly-spaced FFT bins onto a musical log-frequency axis.
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    log_freqs = np.geomspace(f_min, f_max, n_freq_bins)
    bin_pos = np.interp(log_freqs, fft_freqs, np.arange(len(fft_freqs)))
    b0 = np.floor(bin_pos).astype(int)
    b1 = np.minimum(b0 + 1, mag.shape[0] - 1)
    frac = (bin_pos - b0)[:, None]
    S_lin = (1.0 - frac) * mag[b0] + frac * mag[b1]

    # Mild high-frequency presence tilt so vocal formants and harmonics stay legible.
    tilt = np.linspace(0.0, 7.2, n_freq_bins)[:, None]
    S_db = 20.0 * np.log10(S_lin + 1e-6) + tilt

    vmax = np.percentile(S_db, 99.6)
    dyn_db = 50.0
    S_norm = np.clip((S_db - (vmax - dyn_db)) / dyn_db, 0.0, 1.0)
    S_norm = np.power(S_norm, 1.08)

    # Map to RGB and apply soft cosine edge feathering into pure #000000.
    rgb = SPEC_CMAP(S_norm)[..., :3]
    H, W, _ = rgb.shape
    fade_x = np.ones(W, dtype=np.float32)
    fx_n = max(1, int(0.09 * W))
    ramp_x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fx_n)))
    fade_x[:fx_n] = ramp_x
    fade_x[-fx_n:] = ramp_x[::-1]

    fade_y = np.ones(H, dtype=np.float32)
    fy_n = max(1, int(0.08 * H))
    ramp_y = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fy_n)))
    fade_y[:fy_n] = ramp_y
    fade_y[-fy_n:] = ramp_y[::-1]

    mask = fade_y[:, None] * fade_x[None, :]
    rgb = rgb * mask[..., None]

    u0_frac = (0.0 - u_min) / (u_max - u_min)
    return rgb, u0_frac


def draw_handoff_marker(ax: plt.Axes, x0: float, y_lo: float, y_hi: float) -> None:
    """Draw a single razor-thin vertical marker at u = 0, softly feathered at ends."""
    ys = np.linspace(y_lo, y_hi, 240)
    t = np.linspace(0.0, 1.0, len(ys))
    alpha = np.sin(np.pi * t) ** 0.65
    for i in range(len(ys) - 1):
        ax.plot(
            [x0, x0],
            [ys[i], ys[i + 1]],
            color="#ffffff",
            alpha=0.42 * float(alpha[i]),
            lw=0.8,
            solid_capstyle="butt",
        )


def render_banner(out_path: Path) -> None:
    """3200x1000 minimalist dark repository banner."""
    rgb, u0_frac = load_real_spectrogram(u_min=-7.0, u_max=3.0)

    fig = plt.figure(figsize=(16, 5), dpi=200, facecolor=BG)

    # Full-canvas typographic layer (matches Stylus 3200x1000 coordinate system)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=BG)
    ax.set_xlim(0, 3200)
    ax.set_ylim(1000, 0)
    ax.axis("off")

    ax.text(
        155,
        448,
        "Sone",
        fontsize=94,
        family="serif",
        fontweight="normal",
        color="#ffffff",
        va="baseline",
    )
    ax.text(
        160,
        583,
        "Neuro-symbolic DJ transitions and",
        fontsize=20.5,
        family="serif",
        fontweight="normal",
        color="#d4d4d8",
        va="baseline",
    )
    ax.text(
        160,
        650,
        "stem separation on one CPU core.",
        fontsize=20.5,
        family="serif",
        fontweight="normal",
        color="#d4d4d8",
        va="baseline",
    )

    # Right-hand real transition spectrogram, seamlessly blended onto pure black
    spec_left = 0.445
    spec_bottom = 0.135
    spec_w = 0.515
    spec_h = 0.730
    ax_sp = fig.add_axes([spec_left, spec_bottom, spec_w, spec_h], facecolor=BG)
    ax_sp.imshow(
        rgb,
        origin="lower",
        aspect="auto",
        extent=[0.0, 1.0, 0.0, 1.0],
        interpolation="bilinear",
    )
    draw_handoff_marker(ax_sp, u0_frac, 0.02, 0.98)
    ax_sp.set_xlim(0.0, 1.0)
    ax_sp.set_ylim(0.0, 1.0)
    ax_sp.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)


def render_og(out_path: Path) -> None:
    """1200x630 minimalist dark Open Graph social card."""
    rgb, u0_frac = load_real_spectrogram(u_min=-7.0, u_max=3.0)

    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=BG)
    ax.set_xlim(0, 1200)
    ax.set_ylim(630, 0)
    ax.axis("off")

    ax.text(
        78,
        156,
        "Sone",
        fontsize=68,
        family="serif",
        fontweight="normal",
        color="#ffffff",
        va="baseline",
    )
    ax.text(
        82,
        214,
        "Neuro-symbolic DJ transitions and stem separation on one CPU core.",
        fontsize=16.5,
        family="serif",
        fontweight="normal",
        color="#d4d4d8",
        va="baseline",
    )

    ax_sp = fig.add_axes([0.055, 0.095, 0.890, 0.505], facecolor=BG)
    ax_sp.imshow(
        rgb,
        origin="lower",
        aspect="auto",
        extent=[0.0, 1.0, 0.0, 1.0],
        interpolation="bilinear",
    )
    draw_handoff_marker(ax_sp, u0_frac, 0.02, 0.98)
    ax_sp.set_xlim(0.0, 1.0)
    ax_sp.set_ylim(0.0, 1.0)
    ax_sp.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor=BG)
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
