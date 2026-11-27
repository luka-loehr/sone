#!/usr/bin/env python3
"""Pack unified sone-infer/1 safetensors bundles, verify FP32-vs-FP16 parity across
all four organs (Cochlea, Tympanum, Cortex, Cingulate), export side-by-side stem
comparison clips, and render end-to-end studio transitions."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: str, env: dict[str, str] | None = None) -> str:
    print(f"+ {cmd}", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(cmd, shell=True, cwd=str(ROOT), env=merged, text=True, capture_output=True)
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        print(proc.stderr, flush=True)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")
    return proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack Sone bundles and run end-to-end Rust verification.")
    ap.add_argument("--exports", required=True, help="Directory containing organ .safetensors checkpoints")
    ap.add_argument("--tier", default="sone-pro", help="Bundle prefix (sone-6.5M, sone-v2-base, sone-pro, sone-ultra)")
    ap.add_argument("--stems", default="", help="Optional held-out multitrack directory for Cochlea + E2E mix evaluation")
    ap.add_argument("--owner", default="", help="Optional owner.jsonl path for Cortex/Cingulate alignment audit")
    ap.add_argument("--val", default="", help="Optional val.jsonl.gz path for unseen Cortex/Cingulate evaluation")
    ap.add_argument("--out", default="", help="Output directory for verification JSONs and rendered mixes")
    ap.add_argument("--mixes", type=int, default=10, help="Number of studio transitions to render")
    args = ap.parse_args()

    exports = Path(args.exports).resolve()
    out_dir = Path(args.out).resolve() if args.out else exports / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    run("cargo build --release --examples")
    run(f"./target/release/examples/pack_bundle {exports} {args.tier}")

    if args.owner and Path(args.owner).exists():
        res = run(f"./target/release/examples/cortex_cingulate_verify {exports} {args.owner}")
        (out_dir / "cortex_cingulate_owner.txt").write_text(res)

    if args.val and Path(args.val).exists():
        res = run(f"./target/release/examples/cortex_cingulate_verify {exports} {args.val}")
        (out_dir / "cortex_cingulate_val.txt").write_text(res)

    if args.stems and Path(args.stems).exists():
        cochlea_f16 = exports / "sone-cochlea-f16.safetensors"
        if cochlea_f16.exists():
            run(f"./target/release/examples/cochlea_verify {cochlea_f16} {args.stems} {out_dir / 'cochlea'} 6 24.0")
        mix_dir = out_dir / "mixes"
        shutil.rmtree(mix_dir, ignore_errors=True)
        mix_dir.mkdir(parents=True, exist_ok=True)
        env = {
            "SONE_WIN_CACHE": str(out_dir / ".win_cache"),
            "SONE_COCHLEA_PATH": str(exports / "sone-cochlea-f16.safetensors"),
            "SONE_CORTEX_PATH": str(exports / "sone-cortex-f16.safetensors"),
            "SONE_CINGULATE_PATH": str(exports / "sone-cingulate-f16.safetensors"),
        }
        run(f"./target/release/examples/studio_pro_mixes {exports} {args.stems} {mix_dir} {args.mixes}", env=env)


if __name__ == "__main__":
    main()
