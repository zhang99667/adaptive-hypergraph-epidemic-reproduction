#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="依次运行全部复现脚本。")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-mc", action="store_true", help="只对 Fig. 2 生效：跳过 MC 点，仅画 MMCA 理论曲线")
    return p.parse_args()


def run(script: str, *extra: str) -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts" / script), *extra]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    base = ["--quick"] if args.quick else []
    fig2 = base + (["--skip-mc"] if args.skip_mc else [])
    run("smoke_test.py")
    run("reproduce_fig2.py", *fig2)
    run("reproduce_fig3.py", *base)
    run("reproduce_fig4_fig5.py", *base)
    run("reproduce_fig6.py", *base)


if __name__ == "__main__":
    main()
