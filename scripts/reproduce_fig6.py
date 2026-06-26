#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_hypergraph.model import (
    ModelParams,
    ensure_dir,
    mc_sis,
    random_uniform_hypergraph,
    save_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="复现论文 Fig. 6 风格实验：随机重连与度优先重连对比。")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--outdir", default="outputs")
    return p.parse_args()


def degree_hist(deg: np.ndarray, max_k: int = 25) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(max_k + 2)
    counts, edges = np.histogram(np.clip(deg, 0, max_k), bins=bins)
    return edges[:-1], counts / counts.sum()


def main() -> None:
    args = parse_args()
    out = Path(args.outdir)
    data_dir = ensure_dir(out / "data")
    fig_dir = ensure_dir(out / "figures")

    n = 300 if args.quick else 1500
    mean_k = 6
    d = 3
    m = int(round(n * mean_k / d))
    steps = 280 if args.quick else 700
    intervention_start = 100 if args.quick else 400
    params = ModelParams(beta=0.25, mu=0.1, eta=0.6, gamma=0.02)
    rows: list[list[object]] = []
    data = {}

    for mode in ["random", "preferential"]:
        hg = random_uniform_hypergraph(n=n, m=m, d=d, seed=args.seed)
        deg_before = hg.hyperdegrees().copy()
        result = mc_sis(
            hg,
            params,
            steps=steps,
            initial_infected=0.3,
            seed=args.seed + (1 if mode == "random" else 2),
            intervention="ti",
            intervention_w=0.8,
            intervention_start=intervention_start,
            intervention_once=True,
            rewiring=mode,
            rewiring_theta0=0.1,
            rewiring_alpha=1.0,
        )
        deg_after = result.hypergraph.hyperdegrees()
        data[mode] = (result.rho, deg_before, deg_after)
        for t, rho in enumerate(result.rho):
            rows.append(["rho", mode, t, f"{rho:.8g}", "", ""])
        for stage, deg in [("before", deg_before), ("after", deg_after)]:
            xs, ps = degree_hist(deg, max_k=25 if not args.quick else 18)
            for k, p in zip(xs, ps):
                rows.append(["degree", mode, -1, "", stage, f"{k}:{p:.8g}"])

    save_csv(data_dir / "fig6_rewiring.csv", ["kind", "mode", "t", "rho", "stage", "value"], rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"数据已保存，但绘图被跳过：{exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), constrained_layout=True)
    for row, mode in enumerate(["random", "preferential"]):
        rho, deg_before, deg_after = data[mode]
        ax = axes[row, 0]
        ax.plot(rho, color="#1d4ed8", label="infection density")
        ax.axvline(intervention_start, color="k", ls="--", lw=1, label="intervention")
        ax.set_title(f"{mode} rewiring")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\rho^I(t)$")
        ax.legend(fontsize=7)

        ax = axes[row, 1]
        for deg, stage, color in [(deg_before, "before", "#1d4ed8"), (deg_after, "after", "#dc2626")]:
            xs, ps = degree_hist(deg, max_k=25 if not args.quick else 18)
            ax.plot(xs, ps, marker="o", ms=3, label=stage, color=color)
        ax.set_title(f"{mode} degree distribution")
        ax.set_xlabel("k")
        ax.set_ylabel("P(k)")
        ax.legend(fontsize=7)

    fig.savefig(fig_dir / "fig6_reproduction.png", dpi=220)
    print(f"已保存 {fig_dir / 'fig6_reproduction.png'}")


if __name__ == "__main__":
    main()
