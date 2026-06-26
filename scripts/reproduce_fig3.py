#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_hypergraph.model import ensure_dir, homogeneous_mmca, save_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="复现论文 Fig. 3 风格实验：初始条件和自适应参数扫描。")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--outdir", default="outputs")
    return p.parse_args()


def stationary(beta: float, eta: float, gamma: float, rho0: float, steps: int) -> float:
    # Fig. 3 的热图只需要稳态感染密度，因此调用同质 MMCA 后取末尾均值。
    rho, _ = homogeneous_mmca(beta=beta, mean_k=6, d=3, mu=0.1, eta=eta, gamma=gamma, rho0=rho0, steps=steps)
    return float(np.mean(rho[int(0.8 * len(rho)) :]))


def main() -> None:
    args = parse_args()
    out = Path(args.outdir)
    data_dir = ensure_dir(out / "data")
    fig_dir = ensure_dir(out / "figures")
    steps = 400 if args.quick else 1200
    grid_n = 18 if args.quick else 60

    rows: list[list[object]] = []
    ts = {}
    for rho0, eta in [(0.2, 0.6), (0.2, 0.8), (0.8, 0.6), (0.8, 0.8)]:
        # 上排左图：固定 gamma，比较不同初值和 eta 的时间演化。
        rho, _ = homogeneous_mmca(beta=0.2, mean_k=6, d=3, mu=0.1, eta=eta, gamma=0.02, rho0=rho0, steps=steps)
        ts[("eta", rho0, eta)] = rho
        for t, val in enumerate(rho):
            rows.append(["eta_ts", rho0, eta, 0.02, 0.2, t, f"{val:.8g}"])
    for rho0, gamma in [(0.2, 0.02), (0.2, 0.03), (0.8, 0.02), (0.8, 0.03)]:
        # 下排左图：固定 eta，比较不同初值和 gamma 的时间演化。
        rho, _ = homogeneous_mmca(beta=0.2, mean_k=6, d=3, mu=0.1, eta=0.6, gamma=gamma, rho0=rho0, steps=steps)
        ts[("gamma", rho0, gamma)] = rho
        for t, val in enumerate(rho):
            rows.append(["gamma_ts", rho0, 0.6, gamma, 0.2, t, f"{val:.8g}"])

    betas = np.linspace(0.18, 0.22, grid_n)
    etas = np.linspace(0.4, 0.9, grid_n)
    gammas = np.linspace(0.01, 0.05, grid_n)
    heat_eta = {}
    heat_gamma = {}
    for rho0 in [0.2, 0.8]:
        # beta-eta 热图：每个格点都由 MMCA 方程重新迭代到稳态。
        h = np.zeros((etas.size, betas.size))
        for i, eta in enumerate(etas):
            for j, beta in enumerate(betas):
                h[i, j] = stationary(beta=float(beta), eta=float(eta), gamma=0.02, rho0=rho0, steps=steps)
                rows.append(["eta_heat", rho0, f"{eta:.8g}", 0.02, f"{beta:.8g}", -1, f"{h[i,j]:.8g}"])
        heat_eta[rho0] = h

        # beta-gamma 热图：观察活动恢复率 gamma 对稳态流行区的影响。
        g = np.zeros((gammas.size, betas.size))
        for i, gamma in enumerate(gammas):
            for j, beta in enumerate(betas):
                g[i, j] = stationary(beta=float(beta), eta=0.6, gamma=float(gamma), rho0=rho0, steps=steps)
                rows.append(["gamma_heat", rho0, 0.6, f"{gamma:.8g}", f"{beta:.8g}", -1, f"{g[i,j]:.8g}"])
        heat_gamma[rho0] = g

    # 一个 CSV 同时保存时间序列和热图格点；kind 字段用于区分数据来源。
    save_csv(data_dir / "fig3_adaptive_scans.csv", ["kind", "rho0", "eta", "gamma", "beta", "t", "rho"], rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"数据已保存，但绘图被跳过：{exc}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), constrained_layout=True)
    ax = axes[0, 0]
    # 第一列画时间演化；横轴用 log 尺度，对应论文中强调早期和长期暂态。
    for key, rho in ts.items():
        kind, rho0, eta = key
        if kind == "eta":
            ax.plot(rho, label=f"I0={rho0}, eta={eta}")
    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\rho^I(t)$")
    ax.set_title("(a) eta time")
    ax.legend(fontsize=7)

    for col, rho0 in enumerate([0.2, 0.8], start=1):
        # 上排右两图：不同初值下 beta-eta 的稳态感染密度。
        im = axes[0, col].imshow(
            heat_eta[rho0],
            origin="lower",
            aspect="auto",
            extent=[betas.min(), betas.max(), etas.min(), etas.max()],
            cmap="RdBu_r",
            vmin=0,
            vmax=max(0.5, heat_eta[rho0].max()),
        )
        axes[0, col].set_xlabel(r"$\beta$")
        axes[0, col].set_ylabel(r"$\eta$")
        axes[0, col].set_title(f"({chr(ord('a') + col)}) I0={rho0}")
        fig.colorbar(im, ax=axes[0, col], label=r"$\rho^I$")

    ax = axes[1, 0]
    for key, rho in ts.items():
        kind, rho0, gamma = key
        if kind == "gamma":
            ax.plot(rho, label=f"I0={rho0}, gamma={gamma}")
    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\rho^I(t)$")
    ax.set_title("(d) gamma time")
    ax.legend(fontsize=7)

    for col, rho0 in enumerate([0.2, 0.8], start=1):
        # 下排右两图：不同初值下 beta-gamma 的稳态感染密度。
        im = axes[1, col].imshow(
            heat_gamma[rho0],
            origin="lower",
            aspect="auto",
            extent=[betas.min(), betas.max(), gammas.min(), gammas.max()],
            cmap="RdBu_r",
            vmin=0,
            vmax=max(0.5, heat_gamma[rho0].max()),
        )
        axes[1, col].set_xlabel(r"$\beta$")
        axes[1, col].set_ylabel(r"$\gamma$")
        axes[1, col].set_title(f"({chr(ord('d') + col)}) I0={rho0}")
        fig.colorbar(im, ax=axes[1, col], label=r"$\rho^I$")

    fig.savefig(fig_dir / "fig3_reproduction.png", dpi=220)
    print(f"已保存 {fig_dir / 'fig3_reproduction.png'}")


if __name__ == "__main__":
    main()
