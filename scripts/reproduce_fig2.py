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
    homogeneous_mmca,
    homogeneous_stationary_curve,
    mc_sis,
    random_uniform_hypergraph,
    replicate_mc_stationary,
    save_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce paper Fig. 2 synthetic MMCA vs MC.")
    p.add_argument("--quick", action="store_true", help="small fast run")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-mc", action="store_true")
    p.add_argument("--outdir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.outdir)
    data_dir = ensure_dir(out / "data")
    fig_dir = ensure_dir(out / "figures")

    n = 300 if args.quick else 1500
    reps = args.reps if args.reps is not None else (3 if args.quick else 200)
    beta_points = 9 if args.quick else 36
    mmca_steps = 350 if args.quick else 1600
    mc_steps = 260 if args.quick else 900
    d = 3
    params = ModelParams(beta=0.2, mu=0.1, eta=0.6, gamma=0.02)
    betas = np.linspace(0.0, 0.7, beta_points)
    mean_ks = [6, 9, 12]

    rows: list[list[object]] = []
    theory_curves: dict[int, np.ndarray] = {}
    mc_points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for mean_k in mean_ks:
        theory = homogeneous_stationary_curve(
            betas,
            mean_k=mean_k,
            d=d,
            mu=params.mu,
            eta=params.eta,
            gamma=params.gamma,
            rho0=0.3,
            steps=mmca_steps,
        )
        theory_curves[mean_k] = theory
        for beta, rho in zip(betas, theory):
            rows.append(["mmca", mean_k, f"{beta:.8g}", f"{rho:.8g}", 0.0])

        if not args.skip_mc:
            means, errs = [], []
            for beta in betas:
                mean, err = replicate_mc_stationary(
                    n=n,
                    mean_k=mean_k,
                    d=d,
                    params=params,
                    beta=float(beta),
                    reps=reps,
                    steps=mc_steps,
                    initial_infected=0.3,
                    seed=args.seed + mean_k * 1000 + int(beta * 10000),
                )
                means.append(mean)
                errs.append(err)
                rows.append(["mc", mean_k, f"{beta:.8g}", f"{mean:.8g}", f"{err:.8g}"])
            mc_points[mean_k] = (np.asarray(means), np.asarray(errs))

    save_csv(data_dir / "fig2_stationary.csv", ["kind", "mean_k", "beta", "rho", "stderr"], rows)

    ts_rows: list[list[object]] = []
    ts_data = {}
    for beta in [0.08, 0.10]:
        rho_m, _theta = homogeneous_mmca(
            beta=beta,
            mean_k=12,
            d=d,
            mu=params.mu,
            eta=params.eta,
            gamma=params.gamma,
            rho0=0.3,
            steps=300 if not args.quick else 160,
        )
        ts_data[(beta, "mmca")] = rho_m
        for t, rho in enumerate(rho_m):
            ts_rows.append(["mmca", beta, t, f"{rho:.8g}", f"{1-rho:.8g}"])
        if not args.skip_mc:
            traces = []
            for rep in range(reps):
                hg = random_uniform_hypergraph(n=n, m=int(round(n * 12 / d)), d=d, seed=args.seed + rep + int(beta * 1000))
                result = mc_sis(
                    hg,
                    ModelParams(beta=beta, mu=params.mu, eta=params.eta, gamma=params.gamma),
                    steps=len(rho_m) - 1,
                    initial_infected=0.3,
                    seed=args.seed + 10_000 + rep,
                    qs=False,
                )
                traces.append(result.rho)
            avg = np.mean(np.vstack(traces), axis=0)
            ts_data[(beta, "mc")] = avg
            for t, rho in enumerate(avg):
                ts_rows.append(["mc", beta, t, f"{rho:.8g}", f"{1-rho:.8g}"])
    save_csv(data_dir / "fig2_timeseries.csv", ["kind", "beta", "t", "rho_i", "rho_s"], ts_rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local env
        print(f"data saved, plotting skipped: {exc}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    ax = axes[0]
    for mean_k in mean_ks:
        ax.plot(betas, theory_curves[mean_k], label=f"MMCA <k>={mean_k}")
        if mean_k in mc_points:
            means, errs = mc_points[mean_k]
            ax.errorbar(betas, means, yerr=errs, fmt="o", ms=3, capsize=2, label=f"MC <k>={mean_k}")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\rho^I$")
    ax.set_title("(a) stationary")
    ax.legend(fontsize=7)

    for idx, beta in enumerate([0.08, 0.10], start=1):
        ax = axes[idx]
        rho_m = ts_data[(beta, "mmca")]
        t = np.arange(rho_m.size)
        ax.plot(t, 1 - rho_m, color="#0f766e", label="MMCA S")
        ax.plot(t, rho_m, color="#dc2626", label="MMCA I")
        if (beta, "mc") in ts_data:
            rho_mc = ts_data[(beta, "mc")]
            ax.scatter(t[:: max(1, len(t) // 40)], 1 - rho_mc[:: max(1, len(t) // 40)], s=10, color="#0f766e", facecolors="none", label="MC S")
            ax.scatter(t[:: max(1, len(t) // 40)], rho_mc[:: max(1, len(t) // 40)], s=10, color="#dc2626", marker="^", label="MC I")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\rho$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"({chr(ord('a') + idx)}) beta={beta}")
        ax.legend(fontsize=7)

    fig.savefig(fig_dir / "fig2_reproduction.png", dpi=220)
    print(f"saved {fig_dir / 'fig2_reproduction.png'}")


if __name__ == "__main__":
    main()
