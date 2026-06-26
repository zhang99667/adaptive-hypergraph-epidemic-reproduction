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
    replicate_mc_stationary,
    save_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce paper Figs. 4-5 immunization scans.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--si-sort", choices=["descending", "ascending"], default="descending")
    p.add_argument("--outdir", default="outputs")
    return p.parse_args()


def strategy_spec(name: str) -> tuple[str, float]:
    if name == "random":
        return "random", 0.0
    if name == "ti":
        return "ti", 0.0
    if name == "si1":
        return "si", 0.2
    if name == "si2":
        return "si", 0.4
    if name == "si3":
        return "si", 0.6
    raise ValueError(name)


def run_point(
    *,
    n: int,
    mean_k: float,
    beta: float,
    gamma: float,
    w: float,
    strategy: str,
    reps: int,
    steps: int,
    burnin: int,
    rho0: float,
    seed: int,
    si_sort: str,
) -> tuple[float, float]:
    intervention, theta_min = strategy_spec(strategy)
    params = ModelParams(beta=beta, mu=0.1, eta=0.6, gamma=gamma)
    return replicate_mc_stationary(
        n=n,
        mean_k=mean_k,
        d=3,
        params=params,
        beta=beta,
        reps=reps,
        steps=steps,
        initial_infected=rho0,
        seed=seed,
        intervention=intervention,
        intervention_w=w,
        theta_min=theta_min,
        si_sort=si_sort,
        intervention_start=burnin,
        intervention_once=True,
    )


def main() -> None:
    args = parse_args()
    out = Path(args.outdir)
    data_dir = ensure_dir(out / "data")
    fig_dir = ensure_dir(out / "figures")

    n = 180 if args.quick else 1500
    reps = args.reps if args.reps is not None else (2 if args.quick else 200)
    steps = 180 if args.quick else 1100
    burnin = 40 if args.quick else 300
    if args.quick and args.reps is None:
        reps = 1
    ws = np.linspace(0, 1, 5 if args.quick else 31)
    strategies = ["ti", "random", "si1", "si2", "si3"]
    settings = [(6, 0.02), (9, 0.02), (6, 0.03), (9, 0.03)]
    rows: list[list[object]] = []
    fig4 = {}

    for mean_k, gamma in settings:
        for strategy in strategies:
            vals, errs = [], []
            for w in ws:
                mean, err = run_point(
                    n=n,
                    mean_k=mean_k,
                    beta=0.35,
                    gamma=gamma,
                    w=float(w),
                    strategy=strategy,
                    reps=reps,
                    steps=steps,
                    burnin=burnin,
                    rho0=0.3,
                    seed=args.seed + int(mean_k * 1000 + gamma * 100000 + w * 1000),
                    si_sort=args.si_sort,
                )
                vals.append(mean)
                errs.append(err)
                rows.append(["fig4", mean_k, gamma, strategy, f"{w:.8g}", 0.35, 0.3, f"{mean:.8g}", f"{err:.8g}"])
            fig4[(mean_k, gamma, strategy)] = (np.asarray(vals), np.asarray(errs))

    betas = np.linspace(0.25, 0.40, 4 if args.quick else 25)
    w_grid = np.linspace(0, 1, 6 if args.quick else 51)
    hit = {}
    hit_eps = 0.02
    for rho0 in [0.3, 0.5]:
        for strategy in strategies:
            wc_values = []
            for beta in betas:
                wc = 1.0
                for w in w_grid:
                    mean, err = run_point(
                        n=n,
                        mean_k=6,
                        beta=float(beta),
                        gamma=0.02,
                        w=float(w),
                        strategy=strategy,
                        reps=reps,
                        steps=steps,
                        burnin=burnin,
                        rho0=rho0,
                        seed=args.seed + int(beta * 10000 + w * 1000 + rho0 * 100),
                        si_sort=args.si_sort,
                    )
                    rows.append(["fig5_scan", 6, 0.02, strategy, f"{w:.8g}", f"{beta:.8g}", rho0, f"{mean:.8g}", f"{err:.8g}"])
                    if mean <= hit_eps:
                        wc = float(w)
                        break
                wc_values.append(wc)
                rows.append(["fig5_wc", 6, 0.02, strategy, f"{wc:.8g}", f"{beta:.8g}", rho0, "", ""])
            hit[(rho0, strategy)] = np.asarray(wc_values)

    save_csv(
        data_dir / "fig4_fig5_immunization.csv",
        ["kind", "mean_k", "gamma", "strategy", "w", "beta", "rho0", "rho", "stderr"],
        rows,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"data saved, plotting skipped: {exc}")
        return

    labels = {"ti": "TI", "random": "Random", "si1": "SI1", "si2": "SI2", "si3": "SI3"}
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), constrained_layout=True)
    for ax, (mean_k, gamma) in zip(axes.ravel(), settings):
        for strategy in strategies:
            vals, errs = fig4[(mean_k, gamma, strategy)]
            ax.errorbar(ws, vals, yerr=errs, marker="o", ms=3, capsize=2, label=labels[strategy])
        ax.set_title(fr"$\langle k\rangle={mean_k}, \gamma={gamma}$")
        ax.set_xlabel("w")
        ax.set_ylabel(r"$\rho^I$")
        ax.set_ylim(-0.03, 0.75)
        ax.legend(fontsize=7)
    fig.savefig(fig_dir / "fig4_reproduction.png", dpi=220)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
    for ax, rho0 in zip(axes, [0.3, 0.5]):
        for strategy in strategies:
            ax.plot(betas, hit[(rho0, strategy)], marker="o", ms=3, label=labels[strategy])
        ax.set_title(f"I(0)={rho0}")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$w_c$")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
    fig.savefig(fig_dir / "fig5_reproduction.png", dpi=220)
    print(f"saved {fig_dir / 'fig4_reproduction.png'} and {fig_dir / 'fig5_reproduction.png'}")


if __name__ == "__main__":
    main()
