#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_hypergraph import ModelParams, homogeneous_mmca, mc_sis, random_uniform_hypergraph


def main() -> None:
    hg = random_uniform_hypergraph(n=120, m=240, d=3, seed=7)
    params = ModelParams(beta=0.25, mu=0.1, eta=0.6, gamma=0.02)
    rho, theta = homogeneous_mmca(beta=0.25, mean_k=6, d=3, steps=80)
    assert 0 <= rho[-1] <= 1
    assert 0 <= theta[-1] <= 1

    result = mc_sis(hg, params, steps=60, initial_infected=0.3, seed=11)
    assert len(result.rho) == 61
    assert 0 <= result.stationary_rho <= 1

    rewired = mc_sis(
        random_uniform_hypergraph(n=120, m=240, d=3, seed=8),
        params,
        steps=60,
        initial_infected=0.3,
        seed=12,
        intervention="ti",
        intervention_w=0.2,
        intervention_once=True,
        intervention_start=20,
        rewiring="preferential",
    )
    assert 0 <= rewired.stationary_rho <= 1
    print("smoke ok")


if __name__ == "__main__":
    main()
