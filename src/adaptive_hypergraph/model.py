from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Array = np.ndarray
SISort = Literal["descending", "ascending"]
Intervention = Literal["none", "random", "ti", "si"]
Rewiring = Literal["none", "random", "preferential"]


@dataclass(frozen=True)
class ModelParams:
    beta: float = 0.2
    mu: float = 0.1
    eta: float = 0.6
    gamma: float = 0.02
    theta0: float = 1.0


@dataclass
class Hypergraph:
    n: int
    edges: Array

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

    @property
    def d(self) -> int:
        return int(self.edges.shape[1])

    @property
    def mean_hyperdegree(self) -> float:
        return self.m * self.d / self.n

    def hyperdegrees(self) -> Array:
        deg = np.zeros(self.n, dtype=np.int64)
        np.add.at(deg, self.edges.ravel(), 1)
        return deg


@dataclass
class MCResult:
    rho: Array
    theta_mean: Array
    final_infected: Array
    final_theta: Array
    hypergraph: Hypergraph

    @property
    def stationary_rho(self) -> float:
        tail = self.rho[int(0.8 * len(self.rho)) :]
        return float(np.mean(tail))


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_uniform_hypergraph(n: int, m: int, d: int, seed: int | None = None) -> Hypergraph:
    """Sample m distinct d-node hyperedges uniformly enough for sparse ER-like runs."""
    rng = make_rng(seed)
    seen: set[tuple[int, ...]] = set()
    edges: list[tuple[int, ...]] = []
    while len(edges) < m:
        batch = max(1024, 3 * (m - len(edges)))
        candidates = np.sort(rng.integers(0, n, size=(batch, d)), axis=1)
        valid = np.all(np.diff(candidates, axis=1) != 0, axis=1)
        for row in candidates[valid]:
            edge = tuple(row.tolist())
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
            if len(edges) == m:
                break
    return Hypergraph(n=n, edges=np.asarray(edges, dtype=np.int64))


def load_edge_list(path: str | Path, zero_based: bool = True) -> Hypergraph:
    """Load a comma/space separated edge list. Each line is one hyperedge."""
    rows: list[list[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            row = [int(x) for x in parts]
            rows.append(row if zero_based else [x - 1 for x in row])
    if not rows:
        raise ValueError(f"empty edge list: {path}")
    n = max(max(r) for r in rows) + 1
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError("this reproduction code expects a uniform hypergraph")
    return Hypergraph(n=n, edges=np.asarray(rows, dtype=np.int64))


def homogeneous_mmca(
    beta: float,
    mean_k: float,
    d: int,
    mu: float = 0.1,
    eta: float = 0.6,
    gamma: float = 0.02,
    rho0: float = 0.3,
    theta0: float = 1.0,
    steps: int = 1000,
) -> tuple[Array, Array]:
    rho = np.empty(steps + 1, dtype=float)
    theta = np.empty(steps + 1, dtype=float)
    rho[0] = rho0
    theta[0] = theta0
    for t in range(steps):
        avoid = (1.0 - theta[t] * beta * rho[t] ** (d - 1)) ** mean_k
        rho[t + 1] = (1.0 - rho[t]) * (1.0 - avoid) + (1.0 - mu) * rho[t]
        theta[t + 1] = theta[t] * (1.0 - eta * rho[t] ** d) + gamma * (1.0 - theta[t])
        rho[t + 1] = float(np.clip(rho[t + 1], 0.0, 1.0))
        theta[t + 1] = float(np.clip(theta[t + 1], 0.0, 1.0))
    return rho, theta


def homogeneous_stationary_curve(
    betas: Array,
    mean_k: float,
    d: int,
    mu: float = 0.1,
    eta: float = 0.6,
    gamma: float = 0.02,
    rho0: float = 0.3,
    steps: int = 1500,
) -> Array:
    values = []
    for beta in betas:
        rho, _ = homogeneous_mmca(
            float(beta),
            mean_k=mean_k,
            d=d,
            mu=mu,
            eta=eta,
            gamma=gamma,
            rho0=rho0,
            steps=steps,
        )
        values.append(np.mean(rho[int(0.8 * len(rho)) :]))
    return np.asarray(values)


def microscopic_mmca_step(p: Array, theta: Array, hg: Hypergraph, params: ModelParams) -> tuple[Array, Array]:
    edge_p = p[hg.edges]
    phi = np.prod(edge_p, axis=1)
    q = np.ones(hg.n, dtype=float)
    for pos in range(hg.d):
        nodes = hg.edges[:, pos]
        others = np.prod(np.delete(edge_p, pos, axis=1), axis=1)
        factors = 1.0 - theta * params.beta * others
        np.multiply.at(q, nodes, factors)
    p_next = (1.0 - p) * (1.0 - q) + (1.0 - params.mu) * p
    theta_next = theta * (1.0 - params.eta * phi) + params.gamma * (1.0 - theta)
    return np.clip(p_next, 0.0, 1.0), np.clip(theta_next, 0.0, 1.0)


def random_immunize(active: Array, w: float, rng: np.random.Generator) -> Array:
    active_ids = np.flatnonzero(active)
    budget = min(int(np.floor(w * active.size)), active_ids.size)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(active_ids, size=budget, replace=False)


def targeted_immunize(active: Array, phi: Array, w: float) -> Array:
    active_ids = np.flatnonzero(active)
    budget = min(int(np.floor(w * active.size)), active_ids.size)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(phi[active_ids])[::-1]
    return active_ids[order[:budget]]


def spontaneous_isolation(theta: Array, active: Array, threshold: float, w: float, sort: SISort) -> Array:
    candidates = np.flatnonzero(active & (theta < threshold))
    budget = min(int(np.floor(w * active.size)), candidates.size)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(theta[candidates])
    if sort == "descending":
        order = order[::-1]
    return candidates[order[:budget]]


def _edge_set(edges: Array) -> set[tuple[int, ...]]:
    return {tuple(sorted(e.tolist())) for e in edges}


def _sample_unique_uniform_edge(
    n: int,
    d: int,
    existing: set[tuple[int, ...]],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    while True:
        edge = tuple(sorted(rng.choice(n, size=d, replace=False).tolist()))
        if edge not in existing:
            existing.add(edge)
            return edge


def _sample_unique_preferential_edge(
    hg: Hypergraph,
    existing: set[tuple[int, ...]],
    rng: np.random.Generator,
    alpha: float,
) -> tuple[int, ...]:
    deg = hg.hyperdegrees().astype(float)
    prob = deg + alpha
    prob = prob / prob.sum()
    while True:
        edge = tuple(sorted(rng.choice(hg.n, size=hg.d, replace=False, p=prob).tolist()))
        if edge not in existing:
            existing.add(edge)
            return edge


def rewire_edges(
    hg: Hypergraph,
    edge_ids: Array,
    theta: Array,
    active: Array,
    rng: np.random.Generator,
    mode: Rewiring,
    theta0: float = 0.1,
    alpha: float = 1.0,
) -> None:
    if mode == "none" or edge_ids.size == 0:
        return
    existing = _edge_set(np.delete(hg.edges, edge_ids, axis=0))
    for edge_id in edge_ids:
        if mode == "random":
            new_edge = _sample_unique_uniform_edge(hg.n, hg.d, existing, rng)
        elif mode == "preferential":
            new_edge = _sample_unique_preferential_edge(hg, existing, rng, alpha=alpha)
        else:
            raise ValueError(f"unknown rewiring mode: {mode}")
        hg.edges[edge_id] = np.asarray(new_edge, dtype=np.int64)
        theta[edge_id] = theta0
        active[edge_id] = True


def apply_intervention(
    intervention: Intervention,
    infected: Array,
    theta: Array,
    active: Array,
    hg: Hypergraph,
    rng: np.random.Generator,
    w: float,
    theta_min: float,
    si_sort: SISort,
) -> Array:
    if intervention == "none" or w <= 0:
        return np.empty(0, dtype=np.int64)
    infected_counts = infected[hg.edges].sum(axis=1)
    phi = (infected_counts == hg.d).astype(float)
    if intervention == "random":
        chosen = random_immunize(active, w, rng)
    elif intervention == "ti":
        chosen = targeted_immunize(active, phi, w)
    elif intervention == "si":
        chosen = spontaneous_isolation(theta, active, theta_min, w, sort=si_sort)
    else:
        raise ValueError(f"unknown intervention: {intervention}")
    active[chosen] = False
    theta[chosen] = 0.0
    return chosen


def mc_sis(
    hg: Hypergraph,
    params: ModelParams,
    steps: int = 1000,
    initial_infected: float = 0.3,
    seed: int | None = None,
    intervention: Intervention = "none",
    intervention_w: float = 0.0,
    theta_min: float = 0.2,
    si_sort: SISort = "descending",
    intervention_start: int = 0,
    intervention_once: bool = False,
    rewiring: Rewiring = "none",
    rewiring_theta0: float = 0.1,
    rewiring_alpha: float = 1.0,
    qs: bool = True,
    qs_history_size: int = 50,
    qs_update_prob: float = 0.2,
) -> MCResult:
    """Stochastic synchronous s-SIS simulation.

    QS convention: absorption is always replaced by a stored active state; the
    0.2 probability is used to refresh the stored history, matching the common
    quasistationary implementation and the paper's reported value.
    """
    rng = make_rng(seed)
    infected = rng.random(hg.n) < initial_infected
    if not infected.any():
        infected[rng.integers(hg.n)] = True
    theta = np.full(hg.m, params.theta0, dtype=float)
    active = np.ones(hg.m, dtype=bool)
    rho = np.empty(steps + 1, dtype=float)
    theta_mean = np.empty(steps + 1, dtype=float)
    history: list[Array] = [infected.copy()]
    did_once = False

    for t in range(steps + 1):
        rho[t] = infected.mean()
        theta_mean[t] = theta[active].mean() if active.any() else 0.0
        if t == steps:
            break

        if t >= intervention_start and (not intervention_once or not did_once):
            chosen = apply_intervention(
                intervention,
                infected,
                theta,
                active,
                hg,
                rng,
                w=intervention_w,
                theta_min=theta_min,
                si_sort=si_sort,
            )
            if chosen.size > 0:
                did_once = True
                rewire_edges(
                    hg,
                    chosen,
                    theta,
                    active,
                    rng,
                    mode=rewiring,
                    theta0=rewiring_theta0,
                    alpha=rewiring_alpha,
                )

        edge_states = infected[hg.edges]
        infected_counts = edge_states.sum(axis=1)

        phi = (infected_counts == hg.d).astype(float)
        theta_next = theta * (1.0 - params.eta * phi) + params.gamma * (1.0 - theta)
        theta_next[~active] = 0.0
        theta_next = np.clip(theta_next, 0.0, 1.0)

        new_infections = np.zeros(hg.n, dtype=bool)
        candidate_edges = np.flatnonzero(active & (infected_counts == hg.d - 1))
        for edge_id in candidate_edges:
            local = edge_states[edge_id]
            target_positions = np.flatnonzero(~local)
            if target_positions.size != 1:
                continue
            node = hg.edges[edge_id, target_positions[0]]
            if rng.random() < theta[edge_id] * params.beta:
                new_infections[node] = True

        recovered = infected & (rng.random(hg.n) < params.mu)
        infected = (infected & ~recovered) | new_infections
        theta = theta_next

        if qs and not infected.any():
            if history:
                infected = history[rng.integers(len(history))].copy()
            else:
                infected[rng.integers(hg.n)] = True

        if qs and infected.any() and rng.random() < qs_update_prob:
            if len(history) < qs_history_size:
                history.append(infected.copy())
            else:
                history[rng.integers(qs_history_size)] = infected.copy()

    return MCResult(rho=rho, theta_mean=theta_mean, final_infected=infected, final_theta=theta, hypergraph=hg)


def replicate_mc_stationary(
    n: int,
    mean_k: float,
    d: int,
    params: ModelParams,
    beta: float,
    reps: int,
    steps: int,
    initial_infected: float,
    seed: int,
    **kwargs,
) -> tuple[float, float]:
    values = []
    m = int(round(n * mean_k / d))
    for rep in range(reps):
        hg = random_uniform_hypergraph(n=n, m=m, d=d, seed=seed + 10_000 * rep)
        p = ModelParams(beta=beta, mu=params.mu, eta=params.eta, gamma=params.gamma, theta0=params.theta0)
        result = mc_sis(hg, p, steps=steps, initial_infected=initial_infected, seed=seed + rep, **kwargs)
        values.append(result.stationary_rho)
    arr = np.asarray(values, dtype=float)
    stderr = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(arr.mean()), stderr


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(path: str | Path, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
