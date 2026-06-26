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
    """论文模型中的全局动力学参数。"""

    beta: float = 0.2
    mu: float = 0.1
    eta: float = 0.6
    gamma: float = 0.02
    theta0: float = 1.0


@dataclass
class Hypergraph:
    """d-uniform 超图。

    edges 是形状为 (m, d) 的整数数组，每一行是一条超边包含的节点编号。
    """

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
    """一次 MC 仿真的输出。"""

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
    """生成随机 d-uniform 超图。

    论文使用 ER 型均匀随机超图 H(n,m,d)。这里批量采样 d 个不同节点，
    并去掉重复超边；在论文的稀疏设置下，这和均匀无放回采样近似一致，
    但速度比逐条 rng.choice 快很多。
    """
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
    """读取均匀超图 edge-list。

    每一行是一条超边，节点编号可以用空格或逗号分隔。当前复现核心只处理
    uniform hypergraph，因此所有行的长度必须相同。
    """
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
    """同质 MMCA 二维迭代。

    对应论文 Eq. (14)-(16)：把所有节点概率近似为 rho(t)，所有超边活动
    近似为 theta(t)，得到可用于理论曲线和阈值分析的标量动力系统。
    """
    rho = np.empty(steps + 1, dtype=float)
    theta = np.empty(steps + 1, dtype=float)
    rho[0] = rho0
    theta[0] = theta0
    for t in range(steps):
        # q(t) = [1 - theta * beta * rho^(d-1)]^<k>：
        # 一个易感节点避开所有入射超边感染的概率。
        avoid = (1.0 - theta[t] * beta * rho[t] ** (d - 1)) ** mean_k
        rho[t + 1] = (1.0 - rho[t]) * (1.0 - avoid) + (1.0 - mu) * rho[t]
        # theta 的负反馈：感染压力 rho^d 越高，活动下降越强；
        # gamma 项表示群体活动自发恢复。
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
    """扫描 beta，返回同质 MMCA 的稳态感染密度曲线。"""
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
    """微观 MMCA 的一步更新。

    这里不抽样节点状态，而是更新每个节点的感染概率 p_i 和每条超边活动
    theta_e。该函数对应论文 Eq. (2)-(5)。
    """
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
    """随机选择预算内的活跃超边进行免疫。"""
    active_ids = np.flatnonzero(active)
    budget = min(int(np.floor(w * active.size)), active_ids.size)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(active_ids, size=budget, replace=False)


def targeted_immunize(active: Array, phi: Array, w: float) -> Array:
    """按感染压力 phi_e 从高到低选择超边，对应 TI 策略。"""
    active_ids = np.flatnonzero(active)
    budget = min(int(np.floor(w * active.size)), active_ids.size)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(phi[active_ids])[::-1]
    return active_ids[order[:budget]]


def spontaneous_isolation(theta: Array, active: Array, threshold: float, w: float, sort: SISort) -> Array:
    """按活动阈值触发自发隔离，对应 SI 策略。

    论文算法写的是候选集合 theta_e < threshold，然后按 theta_e 降序取预算内
    超边。由于正文解释存在歧义，这里保留 sort 参数，便于检查升序版本。
    """
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
    """把已免疫/失活的超边替换成新超边。

    mode="random" 时均匀随机生成新超边；mode="preferential" 时按节点当前
    超度 k_i + alpha 做度优先采样。新超边活动初始化为 theta0=0.1。
    """
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
    """在当前 MC 状态上施加一次超边免疫策略。"""
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
    """随机同步 s-SIS 仿真。

    QS 约定：如果系统进入全健康吸收态，就从历史活跃构型中恢复；论文提到
    的 0.2 在这里解释为“刷新历史池”的概率，这是准稳态仿真的常见写法。
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

        # 干预先于本轮疾病传播执行。Fig. 4-6 的脚本默认 burn-in 后一次性干预。
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

        # 每条超边里有多少感染节点，用于判断高阶传播条件和活动抑制。
        edge_states = infected[hg.edges]
        infected_counts = edge_states.sum(axis=1)

        phi = (infected_counts == hg.d).astype(float)
        theta_next = theta * (1.0 - params.eta * phi) + params.gamma * (1.0 - theta)
        theta_next[~active] = 0.0
        theta_next = np.clip(theta_next, 0.0, 1.0)

        new_infections = np.zeros(hg.n, dtype=bool)
        # s-SIS 高阶感染规则：一条 d-超边中，除目标节点外其余 d-1 个节点
        # 都感染时，才会尝试感染这个唯一的易感节点。
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

        # QS 防止有限规模仿真过早落入全健康吸收态。
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
    """重复 MC 仿真并返回稳态感染密度的均值和标准误。"""
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
