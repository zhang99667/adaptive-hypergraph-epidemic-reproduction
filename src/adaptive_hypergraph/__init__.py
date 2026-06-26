"""自适应超图传播动力学复现工具包。"""

from .model import (
    Hypergraph,
    ModelParams,
    MCResult,
    homogeneous_mmca,
    homogeneous_stationary_curve,
    mc_sis,
    random_uniform_hypergraph,
)

__all__ = [
    "Hypergraph",
    "ModelParams",
    "MCResult",
    "homogeneous_mmca",
    "homogeneous_stationary_curve",
    "mc_sis",
    "random_uniform_hypergraph",
]
