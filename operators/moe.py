from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from operators.base import Operator
from operators.elementwise import SwiGLUActivation
from operators.gemm import Gemm
from operators.pattern import LabeledOp, OpRole
from operators.v3_config import V3Config
from workload_context import WorkloadContext


@dataclass(frozen=True)
class DispatchMarker(Operator):
    """Zero-cost attachment point for the dispatch all-to-all."""

    def compute_flops(self, ctx: WorkloadContext) -> float:
        del ctx
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        del ctx
        return 0.0


@dataclass(frozen=True)
class CombineMarker(Operator):
    """Zero-cost attachment point for the combine all-to-all."""

    def compute_flops(self, ctx: WorkloadContext) -> float:
        del ctx
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        del ctx
        return 0.0


def build_moe_ffn(
    cfg: V3Config,
    derive_t: Callable[[WorkloadContext], float],
) -> list[LabeledOp]:
    """One MoE FFN: router + dispatch + grouped routed experts + combine + shared expert.

    Emits labeled operators; the strategy decorates with sharding and
    collectives. Routed expert grouped GEMMs carry n_replicas=n_routed_experts
    for parameter accounting; compute/memory are sized to a single (bottleneck)
    expert's per-call work — the experts run in parallel on EP ranks.
    """
    n_routed = cfg.n_routed_experts
    top_k = cfg.top_k
    imb = cfg.moe_imbalance_factor
    dtype = cfg.dtype_bytes

    expert_M: Callable[[WorkloadContext], float] = (
        lambda ctx: derive_t(ctx) * top_k / n_routed * imb
    )

    return [
        LabeledOp(OpRole.ROUTER, Gemm(
            n=n_routed, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t,
        )),
        LabeledOp(OpRole.DISPATCH_MARKER, DispatchMarker()),
        LabeledOp(OpRole.ROUTED_EXPERT_EXPAND, Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        )),
        LabeledOp(OpRole.ROUTED_EXPERT_EXPAND, Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        )),
        LabeledOp(OpRole.ROUTED_EXPERT_ACT, SwiGLUActivation(
            intermediate_size=cfg.m_expert, dtype_bytes=dtype, derive_t=expert_M,
        )),
        LabeledOp(OpRole.ROUTED_EXPERT_REDUCE, Gemm(
            n=cfg.h, k=cfg.m_expert, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        )),
        LabeledOp(OpRole.COMBINE_MARKER, CombineMarker()),
        # Shared experts are just a dense FFN path — re-use the FFN roles.
        LabeledOp(OpRole.FFN_EXPAND, Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t,
        )),
        LabeledOp(OpRole.FFN_EXPAND, Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t,
        )),
        LabeledOp(OpRole.FFN_ACT, SwiGLUActivation(
            intermediate_size=cfg.m_expert, dtype_bytes=dtype, derive_t=derive_t,
        )),
        LabeledOp(OpRole.FFN_REDUCE, Gemm(
            n=cfg.h, k=cfg.m_expert, dtype_bytes=dtype, derive_M=derive_t,
        )),
    ]
