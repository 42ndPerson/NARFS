from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from hardware import HardwareSpec
from network_requirement import ParallelismConfig
from operator_base import Operator
from operators.elementwise import SwiGLUActivation
from operators.gemm import Gemm
from operators.pattern import LabeledOp
from operators.v3_config import V3Config
from workload_context import WorkloadContext


# Canonical MoE labels.
MOE_ROUTER = "moe_router"
MOE_DISPATCH = "moe_dispatch"
MOE_ROUTED_GATE = "moe_routed_gate"
MOE_ROUTED_UP = "moe_routed_up"
MOE_ROUTED_SWIGLU = "moe_routed_swiglu"
MOE_ROUTED_DOWN = "moe_routed_down"
MOE_COMBINE = "moe_combine"
MOE_SHARED_GATE = "moe_shared_gate"
MOE_SHARED_UP = "moe_shared_up"
MOE_SHARED_SWIGLU = "moe_shared_swiglu"
MOE_SHARED_DOWN = "moe_shared_down"


@dataclass(frozen=True)
class DispatchMarker(Operator):
    """Zero-cost attachment point for the dispatch all-to-all."""

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        del ctx, par
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        del ctx, par
        return 0.0

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        del ctx, hw, par
        return False


@dataclass(frozen=True)
class CombineMarker(Operator):
    """Zero-cost attachment point for the combine all-to-all."""

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        del ctx, par
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        del ctx, par
        return 0.0

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        del ctx, hw, par
        return False


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
        LabeledOp(MOE_ROUTER, Gemm(
            n=n_routed, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t,
        )),
        LabeledOp(MOE_DISPATCH, DispatchMarker()),
        LabeledOp(MOE_ROUTED_GATE, Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        )),
        LabeledOp(MOE_ROUTED_UP, Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        )),
        LabeledOp(MOE_ROUTED_SWIGLU, SwiGLUActivation(
            intermediate_size=cfg.m_expert, dtype_bytes=dtype, derive_t=expert_M,
        )),
        LabeledOp(MOE_ROUTED_DOWN, Gemm(
            n=cfg.h, k=cfg.m_expert, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        )),
        LabeledOp(MOE_COMBINE, CombineMarker()),
        LabeledOp(MOE_SHARED_GATE, Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t,
        )),
        LabeledOp(MOE_SHARED_UP, Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t,
        )),
        LabeledOp(MOE_SHARED_SWIGLU, SwiGLUActivation(
            intermediate_size=cfg.m_expert, dtype_bytes=dtype, derive_t=derive_t,
        )),
        LabeledOp(MOE_SHARED_DOWN, Gemm(
            n=cfg.h, k=cfg.m_expert, dtype_bytes=dtype, derive_M=derive_t,
        )),
    ]
