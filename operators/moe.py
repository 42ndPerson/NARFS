from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

from hardware import HardwareSpec
from operator_base import Operator
from operators.elementwise import SwiGLUActivation
from operators.gemm import Gemm
from operators.v3_config import V3Config
from workload_context import WorkloadContext


@dataclass(frozen=True)
class DispatchMarker(Operator):
    """Zero-cost attachment point for the dispatch all-to-all.

    ModelMapping recognizes this class and attaches the AllToAll network op
    to it. The operator itself contributes no compute or memory to the
    timing model.
    """

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        return 0.0

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        return False


@dataclass(frozen=True)
class CombineMarker(Operator):
    """Zero-cost attachment point for the combine all-to-all (post-experts)."""

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        return 0.0

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        return False


def build_moe_ffn(
    cfg: V3Config,
    derive_t: Callable[[WorkloadContext], float],
    m_expert_shared_local: Optional[int] = None,
) -> list[Operator]:
    """One MoE FFN: router + dispatch + grouped routed experts + combine + shared expert.

    The grouped routed-expert ops carry n_replicas=n_routed_experts so
    parameter accounting reflects all expert weights, but compute/memory are
    sized to a single (bottleneck) expert's per-call work — the experts run
    in parallel on different ranks.

    The shared expert is TP-sharded: callers pass `m_expert_shared_local`
    (defaulting to full `cfg.m_expert`). Routed experts are EP-sharded only;
    they keep full `m_expert`.
    """
    n_routed = cfg.n_routed_experts
    top_k = cfg.top_k
    imb = cfg.moe_imbalance_factor
    dtype = cfg.dtype_bytes
    m_shared = cfg.m_expert if m_expert_shared_local is None else m_expert_shared_local

    # Per-expert M for the routed grouped GEMMs: average tokens-per-expert
    # scaled by imbalance to model the slowest-expert bottleneck.
    expert_M: Callable[[WorkloadContext], float] = (
        lambda ctx: derive_t(ctx) * top_k / n_routed * imb
    )

    return [
        # Router scores all tokens against every routed expert.
        Gemm(n=n_routed, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t),
        DispatchMarker(),
        # Grouped routed experts (one Gemm per slice of the FFN).
        Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        ),
        Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        ),
        SwiGLUActivation(intermediate_size=cfg.m_expert, dtype_bytes=dtype, derive_t=expert_M),
        Gemm(
            n=cfg.h, k=cfg.m_expert, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        ),
        CombineMarker(),
        # Shared expert (always active for every token), n_shared_experts == 1.
        # TP-sharded along intermediate dim.
        Gemm(n=m_shared, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t),
        Gemm(n=m_shared, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t),
        SwiGLUActivation(intermediate_size=m_shared, dtype_bytes=dtype, derive_t=derive_t),
        Gemm(n=cfg.h, k=m_shared, dtype_bytes=dtype, derive_M=derive_t),
    ]
