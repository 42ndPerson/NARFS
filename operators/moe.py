from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from hardware import HardwareSpec
from network_requirement import ParallelismConfig
from operator_base import Operator
from operators.elementwise import SwiGLUActivation
from operators.gemm import Gemm
from operators.v3_config import V3Config
from parallelism_role import EPMode, ParallelismRole, TPMode
from workload_context import WorkloadContext


_REP = ParallelismRole()
_COL = ParallelismRole(tp=TPMode.COLUMN_PARALLEL)
_ROW = ParallelismRole(tp=TPMode.ROW_PARALLEL)
_EP = ParallelismRole(ep=EPMode.EXPERT_PARALLEL)


@dataclass(frozen=True)
class DispatchMarker(Operator):
    """Zero-cost attachment point for the dispatch all-to-all.

    ModelMapping recognizes this class and attaches the AllToAll network op
    to it. The operator itself contributes no compute or memory to the
    timing model.
    """

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
    """Zero-cost attachment point for the combine all-to-all (post-experts)."""

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
) -> list[Operator]:
    """One MoE FFN: router + dispatch + grouped routed experts + combine + shared expert.

    Routed-expert grouped GEMMs carry n_replicas=n_routed_experts (parameter
    accounting reflects all expert weights) but compute/memory are sized to a
    single expert's per-call work — the experts run in parallel on EP ranks.
    Their `parallelism_role.ep == EXPERT_PARALLEL` so per-rank parameter
    bytes correctly divide by `par.ep`.

    The shared expert is TP-sharded along the intermediate dim with full
    `m_expert` shape and column/row roles; per-rank size is derived inside
    the operator at compute time.
    """
    n_routed = cfg.n_routed_experts
    top_k = cfg.top_k
    imb = cfg.moe_imbalance_factor
    dtype = cfg.dtype_bytes

    # Per-expert M for the routed grouped GEMMs: average tokens-per-expert
    # scaled by imbalance to model the slowest-expert bottleneck.
    expert_M: Callable[[WorkloadContext], float] = (
        lambda ctx: derive_t(ctx) * top_k / n_routed * imb
    )

    return [
        # Router scores all tokens against every routed expert.
        Gemm(n=n_routed, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t,
             parallelism_role=_REP),
        DispatchMarker(),
        # Grouped routed experts (one Gemm per slice of the FFN). EP-sharded
        # only — no TP split inside an expert.
        Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
            parallelism_role=_EP,
        ),
        Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
            parallelism_role=_EP,
        ),
        SwiGLUActivation(intermediate_size=cfg.m_expert, dtype_bytes=dtype, derive_t=expert_M,
                         parallelism_role=_REP),
        Gemm(
            n=cfg.h, k=cfg.m_expert, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
            parallelism_role=_EP,
        ),
        CombineMarker(),
        # Shared expert (always active for every token), n_shared_experts == 1.
        # TP-sharded with column/row roles along the intermediate dim.
        Gemm(n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t,
             parallelism_role=_COL),
        Gemm(n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t,
             parallelism_role=_COL),
        SwiGLUActivation(intermediate_size=cfg.m_expert, dtype_bytes=dtype, derive_t=derive_t,
                         parallelism_role=_COL),
        Gemm(n=cfg.h, k=cfg.m_expert, dtype_bytes=dtype, derive_M=derive_t,
             parallelism_role=_ROW),
    ]
