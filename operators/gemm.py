from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from hardware import HardwareSpec
from network_requirement import ParallelismConfig
from operator_base import Operator
from parallelism_role import EPMode, TPMode
from workload_context import TokenCount, WorkloadContext


@dataclass(frozen=True)
class Gemm(Operator):
    """Single-replica GEMM cost model.

    `n` and `k` are full-model shape coordinates. The operator's
    `parallelism_role` decides how those shapes split per rank under TP:

      COLUMN_PARALLEL / HEADS_PARALLEL / VOCAB_PARALLEL → split N
      ROW_PARALLEL                                       → split K
      REPLICATED                                          → no split

    For backward compatibility during the Phase-2a migration, builders may
    still pass pre-divided shapes alongside `parallelism_role=REPLICATED`,
    in which case no division is applied.

    EP only affects parameter accounting (per-rank expert count = n_replicas
    / ep when role.ep == EXPERT_PARALLEL); per-call compute/memory is sized
    to a single (bottleneck) expert and does not divide further under EP.
    """

    n: int
    k: int
    # M (number of output rows) is derived per call. Either:
    #  - token_count: a TokenCount enum (PREFILL = B*L, DECODE = B), or
    #  - derive_M: a callable for cases the enum can't express (e.g. MoE
    #    expert grouping where M = t * top_k / n_experts * imbalance).
    # Exactly one must be supplied.
    token_count: Optional[TokenCount] = None
    dtype_bytes: int = 2
    # Weight replication (e.g. n_routed_experts in MoE). Compute and
    # memory_bytes model a single replica's per-call cost; replication only
    # affects parameter accounting.
    n_replicas: int = 1
    replicas_active: int = 1
    derive_M: Optional[Callable[[WorkloadContext], float]] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if (self.token_count is None) == (self.derive_M is None):
            raise ValueError(
                "Gemm requires exactly one of token_count or derive_M"
            )
        if self.replicas_active > self.n_replicas:
            raise ValueError(
                f"replicas_active ({self.replicas_active}) cannot exceed "
                f"n_replicas ({self.n_replicas})"
            )

    def _m(self, ctx: WorkloadContext) -> float:
        if self.derive_M is not None:
            return self.derive_M(ctx)
        return float(self.token_count.resolve(ctx))

    def _shape(self, par: ParallelismConfig) -> tuple[float, float]:
        """Per-rank (n, k) after applying TP role. Validates divisibility."""
        n: float = float(self.n)
        k: float = float(self.k)
        if par.tp > 1:
            role_tp = self.parallelism_role.tp
            if role_tp in (TPMode.COLUMN_PARALLEL, TPMode.HEADS_PARALLEL, TPMode.VOCAB_PARALLEL):
                if self.n % par.tp != 0:
                    raise ValueError(
                        f"Gemm.n ({self.n}) not divisible by tp ({par.tp}) under role {role_tp.name}"
                    )
                n = self.n / par.tp
            elif role_tp == TPMode.ROW_PARALLEL:
                if self.k % par.tp != 0:
                    raise ValueError(
                        f"Gemm.k ({self.k}) not divisible by tp ({par.tp}) under ROW_PARALLEL"
                    )
                k = self.k / par.tp
        return n, k

    def _replicas_per_rank(self, par: ParallelismConfig) -> int:
        if par.ep > 1 and self.parallelism_role.ep == EPMode.EXPERT_PARALLEL:
            if self.n_replicas % par.ep != 0:
                raise ValueError(
                    f"Gemm.n_replicas ({self.n_replicas}) not divisible by ep ({par.ep})"
                )
            return self.n_replicas // par.ep
        return self.n_replicas

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        n, k = self._shape(par)
        return 2.0 * self._m(ctx) * n * k

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        n, k = self._shape(par)
        m = self._m(ctx)
        return float(self.dtype_bytes) * (m * k + k * n + m * n)

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        return self.compute_flops(ctx, par) / self.memory_bytes(ctx, par) > hw.ridge_flops_per_byte

    def is_small_dim(self, ctx: WorkloadContext, par: ParallelismConfig) -> bool:
        n, k = self._shape(par)
        return min(self._m(ctx), n, k) < 1024

    def is_gemv(self, ctx: WorkloadContext) -> bool:
        """True for GEMV-shaped GEMMs (M < 16). Decode steps with batch=1
        produce M=1 GEMMs with K, N >> 1024 — these miss `is_small_dim` (since
        only M is small) but have fundamentally GEMV-style efficiency.
        """
        return self._m(ctx) < 16

    # Three-tier efficiency selection: GEMV < small_dim < normal. GEMV takes
    # precedence over small_dim because GEMV is a stricter shape constraint.
    def compute_efficiency_factor(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> float:
        if self.is_gemv(ctx):
            return hw.gemv_efficiency
        if self.is_small_dim(ctx, par):
            return hw.small_gemm_efficiency
        return hw.compute_efficiency

    def memory_efficiency_factor(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> float:
        del par
        if self.is_gemv(ctx):
            return hw.gemv_memory_efficiency
        return hw.memory_efficiency

    def parameter_bytes(self) -> int:
        """Full-model parameter bytes (all replicas, full N×K)."""
        return self.n_replicas * self.n * self.k * self.dtype_bytes

    def parameter_bytes_per_rank(self, par: ParallelismConfig) -> int:
        n, k = self._shape(par)
        return int(self._replicas_per_rank(par) * n * k * self.dtype_bytes)

    def activated_parameter_bytes(self) -> int:
        return self.replicas_active * self.n * self.k * self.dtype_bytes
