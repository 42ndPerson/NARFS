from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from hardware import HardwareSpec
from operator_base import Operator
from workload_context import TokenCount, WorkloadContext


@dataclass(frozen=True)
class Gemm(Operator):
    """Single-replica GEMM cost model.

    Parallelism-pure: `n` and `k` are full-model shape coordinates and
    compute / memory / efficiency are sized to that full shape. The
    `ParallelismScheme` divides the resulting per-rank arithmetic at
    decoration time (typically `kernel_time / tp_factor` for sharded
    roles); GEMV / small_gemm tiers consult the full shape, which is a
    deliberate ±30% approximation — per-rank tile shape differs between
    COLUMN_PARALLEL (split N) and ROW_PARALLEL (split K) and we don't
    model that asymmetry.

    `n_replicas` (e.g. routed experts) sizes parameter accounting only;
    a per-call `compute_flops` still costs one bottleneck replica.
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

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 2.0 * self._m(ctx) * self.n * self.k

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        m = self._m(ctx)
        return float(self.dtype_bytes) * (m * self.k + self.k * self.n + m * self.n)

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        return self.compute_flops(ctx) / self.memory_bytes(ctx) > hw.ridge_flops_per_byte

    def is_small_dim(self, ctx: WorkloadContext) -> bool:
        return min(self._m(ctx), self.n, self.k) < 1024

    def is_gemv(self, ctx: WorkloadContext) -> bool:
        """True for GEMV-shaped GEMMs (M < 16). Decode steps with batch=1
        produce M=1 GEMMs with K, N >> 1024 — these miss `is_small_dim` (since
        only M is small) but have fundamentally GEMV-style efficiency.
        """
        return self._m(ctx) < 16

    def compute_efficiency_factor(
        self, ctx: WorkloadContext, hw: HardwareSpec
    ) -> float:
        if self.is_gemv(ctx):
            return hw.gemv_efficiency
        if self.is_small_dim(ctx):
            return hw.small_gemm_efficiency
        return hw.compute_efficiency

    def memory_efficiency_factor(
        self, ctx: WorkloadContext, hw: HardwareSpec
    ) -> float:
        if self.is_gemv(ctx):
            return hw.gemv_memory_efficiency
        return hw.memory_efficiency

    def parameter_bytes(self) -> int:
        """Full-model parameter bytes (all replicas, full N×K)."""
        return self.n_replicas * self.n * self.k * self.dtype_bytes

    def activated_parameter_bytes(self) -> int:
        return self.replicas_active * self.n * self.k * self.dtype_bytes
