from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hardware import HardwareSpec
from network_requirement import NetworkRequirement, ParallelismConfig
from parallelism_role import ParallelismRole, REPLICATED
from workload_context import WorkloadContext


@dataclass(frozen=True)
class Operator(ABC):
    """Per-kernel cost function. Subclasses are typically frozen dataclasses
    that add their own shape fields (n, k, head_size, ...) and override the
    cost methods.

    Operators carry a `ParallelismRole` that declares how they shard under
    TP and EP. Cost methods take a `ParallelismConfig` so the operator can
    divide its full-model shape by the appropriate degree.
    """
    # Static fallback used when an operator's "smallness" doesn't depend on ctx.
    # For ctx-aware cases (Gemm with M = microbatch * prompt_len), override
    # is_small_dim instead.
    small_dim: bool = field(default=False, kw_only=True)
    parallelism_role: ParallelismRole = field(default=REPLICATED, kw_only=True)

    @abstractmethod
    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        raise NotImplementedError

    @abstractmethod
    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        raise NotImplementedError

    @abstractmethod
    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        raise NotImplementedError

    def is_small_dim(self, ctx: WorkloadContext, par: ParallelismConfig) -> bool:
        del ctx, par
        return self.small_dim

    # ----- efficiency selection (overridable per operator type) -----------------
    # Default: just `hw.compute_efficiency` / `hw.memory_efficiency`. Operator
    # types whose efficiency depends on shape (notably Gemm with its
    # GEMV / small / normal tiers) override these to consult the workload.

    def compute_efficiency_factor(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> float:
        del ctx, par
        return hw.compute_efficiency

    def memory_efficiency_factor(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> float:
        del ctx, par
        return hw.memory_efficiency

    def compute_time(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> float:
        eff = self.compute_efficiency_factor(ctx, hw, par)
        return self.compute_flops(ctx, par) / (hw.peak_compute_flops * eff)

    def memory_time(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> float:
        eff = self.memory_efficiency_factor(ctx, hw, par)
        return self.memory_bytes(ctx, par) / (hw.peak_memory_bandwidth * eff)

    def kernel_time(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> float:
        return max(self.compute_time(ctx, hw, par), self.memory_time(ctx, hw, par))

    # Static parameter accounting. Default to zero so timing-only ops don't
    # need to override; weight-bearing ops (Gemm, TokenEmbedding) override to
    # report N*K*dtype etc. activated_parameter_bytes diverges from
    # parameter_bytes only for sparse-activation ops (MoE experts).
    def parameter_bytes(self) -> int:
        return 0

    def parameter_bytes_per_rank(self, par: ParallelismConfig) -> int:
        """Default: assume parameters replicate, so per-rank == full. Operators
        that shard parameters along TP / EP override this to divide."""
        del par
        return self.parameter_bytes()

    def activated_parameter_bytes(self) -> int:
        return self.parameter_bytes()

    def network_requirements(
        self, par: ParallelismConfig
    ) -> list[NetworkRequirement]:
        """Collectives this operator triggers under the given parallelism.

        Default: none. Operators with a non-REPLICATED role override to emit
        TP-AR / EP-AllToAll / etc. Phase 1 leaves this as the default — the
        builders still emit requirements alongside operators; Phase 2 moves
        the responsibility onto the operator.
        """
        del par
        return []
