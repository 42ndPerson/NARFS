from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hardware import HardwareSpec
from workload_context import WorkloadContext


@dataclass(frozen=True)
class Operator(ABC):
    """Per-kernel cost function. Subclasses are typically frozen dataclasses
    that add their own shape fields (n, k, head_size, ...) and override the
    cost methods.

    Operators are **parallelism-agnostic**. Their cost methods return
    *full-model* values (full FLOPs, full memory bytes, full parameter
    bytes). A `ParallelismScheme` decides how the operator shards onto
    hardware and bakes the per-rank arithmetic into `CombinedOp`'s
    `compute_time_fn` closure at MM construction time. The operator itself
    never sees parallelism degrees.
    """
    # Static fallback used when an operator's "smallness" doesn't depend on ctx.
    # For ctx-aware cases (Gemm with M = microbatch * prompt_len), override
    # is_small_dim instead.
    small_dim: bool = field(default=False, kw_only=True)

    @abstractmethod
    def compute_flops(self, ctx: WorkloadContext) -> float:
        """Full-model compute FLOPs at the given ctx."""
        raise NotImplementedError

    @abstractmethod
    def memory_bytes(self, ctx: WorkloadContext) -> float:
        """Full-model memory traffic in bytes at the given ctx."""
        raise NotImplementedError

    @abstractmethod
    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec
    ) -> bool:
        raise NotImplementedError

    def is_small_dim(self, ctx: WorkloadContext) -> bool:
        del ctx
        return self.small_dim

    # ----- efficiency selection (overridable per operator type) -----------------
    # Default: just `hw.compute_efficiency` / `hw.memory_efficiency`. Operator
    # types whose efficiency depends on shape (notably Gemm with its
    # GEMV / small / normal tiers) override these to consult the workload.

    def compute_efficiency_factor(
        self, ctx: WorkloadContext, hw: HardwareSpec
    ) -> float:
        del ctx
        return hw.compute_efficiency

    def memory_efficiency_factor(
        self, ctx: WorkloadContext, hw: HardwareSpec
    ) -> float:
        del ctx
        return hw.memory_efficiency

    def compute_time(
        self, ctx: WorkloadContext, hw: HardwareSpec
    ) -> float:
        eff = self.compute_efficiency_factor(ctx, hw)
        return self.compute_flops(ctx) / (hw.peak_compute_flops * eff)

    def memory_time(
        self, ctx: WorkloadContext, hw: HardwareSpec
    ) -> float:
        eff = self.memory_efficiency_factor(ctx, hw)
        return self.memory_bytes(ctx) / (hw.peak_memory_bandwidth * eff)

    def kernel_time(
        self, ctx: WorkloadContext, hw: HardwareSpec
    ) -> float:
        return max(self.compute_time(ctx, hw), self.memory_time(ctx, hw))

    # Static parameter accounting. Default to zero so timing-only ops don't
    # need to override; weight-bearing ops (Gemm, TokenEmbedding) override to
    # report N*K*dtype etc. activated_parameter_bytes diverges from
    # parameter_bytes only for sparse-activation ops (MoE experts).
    def parameter_bytes(self) -> int:
        """Full-model parameter bytes. Schemes apply per-rank division
        externally (typically `parameter_bytes() // (tp_factor * ep_factor)`
        when the op is sharded along those axes)."""
        return 0

    def activated_parameter_bytes(self) -> int:
        return self.parameter_bytes()
