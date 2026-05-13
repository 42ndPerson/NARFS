from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from hardware import HardwareSpec
from operator_base import Operator
from workload_context import TokenCount, WorkloadContext


def _resolve_t(
    ctx: WorkloadContext,
    token_count: Optional[TokenCount],
    derive_t: Optional[Callable[[WorkloadContext], float]],
) -> float:
    if derive_t is not None:
        return derive_t(ctx)
    return float(token_count.resolve(ctx))


def _check_t_source(
    token_count: Optional[TokenCount],
    derive_t: Optional[Callable[[WorkloadContext], float]],
    cls_name: str,
) -> None:
    if (token_count is None) == (derive_t is None):
        raise ValueError(f"{cls_name} requires exactly one of token_count or derive_t")


@dataclass(frozen=True)
class RMSNorm(Operator):
    """Operates on the full hidden state (after upstream AR). Parallelism-pure:
    sized to the full hidden_size; the scheme divides per-rank kernel time."""

    hidden_size: int
    token_count: Optional[TokenCount] = None
    dtype_bytes: int = 2
    derive_t: Optional[Callable[[WorkloadContext], float]] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _check_t_source(self.token_count, self.derive_t, "RMSNorm")

    def _t(self, ctx: WorkloadContext) -> float:
        return _resolve_t(ctx, self.token_count, self.derive_t)

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 5.0 * self._t(ctx) * self.hidden_size

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        return 2.0 * self._t(ctx) * self.hidden_size * self.dtype_bytes

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        del ctx, hw
        return False


@dataclass(frozen=True)
class SwiGLUActivation(Operator):
    """Elementwise on the FFN intermediate. Parallelism-pure: sized to the
    full intermediate_size; the scheme divides per-rank kernel time under
    COLUMN_PARALLEL."""

    intermediate_size: int
    token_count: Optional[TokenCount] = None
    dtype_bytes: int = 2
    derive_t: Optional[Callable[[WorkloadContext], float]] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _check_t_source(self.token_count, self.derive_t, "SwiGLUActivation")

    def _t(self, ctx: WorkloadContext) -> float:
        return _resolve_t(ctx, self.token_count, self.derive_t)

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 2.0 * self._t(ctx) * self.intermediate_size

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        return 3.0 * self._t(ctx) * self.intermediate_size * self.dtype_bytes

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        del ctx, hw
        return False


@dataclass(frozen=True)
class RoPE(Operator):
    """Rotary position embedding, applied to Q (n_q heads) and K (n_kv heads).

    `num_heads` is the total number of heads being rotated — pass the
    full-model n_q + n_kv. Parallelism-pure: the scheme divides per-rank
    kernel time under HEADS_PARALLEL.
    """

    num_heads: int
    head_size: int
    rope_dim: int
    token_count: Optional[TokenCount] = None
    dtype_bytes: int = 2
    derive_t: Optional[Callable[[WorkloadContext], float]] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _check_t_source(self.token_count, self.derive_t, "RoPE")

    def _t(self, ctx: WorkloadContext) -> float:
        return _resolve_t(ctx, self.token_count, self.derive_t)

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 6.0 * self._t(ctx) * self.num_heads * self.rope_dim

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        return 2.0 * self._t(ctx) * self.num_heads * self.rope_dim * self.dtype_bytes

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        del ctx, hw
        return False
