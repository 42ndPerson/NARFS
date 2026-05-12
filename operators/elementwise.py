from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from hardware import HardwareSpec
from network_requirement import ParallelismConfig
from operator_base import Operator
from parallelism_role import TPMode
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
    """Operates on the full hidden state (after upstream AR), so default role
    is REPLICATED. If a builder marks this COLUMN_PARALLEL it would shard
    `hidden_size` across TP, but Llama / V3 don't currently use that path.
    """

    hidden_size: int
    token_count: Optional[TokenCount] = None
    dtype_bytes: int = 2
    derive_t: Optional[Callable[[WorkloadContext], float]] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _check_t_source(self.token_count, self.derive_t, "RMSNorm")

    def _t(self, ctx: WorkloadContext) -> float:
        return _resolve_t(ctx, self.token_count, self.derive_t)

    def _hidden(self, par: ParallelismConfig) -> int:
        if par.tp > 1 and self.parallelism_role.tp == TPMode.COLUMN_PARALLEL:
            if self.hidden_size % par.tp != 0:
                raise ValueError(
                    f"RMSNorm.hidden_size ({self.hidden_size}) not divisible by tp ({par.tp})"
                )
            return self.hidden_size // par.tp
        return self.hidden_size

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return 5.0 * self._t(ctx) * self._hidden(par)

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return 2.0 * self._t(ctx) * self._hidden(par) * self.dtype_bytes

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        del ctx, hw, par
        return False


@dataclass(frozen=True)
class SwiGLUActivation(Operator):
    """Elementwise on the FFN intermediate. Under COLUMN_PARALLEL it splits
    `intermediate_size` per rank — follows the column-parallel gate/up GEMMs."""

    intermediate_size: int
    token_count: Optional[TokenCount] = None
    dtype_bytes: int = 2
    derive_t: Optional[Callable[[WorkloadContext], float]] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _check_t_source(self.token_count, self.derive_t, "SwiGLUActivation")

    def _t(self, ctx: WorkloadContext) -> float:
        return _resolve_t(ctx, self.token_count, self.derive_t)

    def _intermediate(self, par: ParallelismConfig) -> int:
        if par.tp > 1 and self.parallelism_role.tp == TPMode.COLUMN_PARALLEL:
            if self.intermediate_size % par.tp != 0:
                raise ValueError(
                    f"SwiGLUActivation.intermediate_size ({self.intermediate_size}) "
                    f"not divisible by tp ({par.tp})"
                )
            return self.intermediate_size // par.tp
        return self.intermediate_size

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return 2.0 * self._t(ctx) * self._intermediate(par)

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return 3.0 * self._t(ctx) * self._intermediate(par) * self.dtype_bytes

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        del ctx, hw, par
        return False


@dataclass(frozen=True)
class RoPE(Operator):
    """Rotary position embedding, applied to Q (n_q heads) and K (n_kv heads).

    `num_heads` is the total number of heads being rotated — pass the
    full-model n_q + n_kv. Under HEADS_PARALLEL the operator divides per
    rank by `par.tp`.
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

    def _heads(self, par: ParallelismConfig) -> int:
        if par.tp > 1 and self.parallelism_role.tp == TPMode.HEADS_PARALLEL:
            if self.num_heads % par.tp != 0:
                raise ValueError(
                    f"RoPE.num_heads ({self.num_heads}) not divisible by tp ({par.tp})"
                )
            return self.num_heads // par.tp
        return self.num_heads

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return 6.0 * self._t(ctx) * self._heads(par) * self.rope_dim

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return 2.0 * self._t(ctx) * self._heads(par) * self.rope_dim * self.dtype_bytes

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        del ctx, hw, par
        return False
