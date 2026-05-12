from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from hardware import HardwareSpec
from network_requirement import ParallelismConfig
from operator_base import Operator
from parallelism_role import TPMode
from workload_context import WorkloadContext


def _heads_per_rank(role_tp: TPMode, full_heads: int, tp: int, label: str) -> int:
    if tp <= 1 or role_tp != TPMode.HEADS_PARALLEL:
        return full_heads
    if full_heads % tp != 0:
        raise ValueError(
            f"{label} ({full_heads}) not divisible by tp ({tp}) under HEADS_PARALLEL"
        )
    return full_heads // tp


@dataclass(frozen=True)
class PrefillAttention(Operator):
    """GQA prefill attention via FlashAttention.

    Shapes are full-model: under HEADS_PARALLEL the operator divides
    `num_heads` and `num_kv_heads` per rank by `par.tp`. REPLICATED leaves
    them untouched (Phase-2a back-compat with builders that still pre-divide).

    Per request of length l, per query head:
      compute: 2 * l^2 * head_size  (DistServe Appendix A factor)
      memory:  3 * l * head_size * (l / flash_block_size) bytes

    Always memory-bound on A100/H100 for any reasonable shape.
    """

    num_heads: int
    num_kv_heads: int
    head_size: int
    flash_block_size: int = 64
    dtype_bytes: int = 2

    @staticmethod
    def _t2(ctx: WorkloadContext) -> int:
        # Per-invocation work: uniform prompt length assumption ⇒
        # sum_r l_r^2 = microbatch_size * prompt_len^2.
        return ctx.microbatch_size * ctx.prompt_len * ctx.prompt_len

    def _heads(self, par: ParallelismConfig) -> int:
        return _heads_per_rank(self.parallelism_role.tp, self.num_heads, par.tp, "num_heads")

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return 2.0 * self._heads(par) * self.head_size * self._t2(ctx)

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return (
            3.0 * self._heads(par) * self.head_size * self._t2(ctx)
            / self.flash_block_size
            * self.dtype_bytes
        )

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        del ctx, hw, par
        return False  # FlashAttention prefill is memory-bound on A100/H100


@dataclass(frozen=True)
class DecodeAttention(Operator):
    """GQA decode attention. Each step processes 1 new query token per request,
    attending over the running KV cache.

    Shapes are full-model under HEADS_PARALLEL the operator divides
    `num_heads` and `num_kv_heads` per rank by `par.tp`.

    Per invocation:
      compute: 2 * num_heads * head_size * avg_context_length * microbatch_size
      memory:  microbatch_size * avg_context_length * kv_bytes_per_token

    avg_context_length = prompt_len + response_len/2.

    kv_bytes_per_token defaults to 2 * num_kv_heads * head_size * dtype_bytes
    (full GQA K + V reads, post-shard). MLA overrides this to
    (d_kv_compress + d_rope) * dtype_bytes.

    Always memory-bound on A100/H100 for any reasonable shape.
    """

    num_heads: int
    num_kv_heads: int
    head_size: int
    dtype_bytes: int = 2
    kv_bytes_per_token: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kv_bytes_per_token is None:
            object.__setattr__(
                self,
                "kv_bytes_per_token",
                2 * self.num_kv_heads * self.head_size * self.dtype_bytes,
            )

    @staticmethod
    def avg_context_length(ctx: WorkloadContext) -> float:
        return ctx.prompt_len + ctx.response_len / 2.0

    def _heads(self, par: ParallelismConfig) -> int:
        return _heads_per_rank(self.parallelism_role.tp, self.num_heads, par.tp, "num_heads")

    def _kv_bytes_per_rank(self, par: ParallelismConfig) -> float:
        # __post_init__ guarantees kv_bytes_per_token is set; cast for the type checker.
        kv_bytes = float(self.kv_bytes_per_token or 0)
        # Under HEADS_PARALLEL, the per-rank KV-cache footprint shrinks too.
        if par.tp > 1 and self.parallelism_role.tp == TPMode.HEADS_PARALLEL:
            if self.num_kv_heads % par.tp != 0:
                raise ValueError(
                    f"num_kv_heads ({self.num_kv_heads}) not divisible by tp ({par.tp})"
                )
            return kv_bytes / par.tp
        return kv_bytes

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return (
            2.0 * self._heads(par) * self.head_size
            * self.avg_context_length(ctx) * ctx.microbatch_size
        )

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return (
            ctx.microbatch_size * self.avg_context_length(ctx) * self._kv_bytes_per_rank(par)
        )

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        del ctx, hw, par
        return False
