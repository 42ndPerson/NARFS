from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from operators.base import Operator
from workload_context import WorkloadContext


@dataclass(frozen=True)
class PrefillAttention(Operator):
    """GQA prefill attention via FlashAttention.

    Parallelism-pure: shapes are full-model. Under HEADS_PARALLEL the
    `ParallelismScheme` divides the per-rank kernel time by `par.tp`;
    the operator itself sees the full head count.

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

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 2.0 * self.num_heads * self.head_size * self._t2(ctx)

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        return (
            3.0 * self.num_heads * self.head_size * self._t2(ctx)
            / self.flash_block_size
            * self.dtype_bytes
        )


@dataclass(frozen=True)
class DecodeAttention(Operator):
    """GQA decode attention. Each step processes 1 new query token per request,
    attending over the running KV cache.

    Parallelism-pure: full head counts. Under HEADS_PARALLEL the scheme
    divides per-rank arithmetic by `par.tp`.

    Per invocation:
      compute: 2 * num_heads * head_size * avg_context_length * microbatch_size
      memory:  microbatch_size * avg_context_length * kv_bytes_per_token

    avg_context_length = prompt_len + response_len/2.

    kv_bytes_per_token defaults to 2 * num_kv_heads * head_size * dtype_bytes
    (full GQA K + V reads). MLA overrides this to
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

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return (
            2.0 * self.num_heads * self.head_size
            * self.avg_context_length(ctx) * ctx.microbatch_size
        )

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        kv_bytes = float(self.kv_bytes_per_token or 0)
        return ctx.microbatch_size * self.avg_context_length(ctx) * kv_bytes
