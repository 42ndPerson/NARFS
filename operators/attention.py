from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from hardware import HardwareSpec
from operator_base import Operator
from workload_context import WorkloadContext


@dataclass(frozen=True)
class PrefillAttention(Operator):
    """GQA prefill attention via FlashAttention.

    Per request of length l, per query head:
      compute: 2 * l^2 * head_size  (DistServe Appendix A factor)
      memory:  3 * l * head_size * (l / flash_block_size) bytes

    Summing over n_q heads and a uniform batch (t = B*L, t2 = B*L^2):
      compute_flops = 2 * num_heads * head_size * t2
      memory_bytes  = 3 * num_heads * head_size * t2 / flash_block_size * dtype_bytes

    Always memory-bound on A100/H100: AI = 2 * flash_block_size / (3 * dtype_bytes)
    ~= 21 for b=64, dtype=2 — far below the FP16 roofline ridge of either GPU.
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

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        return False  # FlashAttention prefill is memory-bound on A100/H100


@dataclass(frozen=True)
class DecodeAttention(Operator):
    """GQA decode attention. Each step processes 1 new query token per request,
    attending over the running KV cache.

    Per invocation:
      compute: 2 * num_heads * head_size * avg_context_length * microbatch_size
      memory:  microbatch_size * avg_context_length * kv_bytes_per_token

    `microbatch_size` here is the number of concurrent decode requests in
    this invocation — for non-pipelined decode (the common case) it equals
    the generation batch. The naming is shared with training-microbatch
    elsewhere because both denote "per-invocation batch dimension"; only the
    workload semantics differ.

    avg_context_length = prompt_len + response_len/2, the mean attended length
    over a uniform decode trajectory.

    kv_bytes_per_token defaults to 2 * num_kv_heads * head_size * dtype_bytes
    (full GQA K + V reads). MLA overrides this to (d_kv_compress + d_rope) *
    dtype_bytes — its compressed-latent KV cache is the dominant memory win.

    Always memory-bound on A100/H100 for any reasonable GQA/MLA shape.
    """

    num_heads: int
    num_kv_heads: int
    head_size: int
    dtype_bytes: int = 2
    # Resolved in __post_init__ when None (default GQA formula).
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
        return (
            ctx.microbatch_size * self.avg_context_length(ctx) * self.kv_bytes_per_token
        )

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        return False
