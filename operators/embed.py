from __future__ import annotations
from dataclasses import dataclass

from hardware import HardwareSpec
from operator_base import Operator
from workload_context import TokenCount, WorkloadContext


@dataclass(frozen=True)
class TokenEmbedding(Operator):
    """Token id -> hidden lookup. Zero compute; output write dominates.

    Parallelism-pure: `parameter_bytes()` returns the full vocab×h table.
    Under VOCAB_PARALLEL the scheme divides per-rank parameter bytes by tp.
    Per-call memory (the output write) is unchanged — each rank still emits
    the same h-vector for each token.
    """

    vocab: int
    h: int
    token_count: TokenCount
    dtype_bytes: int = 2

    def _t(self, ctx: WorkloadContext) -> float:
        return float(self.token_count.resolve(ctx))

    def compute_flops(self, ctx: WorkloadContext) -> float:
        del ctx
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        # Per-token write of an h-vector; the table read is one row per token,
        # negligible vs. the contiguous output write.
        return self._t(ctx) * self.h * self.dtype_bytes

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        del ctx, hw
        return False

    def parameter_bytes(self) -> int:
        return self.vocab * self.h * self.dtype_bytes
