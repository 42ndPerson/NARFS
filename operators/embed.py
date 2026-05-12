from __future__ import annotations
from dataclasses import dataclass

from hardware import HardwareSpec
from network_requirement import ParallelismConfig
from operator_base import Operator
from parallelism_role import TPMode
from workload_context import TokenCount, WorkloadContext


@dataclass(frozen=True)
class TokenEmbedding(Operator):
    """Token id -> hidden lookup. Zero compute; output write dominates.

    Under VOCAB_PARALLEL, the embedding table shards across TP along the
    vocab dim — `parameter_bytes_per_rank` divides by tp. Per-call memory
    (the output write) does not change since each rank still emits the
    same h-vector for each token.
    """

    vocab: int
    h: int
    token_count: TokenCount
    dtype_bytes: int = 2

    def _t(self, ctx: WorkloadContext) -> float:
        return float(self.token_count.resolve(ctx))

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        del ctx, par
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        del par
        # Per-token write of an h-vector; the table read is one row per token,
        # negligible vs. the contiguous output write.
        return self._t(ctx) * self.h * self.dtype_bytes

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        del ctx, hw, par
        return False

    def parameter_bytes(self) -> int:
        return self.vocab * self.h * self.dtype_bytes

    def parameter_bytes_per_rank(self, par: ParallelismConfig) -> int:
        if par.tp > 1 and self.parallelism_role.tp == TPMode.VOCAB_PARALLEL:
            if self.vocab % par.tp != 0:
                raise ValueError(
                    f"TokenEmbedding.vocab ({self.vocab}) not divisible by tp ({par.tp})"
                )
            return (self.vocab // par.tp) * self.h * self.dtype_bytes
        return self.parameter_bytes()
