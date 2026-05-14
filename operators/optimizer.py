from __future__ import annotations
from dataclasses import dataclass

from operator_base import Operator
from workload_context import WorkloadContext


@dataclass(frozen=True)
class AdamOptimizerOp(Operator):
    """Adam optimizer step over a parameter shard.

    Sized to a single layer's (or boundary group's) parameter count when used
    inside the layer-aware pipeline partition: ModelMapping emits one
    AdamOptimizerOp per `LayerTag` so each PP stage runs the optimizer for
    the parameters it owns.

    compute_flops:  8 * num_parameters  (m,v update, bias correction, param step)
    memory_bytes:   read params + grads + 2 state moments,
                    write params + 2 state moments
                    = (2 * dtype_bytes_param + 3 * dtype_bytes_state) * num_parameters

    Always memory-bound — Adam is famously bandwidth-limited.
    """

    num_parameters: int
    dtype_bytes_param: int = 2
    dtype_bytes_state: int = 4

    def compute_flops(self, ctx: WorkloadContext) -> float:
        del ctx
        return 8.0 * self.num_parameters

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        del ctx
        return float(
            (2 * self.dtype_bytes_param + 3 * self.dtype_bytes_state)
            * self.num_parameters
        )
