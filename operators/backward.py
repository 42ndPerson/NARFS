from __future__ import annotations
from dataclasses import dataclass

from hardware import HardwareSpec
from network_requirement import ParallelismConfig
from operator_base import Operator
from operators.builders import (
    OperatorWithReqs,
    PatternEntry,
    LayerRepeat,
)
from workload_context import WorkloadContext


@dataclass(frozen=True)
class BackwardOp(Operator):
    """Wraps a forward operator and reports backward-pass cost.

    Scaling factors come from the standard "forward + 2 backward GEMMs"
    decomposition. With activation recomputation, the forward is recomputed
    during backward, hence the extra forward-equivalent compute and memory.

    recompute=True   compute = 3 * fwd  memory = 3 * fwd  (recompute fwd + dW + dX)
    recompute=False  compute = 2 * fwd  memory = 1.5 * fwd
    """

    forward: Operator
    recompute: bool = True

    def _compute_scale(self) -> float:
        return 3.0 if self.recompute else 2.0

    def _memory_scale(self) -> float:
        return 3.0 if self.recompute else 1.5

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return self._compute_scale() * self.forward.compute_flops(ctx, par)

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        return self._memory_scale() * self.forward.memory_bytes(ctx, par)

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        # Same arithmetic intensity as forward (compute and memory scale together).
        return self.forward.is_compute_bound(ctx, hw, par)

    def is_small_dim(self, ctx: WorkloadContext, par: ParallelismConfig) -> bool:
        return self.forward.is_small_dim(ctx, par)

    # Backward doesn't carry weight parameters of its own — the forward op owns them.
    def parameter_bytes(self) -> int:
        return 0

    def activated_parameter_bytes(self) -> int:
        return 0


def derive_backward_pattern(
    forward_pattern: list[PatternEntry],
    recompute: bool = True,
) -> list[PatternEntry]:
    """Walk the forward tagged pattern, producing matching BackwardOps.

    Network requirements on backward ops mirror their forward counterparts —
    a TP-AR after attn_out forward maps to a TP-AR after the corresponding
    backward.

    Structural shape is preserved exactly: LayerRepeats stay LayerRepeats
    with the same `count` and the same number of members; top-level entries
    stay top-level at the same position. This means `expand_pattern_with_layers`
    assigns the same `LayerTag` (layer_index + anchor) to corresponding
    forward/backward operators, which is the invariant the layer-aware PP
    partitioner relies on to co-locate a layer's forward and backward.

    Per-operator timing doesn't depend on layer order, so the output
    preserves the forward order; layer-level reversal is handled by
    pipeline scheduling.
    """
    out: list[PatternEntry] = []
    for item in forward_pattern:
        if isinstance(item, LayerRepeat):
            members: list[OperatorWithReqs] = [
                (BackwardOp(forward=op, recompute=recompute), reqs)
                for (op, reqs) in item.members
            ]
            out.append(LayerRepeat(item.count, members))
        else:
            op, reqs = item
            out.append((BackwardOp(forward=op, recompute=recompute), reqs))
    return out
