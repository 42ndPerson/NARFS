"""Per-MM compute sequence walker.

`ComputeModel` walks a sequence of `CombinedOp`s, evaluates per-op wall
times against a contention snapshot, and hands the resulting `list[float]`
to a caller-provided aggregator closure. The aggregator owns any
parallelism-specific math (e.g. `ModelMapping`'s PP bubble formula);
`ComputeModel` itself is parallelism-oblivious — it only knows "walk the
seq, then aggregate."

This separates two concerns that used to live together in
`ModelMapping._simulate_seq`:
  - the per-op evaluation pass (now in ComputeModel),
  - the aggregation of per-op times into a total (now in the closure).

Per-operator TP scaling is unaffected — it remains baked into each
`CombinedOp.compute_time_fn` at scheme decoration time. ComputeModel
just calls that closure.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from hardware import HardwareSpec
    from model_mapping import CombinedOp
    from network_op import LogicalTransfer, NetworkOp
    from workload_context import WorkloadContext


# Aggregator: takes the per-op wall-time list and returns the total time.
# `ModelMapping` provides one of these as a closure capturing its
# partition / bubble formula / PP-P2P timing.
Aggregator = Callable[[list[float]], float]


@dataclass
class ComputeModel:
    """Per-(MM, phase) walker + aggregator.

    `seq` is the ordered list of `CombinedOp`s for one phase (forward
    pattern, decode pattern, training pattern). `aggregate_fn` turns the
    walker's per-op wall-time list into a single total — that's where the
    PP bubble math lives, supplied by `ModelMapping` as a closure.
    """
    seq: "list[CombinedOp]"
    aggregate_fn: Aggregator

    def evaluate(
        self,
        ctx: "WorkloadContext",
        hw: "HardwareSpec",
        op_transfers: "dict[NetworkOp, list[LogicalTransfer]]",
        transfer_times_steady: "dict[LogicalTransfer, float]",
        transfer_times_boundary: "dict[LogicalTransfer, float]",
    ) -> tuple[float, list[float]]:
        """Walk `seq` evaluating each `CombinedOp.wall_time`; aggregate
        the per-op list. Returns `(total_time, per_op_times)`."""
        per_op = [
            co.wall_time(
                ctx, hw,
                op_transfers, transfer_times_steady, transfer_times_boundary,
            )
            for co in self.seq
        ]
        total = self.aggregate_fn(per_op)
        return total, per_op
