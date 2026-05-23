"""Per-MM compute sequence walker, generalized over contention snapshots.

`ComputeModel` walks a sequence of `CombinedOp`s once per provided
contention snapshot, then hands the resulting per-op wall-time lists
(keyed by snapshot name) to a caller-provided aggregator closure. The
aggregator owns any parallelism-specific math (e.g. `ModelMapping`'s
PP bubble formula and the weighted-two-snapshot combine for fill/drain);
`ComputeModel` itself is parallelism-oblivious — it only knows "walk the
seq once per snapshot, then aggregate."

The dict-based snapshot API is the natural generalization for the
weighted two-snapshot PP contention model:

  - Single-snapshot callers pass `{"steady": <dict>}` and the aggregator
    consumes `per_op["steady"]`.
  - Two-snapshot PP callers pass `{"steady": <dict>, "fill_drain": <dict>}`
    and the aggregator weights both per-op lists by slot count.

Per-operator TP scaling is unaffected — it remains baked into each
`CombinedOp.compute_time_fn` at scheme decoration time. ComputeModel
just calls that closure (via `CombinedOp.wall_time`).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from topology.hardware import HardwareSpec
    from mapping.model_mapping import CombinedOp
    from network_op import LogicalTransfer, NetworkOp
    from workload_context import WorkloadContext


# Aggregator: takes a dict[snapshot_name, per_op_wall_times] and returns
# the total wall time. `ModelMapping` provides one of these as a closure
# capturing its partition / bubble formula / PP-P2P timing per snapshot.
Aggregator = Callable[[dict[str, list[float]]], float]


@dataclass
class ComputeModel:
    """Per-(MM, phase) walker + aggregator.

    `seq` is the ordered list of `CombinedOp`s for one phase (forward
    pattern, decode pattern, training pattern). `aggregate_fn` turns the
    walker's per-op wall-time results (one list per contention snapshot)
    into a single total — that's where the PP bubble math lives,
    supplied by `ModelMapping` as a closure.
    """
    seq: "list[CombinedOp]"
    aggregate_fn: Aggregator

    def evaluate(
        self,
        ctx: "WorkloadContext",
        hw: "HardwareSpec",
        op_transfers: "dict[NetworkOp, list[LogicalTransfer]]",
        transfer_times_per_snapshot: "dict[str, dict[LogicalTransfer, float]]",
        transfer_times_boundary: "dict[LogicalTransfer, float]",
    ) -> tuple[float, "dict[str, list[float]]"]:
        """Walk `seq` once per snapshot, evaluating each `CombinedOp.wall_time`.
        Returns `(total_time, per_op_per_snapshot)` where the second is a
        dict keyed by snapshot name (same keys as the input)."""
        per_op_per_snap: dict[str, list[float]] = {
            name: [
                co.wall_time(
                    ctx, hw,
                    op_transfers, transfer_times, transfer_times_boundary,
                )
                for co in self.seq
            ]
            for name, transfer_times in transfer_times_per_snapshot.items()
        }
        total = self.aggregate_fn(per_op_per_snap)
        return total, per_op_per_snap
