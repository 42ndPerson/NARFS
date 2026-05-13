from __future__ import annotations
from dataclasses import dataclass

from hardware import HardwareSpec
from model_mapping import ModelMapping
from network_op import LogicalTransfer, NetworkOp
from relation import NetworkPhase
from topo import HostTopo
from workload_context import WorkloadContext


def resolve_op_transfers(
    workload_ctx: WorkloadContext, model_mappings: list[ModelMapping]
) -> dict[NetworkOp, list[LogicalTransfer]]:
    """Resolve each NetworkOp's current LogicalTransfers against `ctx`.

    For each (req, op) pair in every ModelMapping's `network_ops`, evaluates
    the requirement's `data_GB_fn(ctx)` to get total bytes, then asks the
    op to distribute that total across its endpoint pairs. The result is a
    map from each NetworkOp to its freshly-constructed transfer list —
    consumed both by the contention-snapshot step and by `CombinedOp.wall_time`.
    """
    out: dict[NetworkOp, list[LogicalTransfer]] = {}
    for mm in model_mappings:
        for req, op in mm.network_ops.items():
            total = float(req.data_GB_fn(workload_ctx))
            out[op] = op.logical_transfers(total)
    return out


def steady_transfer_times(
    op_transfers: dict[NetworkOp, list[LogicalTransfer]],
    topo: HostTopo,
) -> dict[LogicalTransfer, float]:
    """Per-transfer time under shared-bandwidth contention for steady ops."""
    transfers = [
        t for op, ts in op_transfers.items()
        if NetworkOp.phase_of(op) is NetworkPhase.STEADY
        for t in ts
    ]
    if not transfers:
        return {}
    return topo.get_simultaneous_logical_transfer_times(transfers)


def boundary_transfer_times(
    op_transfers: dict[NetworkOp, list[LogicalTransfer]],
    topo: HostTopo,
) -> dict[LogicalTransfer, float]:
    """Per-transfer time under flow-time contention for boundary ops."""
    transfers = [
        t for op, ts in op_transfers.items()
        if NetworkOp.phase_of(op) is NetworkPhase.BOUNDARY
        for t in ts
    ]
    if not transfers:
        return {}
    pair_times = topo.get_simultaneous_system_flow_time(transfers)
    return {
        t: pair_times.get((t.src_device, t.dst_device), 0.0) for t in transfers
    }


@dataclass
class ModelMappingGroup:
    """A set of `ModelMapping`s that share a slice of the cluster — typically
    a stage's actor/critic/reward collocated on the same devices.

    The group's device footprint is implicit in the union of the members'
    `shards_to_devices` keys; no separate mesh field is carried because
    nothing in the simulator needed to query a stand-alone mesh.
    """

    model_mappings: list[ModelMapping]

    def simulate_isolated(
        self,
        workload_ctx: WorkloadContext,
        hw: HardwareSpec,
        topo: HostTopo,
    ) -> float:
        """Approximate single-group wall time.

        Cross-group contention is silently ignored — steady/boundary
        snapshots are taken against only this group's own NetworkOps.
        Use `simulate_stage` when stage-wide context is available.
        """
        op_transfers = resolve_op_transfers(workload_ctx, self.model_mappings)
        steady_times = steady_transfer_times(op_transfers, topo)
        boundary_times = boundary_transfer_times(op_transfers, topo)
        return sum(
            mm.simulate(
                workload_ctx, hw, op_transfers, steady_times, boundary_times,
            ).total_time
            for mm in self.model_mappings
        )
