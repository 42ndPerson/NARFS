from __future__ import annotations
from dataclasses import dataclass

from hardware import HardwareSpec
from mesh import Mesh
from model_mapping import ModelMapping
from network_op import LogicalTransfer, NetworkOp
from relation import NetworkPhase
from topo import HostTopo
from workload_context import WorkloadContext


def partition_steady_boundary(
    model_mappings: list[ModelMapping],
) -> tuple[list[NetworkOp], list[NetworkOp]]:
    """Split each ModelMapping's NetworkOps by phase (STEADY vs BOUNDARY).

    Shared by simulate_isolated (this file) and simulate_stage (simulator.py)
    so both apply the same partitioning rule.
    """
    steady: list[NetworkOp] = []
    boundary: list[NetworkOp] = []
    for mm in model_mappings:
        for op in mm.network_ops.values():
            if NetworkOp.phase_of(op) is NetworkPhase.STEADY:
                steady.append(op)
            else:
                boundary.append(op)
    return steady, boundary


def steady_transfer_times(
    steady_ops: list[NetworkOp], topo: HostTopo
) -> dict[LogicalTransfer, float]:
    """Per-transfer time under shared-bandwidth contention.

    `topo.get_simultaneous_logical_transfer_times` is the bandwidth→time
    wrapper around `get_simultaneous_system_shared_bandwidth`, so we get
    seconds directly.
    """
    transfers = [t for op in steady_ops for t in op.logical_transfers]
    if not transfers:
        return {}
    return topo.get_simultaneous_logical_transfer_times(transfers)


def boundary_transfer_times(
    boundary_ops: list[NetworkOp], topo: HostTopo
) -> dict[LogicalTransfer, float]:
    """Per-transfer time under flow-time contention.

    `topo.get_simultaneous_system_flow_time` returns dict[(src, dst), time];
    we expand to per-LogicalTransfer for NetworkOp.get_operator_time.
    """
    transfers = [t for op in boundary_ops for t in op.logical_transfers]
    if not transfers:
        return {}
    pair_times = topo.get_simultaneous_system_flow_time(transfers)
    return {
        t: pair_times.get((t.src_device, t.dst_device), 0.0) for t in transfers
    }


@dataclass
class SubmeshMapping:
    model_mappings: list[ModelMapping]
    submesh: Mesh

    def simulate_isolated(
        self,
        workload_ctx: WorkloadContext,
        hw: HardwareSpec,
        topo: HostTopo,
    ) -> float:
        """Approximate single-submesh wall time.

        Cross-submesh contention is silently ignored — steady/boundary
        snapshots are taken against only this submesh's own NetworkOps.
        Use simulate_stage when stage-wide context is available; this method
        is intended for early-stage exploration where it isn't.

        Models in the submesh run sequentially per the existing
        SubmeshMapping definition, so the result is the sum of their times.
        """
        # Phase 1: refresh workload-dependent collective sizes before snapshot.
        for mm in self.model_mappings:
            mm.refresh_collective_sizes(workload_ctx)
        steady_ops, boundary_ops = partition_steady_boundary(self.model_mappings)
        steady_times = steady_transfer_times(steady_ops, topo)
        boundary_times = boundary_transfer_times(boundary_ops, topo)
        return sum(
            mm.simulate(workload_ctx, hw, steady_times, boundary_times).total_time
            for mm in self.model_mappings
        )
