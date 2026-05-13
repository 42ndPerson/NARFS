from __future__ import annotations
from dataclasses import dataclass, field

from hardware import HardwareSpec
from model_mapping import SimulationResult
from stages import RLHFTimeline
from model_mapping_group import (
    ModelMappingGroup,
    boundary_transfer_times,
    resolve_op_transfers,
    steady_transfer_times,
)
from topo import HostTopo
from workload_context import WorkloadContext


@dataclass
class IterationResult:
    total_time: float
    per_stage_times: list[float] = field(default_factory=list)
    per_boundary_times: list[float] = field(default_factory=list)
    # Flattened across stages and groups in stage-then-group-then-mm order.
    # Useful for validation / introspection.
    per_mm_results: list[SimulationResult] = field(default_factory=list)


def simulate_stage(
    stage: list[ModelMappingGroup],
    workload_ctx: WorkloadContext,
    hw: HardwareSpec,
    topo: HostTopo,
) -> float:
    """Wall time for a stage executed in parallel across groups.

    All groups share the cluster fabric, so steady and boundary
    snapshots are taken once across every NetworkOp from every ModelMapping
    in every ModelMappingGroup. Each group then sums its own ModelMappings'
    times against those snapshots, and the stage wall time is the slowest
    group.
    """
    if not stage:
        return 0.0

    all_mappings = [mm for g in stage for mm in g.model_mappings]
    op_transfers = resolve_op_transfers(workload_ctx, all_mappings)
    steady_times = steady_transfer_times(op_transfers, topo)
    boundary_times = boundary_transfer_times(op_transfers, topo)

    group_times: list[float] = []
    for g in stage:
        g_time = sum(
            mm.simulate(
                workload_ctx, hw, op_transfers, steady_times, boundary_times,
            ).total_time
            for mm in g.model_mappings
        )
        group_times.append(g_time)
    return max(group_times)


def simulate_rlhf_iteration(
    timeline: RLHFTimeline,
    workload_ctx: WorkloadContext,
    hw: HardwareSpec,
    topo: HostTopo,
) -> IterationResult:
    """Sum stage wall times and boundary resharding times for one RLHF iteration."""
    total = 0.0
    per_stage_times: list[float] = []
    per_boundary_times: list[float] = []
    per_mm_results: list[SimulationResult] = []

    for i, stage in enumerate(timeline.stages):
        if not stage:
            stage_time = 0.0
        else:
            all_mappings = [mm for g in stage for mm in g.model_mappings]
            op_transfers = resolve_op_transfers(workload_ctx, all_mappings)
            steady_times = steady_transfer_times(op_transfers, topo)
            boundary_times = boundary_transfer_times(op_transfers, topo)

            group_times: list[float] = []
            for g in stage:
                g_time = 0.0
                for mm in g.model_mappings:
                    sim = mm.simulate(
                        workload_ctx, hw, op_transfers, steady_times, boundary_times,
                    )
                    per_mm_results.append(sim)
                    g_time += sim.total_time
                group_times.append(g_time)
            stage_time = max(group_times)
        total += stage_time
        per_stage_times.append(stage_time)
        if i < len(timeline.boundaries):
            boundary_time = timeline.boundaries[i].simulate(topo, hw)
            total += boundary_time
            per_boundary_times.append(boundary_time)
    return IterationResult(
        total_time=total,
        per_stage_times=per_stage_times,
        per_boundary_times=per_boundary_times,
        per_mm_results=per_mm_results,
    )
