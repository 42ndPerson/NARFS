from __future__ import annotations
from dataclasses import dataclass, field

from hardware import HardwareSpec
from model_mapping import SimulationResult
from stages import RLHFTimeline
from submesh_mapping import (
    SubmeshMapping,
    boundary_transfer_times,
    partition_steady_boundary,
    steady_transfer_times,
)
from topo import HostTopo
from workload_context import WorkloadContext


@dataclass
class IterationResult:
    total_time: float
    per_stage_times: list[float] = field(default_factory=list)
    per_boundary_times: list[float] = field(default_factory=list)
    # Flattened across stages and submeshes in stage-then-submesh-then-mm order.
    # Useful for validation / introspection — pull per-MM prefill/decode/training
    # times without re-running the simulation.
    per_mm_results: list[SimulationResult] = field(default_factory=list)


def simulate_stage(
    stage: list[SubmeshMapping],
    workload_ctx: WorkloadContext,
    hw: HardwareSpec,
    topo: HostTopo,
) -> float:
    """Wall time for a stage executed in parallel across submeshes.

    All submeshes share the cluster fabric, so steady (shared-bandwidth) and
    boundary (flow-time) snapshots are taken once across every NetworkOp from
    every ModelMapping in every SubmeshMapping. Each submesh then sums its
    own ModelMappings' simulate times against those snapshots, and the stage
    wall time is the slowest submesh.
    """
    if not stage:
        return 0.0

    all_mappings = [mm for sm in stage for mm in sm.model_mappings]
    # Phase 1: re-derive workload-dependent collective sizes against the
    # current ctx before any snapshot is taken. NetworkOps are constructed
    # at __post_init__ with placeholder sizes; this populates real values.
    for mm in all_mappings:
        mm.refresh_collective_sizes(workload_ctx)
    steady_ops, boundary_ops = partition_steady_boundary(all_mappings)
    steady_times = steady_transfer_times(steady_ops, topo)
    boundary_times = boundary_transfer_times(boundary_ops, topo)

    submesh_times: list[float] = []
    for sm in stage:
        sm_time = sum(
            mm.simulate(workload_ctx, hw, steady_times, boundary_times).total_time
            for mm in sm.model_mappings
        )
        submesh_times.append(sm_time)
    return max(submesh_times)


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
            # Same shape as simulate_stage but accumulates per-MM results.
            all_mappings = [mm for sm in stage for mm in sm.model_mappings]
            # Phase 1: refresh workload-dependent collective sizes per stage.
            for mm in all_mappings:
                mm.refresh_collective_sizes(workload_ctx)
            steady_ops, boundary_ops = partition_steady_boundary(all_mappings)
            steady_times = steady_transfer_times(steady_ops, topo)
            boundary_times = boundary_transfer_times(boundary_ops, topo)

            submesh_times: list[float] = []
            for sm in stage:
                sm_time = 0.0
                for mm in sm.model_mappings:
                    sim = mm.simulate(workload_ctx, hw, steady_times, boundary_times)
                    per_mm_results.append(sim)
                    sm_time += sim.total_time
                submesh_times.append(sm_time)
            stage_time = max(submesh_times)
        total += stage_time
        per_stage_times.append(stage_time)
        if i < len(timeline.boundaries):
            # All transitions in the boundary share one fabric simultaneously,
            # so contention is modeled with a single joint flow-time snapshot
            # covering every transition's transfers. StageBoundary.simulate
            # owns that scope.
            boundary_time = timeline.boundaries[i].simulate(topo, hw)
            total += boundary_time
            per_boundary_times.append(boundary_time)
    return IterationResult(
        total_time=total,
        per_stage_times=per_stage_times,
        per_boundary_times=per_boundary_times,
        per_mm_results=per_mm_results,
    )
