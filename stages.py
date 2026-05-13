from __future__ import annotations
from dataclasses import dataclass, field

from hardware import HardwareSpec
from network_op import NetworkOp
from model_mapping_group import ModelMappingGroup
from topo import HostTopo
from transition import StageTransition


@dataclass
class StageBoundary:
    """Resharding hops that fire between two consecutive stages.

    `transitions` is empty if no model needs to move (e.g. consecutive stages
    use identical ModelMappings for every shared model). When populated, all
    transitions execute concurrently on the same fabric, so contention is
    modeled with a single joint flow-time snapshot covering every transition's
    transfers — the boundary wall time is the slowest op under that joint
    contention.
    """

    transitions: list[StageTransition] = field(default_factory=list)

    def simulate(self, topo: HostTopo, hw: HardwareSpec) -> float:
        del hw  # network-only; kernel cost is irrelevant for resharding
        all_ops = [op for t in self.transitions for op in t.network_ops]
        if not all_ops:
            return 0.0
        op_times = NetworkOp.time_simultaneous_system_flow(all_ops, topo)
        return max(op_times)


@dataclass
class RLHFTimeline:
    """Sequence of stages with the resharding gaps between them.

    `len(boundaries) == len(stages) - 1`. The setup function (above the
    simulator) examines each consecutive (prev_stage, next_stage) pair and
    emits a StageTransition for every model that exists on both sides with
    different ModelMappings.
    """

    stages: list[list[ModelMappingGroup]]
    boundaries: list[StageBoundary] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected = max(len(self.stages) - 1, 0)
        if len(self.boundaries) != expected:
            raise ValueError(
                f"RLHFTimeline expects {expected} boundaries for "
                f"{len(self.stages)} stages, got {len(self.boundaries)}"
            )
