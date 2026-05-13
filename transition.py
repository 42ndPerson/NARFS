from __future__ import annotations
from dataclasses import dataclass, field

from hardware import HardwareSpec
from model_mapping import ModelMapping
from network_op import NetworkOp
from relation import NetworkPhase
from strategies import MegatronToMegatronP2P, ReshardingStrategy
from topo import HostTopo


@dataclass
class StageTransition:
    """Resharding hop between two ModelMappings of the same model.

    Strategy is pluggable to accommodate optimized 3D-hybrid resharding
    patterns (e.g. HybridFlow's micro-DP AllGather) — those return their own
    NetworkOp subclasses, and StageTransition consumes them uniformly.

    All resharding NetworkOps must be BOUNDARY phase (one-shot, flow-time
    contention). __post_init__ validates this invariant.
    """

    source: ModelMapping
    dest: ModelMapping
    strategy: ReshardingStrategy = field(default_factory=MegatronToMegatronP2P)
    network_ops: list[NetworkOp] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.source.model is not self.dest.model:
            raise ValueError(
                "StageTransition requires identical model on both sides; "
                f"got source={self.source.model.name!r}, dest={self.dest.model.name!r}"
            )
        ops = self.strategy.compute_network_ops(self.source, self.dest)
        for op in ops:
            phase = NetworkOp.phase_of(op)
            if phase is not NetworkPhase.BOUNDARY:
                raise ValueError(
                    f"Resharding NetworkOp {type(op).__name__} has phase {phase}; "
                    "all transition ops must be BOUNDARY"
                )
        self.network_ops = ops

    def simulate(self, topo: HostTopo, hw: HardwareSpec) -> float:
        """Wall time for the resharding hop.

        All boundary ops share a single flow-time snapshot. The transition
        wall time is the slowest op (they overlap in time on the fabric).
        """
        del hw  # network-only; kernel cost is irrelevant for resharding
        if not self.network_ops:
            return 0.0
        op_times = NetworkOp.time_simultaneous_system_flow(self.network_ops, topo)
        return max(op_times)
