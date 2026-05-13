from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Hashable, Optional

from relation import Relation

if TYPE_CHECKING:
    from workload_context import WorkloadContext


class CollectiveKind(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL_DISPATCH = "all_to_all_dispatch"
    ALL_TO_ALL_COMBINE = "all_to_all_combine"
    P2P_PIPELINE_FORWARD = "p2p_pipeline_forward"
    P2P_PIPELINE_BACKWARD = "p2p_pipeline_backward"


class ParallelismGroup(Enum):
    TP = "tp"
    EP = "ep"
    PP = "pp"
    DP = "dp"


@dataclass(frozen=True)
class NetworkRequirement:
    """A declarative communication requirement an operator emits.

    Resolved at ModelMapping `__post_init__` to a concrete `NetworkOp` over
    the shard set spanning `group`. Two requirements with identical `(kind,
    relation, eta, group, payload_id)` share the same NetworkOp instance
    (deduped by equality).

    `payload_id` is an optional discriminator for cases where the
    same-shape collective fires multiple times with *different* sizing —
    most notably DP gradient AllReduce, where each layer's AR carries its
    own parameter byte count. Phases that emit uniformly-sized collectives
    (TP-AR, EP-AllToAll) leave it `None` and rely on the dedup behavior.

    `data_GB_fn` is a workload-dependent sizing closure evaluated at every
    contention snapshot. It captures architecture parameters at build time
    and consumes only the runtime ctx. Required: every NetworkRequirement
    must supply one (use `lambda ctx: <constant>` for static sizes).
    Excluded from equality / hash so two requirements differing only in
    sizing closure still dedupe — same-(kind, group, payload_id)
    requirements always carry the same closure semantics by construction.

    `combine_times` produces the per-op wall time when this requirement's
    collective overlaps with an operator's kernel time. EXCLUSIVE adds the
    two times; CONCURRENT interpolates between max(kernel, network) and
    kernel+network based on `eta` (1.0 = perfect overlap, 0.0 = no overlap).
    """

    kind: CollectiveKind
    relation: Relation
    eta: float
    group: ParallelismGroup
    data_GB_fn: Callable[["WorkloadContext"], float] = field(
        compare=False, hash=False, kw_only=True,
    )
    payload_id: Optional[Hashable] = None

    def combine_times(self, kernel_time: float, network_time: float) -> float:
        if self.relation is Relation.EXCLUSIVE:
            return kernel_time + network_time
        return (
            (1.0 - self.eta) * (kernel_time + network_time)
            + self.eta * max(kernel_time, network_time)
        )


@dataclass(frozen=True)
class ParallelismConfig:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
