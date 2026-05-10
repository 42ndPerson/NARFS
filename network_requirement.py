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
    (TP-AR, EP-AllToAll) leave it `None` and rely on the dedup behavior
    that's been there since Phase 4a.

    `data_GB_fn` is a workload-dependent sizing closure evaluated by the
    simulator before each snapshot (see `ModelMapping.refresh_collective_sizes`).
    Excluded from equality / hash so two requirements differing only in
    sizing closure still dedupe — same-(kind, group, payload_id)
    requirements always carry the same closure semantics by construction
    (both close over the same architecture parameters).
    """

    kind: CollectiveKind
    relation: Relation
    eta: float
    group: ParallelismGroup
    payload_id: Optional[Hashable] = None
    data_GB_fn: Optional[Callable[["WorkloadContext"], float]] = field(
        default=None, compare=False, hash=False
    )


@dataclass(frozen=True)
class ParallelismConfig:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
