from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Optional

from parallelism import Shard
from topology.topo import Device, HostTopo

if TYPE_CHECKING:
    from mapping.model_mapping import ModelMapping


class NetworkPhase(Enum):
    STEADY = "steady"      # repeats throughout a stage; bandwidth-shared contention
    BOUNDARY = "boundary"  # one-shot at stage edges; flow-time contention


@dataclass(frozen=True, eq=False)
class NetworkOp(ABC):
    """A network operator: a structural description of one collective.

    The op carries `endpoint_pairs` — the directed (src, dst) device pairs
    its transfers traverse — but is stateless w.r.t. payload size. Callers
    invoke `logical_transfers(total_data_GB)` to materialize a fresh list
    of `LogicalTransfer`s for a given total payload, then pass that list
    (along with a transfer-times snapshot) to `get_operator_time` to
    compute the op's wall time.

    `total_data_GB` is stored on the op for cases where the size is known
    at construction (e.g. boundary resharding P2Ps with precomputed
    parameter byte counts). Requirement-driven ops leave it at 0; the
    orchestrator supplies the per-ctx total via the requirement's
    `data_GB_fn`.

    Identity equality (`eq=False`): two NetworkOp instances are distinct
    objects, deduped upstream by `NetworkRequirement` equality. This makes
    NetworkOp hashable by identity so `dict[NetworkOp, ...]` works in the
    orchestration layer.
    """

    endpoint_pairs: list[tuple[Device, Device]]
    total_data_GB: float = 0.0
    # Per-instance contention class. Defaults to the subclass's DEFAULT_PHASE
    # (resolved in __post_init__).
    phase: Optional[NetworkPhase] = field(default=None, kw_only=True)

    # Subclasses set this to declare the typical phase for the op type.
    # None means the caller MUST pass `phase=` explicitly.
    DEFAULT_PHASE: ClassVar[Optional[NetworkPhase]] = None

    def __post_init__(self) -> None:
        if self.phase is None and self.DEFAULT_PHASE is not None:
            object.__setattr__(self, "phase", self.DEFAULT_PHASE)

    @staticmethod
    def phase_of(op: NetworkOp) -> Optional[NetworkPhase]:
        return op.phase

    @abstractmethod
    def logical_transfers(self, total_data_GB: float) -> list[LogicalTransfer]:
        """Fresh per-link payload distribution for a given total. Pure."""
        raise NotImplementedError

    @abstractmethod
    def get_operator_time(
        self,
        transfers: list[LogicalTransfer],
        transfer_times: dict[LogicalTransfer, float],
    ) -> float:
        """Wall time for this collective given resolved transfers and a
        per-transfer times dict. `transfers` must be the list returned by
        `logical_transfers(total)` for the relevant total; the caller is
        responsible for passing coherent inputs.
        """
        raise NotImplementedError

    @staticmethod
    def time_simultaneous_system_flow(
        ops: list[NetworkOp], topo: HostTopo
    ) -> list[float]:
        """Per-op wall times under flow-time contention. Used at stage
        boundaries (resharding) where ops carry their fixed `total_data_GB`.
        """
        op_transfers = {op: op.logical_transfers(op.total_data_GB) for op in ops}
        all_transfers = [t for ts in op_transfers.values() for t in ts]
        if not all_transfers:
            return [0.0 for _ in ops]
        pair_times = topo.get_simultaneous_system_flow_time(all_transfers)
        transfer_times: dict[LogicalTransfer, float] = {
            t: pair_times.get((t.src_device, t.dst_device), 0.0) for t in all_transfers
        }
        return [op.get_operator_time(op_transfers[op], transfer_times) for op in ops]


# ---- per-collective subclasses ----------------------------------------------


class AllReduceRing(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.STEADY

    @classmethod
    def generate(
        cls, model_mapping: ModelMapping, shards_ring: list[Shard]
    ) -> AllReduceRing:
        """Build the (src, dst) device pairs for a ring of `shards_ring`.

        Sensitive to ring order (minor effect under tree structure, but
        major effect under highly differentiated path bandwidths).
        """
        pairs: list[tuple[Device, Device]] = []
        n = len(shards_ring)
        if n > 1:
            for i in range(n):
                next_i = (i + 1) % n
                pairs.append((
                    model_mapping.shards_to_devices[shards_ring[i]],
                    model_mapping.shards_to_devices[shards_ring[next_i]],
                ))
        return cls(endpoint_pairs=pairs)

    def logical_transfers(self, total_data_GB: float) -> list[LogicalTransfer]:
        # Ring with n hops; each hop carries total/n.
        n = len(self.endpoint_pairs)
        per = total_data_GB / n if n else 0.0
        return [
            LogicalTransfer(src_device=src, dst_device=dst, data_GB=per)
            for src, dst in self.endpoint_pairs
        ]

    def get_operator_time(
        self,
        transfers: list[LogicalTransfer],
        transfer_times: dict[LogicalTransfer, float],
    ) -> float:
        if not transfers:
            return 0.0
        n = len(transfers)
        max_time = max(transfer_times.get(t, 0.0) for t in transfers)
        # Ring all-reduce is 2*(n-1) steps, each taking the bottleneck transfer time.
        return max_time * 2 * (n - 1) if n > 0 else 0.0


class AllGatherRing(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.STEADY

    @classmethod
    def generate(
        cls, model_mapping: ModelMapping, shards_ring: list[Shard]
    ) -> AllGatherRing:
        pairs: list[tuple[Device, Device]] = []
        n = len(shards_ring)
        if n > 1:
            for i in range(n):
                next_i = (i + 1) % n
                pairs.append((
                    model_mapping.shards_to_devices[shards_ring[i]],
                    model_mapping.shards_to_devices[shards_ring[next_i]],
                ))
        return cls(endpoint_pairs=pairs)

    def logical_transfers(self, total_data_GB: float) -> list[LogicalTransfer]:
        n = len(self.endpoint_pairs)
        per = total_data_GB / n if n else 0.0
        return [
            LogicalTransfer(src_device=src, dst_device=dst, data_GB=per)
            for src, dst in self.endpoint_pairs
        ]

    def get_operator_time(
        self,
        transfers: list[LogicalTransfer],
        transfer_times: dict[LogicalTransfer, float],
    ) -> float:
        if not transfers:
            return 0.0
        n = len(transfers)
        max_time = max(transfer_times.get(t, 0.0) for t in transfers)
        # Ring all-gather takes n-1 steps, each moving data_GB/n over the ring.
        return max_time * (n - 1) if n > 0 else 0.0


class ReduceScatterRing(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.STEADY

    @classmethod
    def generate(
        cls, model_mapping: ModelMapping, shards_ring: list[Shard]
    ) -> ReduceScatterRing:
        # Same logical traffic pattern as ring all-gather.
        base = AllGatherRing.generate(model_mapping, shards_ring)
        return cls(endpoint_pairs=base.endpoint_pairs)

    def logical_transfers(self, total_data_GB: float) -> list[LogicalTransfer]:
        n = len(self.endpoint_pairs)
        per = total_data_GB / n if n else 0.0
        return [
            LogicalTransfer(src_device=src, dst_device=dst, data_GB=per)
            for src, dst in self.endpoint_pairs
        ]

    def get_operator_time(
        self,
        transfers: list[LogicalTransfer],
        transfer_times: dict[LogicalTransfer, float],
    ) -> float:
        if not transfers:
            return 0.0
        n = len(transfers)
        max_time = max(transfer_times.get(t, 0.0) for t in transfers)
        # Ring reduce-scatter takes n-1 steps.
        return max_time * (n - 1) if n > 0 else 0.0


class AllToAll(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.STEADY

    @classmethod
    def generate(
        cls, model_mapping: ModelMapping, shards: list[Shard]
    ) -> AllToAll:
        pairs: list[tuple[Device, Device]] = []
        # Every node sends to every other node (N*(N-1) directed transfers).
        for i in range(len(shards)):
            for j in range(len(shards)):
                if i != j:
                    pairs.append((
                        model_mapping.shards_to_devices[shards[i]],
                        model_mapping.shards_to_devices[shards[j]],
                    ))
        return cls(endpoint_pairs=pairs)

    def logical_transfers(self, total_data_GB: float) -> list[LogicalTransfer]:
        # AllToAll across N nodes has N*(N-1) transfers, each carrying total/N.
        nt = len(self.endpoint_pairs)
        if nt == 0:
            return []
        # N(N-1) = nt → N = (1 + sqrt(1 + 4*nt)) / 2.
        N = (1 + math.isqrt(1 + 4 * nt)) // 2
        per = total_data_GB / N if N * (N - 1) == nt and N > 0 else total_data_GB / nt
        return [
            LogicalTransfer(src_device=src, dst_device=dst, data_GB=per)
            for src, dst in self.endpoint_pairs
        ]

    def get_operator_time(
        self,
        transfers: list[LogicalTransfer],
        transfer_times: dict[LogicalTransfer, float],
    ) -> float:
        if not transfers:
            return 0.0
        # All connections happen simultaneously; op time = slowest transfer.
        return max(transfer_times.get(t, 0.0) for t in transfers)


class Broadcast(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.BOUNDARY

    @classmethod
    def generate(
        cls,
        model_mapping: ModelMapping,
        src_shard: Shard,
        dst_shards: list[Shard],
        total_data_GB: float = 0.0,
    ) -> Broadcast:
        pairs: list[tuple[Device, Device]] = []
        src_device = model_mapping.shards_to_devices[src_shard]
        for dst in dst_shards:
            if src_shard != dst:
                pairs.append((src_device, model_mapping.shards_to_devices[dst]))
        return cls(endpoint_pairs=pairs, total_data_GB=total_data_GB)

    def logical_transfers(self, total_data_GB: float) -> list[LogicalTransfer]:
        # Each receiver gets the full payload.
        return [
            LogicalTransfer(src_device=src, dst_device=dst, data_GB=total_data_GB)
            for src, dst in self.endpoint_pairs
        ]

    def get_operator_time(
        self,
        transfers: list[LogicalTransfer],
        transfer_times: dict[LogicalTransfer, float],
    ) -> float:
        if not transfers:
            return 0.0
        # Star broadcast time is dominated by the slowest link.
        return max(transfer_times.get(t, 0.0) for t in transfers)


class P2P(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.BOUNDARY

    @classmethod
    def generate(
        cls,
        model_mapping: ModelMapping,
        src_shard: Shard,
        dst_shard: Shard,
        total_data_GB: float = 0.0,
    ) -> P2P:
        return cls(
            endpoint_pairs=[(
                model_mapping.shards_to_devices[src_shard],
                model_mapping.shards_to_devices[dst_shard],
            )],
            total_data_GB=total_data_GB,
        )

    def logical_transfers(self, total_data_GB: float) -> list[LogicalTransfer]:
        return [
            LogicalTransfer(src_device=src, dst_device=dst, data_GB=total_data_GB)
            for src, dst in self.endpoint_pairs
        ]

    def get_operator_time(
        self,
        transfers: list[LogicalTransfer],
        transfer_times: dict[LogicalTransfer, float],
    ) -> float:
        if not transfers:
            return 0.0
        return max(transfer_times.get(t, 0.0) for t in transfers)


@dataclass(frozen=True)
class LogicalTransfer:
    src_device: Device
    dst_device: Device
    data_GB: float

    # Alias matching the field name `topo.HostTopo.get_simultaneous_system_flow_time`
    # uses internally. Same value as `data_GB`.
    @property
    def data_size_GB(self) -> float:
        return self.data_GB

    def get_connection_tuple(self) -> tuple[Device, Device]:
        return (self.src_device, self.dst_device)
