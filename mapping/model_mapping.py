from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from mapping.compute_model import ComputeModel
from mapping.contention import (
    Empty,
    NetworkComposition,
    NetworkLeaf,
    Parallel,
    Sequential,
)
from topology.hardware import HardwareSpec
from model import Model
from network_op import (
    AllGatherRing,
    AllReduceRing,
    AllToAll,
    LogicalTransfer,
    NetworkOp,
    P2P,
    ReduceScatterRing,
)
from network_requirement import CollectiveKind, NetworkRequirement
from operators.base import Operator
from operators.backward import derive_backward_pattern
from operators.pattern import (
    LayerTag,
    OperatorWithReqs,
    expand_pattern_with_layers,
)
from operators.optimizer import AdamOptimizerOp
from parallelism import ParallelismConfig, ParallelismGroup
from network_op import NetworkPhase
from network_requirement import Relation
from parallelism import Shard
from topology.topo import Device
from workload_context import Workload, WorkloadContext


@dataclass
class CombinedOp:
    """One position in a model's execution sequence: the operator at this
    position plus zero or more `NetworkRequirement`s for collectives that
    overlap with the op's kernel time.

    `compute_time_fn(ctx, hw)` is the per-rank kernel-time closure baked by
    the parallelism strategy at decoration time — it incorporates the role
    of this op (TP shard direction, EP sharding) so the simulator core
    stays parallelism-agnostic.

    Each requirement's resolved `NetworkOp` is held in the parallel
    `network_ops` list. Requirements provide the `combine_times` rule
    (relation + eta) and the `data_GB_fn` closure that sizes the collective
    against the current ctx. The orchestrator pre-resolves each NetworkOp's
    current transfers into `op_transfers` and passes that dict to
    `wall_time` so per-op lookup is O(1).
    """
    operator: Operator
    compute_time_fn: Callable[[WorkloadContext, HardwareSpec], float]
    network_ops: list[NetworkOp] = field(default_factory=list)
    requirements: list[NetworkRequirement] = field(default_factory=list)
    layer_tag: LayerTag = field(default_factory=LayerTag)

    def wall_time(
        self,
        ctx: WorkloadContext,
        hw: HardwareSpec,
        op_transfers: dict[NetworkOp, list[LogicalTransfer]],
        transfer_times_steady: dict[LogicalTransfer, float],
        transfer_times_boundary: dict[LogicalTransfer, float],
    ) -> float:
        wall = self.compute_time_fn(ctx, hw)
        for net_op, req in zip(self.network_ops, self.requirements):
            phase = NetworkOp.phase_of(net_op)
            tt = (
                transfer_times_steady if phase is NetworkPhase.STEADY
                else transfer_times_boundary
            )
            transfers = op_transfers.get(net_op, [])
            net_time = net_op.get_operator_time(transfers, tt)
            wall = req.combine_times(wall, net_time)
        return wall


@dataclass
class SimulationResult:
    total_time: float
    prefill_time: Optional[float] = None
    decode_time: Optional[float] = None
    training_time: Optional[float] = None
    pp_stage_times: list[float] = field(default_factory=list)
    per_op_breakdown: list[float] = field(default_factory=list)


# Reference HardwareSpec used solely by the layer-aware PP partitioner to
# compute layer cost ratios. The partition is invariant under uniform
# rescaling of compute/memory, so the absolute numbers don't matter — only
# the ratio compute_flops/memory_bytes does.
_PARTITION_REF_HW = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
)


def _partition_layers_by_cost(layer_costs: list[float], num_stages: int) -> list[list[int]]:
    """Contiguous-split layer indices into `num_stages` groups, minimizing the
    maximum group cost (linear-time partition / "painter's partition" DP).
    """
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    n = len(layer_costs)
    if n == 0:
        return [[] for _ in range(num_stages)]
    if num_stages >= n:
        return [[i] if i < n else [] for i in range(num_stages)]

    prefix = [0.0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + layer_costs[i]

    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(num_stages + 1)]
    cut = [[0] * (n + 1) for _ in range(num_stages + 1)]
    dp[0][0] = 0.0
    for s in range(1, num_stages + 1):
        for i in range(1, n + 1):
            for j in range(s - 1, i):
                stage_cost = prefix[i] - prefix[j]
                cost = max(dp[s - 1][j], stage_cost)
                if cost < dp[s][i]:
                    dp[s][i] = cost
                    cut[s][i] = j

    partition: list[list[int]] = [[] for _ in range(num_stages)]
    i = n
    for s in range(num_stages, 0, -1):
        j = cut[s][i]
        partition[s - 1] = list(range(j, i))
        i = j
    return partition


@dataclass
class ModelMapping:
    """One model on one parallelism / workload assignment.

    `__post_init__` builds the model's execution sequence as a list of
    `CombinedOp`s — one per operator position, with each position carrying
    its overlapping NetworkOps + requirements. `simulate(...)` walks the
    sequence and sums per-position wall times.

    For GENERATION, the prefill and decode patterns are built independently
    via `model.forward_sequence(phase)`; each phase's sequence
    has its own pipeline partition. For TRAINING, the forward sequence is
    augmented with derived backward and per-LayerTag Adam optimizer ops.
    """

    model: Model
    workload: Workload
    parallelism: ParallelismConfig
    shards_to_devices: dict[Shard, Device]
    device_to_shard: dict[Device, Shard]

    # Pre-decorated patterns from the strategy. `forward_pattern` is the
    # primary sequence (used for PREPARATION/TRAINING and for GENERATION's
    # prefill leg). `decode_pattern` is GENERATION's decode leg.
    forward_pattern: list = field(default_factory=list)
    decode_pattern: list = field(default_factory=list)

    # Training option: when True, BackwardOp uses 3× forward (full
    # activation recomputation). When False, 2× forward (selective
    # checkpointing — closer to what Megatron-class training actually
    # achieves). Only consulted for `Workload.TRAINING`.
    recompute: bool = False

    # Computed in __post_init__:
    forward_seq: list[CombinedOp] = field(init=False, default_factory=list)
    decode_seq: list[CombinedOp] = field(init=False, default_factory=list)
    pipeline_partition: list[list[int]] = field(init=False, default_factory=list)
    decode_pipeline_partition: list[list[int]] = field(init=False, default_factory=list)
    network_ops: dict[NetworkRequirement, NetworkOp] = field(init=False, default_factory=dict)
    # Network composition for one pipeline slot's worth of activity (or
    # one full forward pass when pp=1). Built from forward_seq +
    # pipeline_partition + pp_p2p_op_*. Consumed by `ContentionModel` at
    # the stage level (composed up through MMG and stage `Parallel`).
    #
    # Two snapshots: `network_composition` for steady state (all PP
    # stages concurrent); `network_composition_fill_drain` for the fill
    # and drain phases (only one stage active at a time, modeled by
    # wrapping per-stage Sequentials in an outer `Sequential` rather
    # than the steady `Parallel`). For pp=1 the two are identical.
    network_composition: NetworkComposition = field(
        init=False, default_factory=Sequential,
    )
    network_composition_fill_drain: NetworkComposition = field(
        init=False, default_factory=Sequential,
    )
    # Indices of optimizer ops within forward_seq (TRAINING only). These
    # are excluded from the per-slot bubble formula — the optimizer runs
    # once per iteration, concurrently across PP stages, not once per
    # microbatch slot. Added back as a one-shot `max(stage_optimizer_times)`.
    optimizer_indices: set[int] = field(init=False, default_factory=set)
    # PP activation P2P NetworkOps. Forward fires every microbatch transition;
    # backward only in TRAINING. None when pp == 1.
    pp_p2p_op_forward: Optional[NetworkOp] = field(init=False, default=None)
    pp_p2p_op_backward: Optional[NetworkOp] = field(init=False, default=None)
    pp_p2p_req_forward: Optional[NetworkRequirement] = field(init=False, default=None)
    pp_p2p_req_backward: Optional[NetworkRequirement] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._validate_axis_consistency()
        if self.workload is Workload.GENERATION:
            self.forward_seq, self.pipeline_partition = self._build_seq_from_pattern(self.forward_pattern)
            self.decode_seq, self.decode_pipeline_partition = self._build_seq_from_pattern(self.decode_pattern)
        elif self.workload is Workload.PREPARATION:
            self.forward_seq, self.pipeline_partition = self._build_seq_from_pattern(self.forward_pattern)
        elif self.workload is Workload.TRAINING:
            self._build_training_seq_from_pattern()
        elif self.workload is Workload.DORMANT:
            pass
        else:
            raise ValueError(f"Unknown workload: {self.workload!r}")

        if self.parallelism.pp > 1 and self.workload is not Workload.DORMANT:
            self._install_pp_p2p_ops()

        # Build network compositions AFTER seqs + pp-p2p ops are in place.
        # Only the forward composition is materialized; the simulator reuses
        # the same `transfer_times_steady` snapshot for the decode leg
        # (decode runs on the same MM and the per-link load picture is
        # close enough that a second snapshot would just duplicate work).
        if self.workload is not Workload.DORMANT:
            self.network_composition = self._build_network_composition(
                self.forward_seq, self.pipeline_partition,
            )
            self.network_composition_fill_drain = self._build_network_composition_fill_drain(
                self.forward_seq, self.pipeline_partition,
            )

    # ----- legacy property surface -------------------------------------------

    @property
    def operator_pattern(self) -> list[Operator]:
        return [co.operator for co in self.forward_seq]

    @property
    def per_op_network_requirements(self) -> list[list[NetworkRequirement]]:
        return [co.requirements for co in self.forward_seq]

    @property
    def per_op_layer_tags(self) -> list[LayerTag]:
        return [co.layer_tag for co in self.forward_seq]

    @property
    def decode_operator_pattern(self) -> list[Operator]:
        return [co.operator for co in self.decode_seq]

    @property
    def decode_per_op_network_requirements(self) -> list[list[NetworkRequirement]]:
        return [co.requirements for co in self.decode_seq]

    @property
    def decode_per_op_layer_tags(self) -> list[LayerTag]:
        return [co.layer_tag for co in self.decode_seq]

    # ----- pattern build / install -------------------------------------------

    def _build_seq_from_pattern(
        self, pattern: list
    ) -> tuple[list[CombinedOp], list[list[int]]]:
        """Consume a strategy-decorated pattern, expand it, attach layer
        tags, and turn it into a CombinedOp sequence + PP partition."""
        expanded = expand_pattern_with_layers(pattern)
        seq = self._expanded_to_seq(expanded)
        partition = self._layer_aware_partition(seq)
        return seq, partition

    def _build_training_seq_from_pattern(self) -> None:
        """Training: take the strategy-decorated forward pattern, derive
        backward + per-layer Adam, then build the CombinedOp sequence.

        Adam is sized from the strategy's pre-computed
        `per_rank_parameter_bytes` on the forward decoration — the operators
        themselves are parallelism-pure and only know full-model footprints.
        """
        forward_pattern = self.forward_pattern
        backward_pattern = derive_backward_pattern(forward_pattern, recompute=self.recompute)

        forward_expanded = expand_pattern_with_layers(forward_pattern)
        backward_expanded = expand_pattern_with_layers(backward_pattern)

        params_by_tag: dict[LayerTag, int] = defaultdict(int)
        for member, tag in forward_expanded:
            params_by_tag[tag] += member.per_rank_parameter_bytes

        dtype_bytes_param = 2
        optimizer_expanded: list[tuple[OperatorWithReqs, LayerTag]] = []
        for tag, total_param_bytes in params_by_tag.items():
            if total_param_bytes <= 0:
                continue
            num_params = total_param_bytes // dtype_bytes_param
            opt = AdamOptimizerOp(num_parameters=num_params, dtype_bytes_param=dtype_bytes_param)
            entry = OperatorWithReqs(
                operator=opt,
                requirements=[],
                compute_time_fn=lambda ctx, hw, _o=opt: _o.kernel_time(ctx, hw),
                per_rank_parameter_bytes=0,
            )
            optimizer_expanded.append((entry, tag))

        full = forward_expanded + backward_expanded + optimizer_expanded
        self.forward_seq = self._expanded_to_seq(full)
        self.pipeline_partition = self._layer_aware_partition(self.forward_seq)
        # Optimizer ops occupy the tail of `full` (after forward+backward).
        # Record their indices so `_simulate_seq` can exclude them from the
        # per-slot bubble multiplier and add them once as a max over stages.
        n_fb = len(forward_expanded) + len(backward_expanded)
        self.optimizer_indices = set(range(n_fb, n_fb + len(optimizer_expanded)))

    def _expanded_to_seq(
        self,
        expanded: list[tuple[OperatorWithReqs, LayerTag]],
    ) -> list[CombinedOp]:
        seq: list[CombinedOp] = []
        for member, tag in expanded:
            net_ops: list[NetworkOp] = []
            for req in member.requirements:
                if req not in self.network_ops:
                    self.network_ops[req] = self._resolve_network_op(req)
                net_ops.append(self.network_ops[req])
            seq.append(CombinedOp(
                operator=member.operator,
                compute_time_fn=member.compute_time_fn,
                network_ops=net_ops,
                requirements=list(member.requirements),
                layer_tag=tag,
            ))
        return seq

    # ----- network composition (for ContentionModel) -------------------------

    @staticmethod
    def _co_to_composition(co: CombinedOp) -> NetworkComposition:
        if not co.network_ops:
            return Empty(1)
        if len(co.network_ops) == 1:
            return NetworkLeaf(co.network_ops[0])
        # Multiple ops at one position genuinely co-occur.
        return Parallel([NetworkLeaf(op) for op in co.network_ops])

    def _stage_to_composition(
        self,
        seq: list[CombinedOp],
        indices: list[int],
    ) -> NetworkComposition:
        """One PP stage's sequential network composition (or the whole
        forward seq when pp=1). Compute-only runs collapse into Empty(N)."""
        children: list[NetworkComposition] = []
        empty_run = 0
        for i in indices:
            comp = self._co_to_composition(seq[i])
            if isinstance(comp, Empty):
                empty_run += comp.length
            else:
                if empty_run > 0:
                    children.append(Empty(empty_run))
                    empty_run = 0
                children.append(comp)
        if empty_run > 0:
            children.append(Empty(empty_run))
        return Sequential(children)

    def _build_network_composition(
        self,
        seq: list[CombinedOp],
        partition: list[list[int]],
    ) -> NetworkComposition:
        """Steady-state composition: one pipeline slot's worth of network
        activity (or one full forward pass when pp=1).

        pp=1 → Sequential over every CombinedOp position: a NetworkLeaf
        per collective, Empty for compute-only positions.

        pp>1 → Parallel over per-stage Sequentials plus the PP-P2P
        leaves. At pipeline steady state every stage is processing some
        microbatch concurrently, and the per-slot P2Ps fire every slot,
        so all live as siblings under one Parallel.
        """
        if not seq:
            return Sequential([])
        if self.parallelism.pp <= 1:
            return self._stage_to_composition(seq, list(range(len(seq))))

        stage_compositions = [
            self._stage_to_composition(seq, indices)
            for indices in partition if indices
        ]
        parallel_children: list[NetworkComposition] = list(stage_compositions)
        if self.pp_p2p_op_forward is not None:
            parallel_children.append(NetworkLeaf(self.pp_p2p_op_forward))
        if self.pp_p2p_op_backward is not None:
            parallel_children.append(NetworkLeaf(self.pp_p2p_op_backward))
        return Parallel(parallel_children)

    def _build_network_composition_fill_drain(
        self,
        seq: list[CombinedOp],
        partition: list[list[int]],
    ) -> NetworkComposition:
        """Fill/drain composition: only one PP stage active at a time.

        Identical to the steady composition for pp=1 (no fill/drain phase).
        For pp>1, the per-stage Sequentials are wrapped in an outer
        `Sequential` instead of `Parallel`, so each stage's ops see no
        background from other stages — modeling the fill/drain phase
        where the pipeline isn't fully loaded. P2P remains a `Parallel`
        sibling (it fires every slot).

        This is the simpler bound of "one stage active." The weighted
        aggregator combines per-op times from this snapshot with the
        steady snapshot, weighted by slot count
        (`fill_drain_slots = 2(pp − 1)`, `steady_slots = nmb − pp + 1`).
        """
        if not seq:
            return Sequential([])
        if self.parallelism.pp <= 1:
            return self._stage_to_composition(seq, list(range(len(seq))))

        stage_compositions = [
            self._stage_to_composition(seq, indices)
            for indices in partition if indices
        ]
        parallel_children: list[NetworkComposition] = [Sequential(stage_compositions)]
        if self.pp_p2p_op_forward is not None:
            parallel_children.append(NetworkLeaf(self.pp_p2p_op_forward))
        if self.pp_p2p_op_backward is not None:
            parallel_children.append(NetworkLeaf(self.pp_p2p_op_backward))
        return Parallel(parallel_children)

    # ----- layer-aware pipeline partition ------------------------------------

    def _reference_partition_ctx(self) -> WorkloadContext:
        return WorkloadContext(
            workload_type=self.workload,
            batch_size=1,
            microbatch_size=1,
            prompt_len=2048,
            response_len=128 if self.workload is Workload.GENERATION else 0,
            num_microbatches=1,
        )

    def _layer_aware_partition(self, seq: list[CombinedOp]) -> list[list[int]]:
        pp = max(self.parallelism.pp, 1)
        n = len(seq)
        if pp == 1 or n == 0:
            return [list(range(n))] + [[] for _ in range(pp - 1)]

        indices_by_layer: dict[int, list[int]] = defaultdict(list)
        first_anchor: list[int] = []
        last_anchor: list[int] = []
        for i, co in enumerate(seq):
            tag = co.layer_tag
            if tag.layer_index is not None:
                indices_by_layer[tag.layer_index].append(i)
            elif tag.anchor == "last":
                last_anchor.append(i)
            else:
                first_anchor.append(i)

        layer_indices_sorted = sorted(indices_by_layer.keys())
        if not layer_indices_sorted:
            stage0 = sorted(first_anchor + last_anchor)
            return [stage0] + [[] for _ in range(pp - 1)]

        ref_ctx = self._reference_partition_ctx()
        layer_costs: list[float] = []
        for lidx in layer_indices_sorted:
            cost = sum(
                seq[i].compute_time_fn(ref_ctx, _PARTITION_REF_HW)
                for i in indices_by_layer[lidx]
            )
            layer_costs.append(cost)

        layer_partition = _partition_layers_by_cost(layer_costs, pp)

        out: list[list[int]] = [[] for _ in range(pp)]
        for stage_idx, layer_positions in enumerate(layer_partition):
            for layer_pos in layer_positions:
                lidx = layer_indices_sorted[layer_pos]
                out[stage_idx].extend(sorted(indices_by_layer[lidx]))

        out[0] = sorted(first_anchor) + out[0]
        out[-1] = out[-1] + sorted(last_anchor)
        return out

    # ----- network requirement resolution ------------------------------------

    def _shards_in_group(self, group: ParallelismGroup) -> list[Shard]:
        all_shards = list(self.shards_to_devices.keys())
        if not all_shards:
            return []
        ref = all_shards[0]
        if group is ParallelismGroup.TP:
            return sorted(
                (s for s in all_shards
                 if s.dp == ref.dp and s.pp == ref.pp and s.ep == ref.ep),
                key=lambda s: s.tp,
            )
        if group is ParallelismGroup.EP:
            return sorted(
                (s for s in all_shards
                 if s.dp == ref.dp and s.pp == ref.pp and s.tp == ref.tp),
                key=lambda s: s.ep,
            )
        if group is ParallelismGroup.PP:
            return sorted(
                (s for s in all_shards
                 if s.dp == ref.dp and s.tp == ref.tp and s.ep == ref.ep),
                key=lambda s: s.pp,
            )
        if group is ParallelismGroup.DP:
            return sorted(
                (s for s in all_shards
                 if s.pp == ref.pp and s.tp == ref.tp and s.ep == ref.ep),
                key=lambda s: s.dp,
            )
        raise ValueError(f"Unknown ParallelismGroup: {group!r}")

    def _validate_axis_consistency(self) -> None:
        for group, expected in (
            (ParallelismGroup.TP, self.parallelism.tp),
            (ParallelismGroup.EP, self.parallelism.ep),
            (ParallelismGroup.PP, self.parallelism.pp),
            (ParallelismGroup.DP, self.parallelism.dp),
        ):
            if expected <= 1:
                continue
            actual = len(self._shards_in_group(group))
            if actual < expected:
                raise ValueError(
                    f"ModelMapping: shard layout has {actual} shard(s) along "
                    f"{group.name} (with other axes fixed at the reference "
                    f"shard) but parallelism.{group.name.lower()}={expected}. "
                    f"Construct shards covering all {expected} ranks or lower "
                    f"the parallelism degree."
                )

    def _install_pp_p2p_ops(self) -> None:
        """Build the per-microbatch PP activation P2P ops + their requirements.

        Forward leg fires every microbatch (any non-dormant workload);
        backward leg only in TRAINING. Sized to microbatch-prefill activation
        (`microbatch_size * prompt_len * h * dtype`).

        Phase override is `STEADY`: P2P normally defaults to BOUNDARY (one-
        shot pipeline send/recv at handover boundaries). Here it streams
        per-microbatch through the same bandwidth pool as compute-phase
        steady-state collectives.
        """
        pp_shards = self._shards_in_group(ParallelismGroup.PP)
        if len(pp_shards) < 2:
            return
        src, dst = pp_shards[0], pp_shards[1]
        arch = self.model.architecture
        h = getattr(arch, "h", None)
        dtype = getattr(arch, "dtype_bytes", None)
        if h is None or dtype is None:
            return

        def _activation_size_fn(ctx: WorkloadContext) -> float:
            T = ctx.microbatch_size * max(ctx.prompt_len, 1)
            return T * h * dtype / 1e9

        fwd_req = NetworkRequirement(
            kind=CollectiveKind.P2P_PIPELINE_FORWARD,
            relation=Relation.EXCLUSIVE,
            eta=1.0,
            group=ParallelismGroup.PP,
            payload_id=("pp_p2p_forward",),
            data_GB_fn=_activation_size_fn,
        )
        fwd_op = P2P(
            endpoint_pairs=[(
                self.shards_to_devices[src],
                self.shards_to_devices[dst],
            )],
            phase=NetworkPhase.STEADY,
        )
        self.network_ops[fwd_req] = fwd_op
        self.pp_p2p_op_forward = fwd_op
        self.pp_p2p_req_forward = fwd_req

        if self.workload is Workload.TRAINING:
            bwd_req = NetworkRequirement(
                kind=CollectiveKind.P2P_PIPELINE_BACKWARD,
                relation=Relation.EXCLUSIVE,
                eta=1.0,
                group=ParallelismGroup.PP,
                payload_id=("pp_p2p_backward",),
                data_GB_fn=_activation_size_fn,
            )
            bwd_op = P2P(
                endpoint_pairs=[(
                    self.shards_to_devices[dst],
                    self.shards_to_devices[src],
                )],
                phase=NetworkPhase.STEADY,
            )
            self.network_ops[bwd_req] = bwd_op
            self.pp_p2p_op_backward = bwd_op
            self.pp_p2p_req_backward = bwd_req

    def _resolve_network_op(self, req: NetworkRequirement) -> NetworkOp:
        """Concrete NetworkOp from a declarative requirement.

        Builds the structural (src, dst) pairs only — payload sizing is
        deferred to `op.logical_transfers(total)` at simulate time.
        """
        shards = self._shards_in_group(req.group)
        kind = req.kind
        if kind is CollectiveKind.ALL_REDUCE:
            return AllReduceRing.generate(self, shards)
        if kind is CollectiveKind.ALL_GATHER:
            return AllGatherRing.generate(self, shards)
        if kind is CollectiveKind.REDUCE_SCATTER:
            return ReduceScatterRing.generate(self, shards)
        if kind in (CollectiveKind.ALL_TO_ALL_DISPATCH, CollectiveKind.ALL_TO_ALL_COMBINE):
            return AllToAll.generate(self, shards)
        if kind in (CollectiveKind.P2P_PIPELINE_FORWARD, CollectiveKind.P2P_PIPELINE_BACKWARD):
            if len(shards) < 2:
                return P2P(endpoint_pairs=[])
            return P2P.generate(self, shards[0], shards[1])
        raise ValueError(f"Unknown CollectiveKind: {kind!r}")

    # ----- simulation --------------------------------------------------------

    def simulate(
        self,
        workload_ctx: WorkloadContext,
        hw: HardwareSpec,
        op_transfers: dict[NetworkOp, list[LogicalTransfer]],
        transfer_times_steady: dict[LogicalTransfer, float],
        transfer_times_boundary: dict[LogicalTransfer, float],
        transfer_times_fill_drain: Optional[dict[LogicalTransfer, float]] = None,
    ) -> SimulationResult:
        """Per-MM simulation. When `transfer_times_fill_drain` is provided
        (and the MM is PP), the bubble formula uses weighted two-snapshot
        contention: `steady_slots × max_stage_steady` for the steady
        portion plus `fill_drain_slots × max_stage_fill_drain` for the
        fill+drain portion. When `None`, the steady snapshot is used for
        all slots (legacy single-snapshot behavior)."""
        if self.workload is Workload.GENERATION:
            prefill = self._simulate_seq(
                self.forward_seq, self.pipeline_partition,
                workload_ctx, hw,
                op_transfers, transfer_times_steady, transfer_times_boundary,
                transfer_times_fill_drain=transfer_times_fill_drain,
                pipelined=True,
            )
            decode_step = self._simulate_seq(
                self.decode_seq, self.decode_pipeline_partition,
                workload_ctx, hw,
                op_transfers, transfer_times_steady, transfer_times_boundary,
                transfer_times_fill_drain=transfer_times_fill_drain,
                pipelined=False,
            )
            # Per-token decode-step time plus the configured framework /
            # kernel-launch overhead (constant the roofline doesn't model).
            decode_overhead_s = hw.decode_overhead_per_token_ms / 1000.0
            decode_time = (
                (decode_step.total_time + decode_overhead_s)
                * max(workload_ctx.response_len, 0)
            )
            return SimulationResult(
                total_time=prefill.total_time + decode_time,
                prefill_time=prefill.total_time,
                decode_time=decode_time,
                pp_stage_times=prefill.pp_stage_times,
                per_op_breakdown=prefill.per_op_breakdown + decode_step.per_op_breakdown,
            )

        if self.workload is Workload.DORMANT:
            return SimulationResult(total_time=0.0, pp_stage_times=[0.0])

        result = self._simulate_seq(
            self.forward_seq, self.pipeline_partition,
            workload_ctx, hw,
            op_transfers, transfer_times_steady, transfer_times_boundary,
            transfer_times_fill_drain=transfer_times_fill_drain,
            pipelined=True,
        )
        if self.workload is Workload.TRAINING:
            return SimulationResult(
                total_time=result.total_time,
                training_time=result.total_time,
                pp_stage_times=result.pp_stage_times,
                per_op_breakdown=result.per_op_breakdown,
            )
        return result

    def _simulate_seq(
        self,
        seq: list[CombinedOp],
        partition: list[list[int]],
        workload_ctx: WorkloadContext,
        hw: HardwareSpec,
        op_transfers: dict[NetworkOp, list[LogicalTransfer]],
        transfer_times_steady: dict[LogicalTransfer, float],
        transfer_times_boundary: dict[LogicalTransfer, float],
        pipelined: bool,
        transfer_times_fill_drain: Optional[dict[LogicalTransfer, float]] = None,
    ) -> SimulationResult:
        """Walk `seq` via `ComputeModel` (once per provided snapshot) and
        aggregate with a partition/bubble-aware closure. The walk is
        parallelism-oblivious; the closure owns the PP math.

        When `transfer_times_fill_drain` is provided AND the MM is PP, the
        bubble formula uses the weighted two-snapshot form:

            total = steady_slots × (max_stage_steady + p2p_steady)
                  + fill_drain_slots × (max_stage_fill_drain + p2p_fill_drain)

        where `steady_slots = max(0, nmb − pp + 1)` and
        `fill_drain_slots = 2 × (pp − 1)`. Each snapshot drives both its
        own per-op times and its own P2P time.

        When `transfer_times_fill_drain` is None (or pp ≤ 1), the steady
        snapshot is used for all slots (legacy single-snapshot behavior).
        """
        if not seq:
            return SimulationResult(total_time=0.0, pp_stage_times=[0.0])

        pp = self.parallelism.pp
        nmb = max(workload_ctx.num_microbatches, 1)
        is_pipelined = pipelined and pp > 1
        use_two_snapshot = transfer_times_fill_drain is not None and pp > 1

        # Per-slot PP-P2P time, sized against a given steady-class snapshot.
        def _p2p_under(transfer_times: dict[LogicalTransfer, float]) -> float:
            p2p = 0.0
            if self.pp_p2p_op_forward is not None:
                transfers = op_transfers.get(self.pp_p2p_op_forward, [])
                p2p += self.pp_p2p_op_forward.get_operator_time(
                    transfers, transfer_times,
                )
            if self.pp_p2p_op_backward is not None:
                transfers = op_transfers.get(self.pp_p2p_op_backward, [])
                p2p += self.pp_p2p_op_backward.get_operator_time(
                    transfers, transfer_times,
                )
            return p2p

        p2p_steady = _p2p_under(transfer_times_steady) if pp > 1 else 0.0
        if use_two_snapshot and transfer_times_fill_drain is not None:
            p2p_fill_drain = _p2p_under(transfer_times_fill_drain)
        else:
            p2p_fill_drain = p2p_steady

        # Optimizer indices (if any): excluded from the per-slot bubble
        # cost and added once as a max over stages (optimizer runs once
        # per iteration, in parallel across PP stages).
        opt_idx = self.optimizer_indices

        def _stage_sum(per_op: list[float], stage: list[int], exclude: set[int]) -> float:
            if not exclude:
                return sum(per_op[i] for i in stage)
            return sum(per_op[i] for i in stage if i not in exclude)

        def aggregate(per_op_per_snap: dict[str, list[float]]) -> float:
            per_op_steady = per_op_per_snap["steady"]
            # Forward+backward stage times (excluding optimizer).
            stage_times_steady = [
                _stage_sum(per_op_steady, stage, opt_idx) for stage in partition
            ]
            # Optimizer contribution per stage; runs once per iteration,
            # concurrently across PP stages → max, not sum or multiplied.
            optimizer_stage_times = [
                sum(per_op_steady[i] for i in stage if i in opt_idx)
                for stage in partition
            ]
            optimizer_total = max(optimizer_stage_times, default=0.0)

            if is_pipelined:
                max_stage_steady = max(stage_times_steady)
                if use_two_snapshot:
                    per_op_fd = per_op_per_snap["fill_drain"]
                    stage_times_fd = [
                        _stage_sum(per_op_fd, stage, opt_idx) for stage in partition
                    ]
                    max_stage_fd = max(stage_times_fd)
                    steady_slots = max(0, nmb - pp + 1)
                    fill_drain_slots = 2 * (pp - 1)
                    bubble = (
                        steady_slots * (max_stage_steady + p2p_steady)
                        + fill_drain_slots * (max_stage_fd + p2p_fill_drain)
                    )
                else:
                    bubble = (nmb + pp - 1) * (max_stage_steady + p2p_steady)
                return bubble + optimizer_total
            # Unpipelined: stages run one after another per invocation.
            # Under PP this includes (pp-1) hand-offs per pass.
            return (
                sum(stage_times_steady) + (pp - 1) * p2p_steady + optimizer_total
            )

        snapshots: dict[str, dict[LogicalTransfer, float]] = {"steady": transfer_times_steady}
        if use_two_snapshot:
            snapshots["fill_drain"] = transfer_times_fill_drain  # type: ignore[assignment]

        cm = ComputeModel(seq=seq, aggregate_fn=aggregate)
        total, per_op_per_snap = cm.evaluate(
            workload_ctx, hw,
            op_transfers, snapshots, transfer_times_boundary,
        )
        per_op_steady = per_op_per_snap["steady"]
        stage_times = [sum(per_op_steady[i] for i in stage) for stage in partition]
        return SimulationResult(
            total_time=total,
            pp_stage_times=stage_times,
            per_op_breakdown=per_op_steady,
        )
