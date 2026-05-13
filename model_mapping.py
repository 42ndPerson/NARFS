from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from hardware import HardwareSpec
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
from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operator_base import Operator
from operators.backward import derive_backward_pattern
from operators.pattern import (
    LayerTag,
    OperatorWithReqs,
    expand_pattern_with_layers,
)
from operators.optimizer import AdamOptimizerOp
from relation import NetworkPhase, Relation
from shard import Shard
from topo import Device
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
    ridge_flops_per_byte=156.0,
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
    via `model.forward_sequence(phase, parallelism)`; each phase's sequence
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

    # Computed in __post_init__:
    forward_seq: list[CombinedOp] = field(init=False, default_factory=list)
    decode_seq: list[CombinedOp] = field(init=False, default_factory=list)
    pipeline_partition: list[list[int]] = field(init=False, default_factory=list)
    decode_pipeline_partition: list[list[int]] = field(init=False, default_factory=list)
    network_ops: dict[NetworkRequirement, NetworkOp] = field(init=False, default_factory=dict)
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
        backward_pattern = derive_backward_pattern(forward_pattern, recompute=True)

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
    ) -> SimulationResult:
        if self.workload is Workload.GENERATION:
            prefill = self._simulate_seq(
                self.forward_seq, self.pipeline_partition,
                workload_ctx, hw,
                op_transfers, transfer_times_steady, transfer_times_boundary,
                pipelined=True,
            )
            decode_step = self._simulate_seq(
                self.decode_seq, self.decode_pipeline_partition,
                workload_ctx, hw,
                op_transfers, transfer_times_steady, transfer_times_boundary,
                pipelined=False,
            )
            decode_time = decode_step.total_time * max(workload_ctx.response_len, 0)
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
    ) -> SimulationResult:
        if not seq:
            return SimulationResult(total_time=0.0, pp_stage_times=[0.0])

        per_op = [
            co.wall_time(
                workload_ctx, hw,
                op_transfers, transfer_times_steady, transfer_times_boundary,
            )
            for co in seq
        ]
        stage_times = [sum(per_op[i] for i in stage) for stage in partition]

        if not pipelined or self.parallelism.pp <= 1:
            total = sum(stage_times)
            return SimulationResult(
                total_time=total,
                pp_stage_times=stage_times,
                per_op_breakdown=per_op,
            )

        num_microbatches = max(workload_ctx.num_microbatches, 1)
        pp = self.parallelism.pp
        max_stage = max(stage_times)
        p2p_per_mb = 0.0
        if self.pp_p2p_op_forward is not None:
            transfers = op_transfers.get(self.pp_p2p_op_forward, [])
            p2p_per_mb += self.pp_p2p_op_forward.get_operator_time(
                transfers, transfer_times_steady,
            )
        if self.pp_p2p_op_backward is not None:
            transfers = op_transfers.get(self.pp_p2p_op_backward, [])
            p2p_per_mb += self.pp_p2p_op_backward.get_operator_time(
                transfers, transfer_times_steady,
            )
        total = (num_microbatches + pp - 1) * (max_stage + p2p_per_mb)
        return SimulationResult(
            total_time=total,
            pp_stage_times=stage_times,
            per_op_breakdown=per_op,
        )
