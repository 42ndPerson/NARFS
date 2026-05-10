from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from hardware import HardwareSpec
from mesh import Mesh
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
from operators.builders import (
    LayerTag,
    OperatorWithReqs,
    PatternEntry,
    expand_pattern_with_layers,
    split_pattern_with_layers,
)
from operators.optimizer import AdamOptimizerOp
from relation import NetworkAssociation, NetworkPhase, Relation
from shard import Shard
from workload_context import Workload, WorkloadContext


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
# the ratio compute_flops/memory_bytes does. Using A100-class avoids
# pathological GEMV vs. compute-bound inversions.
_PARTITION_REF_HW = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
    ridge_flops_per_byte=156.0,
)


def _partition_layers_by_cost(layer_costs: list[float], num_stages: int) -> list[list[int]]:
    """Contiguous-split layer indices into `num_stages` groups, minimizing the
    maximum group cost (linear-time partition / "painter's partition" DP).

    Returns list of length `num_stages`; each entry is a list of layer
    positions (0-indexed) belonging to that stage. Stages may be empty if
    `num_stages > len(layer_costs)`.

    Complexity: O(num_stages × n²) where n = len(layer_costs). Trivial for
    realistic models (n ≤ ~80 layers, num_stages ≤ ~16).
    """
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    n = len(layer_costs)
    if n == 0:
        return [[] for _ in range(num_stages)]
    if num_stages >= n:
        return [[i] if i < n else [] for i in range(num_stages)]

    # Prefix sums for O(1) range cost queries.
    prefix = [0.0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + layer_costs[i]

    inf = float("inf")
    # dp[s][i] = best (min) max-stage-cost partitioning layers[0..i) into s stages.
    dp = [[inf] * (n + 1) for _ in range(num_stages + 1)]
    cut = [[0] * (n + 1) for _ in range(num_stages + 1)]
    dp[0][0] = 0.0
    for s in range(1, num_stages + 1):
        for i in range(1, n + 1):
            # Last stage covers layers[j..i); j must leave at least s-1 layers for previous stages.
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

    `simulate` consumes pre-computed transfer-times dicts (steady + boundary)
    and returns a SimulationResult. NetworkOps are constructed once at
    __post_init__ with placeholder data_GB; the upstream stage-orchestration
    layer (Phase 4b) is responsible for populating realistic transfer times.
    """

    model: Model
    mesh: Mesh
    workload: Workload
    parallelism: ParallelismConfig
    shards_to_host_ids: dict[Shard, int]
    host_id_to_shard: dict[int, Shard]
    # Reference data_GB per collective kind for sizing the constructed
    # NetworkOps. Empty dict (default) treats every collective as 0-byte —
    # callers that want realistic contention should populate this with
    # estimated activation sizes per collective. Phase 4b's stage orchestrator
    # is the natural source.
    data_GB_estimates: dict["CollectiveKind", float] = field(default_factory=dict)

    # Computed in __post_init__:
    operator_pattern: list[Operator] = field(init=False, default_factory=list)
    per_op_network_requirements: list[list[NetworkRequirement]] = field(init=False, default_factory=list)
    per_op_layer_tags: list[LayerTag] = field(init=False, default_factory=list)
    network_ops: dict[NetworkRequirement, NetworkOp] = field(init=False, default_factory=dict)
    pipeline_partition: list[list[int]] = field(init=False, default_factory=list)
    # GENERATION holds a parallel decode-shaped graph; everything else is empty.
    decode_operator_pattern: list[Operator] = field(init=False, default_factory=list)
    decode_per_op_network_requirements: list[list[NetworkRequirement]] = field(init=False, default_factory=list)
    decode_per_op_layer_tags: list[LayerTag] = field(init=False, default_factory=list)
    decode_pipeline_partition: list[list[int]] = field(init=False, default_factory=list)
    # Phase 5: PP activation P2P NetworkOps. Forward fires every microbatch
    # transition; backward only in TRAINING. None when pp == 1.
    pp_p2p_op_forward: Optional[NetworkOp] = field(init=False, default=None)
    pp_p2p_op_backward: Optional[NetworkOp] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._validate_axis_consistency()
        if self.workload is Workload.GENERATION:
            self._install_pattern(self._build_pattern("prefill"), is_decode=False)
            self._install_pattern(self._build_pattern("decode"), is_decode=True)
        elif self.workload is Workload.PREPARATION:
            self._install_pattern(self._build_pattern("prefill"), is_decode=False)
        elif self.workload is Workload.TRAINING:
            self._install_training_pattern()
        elif self.workload is Workload.DORMANT:
            pass  # leaves operator_pattern empty
        else:
            raise ValueError(f"Unknown workload: {self.workload!r}")

        # Materialize one NetworkOp per unique NetworkRequirement.
        for reqs_list in (self.per_op_network_requirements, self.decode_per_op_network_requirements):
            for reqs in reqs_list:
                for req in reqs:
                    if req not in self.network_ops:
                        self.network_ops[req] = self._resolve_network_op(req)

        # Phase 5: install PP activation P2P NetworkOps. Sized off
        # microbatch activations; STEADY phase (one P2P per microbatch
        # transition, not a one-shot boundary). DORMANT skips this.
        if self.parallelism.pp > 1 and self.workload is not Workload.DORMANT:
            self._install_pp_p2p_ops()

    # ----- pattern build / install --------------------------------------------------

    def _build_pattern(self, phase: str) -> list[PatternEntry]:
        return self.model.build_pattern(self.model.architecture, phase, self.parallelism)

    def _install_pattern(self, pattern: list[PatternEntry], is_decode: bool) -> None:
        expanded = expand_pattern_with_layers(pattern)
        ops, reqs, tags = split_pattern_with_layers(expanded)
        partition = self._layer_aware_partition(ops, tags)
        if is_decode:
            self.decode_operator_pattern = ops
            self.decode_per_op_network_requirements = reqs
            self.decode_per_op_layer_tags = tags
            self.decode_pipeline_partition = partition
        else:
            self.operator_pattern = ops
            self.per_op_network_requirements = reqs
            self.per_op_layer_tags = tags
            self.pipeline_partition = partition

    def _install_training_pattern(self) -> None:
        """Build forward + backward + per-layer optimizer with consistent layer tags.

        Forward and backward share `LayerTag`s by construction (derive_backward_pattern
        preserves the structural shape). Optimizer ops are emitted one per
        layer-tag-with-non-zero-params; each carries the same tag so the
        partitioner co-locates a layer's forward, backward, and optimizer.
        """
        forward = self._build_pattern("training")
        backward = derive_backward_pattern(forward, recompute=True)

        forward_expanded = expand_pattern_with_layers(forward)
        backward_expanded = expand_pattern_with_layers(backward)

        # Aggregate parameter bytes per LayerTag.
        params_by_tag: dict[LayerTag, int] = defaultdict(int)
        for (op, _reqs), tag in forward_expanded:
            params_by_tag[tag] += op.parameter_bytes()

        # One AdamOptimizerOp per tag with non-zero params.
        dtype_bytes_param = 2
        optimizer_expanded: list[tuple[OperatorWithReqs, LayerTag]] = []
        for tag, total_param_bytes in params_by_tag.items():
            if total_param_bytes <= 0:
                continue
            num_params = total_param_bytes // dtype_bytes_param
            opt = AdamOptimizerOp(num_parameters=num_params, dtype_bytes_param=dtype_bytes_param)
            optimizer_expanded.append(((opt, []), tag))

        full = forward_expanded + backward_expanded + optimizer_expanded
        ops, reqs, tags = split_pattern_with_layers(full)
        partition = self._layer_aware_partition(ops, tags)
        self.operator_pattern = ops
        self.per_op_network_requirements = reqs
        self.per_op_layer_tags = tags
        self.pipeline_partition = partition

    # ----- layer-aware pipeline partition --------------------------------------------

    def _reference_partition_ctx(self) -> WorkloadContext:
        """Synthetic reference workload used only for measuring relative layer cost.

        Operators are sized off `microbatch_size × prompt_len` per invocation,
        so a microbatch_size of 1 keeps the partition stable across the
        eventual real workload. Only the *ratio* between layer costs matters
        for the partition.
        """
        return WorkloadContext(
            workload_type=self.workload,
            batch_size=1,
            microbatch_size=1,
            prompt_len=2048,
            response_len=128 if self.workload is Workload.GENERATION else 0,
            num_microbatches=1,
        )

    def _layer_aware_partition(
        self,
        ops: list[Operator],
        tags: list[LayerTag],
    ) -> list[list[int]]:
        """Layer-co-locating PP partition.

        Operators sharing a `layer_index` always land on the same stage
        (real PP keeps a layer's forward and backward on the same hardware).
        Layers are split into `pp` contiguous groups by min-max cost using
        per-layer kernel time at a synthetic reference workload. Boundary
        ops (`anchor="first"` / `"last"`) pin to the first / last stage.
        """
        pp = max(self.parallelism.pp, 1)
        n = len(ops)
        if pp == 1 or n == 0:
            return [list(range(n))] + [[] for _ in range(pp - 1)]

        # Group op indices by layer tag.
        indices_by_layer: dict[int, list[int]] = defaultdict(list)
        first_anchor: list[int] = []
        last_anchor: list[int] = []
        for i, tag in enumerate(tags):
            if tag.layer_index is not None:
                indices_by_layer[tag.layer_index].append(i)
            elif tag.anchor == "last":
                last_anchor.append(i)
            else:
                # "first" or fallback default.
                first_anchor.append(i)

        layer_indices_sorted = sorted(indices_by_layer.keys())
        if not layer_indices_sorted:
            # No numeric layers — put boundary ops on first stage.
            stage0 = sorted(first_anchor + last_anchor)
            return [stage0] + [[] for _ in range(pp - 1)]

        # Per-layer cost = sum of kernel_time of all ops sharing this layer index,
        # at the reference ctx + reference hw.
        ref_ctx = self._reference_partition_ctx()
        layer_costs: list[float] = []
        for lidx in layer_indices_sorted:
            cost = sum(
                ops[i].kernel_time(ref_ctx, _PARTITION_REF_HW)
                for i in indices_by_layer[lidx]
            )
            layer_costs.append(cost)

        layer_partition = _partition_layers_by_cost(layer_costs, pp)

        out: list[list[int]] = [[] for _ in range(pp)]
        for stage_idx, layer_positions in enumerate(layer_partition):
            for layer_pos in layer_positions:
                lidx = layer_indices_sorted[layer_pos]
                out[stage_idx].extend(sorted(indices_by_layer[lidx]))

        # Pin boundary ops (embed → first stage, final norm + LM head → last stage).
        out[0] = sorted(first_anchor) + out[0]
        out[-1] = out[-1] + sorted(last_anchor)
        return out

    # ----- collective sizing refresh (Phase 1) ---------------------------------------

    def refresh_collective_sizes(self, workload_ctx: WorkloadContext) -> None:
        """Re-derive each NetworkOp's transfer sizes against the current ctx.

        Called by the simulator at the top of `simulate_stage` /
        `simulate_isolated` / `simulate_rlhf_iteration`, before contention
        snapshots are taken. For each unique `NetworkRequirement` in
        `network_ops`, evaluates its `data_GB_fn(ctx)` (when present) and
        calls the NetworkOp's `refresh_data_size`. Requirements without a
        sizing closure fall back to the legacy `data_GB_estimates` dict.
        """
        for req, op in self.network_ops.items():
            fn = req.data_GB_fn
            if fn is not None:
                data_GB = float(fn(workload_ctx))
            else:
                data_GB = float(self.data_GB_estimates.get(req.kind, 0.0))
            op.refresh_data_size(data_GB)

    # ----- network requirement resolution --------------------------------------------

    def _shards_in_group(self, group: ParallelismGroup) -> list[Shard]:
        """Return the shard set spanning the requested parallelism axis.

        Each axis varies one rank dimension and holds the *other three*
        fixed to the reference shard's values. With Phase 2's `Shard.ep`,
        TP and EP are independent axes and can coexist on the same mesh.
        """
        all_shards = list(self.shards_to_host_ids.keys())
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
        """Verify the shard layout actually spans the parallelism degrees.

        For each axis with `parallelism.<axis> > 1`, the shard set's reach
        along that axis (holding the other axes fixed at the reference shard)
        must equal the configured degree. Catches off-by-one mesh
        construction errors before they silently shrink a collective.
        """
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
        """Phase 5: build the per-microbatch PP activation P2P ops.

        Forward leg fires every microbatch (any non-dormant workload);
        backward leg only in TRAINING. Sized to microbatch-prefill activation
        (`microbatch_size * prompt_len * h * dtype`); the closure is
        evaluated by `refresh_collective_sizes` against the current ctx.

        Phase override is `STEADY`: P2P normally defaults to BOUNDARY (one-
        shot pipeline send/recv at handover boundaries). Here it streams
        per-microbatch through the same bandwidth pool as compute-phase
        steady-state collectives.

        Both ops are registered in `network_ops` so refresh sees them, and
        also stashed on the mapping for the bubble-formula in `_simulate_one`.
        Cross-axis exists for each (dp, tp, ep) coord; we instantiate one
        representative pair (the (0,0,0) coord's pp=0→1 hop) and let the
        topology contention model amortize sharing across coords.
        """
        pp_shards = self._shards_in_group(ParallelismGroup.PP)
        if len(pp_shards) < 2:
            return
        src, dst = pp_shards[0], pp_shards[1]
        arch = self.model.architecture
        h = getattr(arch, "h", None)
        dtype = getattr(arch, "dtype_bytes", None)
        if h is None or dtype is None:
            return  # architectures lacking shape — skip

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
            logical_transfers=[
                LogicalTransfer(
                    src_host_id=self.shards_to_host_ids[src],
                    dst_host_id=self.shards_to_host_ids[dst],
                    data_GB=0.0,
                )
            ],
            phase=NetworkPhase.STEADY,
        )
        self.network_ops[fwd_req] = fwd_op
        self.pp_p2p_op_forward = fwd_op

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
                logical_transfers=[
                    LogicalTransfer(
                        src_host_id=self.shards_to_host_ids[dst],
                        dst_host_id=self.shards_to_host_ids[src],
                        data_GB=0.0,
                    )
                ],
                phase=NetworkPhase.STEADY,
            )
            self.network_ops[bwd_req] = bwd_op
            self.pp_p2p_op_backward = bwd_op

    def _resolve_network_op(self, req: NetworkRequirement) -> NetworkOp:
        """Concrete NetworkOp from a declarative requirement.

        data_GB is a placeholder — Phase 4b's stage orchestrator computes
        realistic activation sizes and populates transfer-times dicts. Empty
        dicts return 0.0 via .get(t, 0.0).
        """
        shards = self._shards_in_group(req.group)
        kind = req.kind
        data_GB = self.data_GB_estimates.get(kind, 0.0)
        if kind is CollectiveKind.ALL_REDUCE:
            return AllReduceRing.generate(self, shards, data_GB=data_GB)
        if kind is CollectiveKind.ALL_GATHER:
            return AllGatherRing.generate(self, shards, data_GB=data_GB)
        if kind is CollectiveKind.REDUCE_SCATTER:
            return ReduceScatterRing.generate(self, shards, data_GB=data_GB)
        if kind in (CollectiveKind.ALL_TO_ALL_DISPATCH, CollectiveKind.ALL_TO_ALL_COMBINE):
            return AllToAll.generate(self, shards, data_GB=data_GB)
        if kind in (CollectiveKind.P2P_PIPELINE_FORWARD, CollectiveKind.P2P_PIPELINE_BACKWARD):
            if len(shards) < 2:
                # Degenerate — single-rank "pipeline". No transfer.
                return P2P(logical_transfers=[])
            return P2P.generate(self, shards[0], shards[1], data_GB=data_GB)
        raise ValueError(f"Unknown CollectiveKind: {kind!r}")

    # ----- simulation ----------------------------------------------------------------

    def simulate(
        self,
        workload_ctx: WorkloadContext,
        hw: HardwareSpec,
        transfer_times_steady: dict[LogicalTransfer, float],
        transfer_times_boundary: dict[LogicalTransfer, float],
    ) -> SimulationResult:
        if self.workload is Workload.GENERATION:
            prefill = self._simulate_one(
                self.operator_pattern,
                self.per_op_network_requirements,
                self.pipeline_partition,
                workload_ctx, hw,
                transfer_times_steady, transfer_times_boundary,
                pipelined=True,
            )
            # Per-token decode = sum(stage_times) — no pipelining benefit at
            # decode (one new token per request per step, no microbatching).
            decode_step = self._simulate_one(
                self.decode_operator_pattern,
                self.decode_per_op_network_requirements,
                self.decode_pipeline_partition,
                workload_ctx, hw,
                transfer_times_steady, transfer_times_boundary,
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

        result = self._simulate_one(
            self.operator_pattern,
            self.per_op_network_requirements,
            self.pipeline_partition,
            workload_ctx, hw,
            transfer_times_steady, transfer_times_boundary,
            pipelined=True,
        )
        if self.workload is Workload.TRAINING:
            return SimulationResult(
                total_time=result.total_time,
                training_time=result.total_time,
                pp_stage_times=result.pp_stage_times,
                per_op_breakdown=result.per_op_breakdown,
            )
        # PREPARATION
        return result

    def _simulate_one(
        self,
        ops: list[Operator],
        reqs: list[list[NetworkRequirement]],
        partition: list[list[int]],
        workload_ctx: WorkloadContext,
        hw: HardwareSpec,
        transfer_times_steady: dict[LogicalTransfer, float],
        transfer_times_boundary: dict[LogicalTransfer, float],
        pipelined: bool,
    ) -> SimulationResult:
        if not ops:
            return SimulationResult(total_time=0.0, pp_stage_times=[0.0])

        per_op: list[float] = []
        for op, req_list in zip(ops, reqs):
            kernel = op.kernel_time(workload_ctx, hw)
            wall = kernel
            for req in req_list:
                net_op = self.network_ops[req]
                phase = NetworkOp.phase_of(net_op)
                tt = transfer_times_steady if phase is NetworkPhase.STEADY else transfer_times_boundary
                net_time = net_op.get_operator_time(tt)
                # Chain multiple requirements: each refines the running `wall`.
                assoc = NetworkAssociation(
                    network_op=net_op, relation=req.relation, eta=req.eta
                )
                wall = assoc.combine_times(wall, net_time)
            per_op.append(wall)

        # Stage time = sum of per-op walls for ops in that stage. Partition is
        # an explicit list of indices (layer-aware), so explicit indexing
        # (not slicing) is required.
        stage_times = [sum(per_op[i] for i in stage) for stage in partition]

        if not pipelined or self.parallelism.pp <= 1:
            total = sum(stage_times)
            return SimulationResult(
                total_time=total,
                pp_stage_times=stage_times,
                per_op_breakdown=per_op,
            )

        # 1F1B-shaped iteration time. Bubble correction is implicit in
        # (num_microbatches + pp - 1). Phase 5 adds the per-microbatch PP
        # activation P2P (forward + backward in training) additively to
        # max_stage — STEADY-phase overhead per microbatch transition.
        num_microbatches = max(workload_ctx.num_microbatches, 1)
        pp = self.parallelism.pp
        max_stage = max(stage_times)
        p2p_per_mb = 0.0
        if self.pp_p2p_op_forward is not None:
            p2p_per_mb += self.pp_p2p_op_forward.get_operator_time(transfer_times_steady)
        if self.pp_p2p_op_backward is not None:
            p2p_per_mb += self.pp_p2p_op_backward.get_operator_time(transfer_times_steady)
        total = (num_microbatches + pp - 1) * (max_stage + p2p_per_mb)
        return SimulationResult(
            total_time=total,
            pp_stage_times=stage_times,
            per_op_breakdown=per_op,
        )
