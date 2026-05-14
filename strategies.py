"""Parallelism schemes + placements — the two-layer split that owns
*how* a model is sharded onto hardware.

The split:

- **ParallelismScheme** owns parallelism-only decisions: which operators
  get sharded along which axis, what collectives fire where, how compute
  divides per rank, parallelism-vs-architecture divisibility. Schemes
  produce a `Plan` containing the decorated forward/decode patterns plus
  the abstract `Shard`s the scheme requires. A `Plan` is device-agnostic.

- **Placement** is a function `(plan, devices, topology) -> dict[Shard, Device]`
  that binds shards to physical hardware. Placement is independent of the
  scheme: different placements can be tried against the same plan to
  optimize cross-stage resharding cost (as in HybridFlow's 3D HybridEngine).

- **`assemble(plan, placement, devices, topology)`** combines the two,
  producing a `ModelMapping` ready for simulation. NetworkOps' physical
  `endpoint_pairs` resolve at this step.

The core simulator (CombinedOp, ModelMapping, simulator.py, RLHFTimeline)
is intentionally unaware of parallelism or placement semantics.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from hardware import HardwareSpec
from model import Model
from model_mapping import ModelMapping
from network_op import NetworkOp
from network_requirement import CollectiveKind, NetworkRequirement
from operator_base import Operator
from operators.llama_config import LlamaConfig
from operators.pattern import (
    LabeledOp,
    LayerRepeat,
    OpRole,
    OperatorWithReqs,
    PatternEntry,
)
from operators.v3_config import V3Config
from parallelism import (
    EPMode,
    ParallelismConfig,
    ParallelismGroup,
    ParallelismRole,
    TPMode,
)
from relation import Relation
from shard import Shard
from topo import Device, HostTopo
from workload_context import TokenCount, Workload, WorkloadContext


# Canonical role aliases used by Megatron-family schemes.
_REP = ParallelismRole()
_COL = ParallelismRole(tp=TPMode.COLUMN_PARALLEL)
_ROW = ParallelismRole(tp=TPMode.ROW_PARALLEL)
_HEADS = ParallelismRole(tp=TPMode.HEADS_PARALLEL)
_VOCAB = ParallelismRole(tp=TPMode.VOCAB_PARALLEL)
_EP = ParallelismRole(ep=EPMode.EXPERT_PARALLEL)


# Collective sentinels — what kind of collective fires after an op.
_TP_AR = "tp_ar"
_TP_AG = "tp_ag"
_EP_DISPATCH = "ep_dispatch"
_EP_COMBINE = "ep_combine"


@dataclass(frozen=True)
class _Decoration:
    """Per-label decoration: the sharding role to set on the operator and
    the collective (if any) to emit after it.
    """
    role: ParallelismRole
    collective: Optional[str] = None


# Megatron canonical role → decoration table. The architecture-agnostic
# `OpRole` taxonomy means this table covers Llama, V3 dense, V3 MLA, and
# V3 MoE in one ~17-entry mapping — adding a new architecture that uses
# the existing roles requires no scheme changes.
_MEGATRON_DECORATIONS: dict[OpRole, _Decoration] = {
    # Embedding + final projection
    OpRole.TOKEN_EMBED:           _Decoration(role=_REP),
    OpRole.LM_HEAD:               _Decoration(role=_VOCAB, collective=_TP_AG),
    # Residual-stream norms
    OpRole.NORM:                  _Decoration(role=_REP),
    # Attention block
    OpRole.QKV_PROJ:              _Decoration(role=_COL),
    OpRole.LATENT_DOWN_PROJ:      _Decoration(role=_REP),
    OpRole.ROPE:                  _Decoration(role=_HEADS),
    OpRole.ATTN_HEADS:            _Decoration(role=_HEADS),
    OpRole.ATTN_OUT:              _Decoration(role=_ROW, collective=_TP_AR),
    # Dense FFN (shared MoE experts re-use these)
    OpRole.FFN_EXPAND:            _Decoration(role=_COL),
    OpRole.FFN_ACT:               _Decoration(role=_COL),
    OpRole.FFN_REDUCE:            _Decoration(role=_ROW, collective=_TP_AR),
    # MoE routing + routed experts
    OpRole.ROUTER:                _Decoration(role=_REP),
    OpRole.DISPATCH_MARKER:       _Decoration(role=_REP, collective=_EP_DISPATCH),
    OpRole.COMBINE_MARKER:        _Decoration(role=_REP, collective=_EP_COMBINE),
    OpRole.ROUTED_EXPERT_EXPAND:  _Decoration(role=_EP),
    OpRole.ROUTED_EXPERT_ACT:     _Decoration(role=_REP),
    OpRole.ROUTED_EXPERT_REDUCE:  _Decoration(role=_EP),
}


# ---- per-rank baking (role → kernel-time / param-byte scaling) -------------


def _tp_divides_compute(role_tp: TPMode) -> bool:
    return role_tp in (
        TPMode.COLUMN_PARALLEL,
        TPMode.ROW_PARALLEL,
        TPMode.HEADS_PARALLEL,
        TPMode.VOCAB_PARALLEL,
    )


def _make_compute_time_fn(
    op: Operator, role: ParallelismRole, par: ParallelismConfig
):
    """Per-rank kernel time closure for `op` under `role` and `par`.

    Approximation: per-rank time = full-model `kernel_time` divided by the
    TP factor when the role shards along TP. EP doesn't divide per-call
    compute since the operator is sized to a single bottleneck expert. The
    full-shape kernel still drives GEMV / small-dim tier selection — an
    acknowledged ±30% asymmetry (per-rank tile shape under COL vs ROW
    parallel isn't modeled).
    """
    tp_factor = par.tp if par.tp > 1 and _tp_divides_compute(role.tp) else 1
    if tp_factor == 1:
        return lambda ctx, hw: op.kernel_time(ctx, hw)
    return lambda ctx, hw: op.kernel_time(ctx, hw) / tp_factor


def _per_rank_parameter_bytes(
    op: Operator, role: ParallelismRole, par: ParallelismConfig,
) -> int:
    """Per-rank weight footprint after applying the role's TP / EP shards.

    Schemes shard weights along whichever TP axis is set (COL, ROW, HEADS,
    VOCAB) and along EP for routed experts. Validates divisibility at the
    full-parameter level via the same factor — a misalignment yields a
    fractional per-rank count which we surface as a ValueError.
    """
    full = op.parameter_bytes()
    if full == 0:
        return 0
    if par.tp > 1 and _tp_divides_compute(role.tp):
        if full % par.tp != 0:
            raise ValueError(
                f"per_rank_parameter_bytes: full={full} not divisible by tp={par.tp} "
                f"for op {type(op).__name__} under role.tp={role.tp.name}"
            )
        full //= par.tp
    if par.ep > 1 and role.ep == EPMode.EXPERT_PARALLEL:
        if full % par.ep != 0:
            raise ValueError(
                f"per_rank_parameter_bytes: full={full} not divisible by ep={par.ep} "
                f"for op {type(op).__name__} under role.ep={role.ep.name}"
            )
        full //= par.ep
    return full


# ---- size closures (architecture-agnostic) ---------------------------------


def _activation_size_fn(h: int, dtype_bytes: int, tc: TokenCount):
    def _fn(ctx) -> float:
        return tc.resolve(ctx) * h * dtype_bytes / 1e9
    return _fn


def _lm_head_gather_size_fn(vocab: int, dtype_bytes: int, tc: TokenCount):
    def _fn(ctx) -> float:
        return tc.resolve(ctx) * vocab * dtype_bytes / 1e9
    return _fn


def _ep_dispatched_size_fn(h: int, dtype_bytes: int, top_k: int, tc: TokenCount):
    def _fn(ctx) -> float:
        return top_k * tc.resolve(ctx) * h * dtype_bytes / 1e9
    return _fn


# ---- Plan -------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """A device-agnostic execution plan produced by a `ParallelismScheme`.

    Holds the strategy-decorated patterns (operators have sharding roles
    applied and NetworkRequirements attached based on labels), the abstract
    `Shard`s the scheme uses, plus the originating model / parallelism /
    workload. A `Placement` binds the shards to physical devices; the
    `assemble(...)` step produces the final ModelMapping.
    """
    model: Model
    workload: Workload
    parallelism: ParallelismConfig
    forward_pattern: list = field(default_factory=list)
    decode_pattern: list = field(default_factory=list)
    shards: tuple[Shard, ...] = field(default_factory=tuple)


# ---- Placement --------------------------------------------------------------


# A placement is a callable that, given a plan, a device pool, and (optionally)
# the topology, returns the binding `dict[Shard, Device]`. Concrete placements
# implement particular allocation strategies (canonical Megatron, rail-aligned,
# HybridFlow-3D-engine, etc.). A precomputed `dict[Shard, Device]` is also
# acceptable when the caller wants to express the mapping directly.
Placement = Callable[
    [Plan, list[Device], Optional[HostTopo]],
    dict[Shard, Device],
]


def canonical_megatron_placement(
    plan: Plan,
    devices: list[Device],
    topology: Optional[HostTopo] = None,
) -> dict[Shard, Device]:
    """The default Megatron placement: walks plan.shards in their canonical
    (dp, pp, ep, tp) order (TP innermost) and pairs with `devices` in input
    order. Consecutive devices become TP-adjacent ranks — matching the
    typical intra-host NVLink/NVSwitch layout where TP collectives benefit
    most from device proximity. Caller is responsible for ordering `devices`
    to match topology (devices on the same host first, then next host).

    `topology` is accepted for interface symmetry but ignored.
    """
    del topology
    if len(devices) != len(plan.shards):
        par = plan.parallelism
        raise ValueError(
            f"canonical_megatron_placement: requires {len(plan.shards)} devices "
            f"for DP={par.dp} PP={par.pp} EP={par.ep} TP={par.tp}, "
            f"got {len(devices)}"
        )
    return dict(zip(plan.shards, devices))


# ---- base scheme ------------------------------------------------------------


class ParallelismScheme(ABC):
    """A scheme owns parallelism-only decisions: which operators shard along
    which axis, what collectives fire where, how compute divides per rank.

    `build_plan(...)` produces a device-agnostic `Plan`. Use `assemble(...)`
    (or the `build_mm(...)` facade method) to bind devices and produce a
    `ModelMapping`.
    """

    parallelism: ParallelismConfig

    @abstractmethod
    def build_plan(self, model: Model, workload: Workload) -> Plan:
        ...

    def build_mm(
        self,
        model: Model,
        workload: Workload,
        devices: list[Device],
        placement: Placement = canonical_megatron_placement,
        topology: Optional[HostTopo] = None,
        name: str = "",
        role: str = "",
    ) -> ModelMapping:
        """Convenience facade: build a Plan and assemble it with the given
        placement. Defaults to `canonical_megatron_placement` for backwards
        compatibility with pre-split callers.
        """
        del name, role
        plan = self.build_plan(model, workload)
        return assemble(plan, placement, devices, topology)


# ---- MegatronScheme ---------------------------------------------------------


@dataclass(frozen=True)
class MegatronScheme(ParallelismScheme):
    """Canonical Megatron-style sharding.

    Scheme rules (applied via label → decoration table):
      - QKV, gate, up, LM head:    column-parallel
      - O, down, shared-expert-down: row-parallel (TP-AR after)
      - Attention, RoPE:           heads-parallel
      - Routed MoE experts:        EP-sharded
      - Embedding, RMSNorm:        replicated
      - DP gradient AR per layer template in TRAINING when dp > 1
      - LM-head AllGather under TP > 1
      - PP activation P2P per microbatch when pp > 1 (installed at assemble)
    """

    parallelism: ParallelismConfig

    def build_plan(self, model: Model, workload: Workload) -> Plan:
        self._validate_against_arch(model.architecture)
        shards = self._enumerate_shards(model)
        forward_pattern: list[PatternEntry] = []
        decode_pattern: list[PatternEntry] = []
        if workload is Workload.GENERATION:
            forward_pattern = self._decorate_pattern(
                model.forward_sequence("prefill"), model.architecture, "prefill",
            )
            decode_pattern = self._decorate_pattern(
                model.forward_sequence("decode"), model.architecture, "decode",
            )
        elif workload is Workload.PREPARATION:
            forward_pattern = self._decorate_pattern(
                model.forward_sequence("prefill"), model.architecture, "prefill",
            )
        elif workload is Workload.TRAINING:
            forward_pattern = self._decorate_pattern(
                model.forward_sequence("training"), model.architecture, "training",
            )
        # DORMANT: empty patterns.
        return Plan(
            model=model,
            workload=workload,
            parallelism=self.parallelism,
            forward_pattern=forward_pattern,
            decode_pattern=decode_pattern,
            shards=tuple(shards),
        )

    # ---- shard enumeration -------------------------------------------------

    def _enumerate_shards(self, model: Model) -> list[Shard]:
        """Produces the scheme's canonical-order shard list. TP varies
        innermost (so consecutive shards are TP-adjacent), then EP, PP, DP.
        Placement functions decide how these abstract coordinates bind to
        physical devices.
        """
        shards: list[Shard] = []
        for d in range(self.parallelism.dp):
            for p in range(self.parallelism.pp):
                for e in range(self.parallelism.ep):
                    for t in range(self.parallelism.tp):
                        shards.append(Shard(model=model, dp=d, pp=p, tp=t, ep=e))
        return shards

    # ---- architecture validation -------------------------------------------

    def _validate_against_arch(self, arch) -> None:
        tp = self.parallelism.tp
        ep = self.parallelism.ep
        if isinstance(arch, LlamaConfig):
            if tp > 1:
                if arch.vocab % tp != 0:
                    raise ValueError(f"vocab ({arch.vocab}) must be divisible by tp ({tp})")
                if arch.n_q % tp != 0:
                    raise ValueError(f"n_q ({arch.n_q}) must be divisible by tp ({tp})")
                if arch.n_kv % tp != 0:
                    raise ValueError(f"n_kv ({arch.n_kv}) must be divisible by tp ({tp})")
                if arch.m % tp != 0:
                    raise ValueError(f"m ({arch.m}) must be divisible by tp ({tp})")
        elif isinstance(arch, V3Config):
            if tp > 1:
                if arch.vocab % tp != 0:
                    raise ValueError(f"vocab ({arch.vocab}) must be divisible by tp ({tp})")
                if arch.n_h % tp != 0:
                    raise ValueError(f"n_h ({arch.n_h}) must be divisible by tp ({tp})")
                if arch.m_dense % tp != 0:
                    raise ValueError(f"m_dense ({arch.m_dense}) must be divisible by tp ({tp})")
                if arch.m_expert % tp != 0:
                    raise ValueError(f"m_expert ({arch.m_expert}) must be divisible by tp ({tp})")
            if ep > 1 and arch.n_routed_experts % ep != 0:
                raise ValueError(
                    f"n_routed_experts ({arch.n_routed_experts}) must be divisible by ep ({ep})"
                )

    # ---- inspection hook ---------------------------------------------------

    def decorated_pattern(
        self, model: Model, phase: str
    ) -> list[PatternEntry]:
        """Public hook for inspecting the scheme's decorated pattern for one
        phase without constructing a full Plan or ModelMapping. Used by tests."""
        self._validate_against_arch(model.architecture)
        return self._decorate_pattern(
            model.forward_sequence(phase), model.architecture, phase,
        )

    # ---- pattern decoration ------------------------------------------------

    def _decorate_pattern(
        self,
        labeled: list,
        arch,
        phase: str,
    ) -> list[PatternEntry]:
        """Walk a labeled pattern; produce an OperatorWithReqs-decorated form."""
        out: list[PatternEntry] = []
        tc = TokenCount.from_phase(phase)
        for item in labeled:
            if isinstance(item, LayerRepeat):
                decorated_members: list[OperatorWithReqs] = [
                    self._decorate_op(m, arch, tc) for m in item.members
                ]
                if phase == "training" and self.parallelism.dp > 1:
                    template_roles = tuple(m.role for m in item.members)
                    dp_ar = self._dp_grad_all_reduce(decorated_members, template_roles)
                    last = decorated_members[-1]
                    decorated_members[-1] = OperatorWithReqs(
                        operator=last.operator,
                        requirements=last.requirements + [dp_ar],
                        compute_time_fn=last.compute_time_fn,
                        per_rank_parameter_bytes=last.per_rank_parameter_bytes,
                    )
                out.append(LayerRepeat(item.count, decorated_members))
            elif isinstance(item, LabeledOp):
                out.append(self._decorate_op(item, arch, tc))
            else:
                raise ValueError(
                    f"_decorate_pattern: expected LabeledOp or LayerRepeat, got {type(item).__name__}"
                )
        return out

    def _decorate_op(
        self, labeled: LabeledOp, arch, tc: TokenCount
    ) -> OperatorWithReqs:
        try:
            dec = _MEGATRON_DECORATIONS[labeled.role]
        except KeyError:
            raise ValueError(
                f"MegatronScheme: no decoration rule for role {labeled.role.name}"
            )
        op = labeled.operator
        reqs: list[NetworkRequirement] = []
        if dec.collective is not None:
            req = self._emit_collective(dec.collective, arch, tc)
            if req is not None:
                reqs.append(req)
        return OperatorWithReqs(
            operator=op,
            requirements=reqs,
            compute_time_fn=_make_compute_time_fn(op, dec.role, self.parallelism),
            per_rank_parameter_bytes=_per_rank_parameter_bytes(op, dec.role, self.parallelism),
        )

    def _emit_collective(
        self, kind_tag: str, arch, tc: TokenCount
    ) -> Optional[NetworkRequirement]:
        if kind_tag == _TP_AR:
            if self.parallelism.tp <= 1:
                return None
            return NetworkRequirement(
                kind=CollectiveKind.ALL_REDUCE,
                relation=Relation.EXCLUSIVE,
                eta=1.0,
                group=ParallelismGroup.TP,
                data_GB_fn=_activation_size_fn(arch.h, arch.dtype_bytes, tc),
            )
        if kind_tag == _TP_AG:
            if self.parallelism.tp <= 1:
                return None
            return NetworkRequirement(
                kind=CollectiveKind.ALL_GATHER,
                relation=Relation.EXCLUSIVE,
                eta=1.0,
                group=ParallelismGroup.TP,
                data_GB_fn=_lm_head_gather_size_fn(arch.vocab, arch.dtype_bytes, tc),
            )
        if kind_tag == _EP_DISPATCH:
            if self.parallelism.ep <= 1:
                return None
            return NetworkRequirement(
                kind=CollectiveKind.ALL_TO_ALL_DISPATCH,
                relation=Relation.EXCLUSIVE,
                eta=1.0,
                group=ParallelismGroup.EP,
                data_GB_fn=_ep_dispatched_size_fn(arch.h, arch.dtype_bytes, arch.top_k, tc),
            )
        if kind_tag == _EP_COMBINE:
            if self.parallelism.ep <= 1:
                return None
            return NetworkRequirement(
                kind=CollectiveKind.ALL_TO_ALL_COMBINE,
                relation=Relation.EXCLUSIVE,
                eta=1.0,
                group=ParallelismGroup.EP,
                data_GB_fn=_ep_dispatched_size_fn(arch.h, arch.dtype_bytes, arch.top_k, tc),
            )
        raise ValueError(f"Unknown collective tag: {kind_tag!r}")

    def _dp_grad_all_reduce(
        self, layer_members: list[OperatorWithReqs], template_roles: tuple,
    ) -> NetworkRequirement:
        total_GB = sum(m.per_rank_parameter_bytes for m in layer_members) / 1e9
        return NetworkRequirement(
            kind=CollectiveKind.ALL_REDUCE,
            relation=Relation.EXCLUSIVE,
            eta=1.0,
            group=ParallelismGroup.DP,
            payload_id=("dp_grad", template_roles),
            data_GB_fn=lambda _ctx, _v=total_GB: _v,
        )


# ---- assemble ---------------------------------------------------------------


def assemble(
    plan: Plan,
    placement: Placement,
    devices: list[Device],
    topology: Optional[HostTopo] = None,
) -> ModelMapping:
    """Combine a scheme-produced `Plan` with a `Placement` to produce a
    fully-resolved `ModelMapping`. Device-binding happens here.

    Validates device count and resolves each shard's device via the
    placement callable; constructs the ModelMapping with the decorated
    patterns from the plan, leaving NetworkOp resolution to ModelMapping's
    own __post_init__.
    """
    s2d = placement(plan, devices, topology)
    if set(s2d.keys()) != set(plan.shards):
        missing = set(plan.shards) - set(s2d.keys())
        extra = set(s2d.keys()) - set(plan.shards)
        raise ValueError(
            f"placement returned mapping that doesn't cover plan.shards: "
            f"missing {len(missing)} shard(s), extra {len(extra)} key(s)"
        )
    d2s = {dev: s for s, dev in s2d.items()}
    return ModelMapping(
        model=plan.model,
        workload=plan.workload,
        parallelism=plan.parallelism,
        shards_to_devices=s2d,
        device_to_shard=d2s,
        forward_pattern=plan.forward_pattern,
        decode_pattern=plan.decode_pattern,
    )


# ---- resharding -------------------------------------------------------------


class ReshardingScheme(ABC):
    """Computes the NetworkOps that physically move parameter bytes from one
    parallelism layout to another.

    Concrete schemes are paired with the parallelism schemes on both sides
    — `MegatronToMegatronP2P` (in `operators/llama_resharding.py`) handles
    two Megatron layouts; a future `HybridFlowMicroDPAllGather` would handle
    HybridFlow's 3D HybridEngine resharding. The byte-movement pattern is
    determined by how each side's scheme lays weights out across ranks,
    which is why resharding lives in the strategy layer rather than the
    core simulator.
    """

    @abstractmethod
    def compute_network_ops(
        self, source: ModelMapping, dest: ModelMapping,
    ) -> list[NetworkOp]:
        ...


