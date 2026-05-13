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

The legacy names `ParallelismStrategy` and `MegatronStrategy` are aliases
for backwards compatibility.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from hardware import HardwareSpec
from model import Model
from model_mapping import ModelMapping
from network_op import NetworkOp, P2P
from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operator_base import Operator
from operators.llama_config import LlamaConfig
from operators.pattern import (
    LabeledOp,
    LayerRepeat,
    OperatorWithReqs,
    PatternEntry,
)
from operators.v3_config import V3Config
from parallelism_role import EPMode, ParallelismRole, TPMode
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


# Megatron canonical label → decoration table. Covers Llama and V3 vocabularies.
_MEGATRON_LABELS: dict[str, _Decoration] = {
    # ---- Llama labels ----
    "llama_embed":      _Decoration(role=_REP),
    "llama_attn_norm":  _Decoration(role=_REP),
    "llama_qkv":        _Decoration(role=_COL),
    "llama_rope":       _Decoration(role=_HEADS),
    "llama_attention":  _Decoration(role=_HEADS),
    "llama_attn_out":   _Decoration(role=_ROW, collective=_TP_AR),
    "llama_ffn_norm":   _Decoration(role=_REP),
    "llama_ffn_gate":   _Decoration(role=_COL),
    "llama_ffn_up":     _Decoration(role=_COL),
    "llama_ffn_swiglu": _Decoration(role=_COL),
    "llama_ffn_down":   _Decoration(role=_ROW, collective=_TP_AR),
    "llama_final_norm": _Decoration(role=_REP),
    "llama_lm_head":    _Decoration(role=_VOCAB, collective=_TP_AG),
    # ---- V3 top-level + dense FFN labels ----
    "v3_embed":         _Decoration(role=_REP),
    "v3_attn_norm":     _Decoration(role=_REP),
    "v3_ffn_norm":      _Decoration(role=_REP),
    "v3_ffn_gate":      _Decoration(role=_COL),
    "v3_ffn_up":        _Decoration(role=_COL),
    "v3_ffn_swiglu":    _Decoration(role=_COL),
    "v3_ffn_down":      _Decoration(role=_ROW, collective=_TP_AR),
    "v3_final_norm":    _Decoration(role=_REP),
    "v3_lm_head":       _Decoration(role=_VOCAB, collective=_TP_AG),
    # ---- MLA labels ----
    "mla_q_down":       _Decoration(role=_REP),
    "mla_q_up":         _Decoration(role=_COL),
    "mla_kv_down":      _Decoration(role=_REP),
    "mla_kv_up":        _Decoration(role=_COL),
    "mla_rope":         _Decoration(role=_HEADS),
    "mla_attention":    _Decoration(role=_HEADS),
    "mla_attn_out":     _Decoration(role=_ROW, collective=_TP_AR),
    # ---- MoE labels ----
    "moe_router":       _Decoration(role=_REP),
    "moe_dispatch":     _Decoration(role=_REP, collective=_EP_DISPATCH),
    "moe_routed_gate":  _Decoration(role=_EP),
    "moe_routed_up":    _Decoration(role=_EP),
    "moe_routed_swiglu":_Decoration(role=_REP),
    "moe_routed_down":  _Decoration(role=_EP),
    "moe_combine":      _Decoration(role=_REP, collective=_EP_COMBINE),
    "moe_shared_gate":  _Decoration(role=_COL),
    "moe_shared_up":    _Decoration(role=_COL),
    "moe_shared_swiglu":_Decoration(role=_COL),
    "moe_shared_down":  _Decoration(role=_ROW, collective=_TP_AR),
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
    full-shape kernel still drives GEMV / small-dim tier selection — that's
    the ±30% asymmetry we accept in Phase 2c (per-rank tile shape under
    COL vs ROW parallel isn't modeled).
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


def _phase_to_token_count(phase: str) -> TokenCount:
    if phase in ("prefill", "training"):
        return TokenCount.PREFILL
    if phase == "decode":
        return TokenCount.DECODE
    raise ValueError(f"Unknown phase: {phase!r}")


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
        tc = _phase_to_token_count(phase)
        for item in labeled:
            if isinstance(item, LayerRepeat):
                decorated_members: list[OperatorWithReqs] = [
                    self._decorate_op(m, arch, tc) for m in item.members
                ]
                if phase == "training" and self.parallelism.dp > 1:
                    template_labels = tuple(m.label for m in item.members)
                    dp_ar = self._dp_grad_all_reduce(decorated_members, template_labels)
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
            dec = _MEGATRON_LABELS[labeled.label]
        except KeyError:
            raise ValueError(
                f"MegatronScheme: no decoration rule for label {labeled.label!r}"
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
        self, layer_members: list[OperatorWithReqs], template_labels: tuple,
    ) -> NetworkRequirement:
        total_GB = sum(m.per_rank_parameter_bytes for m in layer_members) / 1e9
        return NetworkRequirement(
            kind=CollectiveKind.ALL_REDUCE,
            relation=Relation.EXCLUSIVE,
            eta=1.0,
            group=ParallelismGroup.DP,
            payload_id=("dp_grad", template_labels),
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
#
# A `ReshardingStrategy` converts a `(source_mapping, dest_mapping)` pair
# into `NetworkOp`s that physically move parameter bytes from the source's
# parallelism layout to the destination's. Resharding lives alongside the
# parallelism schemes because the byte-movement pattern is determined by
# how each side's scheme lays weights out across ranks — Megatron-to-
# Megatron is the v1 case; HybridFlow's 3D HybridEngine wants its own
# strategy that exploits micro-DP AllGather instead of P2P.


@dataclass(frozen=True)
class _ParameterTensor:
    """A single weight tensor with the info needed to compute per-rank slices.

    shard_axis:
      0  → row-parallel (split rows across TP)
      1  → column-parallel (split cols across TP)
      None → fully replicated across TP
    pp_anchor:
      "first" → lives on PP rank 0 (embedding)
      "last"  → lives on PP rank pp_size - 1 (LM head)
      "layer" → derived from layer_id × pp_size // n_layers
    """

    name: str
    shape: tuple[int, int]
    dtype_bytes: int
    layer_id: Optional[int] = None
    shard_axis: Optional[int] = None
    pp_anchor: str = "layer"


def _llama_parameters(arch: LlamaConfig) -> list[_ParameterTensor]:
    """Llama parameter list under the Megatron-style natural sharding scheme.

    QKV column-parallel, attention output row-parallel, FFN gate/up
    column-parallel, FFN down row-parallel, embedding+LM head vocab-parallel.
    Norm scales (RMSNorm) are negligible and intentionally omitted; total
    parameter accounting differs by O(n_layers · h) bytes.
    """
    out: list[_ParameterTensor] = []
    out.append(_ParameterTensor(
        name="embed",
        shape=(arch.vocab, arch.h),
        dtype_bytes=arch.dtype_bytes,
        shard_axis=0,
        pp_anchor="first",
    ))
    n_qkv = arch.n_q + 2 * arch.n_kv
    for layer_id in range(arch.n_layers):
        out.append(_ParameterTensor(
            name=f"qkv_l{layer_id}",
            shape=(n_qkv * arch.head_size, arch.h),
            dtype_bytes=arch.dtype_bytes,
            layer_id=layer_id, shard_axis=0, pp_anchor="layer",
        ))
        out.append(_ParameterTensor(
            name=f"o_l{layer_id}",
            shape=(arch.h, arch.n_q * arch.head_size),
            dtype_bytes=arch.dtype_bytes,
            layer_id=layer_id, shard_axis=1, pp_anchor="layer",
        ))
        out.append(_ParameterTensor(
            name=f"gate_l{layer_id}",
            shape=(arch.m, arch.h),
            dtype_bytes=arch.dtype_bytes,
            layer_id=layer_id, shard_axis=0, pp_anchor="layer",
        ))
        out.append(_ParameterTensor(
            name=f"up_l{layer_id}",
            shape=(arch.m, arch.h),
            dtype_bytes=arch.dtype_bytes,
            layer_id=layer_id, shard_axis=0, pp_anchor="layer",
        ))
        out.append(_ParameterTensor(
            name=f"down_l{layer_id}",
            shape=(arch.h, arch.m),
            dtype_bytes=arch.dtype_bytes,
            layer_id=layer_id, shard_axis=1, pp_anchor="layer",
        ))
    out.append(_ParameterTensor(
        name="lm_head",
        shape=(arch.vocab, arch.h),
        dtype_bytes=arch.dtype_bytes,
        shard_axis=0, pp_anchor="last",
    ))
    return out


def _pp_rank_for(param: _ParameterTensor, pp_size: int, n_layers: int) -> int:
    if param.pp_anchor == "first":
        return 0
    if param.pp_anchor == "last":
        return pp_size - 1
    if param.pp_anchor == "layer":
        if param.layer_id is None:
            raise ValueError(f"layer-anchored param {param.name} missing layer_id")
        return (param.layer_id * pp_size) // n_layers
    raise ValueError(f"Unknown pp_anchor: {param.pp_anchor!r}")


def _shards_at(mapping: ModelMapping, dp: int, pp: int) -> dict[int, Shard]:
    """Returns {tp_rank: Shard} for the given (dp, pp) coordinate."""
    return {
        s.tp: s
        for s in mapping.shards_to_devices
        if s.dp == dp and s.pp == pp
    }


def llama_total_parameter_bytes(arch: LlamaConfig) -> int:
    """Sum of parameter bytes for a Llama model under the natural sharding scheme.

    Exposed so tests can compare resharding bytes against the architectural
    target. Mirrors `_llama_parameters`.
    """
    return sum(p.shape[0] * p.shape[1] * p.dtype_bytes for p in _llama_parameters(arch))


class ReshardingStrategy(ABC):
    """Computes the NetworkOps that physically move parameter bytes from one
    parallelism layout to another. Concrete strategies are typically paired
    with the schemes on both sides (e.g. `MegatronToMegatronP2P` between
    two Megatron layouts; a future `HybridFlowMicroDPAllGather` for
    HybridFlow's 3D HybridEngine resharding).
    """

    @abstractmethod
    def compute_network_ops(
        self, source: ModelMapping, dest: ModelMapping,
    ) -> list[NetworkOp]:
        ...


class MegatronToMegatronP2P(ReshardingStrategy):
    """Resharding between two Megatron layouts via per-host-pair P2P.

    For each parameter tensor, computes (src_device, dst_device) byte
    movements under the natural Megatron sharding scheme (QKV column-
    parallel, O row-parallel, gate/up column-parallel, down row-parallel,
    embed + LM head vocab-parallel), aggregates by device pair, and emits
    one P2P per non-zero pair.

    Treats DP as canonical (dp=0 source → dp=0 dest); a real implementation
    would also broadcast to other DP replicas, but the extra fan-out is the
    same parameter set replicated and orthogonal to the resharding cost
    modeled here.

    Currently supports only LlamaConfig models. V3 (with MoE expert
    sharding) requires a separate enumeration and is left for a later phase.
    """

    def compute_network_ops(
        self, source: ModelMapping, dest: ModelMapping,
    ) -> list[NetworkOp]:
        arch = source.model.architecture
        if not isinstance(arch, LlamaConfig):
            raise NotImplementedError(
                "MegatronToMegatronP2P only supports LlamaConfig models in v1; "
                "V3 / MoE resharding will need its own enumeration."
            )

        params = _llama_parameters(arch)
        n_layers = arch.n_layers

        agg: dict[tuple[Device, Device], int] = defaultdict(int)
        for p in params:
            for pair, bytes_moved in self._movements_for_param(
                p, source, dest, n_layers
            ).items():
                agg[pair] += bytes_moved

        ops: list[NetworkOp] = []
        for (src_device, dst_device), bytes_total in agg.items():
            if bytes_total <= 0:
                continue
            ops.append(P2P(
                endpoint_pairs=[(src_device, dst_device)],
                total_data_GB=bytes_total / 1e9,
            ))
        return ops

    def _movements_for_param(
        self,
        param: _ParameterTensor,
        source: ModelMapping,
        dest: ModelMapping,
        n_layers: int,
    ) -> dict[tuple[Device, Device], int]:
        src_pp = _pp_rank_for(param, source.parallelism.pp, n_layers)
        dst_pp = _pp_rank_for(param, dest.parallelism.pp, n_layers)
        src_shards = _shards_at(source, dp=0, pp=src_pp)
        dst_shards = _shards_at(dest, dp=0, pp=dst_pp)
        if not src_shards or not dst_shards:
            return {}

        rows, cols = param.shape
        movements: dict[tuple[Device, Device], int] = defaultdict(int)

        if param.shard_axis is None:
            src_device = source.shards_to_devices[src_shards[0]]
            dst_device = dest.shards_to_devices[dst_shards[0]]
            movements[(src_device, dst_device)] = rows * cols * param.dtype_bytes
            return movements

        sharded_dim = param.shape[param.shard_axis]
        other_dim = param.shape[1 - param.shard_axis]
        src_tp = source.parallelism.tp
        dst_tp = dest.parallelism.tp

        if sharded_dim % src_tp != 0:
            raise ValueError(
                f"Param {param.name}: shard dim {sharded_dim} not divisible by "
                f"source tp {src_tp}"
            )
        if sharded_dim % dst_tp != 0:
            raise ValueError(
                f"Param {param.name}: shard dim {sharded_dim} not divisible by "
                f"dest tp {dst_tp}"
            )
        src_chunk = sharded_dim // src_tp
        dst_chunk = sharded_dim // dst_tp

        for src_t in range(src_tp):
            src_lo, src_hi = src_t * src_chunk, (src_t + 1) * src_chunk
            src_device = source.shards_to_devices[src_shards[src_t]]
            for dst_t in range(dst_tp):
                dst_lo, dst_hi = dst_t * dst_chunk, (dst_t + 1) * dst_chunk
                overlap = max(0, min(src_hi, dst_hi) - max(src_lo, dst_lo))
                if overlap == 0:
                    continue
                dst_device = dest.shards_to_devices[dst_shards[dst_t]]
                movements[(src_device, dst_device)] += overlap * other_dim * param.dtype_bytes
        return movements


# ---- backward-compat aliases ------------------------------------------------

# Callers that imported the previous `ParallelismStrategy` / `MegatronStrategy`
# names continue to work. New code should prefer `ParallelismScheme` /
# `MegatronScheme`.
ParallelismStrategy = ParallelismScheme
MegatronStrategy = MegatronScheme
# Resharding alias (old `NaiveP2PStrategy` name).
NaiveP2PStrategy = MegatronToMegatronP2P
