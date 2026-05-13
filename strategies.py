"""Parallelism strategies — the layer that owns *how* a model is sharded
onto hardware. The core simulator (CombinedOp, ModelMapping, simulator.py,
RLHFTimeline) is intentionally unaware of parallelism semantics; a strategy
is what bridges the parallelism-free architecture (`Model.forward_sequence`)
to the resolved simulation plan (`ModelMapping`).

A strategy carries a `ParallelismConfig` and exposes `build_mm(...)` to
produce a `ModelMapping` from an ordered device pool. Internally the
strategy walks the architecture's labeled forward sequence, applies
scheme-specific sharding role decorations to each operator, and emits
collectives (TP-AR, EP-AllToAll, DP-grad-AR, LM-head AllGather) based on
labels. The result is an `OperatorWithReqs`-decorated pattern that
ModelMapping consumes.
"""
from __future__ import annotations
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from mesh import Mesh
from model import Model
from model_mapping import ModelMapping
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
from topo import Device
from workload_context import TokenCount, Workload


# Canonical role aliases used by Megatron-family strategies.
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


# ---- base class -------------------------------------------------------------


class ParallelismStrategy(ABC):
    """Owns parallelism / sharding / collective-emission logic for one scheme."""

    parallelism: ParallelismConfig

    @abstractmethod
    def build_mm(
        self,
        model: Model,
        workload: Workload,
        devices: list[Device],
        name: str = "",
        role: str = "",
    ) -> ModelMapping:
        ...


# ---- MegatronStrategy -------------------------------------------------------


@dataclass(frozen=True)
class MegatronStrategy(ParallelismStrategy):
    """Canonical Megatron-style sharding.

    Sharding scheme (applied via label → role mapping):
      - QKV, gate, up, LM head:   column-parallel
      - O, down, shared-expert-down: row-parallel (TP-AR after)
      - Attention, RoPE:          heads-parallel
      - Routed MoE experts:       EP-sharded
      - Embedding, RMSNorm:       replicated
      - DP gradient AR per layer template in TRAINING when dp > 1
      - LM-head AllGather under TP > 1
      - PP activation P2P per microbatch when pp > 1 (installed by ModelMapping)
    """

    parallelism: ParallelismConfig

    def build_mm(
        self,
        model: Model,
        workload: Workload,
        devices: list[Device],
        name: str = "",
        role: str = "",
    ) -> ModelMapping:
        del name, role
        self._validate_devices(devices)
        self._validate_against_arch(model.architecture)

        s2d = self._assign_shards(model, devices)
        d2s = {dev: s for s, dev in s2d.items()}

        # Build per-phase decorated patterns and pass them in to ModelMapping.
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

        return ModelMapping(
            model=model,
            mesh=Mesh(devices=list(devices)),
            workload=workload,
            parallelism=self.parallelism,
            shards_to_devices=s2d,
            device_to_shard=d2s,
            forward_pattern=forward_pattern,
            decode_pattern=decode_pattern,
        )

    # ---- shard placement ---------------------------------------------------

    def _validate_devices(self, devices: list[Device]) -> None:
        expected = (
            self.parallelism.dp
            * self.parallelism.pp
            * self.parallelism.ep
            * self.parallelism.tp
        )
        if len(devices) != expected:
            raise ValueError(
                f"MegatronStrategy.build_mm: requires {expected} devices for "
                f"DP={self.parallelism.dp} PP={self.parallelism.pp} "
                f"EP={self.parallelism.ep} TP={self.parallelism.tp}, "
                f"got {len(devices)}"
            )

    def _assign_shards(
        self, model: Model, devices: list[Device]
    ) -> dict[Shard, Device]:
        s2d: dict[Shard, Device] = {}
        idx = 0
        # TP innermost so consecutive devices become TP-adjacent ranks.
        for d in range(self.parallelism.dp):
            for p in range(self.parallelism.pp):
                for e in range(self.parallelism.ep):
                    for t in range(self.parallelism.tp):
                        s2d[Shard(model=model, dp=d, pp=p, tp=t, ep=e)] = devices[idx]
                        idx += 1
        return s2d

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

    # ---- pattern decoration ------------------------------------------------

    def decorated_pattern(
        self, model: Model, phase: str
    ) -> list[PatternEntry]:
        """Public hook for inspecting the strategy's decorated pattern
        without constructing a full ModelMapping. Validates the architecture
        against the strategy's parallelism, then walks
        `model.forward_sequence(phase)` and applies sharding roles +
        collectives.
        """
        self._validate_against_arch(model.architecture)
        return self._decorate_pattern(
            model.forward_sequence(phase), model.architecture, phase,
        )

    def _decorate_pattern(
        self,
        labeled: list,
        arch,
        phase: str,
    ) -> list[PatternEntry]:
        """Walk a labeled pattern; produce an OperatorWithReqs-decorated form.

        - Each LabeledOp's operator has its `parallelism_role` set per the
          Megatron table.
        - Collectives (TP-AR, EP-AllToAll, LM-head AG) are attached as
          NetworkRequirements alongside the operators.
        - For TRAINING when dp > 1, a per-LayerRepeat-template DP gradient AR
          is appended to the last op of each repeated layer.
        """
        out: list[PatternEntry] = []
        tc = _phase_to_token_count(phase)
        for item in labeled:
            if isinstance(item, LayerRepeat):
                decorated_members: list[OperatorWithReqs] = [
                    self._decorate_op(m, arch, tc) for m in item.members
                ]
                if phase == "training" and self.parallelism.dp > 1:
                    template_labels = tuple(m.label for m in item.members)
                    template_ops = [op for op, _ in decorated_members]
                    dp_ar = self._dp_grad_all_reduce(template_ops, template_labels)
                    last_op, last_reqs = decorated_members[-1]
                    decorated_members[-1] = (last_op, last_reqs + [dp_ar])
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
                f"MegatronStrategy: no decoration rule for label {labeled.label!r}"
            )
        # Apply the role to the operator. Operators are frozen dataclasses;
        # use dataclasses.replace.
        op = dataclasses.replace(labeled.operator, parallelism_role=dec.role)
        reqs: list[NetworkRequirement] = []
        if dec.collective is not None:
            req = self._emit_collective(dec.collective, arch, tc)
            if req is not None:
                reqs.append(req)
        return (op, reqs)

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
        self, layer_ops: list[Operator], template_labels: tuple,
    ) -> NetworkRequirement:
        """DP gradient AllReduce sized to the per-rank parameter byte count
        of one layer template. `template_labels` is the tuple of label
        strings for the ops in this LayerRepeat, used as the dedup
        discriminator so different layer templates (e.g. V3 dense vs MoE)
        don't collapse to one requirement.
        """
        total_GB = sum(
            op.parameter_bytes_per_rank(self.parallelism) for op in layer_ops
        ) / 1e9
        return NetworkRequirement(
            kind=CollectiveKind.ALL_REDUCE,
            relation=Relation.EXCLUSIVE,
            eta=1.0,
            group=ParallelismGroup.DP,
            payload_id=("dp_grad", template_labels),
            data_GB_fn=lambda _ctx, _v=total_GB: _v,
        )
