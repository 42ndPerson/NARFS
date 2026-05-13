from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

from model import Model
from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operator_base import Operator
from operators.pattern import (
    OperatorWithReqs,
    PatternEntry,
    LayerRepeat,
)
from operators.elementwise import RMSNorm, SwiGLUActivation
from operators.embed import TokenEmbedding
from operators.gemm import Gemm
from operators.mla import build_mla_decode, build_mla_prefill
from operators.moe import (
    CombineMarker,
    DispatchMarker,
    build_moe_ffn,
)
from operators.v3_config import V3Config
from parallelism_role import ParallelismRole, TPMode
from relation import Relation
from workload_context import TokenCount, WorkloadContext


_COL = ParallelismRole(tp=TPMode.COLUMN_PARALLEL)
_ROW = ParallelismRole(tp=TPMode.ROW_PARALLEL)
_VOCAB = ParallelismRole(tp=TPMode.VOCAB_PARALLEL)


def _phase_to_token_count(phase) -> TokenCount:
    # Accept either a TokenCount (legacy callers from Phase 3 tests) or a
    # phase string ("prefill" / "decode" / "training").
    if isinstance(phase, TokenCount):
        return phase
    if phase in ("prefill", "training"):
        return TokenCount.PREFILL
    if phase == "decode":
        return TokenCount.DECODE
    raise ValueError(f"Unknown phase: {phase!r}")


def _activation_size_fn(h: int, dtype_bytes: int, tc: TokenCount):
    """Closure for hidden-state activation: T * h * dtype_bytes / 1e9."""
    def _fn(ctx) -> float:
        return tc.resolve(ctx) * h * dtype_bytes / 1e9
    return _fn


def _ep_dispatched_size_fn(h: int, dtype_bytes: int, top_k: int, tc: TokenCount):
    """Closure for MoE dispatch/combine: top_k * T * h * dtype_bytes / 1e9."""
    def _fn(ctx) -> float:
        return top_k * tc.resolve(ctx) * h * dtype_bytes / 1e9
    return _fn


def _lm_head_gather_size_fn(vocab: int, dtype_bytes: int, tc: TokenCount):
    """Closure for LM-head AllGather: T * vocab * dtype_bytes / 1e9."""
    def _fn(ctx) -> float:
        return tc.resolve(ctx) * vocab * dtype_bytes / 1e9
    return _fn


def _tp_all_reduce(cfg: V3Config, tc: TokenCount) -> NetworkRequirement:
    return NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
        data_GB_fn=_activation_size_fn(cfg.h, cfg.dtype_bytes, tc),
    )


def _ep_dispatch(cfg: V3Config, tc: TokenCount) -> NetworkRequirement:
    return NetworkRequirement(
        kind=CollectiveKind.ALL_TO_ALL_DISPATCH,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.EP,
        data_GB_fn=_ep_dispatched_size_fn(cfg.h, cfg.dtype_bytes, cfg.top_k, tc),
    )


def _ep_combine(cfg: V3Config, tc: TokenCount) -> NetworkRequirement:
    return NetworkRequirement(
        kind=CollectiveKind.ALL_TO_ALL_COMBINE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.EP,
        data_GB_fn=_ep_dispatched_size_fn(cfg.h, cfg.dtype_bytes, cfg.top_k, tc),
    )


def _lm_head_all_gather(cfg: V3Config, tc: TokenCount) -> NetworkRequirement:
    return NetworkRequirement(
        kind=CollectiveKind.ALL_GATHER,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
        data_GB_fn=_lm_head_gather_size_fn(cfg.vocab, cfg.dtype_bytes, tc),
    )


def _dp_grad_all_reduce(
    ops: list[Operator], parallelism: ParallelismConfig, payload_tag: str
) -> NetworkRequirement:
    """DP gradient AllReduce sized to the per-rank parameter byte count."""
    total_GB = sum(op.parameter_bytes_per_rank(parallelism) for op in ops) / 1e9
    return NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.DP,
        payload_id=("dp_grad", payload_tag),
        data_GB_fn=lambda _ctx, _v=total_GB: _v,
    )


def _derive_t_for(phase: TokenCount) -> Callable[[WorkloadContext], float]:
    if phase is TokenCount.PREFILL:
        return lambda ctx: float(ctx.microbatch_size * ctx.prompt_len)
    return lambda ctx: float(ctx.microbatch_size)


def _dense_ffn(cfg: V3Config, phase: TokenCount) -> list[Operator]:
    """Dense FFN with column/row TP roles along the intermediate dim."""
    return [
        Gemm(n=cfg.m_dense, k=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes,
             parallelism_role=_COL),
        Gemm(n=cfg.m_dense, k=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes,
             parallelism_role=_COL),
        SwiGLUActivation(intermediate_size=cfg.m_dense, token_count=phase,
                         dtype_bytes=cfg.dtype_bytes, parallelism_role=_COL),
        Gemm(n=cfg.h, k=cfg.m_dense, token_count=phase, dtype_bytes=cfg.dtype_bytes,
             parallelism_role=_ROW),
    ]


def build_v3_dense_layer(
    cfg: V3Config,
    phase,
    parallelism: Optional[ParallelismConfig] = None,
) -> list[OperatorWithReqs]:
    """One V3 dense decoder layer with full-model shapes + TP roles."""
    parallelism = parallelism or ParallelismConfig()
    tp = parallelism.tp
    if tp > 1 and cfg.m_dense % tp != 0:
        raise ValueError(f"m_dense ({cfg.m_dense}) must be divisible by tp ({tp})")

    tc = _phase_to_token_count(phase)
    mla = build_mla_prefill(cfg, tp) if tc is TokenCount.PREFILL else build_mla_decode(cfg, tp)
    ffn = _dense_ffn(cfg, tc)

    no_req: list[NetworkRequirement] = []
    ar = [_tp_all_reduce(cfg, tc)] if tp > 1 else no_req

    out: list[OperatorWithReqs] = [
        (RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), no_req),
    ]
    for i, op in enumerate(mla):
        is_attn_out = i == len(mla) - 1
        out.append((op, ar if is_attn_out else no_req))
    out.append((RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), no_req))
    for i, op in enumerate(ffn):
        is_ffn_down = i == len(ffn) - 1
        out.append((op, ar if is_ffn_down else no_req))

    if phase == "training" and parallelism.dp > 1:
        dp_ar = _dp_grad_all_reduce(
            [op for op, _ in out], parallelism, "v3_dense_layer"
        )
        last_op, last_reqs = out[-1]
        out[-1] = (last_op, last_reqs + [dp_ar])
    return out


def build_v3_moe_layer(
    cfg: V3Config,
    phase,
    parallelism: Optional[ParallelismConfig] = None,
) -> list[OperatorWithReqs]:
    """One V3 MoE decoder layer.

    TP shards: MLA heads, shared-expert intermediate. Routed experts are
    EP-sharded only (their per-expert intermediate stays at full `m_expert`).
    """
    parallelism = parallelism or ParallelismConfig()
    tp = parallelism.tp
    if tp > 1 and cfg.m_expert % tp != 0:
        raise ValueError(f"m_expert ({cfg.m_expert}) must be divisible by tp ({tp})")

    tc = _phase_to_token_count(phase)
    mla = build_mla_prefill(cfg, tp) if tc is TokenCount.PREFILL else build_mla_decode(cfg, tp)
    moe = build_moe_ffn(cfg, _derive_t_for(tc))

    no_req: list[NetworkRequirement] = []
    ar = [_tp_all_reduce(cfg, tc)] if parallelism.tp > 1 else no_req
    dispatch = [_ep_dispatch(cfg, tc)] if parallelism.ep > 1 else no_req
    combine = [_ep_combine(cfg, tc)] if parallelism.ep > 1 else no_req

    out: list[OperatorWithReqs] = [
        (RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), no_req),
    ]
    for i, op in enumerate(mla):
        is_attn_out = i == len(mla) - 1
        out.append((op, ar if is_attn_out else no_req))
    out.append((RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), no_req))
    # MoE layout from build_moe_ffn:
    # [router, dispatch_marker, grouped_gate, grouped_up, grouped_swiglu, grouped_down,
    #  combine_marker, shared_gate, shared_up, shared_swiglu, shared_down]
    last_idx = len(moe) - 1
    for i, op in enumerate(moe):
        if isinstance(op, DispatchMarker):
            reqs = dispatch
        elif isinstance(op, CombineMarker):
            reqs = combine
        elif i == last_idx:
            reqs = ar
        else:
            reqs = no_req
        out.append((op, reqs))

    if phase == "training" and parallelism.dp > 1:
        dp_ar = _dp_grad_all_reduce(
            [op for op, _ in out], parallelism, "v3_moe_layer"
        )
        last_op, last_reqs = out[-1]
        out[-1] = (last_op, last_reqs + [dp_ar])
    return out


def build_v3_model(
    cfg: V3Config,
    phase,
    parallelism: Optional[ParallelismConfig] = None,
) -> list[PatternEntry]:
    """Full V3 model graph: embed, dense layers, MoE layers, final norm, LM head."""
    parallelism = parallelism or ParallelismConfig()
    tp = parallelism.tp
    if tp > 1 and cfg.vocab % tp != 0:
        raise ValueError(f"vocab ({cfg.vocab}) must be divisible by tp ({tp})")

    tc = _phase_to_token_count(phase)
    dense = build_v3_dense_layer(cfg, phase, parallelism)
    moe = build_v3_moe_layer(cfg, phase, parallelism)
    lm_head_reqs: list[NetworkRequirement] = (
        [_lm_head_all_gather(cfg, tc)] if tp > 1 else []
    )
    return [
        # Embedding is replicated across TP.
        (TokenEmbedding(vocab=cfg.vocab, h=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), []),
        LayerRepeat(cfg.n_dense_layers, dense),
        LayerRepeat(cfg.n_layers - cfg.n_dense_layers, moe),
        (RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), []),
        # LM head: vocab-parallel — splits N along the vocab dim.
        (Gemm(n=cfg.vocab, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
              parallelism_role=_VOCAB), lm_head_reqs),
    ]


@dataclass(frozen=True)
class V3Model(Model):
    """DeepSeek-V3-style decoder with MLA + dense + MoE layers. Architecture
    must be a `V3Config`.
    """

    def forward_sequence(
        self, phase: str, parallelism: ParallelismConfig
    ) -> list[PatternEntry]:
        arch = self.architecture
        if not isinstance(arch, V3Config):
            raise TypeError(
                f"V3Model requires V3Config, got {type(arch).__name__}"
            )
        return build_v3_model(arch, phase, parallelism)
