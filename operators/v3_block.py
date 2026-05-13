"""V3 architecture builder. Produces labeled operator patterns; the
ParallelismStrategy decorates with sharding roles and collectives.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from model import Model
from network_requirement import ParallelismConfig
from operators.elementwise import RMSNorm, SwiGLUActivation
from operators.embed import TokenEmbedding
from operators.gemm import Gemm
from operators.mla import build_mla_decode, build_mla_prefill
from operators.moe import build_moe_ffn
from operators.pattern import (
    LabeledOp,
    LayerRepeat,
    PatternEntry,
)
from operators.v3_config import V3Config
from workload_context import TokenCount, WorkloadContext


# Canonical V3 top-level labels.
V3_EMBED = "v3_embed"
V3_ATTN_NORM = "v3_attn_norm"
V3_FFN_NORM = "v3_ffn_norm"
V3_FFN_GATE = "v3_ffn_gate"
V3_FFN_UP = "v3_ffn_up"
V3_FFN_SWIGLU = "v3_ffn_swiglu"
V3_FFN_DOWN = "v3_ffn_down"
V3_FINAL_NORM = "v3_final_norm"
V3_LM_HEAD = "v3_lm_head"


def _phase_to_token_count(phase) -> TokenCount:
    if isinstance(phase, TokenCount):
        return phase
    if phase in ("prefill", "training"):
        return TokenCount.PREFILL
    if phase == "decode":
        return TokenCount.DECODE
    raise ValueError(f"Unknown phase: {phase!r}")


def _derive_t_for(phase: TokenCount) -> Callable[[WorkloadContext], float]:
    if phase is TokenCount.PREFILL:
        return lambda ctx: float(ctx.microbatch_size * ctx.prompt_len)
    return lambda ctx: float(ctx.microbatch_size)


def _dense_ffn(cfg: V3Config, tc: TokenCount) -> list[LabeledOp]:
    """Dense FFN as labeled ops. Strategy applies column/row roles via labels."""
    return [
        LabeledOp(V3_FFN_GATE, Gemm(
            n=cfg.m_dense, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(V3_FFN_UP, Gemm(
            n=cfg.m_dense, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(V3_FFN_SWIGLU, SwiGLUActivation(
            intermediate_size=cfg.m_dense, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(V3_FFN_DOWN, Gemm(
            n=cfg.h, k=cfg.m_dense, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
    ]


def build_v3_dense_layer(cfg: V3Config, phase) -> list[LabeledOp]:
    """One V3 dense decoder layer as a flat list of LabeledOps.

    Sequence: pre-attn norm, *MLA, pre-ffn norm, *FFN.
    """
    tc = _phase_to_token_count(phase)
    mla = build_mla_prefill(cfg) if tc is TokenCount.PREFILL else build_mla_decode(cfg)
    ffn = _dense_ffn(cfg, tc)

    out: list[LabeledOp] = [
        LabeledOp(V3_ATTN_NORM, RMSNorm(
            hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
    ]
    out.extend(mla)
    out.append(LabeledOp(V3_FFN_NORM, RMSNorm(
        hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
    )))
    out.extend(ffn)
    return out


def build_v3_moe_layer(cfg: V3Config, phase) -> list[LabeledOp]:
    """One V3 MoE decoder layer as a flat list of LabeledOps.

    Sequence: pre-attn norm, *MLA, pre-ffn norm, *MoE FFN.
    """
    tc = _phase_to_token_count(phase)
    mla = build_mla_prefill(cfg) if tc is TokenCount.PREFILL else build_mla_decode(cfg)
    moe = build_moe_ffn(cfg, _derive_t_for(tc))

    out: list[LabeledOp] = [
        LabeledOp(V3_ATTN_NORM, RMSNorm(
            hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
    ]
    out.extend(mla)
    out.append(LabeledOp(V3_FFN_NORM, RMSNorm(
        hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
    )))
    out.extend(moe)
    return out


def build_v3_model(cfg: V3Config, phase) -> list[PatternEntry]:
    """Full V3 model graph as a labeled pattern."""
    tc = _phase_to_token_count(phase)
    return [
        LabeledOp(V3_EMBED, TokenEmbedding(
            vocab=cfg.vocab, h=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LayerRepeat(cfg.n_dense_layers, build_v3_dense_layer(cfg, phase)),
        LayerRepeat(cfg.n_layers - cfg.n_dense_layers, build_v3_moe_layer(cfg, phase)),
        LabeledOp(V3_FINAL_NORM, RMSNorm(
            hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(V3_LM_HEAD, Gemm(
            n=cfg.vocab, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
    ]


@dataclass(frozen=True)
class V3Model(Model):
    """DeepSeek-V3-style decoder with MLA + dense + MoE layers.
    Architecture must be a `V3Config`.
    """

    def forward_sequence(
        self, phase: str, parallelism: ParallelismConfig | None = None
    ) -> list[PatternEntry]:
        # `parallelism` accepted for legacy callers but unused — strategy
        # owns parallelism decisions. Phase 2b will drop it from the ABC.
        del parallelism
        arch = self.architecture
        if not isinstance(arch, V3Config):
            raise TypeError(
                f"V3Model requires V3Config, got {type(arch).__name__}"
            )
        return build_v3_model(arch, phase)
