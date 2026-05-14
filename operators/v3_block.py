"""V3 architecture builder. Produces labeled operator patterns; the
`ParallelismScheme` decorates with sharding roles and collectives.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from model import Model
from operators.elementwise import RMSNorm, SwiGLUActivation
from operators.embed import TokenEmbedding
from operators.gemm import Gemm
from operators.mla import build_mla_decode, build_mla_prefill
from operators.moe import build_moe_ffn
from operators.pattern import (
    LabeledOp,
    LayerRepeat,
    OpRole,
    PatternEntry,
)
from operators.v3_config import V3Config
from workload_context import TokenCount, WorkloadContext


def _derive_t_for(phase: TokenCount) -> Callable[[WorkloadContext], float]:
    if phase is TokenCount.PREFILL:
        return lambda ctx: float(ctx.microbatch_size * ctx.prompt_len)
    return lambda ctx: float(ctx.microbatch_size)


def _dense_ffn(cfg: V3Config, tc: TokenCount) -> list[LabeledOp]:
    """Dense FFN as labeled ops. Strategy applies column/row roles via labels."""
    return [
        LabeledOp(OpRole.FFN_EXPAND, Gemm(
            n=cfg.m_dense, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(OpRole.FFN_EXPAND, Gemm(
            n=cfg.m_dense, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(OpRole.FFN_ACT, SwiGLUActivation(
            intermediate_size=cfg.m_dense, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(OpRole.FFN_REDUCE, Gemm(
            n=cfg.h, k=cfg.m_dense, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
    ]


def build_v3_dense_layer(cfg: V3Config, phase) -> list[LabeledOp]:
    """One V3 dense decoder layer as a flat list of LabeledOps.

    Sequence: pre-attn norm, *MLA, pre-ffn norm, *FFN.
    """
    tc = TokenCount.from_phase(phase)
    mla = build_mla_prefill(cfg) if tc is TokenCount.PREFILL else build_mla_decode(cfg)
    ffn = _dense_ffn(cfg, tc)

    out: list[LabeledOp] = [
        LabeledOp(OpRole.NORM, RMSNorm(
            hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
    ]
    out.extend(mla)
    out.append(LabeledOp(OpRole.NORM, RMSNorm(
        hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
    )))
    out.extend(ffn)
    return out


def build_v3_moe_layer(cfg: V3Config, phase) -> list[LabeledOp]:
    """One V3 MoE decoder layer as a flat list of LabeledOps.

    Sequence: pre-attn norm, *MLA, pre-ffn norm, *MoE FFN.
    """
    tc = TokenCount.from_phase(phase)
    mla = build_mla_prefill(cfg) if tc is TokenCount.PREFILL else build_mla_decode(cfg)
    moe = build_moe_ffn(cfg, _derive_t_for(tc))

    out: list[LabeledOp] = [
        LabeledOp(OpRole.NORM, RMSNorm(
            hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
    ]
    out.extend(mla)
    out.append(LabeledOp(OpRole.NORM, RMSNorm(
        hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
    )))
    out.extend(moe)
    return out


def build_v3_model(cfg: V3Config, phase) -> list[PatternEntry]:
    """Full V3 model graph as a labeled pattern."""
    tc = TokenCount.from_phase(phase)
    return [
        LabeledOp(OpRole.TOKEN_EMBED, TokenEmbedding(
            vocab=cfg.vocab, h=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LayerRepeat(cfg.n_dense_layers, build_v3_dense_layer(cfg, phase)),
        LayerRepeat(cfg.n_layers - cfg.n_dense_layers, build_v3_moe_layer(cfg, phase)),
        LabeledOp(OpRole.NORM, RMSNorm(
            hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(OpRole.LM_HEAD, Gemm(
            n=cfg.vocab, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
    ]


@dataclass(frozen=True)
class V3Model(Model):
    """DeepSeek-V3-style decoder with MLA + dense + MoE layers.
    Architecture must be a `V3Config`.
    """

    def forward_sequence(self, phase: str) -> list[PatternEntry]:
        arch = self.architecture
        if not isinstance(arch, V3Config):
            raise TypeError(
                f"V3Model requires V3Config, got {type(arch).__name__}"
            )
        return build_v3_model(arch, phase)
