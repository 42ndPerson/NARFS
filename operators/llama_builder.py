"""Llama architecture builder. Emits operators tagged with semantic
`OpRole`s; the `ParallelismScheme` decorates them with sharding decisions.

This module is parallelism-agnostic: it emits the structural sequence of
a Llama transformer using the shared `OpRole` taxonomy, so a new scheme
can support Llama (and every other architecture using the same roles) by
defining one role→decoration table.
"""
from __future__ import annotations
from dataclasses import dataclass

from model import Model
from operators.attention import DecodeAttention, PrefillAttention
from operators.pattern import (
    LabeledOp,
    LayerRepeat,
    OpRole,
    PatternEntry,
)
from operators.elementwise import RMSNorm, RoPE, SwiGLUActivation
from operators.embed import TokenEmbedding
from operators.gemm import Gemm
from operators.llama_config import LlamaConfig
from workload_context import TokenCount


def build_llama_layer(arch: LlamaConfig, phase: str) -> list[LabeledOp]:
    """One Llama decoder layer as a flat list of LabeledOps.

    Operators carry full-model shapes and are parallelism-pure. The
    strategy decorates the labeled list with sharding roles and collective
    requirements externally and bakes per-rank arithmetic into the
    CombinedOp closure.
    """
    tc = TokenCount.from_phase(phase)
    qkv_n = (arch.n_q + 2 * arch.n_kv) * arch.head_size
    q_dim = arch.n_q * arch.head_size

    if tc is TokenCount.PREFILL:
        attention = PrefillAttention(
            num_heads=arch.n_q,
            num_kv_heads=arch.n_kv,
            head_size=arch.head_size,
            flash_block_size=arch.flash_block_size,
            dtype_bytes=arch.dtype_bytes,
        )
    else:
        attention = DecodeAttention(
            num_heads=arch.n_q,
            num_kv_heads=arch.n_kv,
            head_size=arch.head_size,
            dtype_bytes=arch.dtype_bytes,
        )

    return [
        LabeledOp(OpRole.NORM, RMSNorm(arch.h, tc, arch.dtype_bytes)),
        LabeledOp(OpRole.QKV_PROJ, Gemm(n=qkv_n, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes)),
        LabeledOp(OpRole.ROPE, RoPE(
            num_heads=arch.n_q + arch.n_kv,
            head_size=arch.head_size,
            rope_dim=arch.rope_dim,
            token_count=tc,
            dtype_bytes=arch.dtype_bytes,
        )),
        LabeledOp(OpRole.ATTN_HEADS, attention),
        LabeledOp(OpRole.ATTN_OUT, Gemm(n=arch.h, k=q_dim, token_count=tc, dtype_bytes=arch.dtype_bytes)),
        LabeledOp(OpRole.NORM, RMSNorm(arch.h, tc, arch.dtype_bytes)),
        LabeledOp(OpRole.FFN_EXPAND, Gemm(n=arch.m, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes)),
        LabeledOp(OpRole.FFN_EXPAND, Gemm(n=arch.m, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes)),
        LabeledOp(OpRole.FFN_ACT, SwiGLUActivation(
            intermediate_size=arch.m, token_count=tc, dtype_bytes=arch.dtype_bytes,
        )),
        LabeledOp(OpRole.FFN_REDUCE, Gemm(n=arch.h, k=arch.m, token_count=tc, dtype_bytes=arch.dtype_bytes)),
    ]


@dataclass(frozen=True)
class LlamaModel(Model):
    """Llama-style decoder model. Architecture must be a `LlamaConfig`.

    Full graph: embed, n_layers transformer blocks, final norm, LM head.
    Emits a labeled pattern; the strategy applies sharding and collectives.
    """

    def forward_sequence(self, phase: str) -> list[PatternEntry]:
        arch = self.architecture
        if not isinstance(arch, LlamaConfig):
            raise TypeError(
                f"LlamaModel requires LlamaConfig, got {type(arch).__name__}"
            )

        tc = TokenCount.from_phase(phase)
        return [
            LabeledOp(OpRole.TOKEN_EMBED, TokenEmbedding(
                vocab=arch.vocab, h=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes,
            )),
            LayerRepeat(arch.n_layers, build_llama_layer(arch, phase)),
            LabeledOp(OpRole.NORM, RMSNorm(arch.h, tc, arch.dtype_bytes)),
            LabeledOp(OpRole.LM_HEAD, Gemm(
                n=arch.vocab, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes,
            )),
        ]
