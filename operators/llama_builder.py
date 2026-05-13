"""Llama architecture builder. Produces labeled operator patterns that
strategies decorate with sharding roles and collectives.

This module is parallelism-agnostic: it emits the structural sequence of a
Llama transformer, tagging each operator with a canonical label. A
ParallelismStrategy walks the labeled pattern and applies its scheme-specific
rules (Megatron shard direction, requirement emission, divisibility
validation).
"""
from __future__ import annotations
from dataclasses import dataclass

from model import Model
from network_requirement import ParallelismConfig
from operators.attention import DecodeAttention, PrefillAttention
from operators.pattern import (
    LabeledOp,
    LayerRepeat,
    PatternEntry,
)
from operators.elementwise import RMSNorm, RoPE, SwiGLUActivation
from operators.embed import TokenEmbedding
from operators.gemm import Gemm
from operators.llama_config import LlamaConfig
from workload_context import TokenCount


# Canonical labels for Llama operators. Strategies look these up in their
# label→decision tables to decide sharding, collective emission, etc.
LLAMA_EMBED = "llama_embed"
LLAMA_ATTN_NORM = "llama_attn_norm"
LLAMA_QKV = "llama_qkv"
LLAMA_ROPE = "llama_rope"
LLAMA_ATTENTION = "llama_attention"
LLAMA_ATTN_OUT = "llama_attn_out"
LLAMA_FFN_NORM = "llama_ffn_norm"
LLAMA_FFN_GATE = "llama_ffn_gate"
LLAMA_FFN_UP = "llama_ffn_up"
LLAMA_FFN_SWIGLU = "llama_ffn_swiglu"
LLAMA_FFN_DOWN = "llama_ffn_down"
LLAMA_FINAL_NORM = "llama_final_norm"
LLAMA_LM_HEAD = "llama_lm_head"


def _phase_to_token_count(phase: str) -> TokenCount:
    if phase in ("prefill", "training"):
        return TokenCount.PREFILL
    if phase == "decode":
        return TokenCount.DECODE
    raise ValueError(f"Unknown phase: {phase!r}")


def build_llama_layer(arch: LlamaConfig, phase: str) -> list[LabeledOp]:
    """One Llama decoder layer as a flat list of LabeledOps.

    Operators carry full-model shapes. Sharding roles are NOT set here —
    a parallelism strategy decorates them later based on labels. Per-rank
    division is handled at the operator level via `parallelism_role` set
    by the strategy.

    (Phase 2c will strip `parallelism_role` from operators entirely; for
    now the operators still default to REPLICATED and strategies override
    via construction.)
    """
    tc = _phase_to_token_count(phase)
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
        LabeledOp(LLAMA_ATTN_NORM, RMSNorm(arch.h, tc, arch.dtype_bytes)),
        LabeledOp(LLAMA_QKV, Gemm(n=qkv_n, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes)),
        LabeledOp(LLAMA_ROPE, RoPE(
            num_heads=arch.n_q + arch.n_kv,
            head_size=arch.head_size,
            rope_dim=arch.rope_dim,
            token_count=tc,
            dtype_bytes=arch.dtype_bytes,
        )),
        LabeledOp(LLAMA_ATTENTION, attention),
        LabeledOp(LLAMA_ATTN_OUT, Gemm(n=arch.h, k=q_dim, token_count=tc, dtype_bytes=arch.dtype_bytes)),
        LabeledOp(LLAMA_FFN_NORM, RMSNorm(arch.h, tc, arch.dtype_bytes)),
        LabeledOp(LLAMA_FFN_GATE, Gemm(n=arch.m, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes)),
        LabeledOp(LLAMA_FFN_UP, Gemm(n=arch.m, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes)),
        LabeledOp(LLAMA_FFN_SWIGLU, SwiGLUActivation(
            intermediate_size=arch.m, token_count=tc, dtype_bytes=arch.dtype_bytes,
        )),
        LabeledOp(LLAMA_FFN_DOWN, Gemm(n=arch.h, k=arch.m, token_count=tc, dtype_bytes=arch.dtype_bytes)),
    ]


@dataclass(frozen=True)
class LlamaModel(Model):
    """Llama-style decoder model. Architecture must be a `LlamaConfig`.

    Full graph: embed, n_layers transformer blocks, final norm, LM head.
    Emits a labeled pattern; the strategy applies sharding and collectives.
    """

    def forward_sequence(
        self, phase: str, parallelism: ParallelismConfig | None = None
    ) -> list[PatternEntry]:
        # `parallelism` is accepted for legacy callers but no longer used —
        # the strategy now owns parallelism decisions. Phase 2b drops the
        # parameter from the ABC signature.
        del parallelism
        arch = self.architecture
        if not isinstance(arch, LlamaConfig):
            raise TypeError(
                f"LlamaModel requires LlamaConfig, got {type(arch).__name__}"
            )

        tc = _phase_to_token_count(phase)
        return [
            LabeledOp(LLAMA_EMBED, TokenEmbedding(
                vocab=arch.vocab, h=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes,
            )),
            LayerRepeat(arch.n_layers, build_llama_layer(arch, phase)),
            LabeledOp(LLAMA_FINAL_NORM, RMSNorm(arch.h, tc, arch.dtype_bytes)),
            LabeledOp(LLAMA_LM_HEAD, Gemm(
                n=arch.vocab, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes,
            )),
        ]
