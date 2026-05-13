"""MLA (Multi-head Latent Attention) operator sub-builders for V3.

Emits the operator sequence for one MLA block (prefill or decode) without
sharding decisions. The parallelism strategy decorates these with roles
and collectives via labels.
"""
from __future__ import annotations

from operators.attention import DecodeAttention, PrefillAttention
from operators.elementwise import RoPE
from operators.gemm import Gemm
from operators.pattern import LabeledOp
from operators.v3_config import V3Config
from workload_context import TokenCount


# Canonical MLA labels.
MLA_Q_DOWN = "mla_q_down"
MLA_Q_UP = "mla_q_up"
MLA_KV_DOWN = "mla_kv_down"
MLA_KV_UP = "mla_kv_up"
MLA_ROPE = "mla_rope"
MLA_ATTENTION = "mla_attention"
MLA_ATTN_OUT = "mla_attn_out"


def _mla_projections(cfg: V3Config, tc: TokenCount) -> list[LabeledOp]:
    """The 4 MLA projection GEMMs + decoupled-RoPE op shared by prefill/decode.

    Megatron canonical sharding (applied by the strategy via labels):
      - q_down: replicated; output is the q-latent
      - q_up:   column-parallel — N spans heads
      - kv_down: replicated; output is the kv-latent
      - kv_up:  column-parallel — N spans heads
      - rope:   heads-parallel
    """
    return [
        LabeledOp(MLA_Q_DOWN, Gemm(
            n=cfg.d_q_compress, k=cfg.h,
            token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(MLA_Q_UP, Gemm(
            n=cfg.n_h * (cfg.head_size + cfg.d_rope),
            k=cfg.d_q_compress,
            token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(MLA_KV_DOWN, Gemm(
            n=cfg.d_kv_compress + cfg.d_rope, k=cfg.h,
            token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(MLA_KV_UP, Gemm(
            n=cfg.n_h * 2 * cfg.head_size,
            k=cfg.d_kv_compress,
            token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
        LabeledOp(MLA_ROPE, RoPE(
            num_heads=cfg.n_h, head_size=cfg.head_size, rope_dim=cfg.d_rope,
            token_count=tc, dtype_bytes=cfg.dtype_bytes,
        )),
    ]


def build_mla_prefill(cfg: V3Config) -> list[LabeledOp]:
    """One MLA layer's operator sequence in prefill mode.

    Order: q_down, q_up, kv_down, kv_up, rope, attn_core, attn_out.
    """
    tc = TokenCount.PREFILL
    ops = _mla_projections(cfg, tc)
    ops.append(LabeledOp(MLA_ATTENTION, PrefillAttention(
        num_heads=cfg.n_h, num_kv_heads=cfg.n_h,
        head_size=cfg.head_size + cfg.d_rope,
        flash_block_size=cfg.flash_block_size,
        dtype_bytes=cfg.dtype_bytes,
    )))
    ops.append(LabeledOp(MLA_ATTN_OUT, Gemm(
        n=cfg.h, k=cfg.n_h * cfg.head_size,
        token_count=tc, dtype_bytes=cfg.dtype_bytes,
    )))
    return ops


def build_mla_decode(cfg: V3Config) -> list[LabeledOp]:
    """One MLA layer in decode mode.

    DecodeAttention's kv_bytes_per_token is overridden to the MLA latent
    footprint — the dominant memory win MLA buys at decode.
    """
    tc = TokenCount.DECODE
    ops = _mla_projections(cfg, tc)
    ops.append(LabeledOp(MLA_ATTENTION, DecodeAttention(
        num_heads=cfg.n_h, num_kv_heads=cfg.n_h,
        head_size=cfg.head_size + cfg.d_rope,
        dtype_bytes=cfg.dtype_bytes,
        kv_bytes_per_token=(cfg.d_kv_compress + cfg.d_rope) * cfg.dtype_bytes,
    )))
    ops.append(LabeledOp(MLA_ATTN_OUT, Gemm(
        n=cfg.h, k=cfg.n_h * cfg.head_size,
        token_count=tc, dtype_bytes=cfg.dtype_bytes,
    )))
    return ops
