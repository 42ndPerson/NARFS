from __future__ import annotations

from operator_base import Operator
from operators.attention import DecodeAttention, PrefillAttention
from operators.elementwise import RoPE
from operators.gemm import Gemm
from operators.v3_config import V3Config
from workload_context import TokenCount


def _mla_projections(cfg: V3Config, tc: TokenCount, n_h_local: int) -> list[Operator]:
    """The 4 MLA projection GEMMs + decoupled-RoPE op shared by prefill/decode.

    Megatron-style TP: q_up, kv_up, and rope are sharded by head count
    (`n_h_local = n_h / tp`). q_down and kv_down are not sharded — their
    output is the latent representation, which is replicated across TP ranks.
    """
    return [
        Gemm(n=cfg.d_q_compress, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes),
        Gemm(
            n=n_h_local * (cfg.head_size + cfg.d_rope),
            k=cfg.d_q_compress,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        ),
        Gemm(
            n=cfg.d_kv_compress + cfg.d_rope,
            k=cfg.h,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        ),
        Gemm(
            n=n_h_local * 2 * cfg.head_size,
            k=cfg.d_kv_compress,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        ),
        RoPE(
            num_heads=n_h_local,
            head_size=cfg.head_size,
            rope_dim=cfg.d_rope,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        ),
    ]


def build_mla_prefill(cfg: V3Config, tp: int = 1) -> list[Operator]:
    """One MLA layer's operator sequence in prefill mode.

    Order: q_down, q_up, kv_down, kv_up, rope, attn_core, attn_out.
    Effective head_size for attn_core is head_size + d_rope (Q/K concat the
    decoupled rope dim for scoring); the value path keeps head_size, so
    attn_out's K is n_h_local * head_size.
    """
    if cfg.n_h % tp != 0:
        raise ValueError(f"n_h ({cfg.n_h}) must be divisible by tp ({tp})")
    n_h_local = cfg.n_h // tp
    tc = TokenCount.PREFILL
    ops = _mla_projections(cfg, tc, n_h_local)
    ops.append(
        PrefillAttention(
            num_heads=n_h_local,
            num_kv_heads=n_h_local,
            head_size=cfg.head_size + cfg.d_rope,
            flash_block_size=cfg.flash_block_size,
            dtype_bytes=cfg.dtype_bytes,
        )
    )
    ops.append(
        Gemm(
            n=cfg.h,
            k=n_h_local * cfg.head_size,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        )
    )
    return ops


def build_mla_decode(cfg: V3Config, tp: int = 1) -> list[Operator]:
    """One MLA layer in decode mode.

    Same operator structure as prefill but decode-shaped tokens, and the key
    difference: DecodeAttention's kv_bytes_per_token is overridden to the
    MLA latent footprint (d_kv_compress + d_rope) * dtype_bytes — this is the
    dominant memory win MLA buys at decode.
    """
    if cfg.n_h % tp != 0:
        raise ValueError(f"n_h ({cfg.n_h}) must be divisible by tp ({tp})")
    n_h_local = cfg.n_h // tp
    tc = TokenCount.DECODE
    ops = _mla_projections(cfg, tc, n_h_local)
    ops.append(
        DecodeAttention(
            num_heads=n_h_local,
            num_kv_heads=n_h_local,
            head_size=cfg.head_size + cfg.d_rope,
            dtype_bytes=cfg.dtype_bytes,
            kv_bytes_per_token=(cfg.d_kv_compress + cfg.d_rope) * cfg.dtype_bytes,
        )
    )
    ops.append(
        Gemm(
            n=cfg.h,
            k=n_h_local * cfg.head_size,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        )
    )
    return ops
