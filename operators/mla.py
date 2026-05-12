from __future__ import annotations

from operator_base import Operator
from operators.attention import DecodeAttention, PrefillAttention
from operators.elementwise import RoPE
from operators.gemm import Gemm
from operators.v3_config import V3Config
from parallelism_role import ParallelismRole, TPMode
from workload_context import TokenCount


_COL = ParallelismRole(tp=TPMode.COLUMN_PARALLEL)
_ROW = ParallelismRole(tp=TPMode.ROW_PARALLEL)
_HEADS = ParallelismRole(tp=TPMode.HEADS_PARALLEL)
_REP = ParallelismRole()


def _mla_projections(cfg: V3Config, tc: TokenCount) -> list[Operator]:
    """The 4 MLA projection GEMMs + decoupled-RoPE op shared by prefill/decode.

    Megatron-style TP roles:
      - q_down: not sharded; output is the q-latent (replicated)
      - q_up:   column-parallel — N spans heads
      - kv_down: not sharded; output is the kv-latent (replicated)
      - kv_up:  column-parallel — N spans heads
      - rope:   heads-parallel
    """
    return [
        Gemm(n=cfg.d_q_compress, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes,
             parallelism_role=_REP),
        Gemm(
            n=cfg.n_h * (cfg.head_size + cfg.d_rope),
            k=cfg.d_q_compress,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
            parallelism_role=_COL,
        ),
        Gemm(n=cfg.d_kv_compress + cfg.d_rope, k=cfg.h, token_count=tc,
             dtype_bytes=cfg.dtype_bytes, parallelism_role=_REP),
        Gemm(
            n=cfg.n_h * 2 * cfg.head_size,
            k=cfg.d_kv_compress,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
            parallelism_role=_COL,
        ),
        RoPE(
            num_heads=cfg.n_h,
            head_size=cfg.head_size,
            rope_dim=cfg.d_rope,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
            parallelism_role=_HEADS,
        ),
    ]


def build_mla_prefill(cfg: V3Config, tp: int = 1) -> list[Operator]:
    """One MLA layer's operator sequence in prefill mode.

    Order: q_down, q_up, kv_down, kv_up, rope, attn_core, attn_out.
    """
    if tp > 1 and cfg.n_h % tp != 0:
        raise ValueError(f"n_h ({cfg.n_h}) must be divisible by tp ({tp})")
    tc = TokenCount.PREFILL
    ops = _mla_projections(cfg, tc)
    ops.append(
        PrefillAttention(
            num_heads=cfg.n_h,
            num_kv_heads=cfg.n_h,
            head_size=cfg.head_size + cfg.d_rope,
            flash_block_size=cfg.flash_block_size,
            dtype_bytes=cfg.dtype_bytes,
            parallelism_role=_HEADS,
        )
    )
    # attn_out: row-parallel — input dim (n_h * head_size) is sharded.
    ops.append(
        Gemm(
            n=cfg.h,
            k=cfg.n_h * cfg.head_size,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
            parallelism_role=_ROW,
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
    if tp > 1 and cfg.n_h % tp != 0:
        raise ValueError(f"n_h ({cfg.n_h}) must be divisible by tp ({tp})")
    tc = TokenCount.DECODE
    ops = _mla_projections(cfg, tc)
    ops.append(
        DecodeAttention(
            num_heads=cfg.n_h,
            num_kv_heads=cfg.n_h,
            head_size=cfg.head_size + cfg.d_rope,
            dtype_bytes=cfg.dtype_bytes,
            kv_bytes_per_token=(cfg.d_kv_compress + cfg.d_rope) * cfg.dtype_bytes,
            parallelism_role=_HEADS,
        )
    )
    ops.append(
        Gemm(
            n=cfg.h,
            k=cfg.n_h * cfg.head_size,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
            parallelism_role=_ROW,
        )
    )
    return ops
