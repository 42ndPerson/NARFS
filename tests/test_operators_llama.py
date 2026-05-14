from __future__ import annotations
from dataclasses import dataclass

import pytest

from hardware import HardwareSpec
from operator_base import Operator
from operators.attention import DecodeAttention, PrefillAttention
from operators.elementwise import RMSNorm, RoPE, SwiGLUActivation
from operators.gemm import Gemm
from workload_context import (
    TokenCount,
    Workload,
    WorkloadContext,
)


# A100-style synthetic spec.
A100 = HardwareSpec(
    peak_compute_flops=312e12,        # FP16 peak
    peak_memory_bandwidth=2.0e12,     # ~2 TB/s HBM
)


# ---- local transformer-block fixture ----------------------------------------
# This is test scaffolding for whole-model FLOP accounting — it predates the
# labeled-pattern builders and stays here so the test owns its own fixture
# rather than depending on a production module.


@dataclass(frozen=True)
class TransformerBlockConfig:
    hidden_size: int
    num_heads: int          # query heads
    num_kv_heads: int       # KV heads (GQA)
    head_size: int
    intermediate_size: int  # FFN inner dim
    rope_dim: int
    flash_block_size: int = 64
    dtype_bytes: int = 2


def _transformer_block_operators(
    c: TransformerBlockConfig, tc: TokenCount
) -> list[Operator]:
    """One decoder layer's operator sequence (residual adds folded in)."""
    q_dim = c.num_heads * c.head_size
    kv_dim = c.num_kv_heads * c.head_size

    if tc is TokenCount.PREFILL:
        attention: Operator = PrefillAttention(
            num_heads=c.num_heads,
            num_kv_heads=c.num_kv_heads,
            head_size=c.head_size,
            flash_block_size=c.flash_block_size,
            dtype_bytes=c.dtype_bytes,
        )
    else:
        attention = DecodeAttention(
            num_heads=c.num_heads,
            num_kv_heads=c.num_kv_heads,
            head_size=c.head_size,
            dtype_bytes=c.dtype_bytes,
        )

    return [
        RMSNorm(c.hidden_size, tc, c.dtype_bytes),
        Gemm(n=q_dim + 2 * kv_dim, k=c.hidden_size, token_count=tc, dtype_bytes=c.dtype_bytes),
        RoPE(
            num_heads=c.num_heads + c.num_kv_heads,
            head_size=c.head_size,
            rope_dim=c.rope_dim,
            token_count=tc,
            dtype_bytes=c.dtype_bytes,
        ),
        attention,
        Gemm(n=c.hidden_size, k=q_dim, token_count=tc, dtype_bytes=c.dtype_bytes),
        RMSNorm(c.hidden_size, tc, c.dtype_bytes),
        Gemm(n=c.intermediate_size, k=c.hidden_size, token_count=tc, dtype_bytes=c.dtype_bytes),
        Gemm(n=c.intermediate_size, k=c.hidden_size, token_count=tc, dtype_bytes=c.dtype_bytes),
        SwiGLUActivation(c.intermediate_size, tc, c.dtype_bytes),
        Gemm(n=c.hidden_size, k=c.intermediate_size, token_count=tc, dtype_bytes=c.dtype_bytes),
    ]


# Llama-3-8B architecture
LLAMA3_8B = TransformerBlockConfig(
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
    head_size=128,
    intermediate_size=14336,
    rope_dim=128,
    flash_block_size=64,
    dtype_bytes=2,
)
LLAMA3_8B_LAYERS = 32
LLAMA3_8B_VOCAB = 128_000
LLAMA3_8B_PARAMS = 8.03e9  # public number for Llama-3-8B


def _prefill_ctx(batch: int, prompt_len: int) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.GENERATION,
        batch_size=batch,
        microbatch_size=batch,
        prompt_len=prompt_len,
        response_len=0,
        num_microbatches=1,
    )


def _llama3_8b_operators() -> list[Operator]:
    """Flat operator list for the whole Llama-3-8B forward graph."""
    block = _transformer_block_operators(LLAMA3_8B, TokenCount.PREFILL)
    final = [
        RMSNorm(LLAMA3_8B.hidden_size, TokenCount.PREFILL, LLAMA3_8B.dtype_bytes),
        Gemm(
            n=LLAMA3_8B_VOCAB,
            k=LLAMA3_8B.hidden_size,
            token_count=TokenCount.PREFILL,
            dtype_bytes=LLAMA3_8B.dtype_bytes,
        ),
    ]
    return block * LLAMA3_8B_LAYERS + final


def test_llama3_8b_prefill_total_flops():
    # Forward-only rule of thumb: ~2 * P * T FLOPs for a dense decoder LLM.
    # For Llama-3-8B at 2048 tokens that's ~2 * 8.03e9 * 2048 ≈ 32.9 TFLOPs.
    batch, prompt_len = 1, 2048
    ctx = _prefill_ctx(batch, prompt_len)

    total_flops = sum(op.compute_flops(ctx) for op in _llama3_8b_operators())

    expected_flops = 2.0 * LLAMA3_8B_PARAMS * batch * prompt_len
    rel_err = abs(total_flops - expected_flops) / expected_flops
    assert rel_err < 0.20, (
        f"total_flops={total_flops:.3e} expected≈{expected_flops:.3e} "
        f"rel_err={rel_err:.3f}"
    )


def test_prefill_attention_is_memory_bound():
    """FlashAttention prefill sits below the roofline ridge — memory_time
    dominates compute_time on A100/H100 for any reasonable shape."""
    attn = PrefillAttention(
        num_heads=LLAMA3_8B.num_heads,
        num_kv_heads=LLAMA3_8B.num_kv_heads,
        head_size=LLAMA3_8B.head_size,
        flash_block_size=LLAMA3_8B.flash_block_size,
        dtype_bytes=LLAMA3_8B.dtype_bytes,
    )
    ctx = _prefill_ctx(batch=1, prompt_len=2048)
    assert attn.memory_time(ctx, A100) > attn.compute_time(ctx, A100)

    # Sanity-check the analytical arithmetic intensity: 2 * b / (3 * dtype_bytes).
    ai = attn.compute_flops(ctx) / attn.memory_bytes(ctx)
    expected_ai = 2.0 * LLAMA3_8B.flash_block_size / (3.0 * LLAMA3_8B.dtype_bytes)
    assert ai == pytest.approx(expected_ai)


def test_qkv_gemm_compute_bound_at_prompt_2048():
    """A large prefill QKV GEMM is compute-bound (compute_time > memory_time);
    the same GEMM at decode M=1 flips memory-bound."""
    qkv_n = (LLAMA3_8B.num_heads + 2 * LLAMA3_8B.num_kv_heads) * LLAMA3_8B.head_size
    qkv = Gemm(
        n=qkv_n,
        k=LLAMA3_8B.hidden_size,
        token_count=TokenCount.PREFILL,
        dtype_bytes=LLAMA3_8B.dtype_bytes,
    )
    ctx = _prefill_ctx(batch=1, prompt_len=2048)
    assert qkv.compute_time(ctx, A100) > qkv.memory_time(ctx, A100)

    decode_qkv = Gemm(
        n=qkv_n,
        k=LLAMA3_8B.hidden_size,
        token_count=TokenCount.DECODE,
        dtype_bytes=LLAMA3_8B.dtype_bytes,
    )
    decode_ctx = WorkloadContext(
        workload_type=Workload.GENERATION,
        batch_size=1,
        microbatch_size=1,
        prompt_len=2048,
        response_len=128,
        num_microbatches=1,
    )
    assert decode_qkv.memory_time(decode_ctx, A100) > decode_qkv.compute_time(decode_ctx, A100)
