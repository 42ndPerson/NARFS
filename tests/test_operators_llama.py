from __future__ import annotations

import pytest

from hardware import HardwareSpec
from operators.attention import PrefillAttention
from operators.composite import (
    Repeat,
    TransformerBlock,
    TransformerBlockConfig,
    expand_layers,
)
from operators.elementwise import RMSNorm
from operators.gemm import Gemm
from workload_context import (
    TokenCount,
    Workload,
    WorkloadContext,
)


# A100-style synthetic spec — only ridge_flops_per_byte affects is_compute_bound.
A100 = HardwareSpec(
    peak_compute_flops=312e12,        # FP16 peak
    peak_memory_bandwidth=2.0e12,     # ~2 TB/s HBM
    ridge_flops_per_byte=156.0,
)


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


def _llama3_8b_layers() -> list[Repeat]:
    block_ops = TransformerBlock(LLAMA3_8B, TokenCount.PREFILL).operators()
    final = [
        RMSNorm(LLAMA3_8B.hidden_size, TokenCount.PREFILL, LLAMA3_8B.dtype_bytes),
        Gemm(
            n=LLAMA3_8B_VOCAB,
            k=LLAMA3_8B.hidden_size,
            token_count=TokenCount.PREFILL,
            dtype_bytes=LLAMA3_8B.dtype_bytes,
        ),
    ]
    return [Repeat(LLAMA3_8B_LAYERS, block_ops), Repeat(1, final)]


def test_llama3_8b_prefill_total_flops():
    # Forward-only rule of thumb: ~2 * P * T FLOPs for a dense decoder LLM.
    # For Llama-3-8B at 2048 tokens that's ~2 * 8.03e9 * 2048 ≈ 32.9 TFLOPs.
    # (The task brief mentioned 65 TFLOPs but that double-counts — the
    # 2*P*T rule already captures forward compute. The 20% tolerance from
    # the brief is preserved here against the corrected target.)
    batch, prompt_len = 1, 2048
    ctx = _prefill_ctx(batch, prompt_len)

    total_flops = sum(op.compute_flops(ctx) for op in expand_layers(_llama3_8b_layers()))

    expected_flops = 2.0 * LLAMA3_8B_PARAMS * batch * prompt_len
    rel_err = abs(total_flops - expected_flops) / expected_flops
    assert rel_err < 0.20, (
        f"total_flops={total_flops:.3e} expected≈{expected_flops:.3e} "
        f"rel_err={rel_err:.3f}"
    )


def test_prefill_attention_is_memory_bound():
    attn = PrefillAttention(
        num_heads=LLAMA3_8B.num_heads,
        num_kv_heads=LLAMA3_8B.num_kv_heads,
        head_size=LLAMA3_8B.head_size,
        flash_block_size=LLAMA3_8B.flash_block_size,
        dtype_bytes=LLAMA3_8B.dtype_bytes,
    )
    ctx = _prefill_ctx(batch=1, prompt_len=2048)
    assert not attn.is_compute_bound(ctx, A100)

    # Sanity-check the analytical AI: 2 * b / (3 * dtype_bytes), well below the ridge.
    ai = attn.compute_flops(ctx) / attn.memory_bytes(ctx)
    expected_ai = 2.0 * LLAMA3_8B.flash_block_size / (3.0 * LLAMA3_8B.dtype_bytes)
    assert ai == pytest.approx(expected_ai)
    assert ai < A100.ridge_flops_per_byte


def test_qkv_gemm_is_compute_bound_at_prompt_2048():
    qkv_n = (LLAMA3_8B.num_heads + 2 * LLAMA3_8B.num_kv_heads) * LLAMA3_8B.head_size
    qkv = Gemm(
        n=qkv_n,
        k=LLAMA3_8B.hidden_size,
        token_count=TokenCount.PREFILL,
        dtype_bytes=LLAMA3_8B.dtype_bytes,
    )
    ctx = _prefill_ctx(batch=1, prompt_len=2048)
    assert qkv.is_compute_bound(ctx, A100)
    # And the ridge boundary: at very small M (single decode token) the same
    # GEMM flips memory-bound.
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
    assert not decode_qkv.is_compute_bound(decode_ctx, A100)
