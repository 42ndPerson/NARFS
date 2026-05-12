"""GEMV efficiency tier on Gemm.

Decode at batch=1 produces M=1 GEMMs with K, N >> 1024. These miss the
small_dim threshold (since only M is small) but have fundamentally lower
efficiency than batched GEMMs. Gemm.compute_efficiency_factor /
memory_efficiency_factor implement a three-tier dispatch:

    is_gemv (M < 16)         → hw.gemv_efficiency / gemv_memory_efficiency
    is_small_dim (any < 1024) → hw.small_gemm_efficiency / memory_efficiency
    otherwise                 → hw.compute_efficiency / memory_efficiency
"""
from __future__ import annotations

import pytest

from hardware import HardwareSpec
from network_requirement import ParallelismConfig
from operators.gemm import Gemm
from workload_context import (
    TokenCount,
    Workload,
    WorkloadContext,
)


PAR = ParallelismConfig()


A100 = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
    ridge_flops_per_byte=156.0,
    # explicit defaults so tests don't break if HardwareSpec defaults shift
    compute_efficiency=0.7,
    memory_efficiency=0.8,
    small_gemm_efficiency=0.4,
    gemv_efficiency=0.3,
    gemv_memory_efficiency=0.65,
)


def _decode_ctx(microbatch: int = 1) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.GENERATION,
        batch_size=microbatch, microbatch_size=microbatch,
        prompt_len=2048, response_len=128, num_microbatches=1,
    )


def _prefill_ctx(microbatch: int = 4) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.PREPARATION,
        batch_size=microbatch, microbatch_size=microbatch,
        prompt_len=2048, response_len=0, num_microbatches=1,
    )


def test_gemv_uses_gemv_efficiency_factors():
    """Decode-shape GEMM (M=1, K=4096, N=12288) hits the GEMV tier."""
    decode_qkv = Gemm(n=12288, k=4096, token_count=TokenCount.DECODE, dtype_bytes=2)
    ctx = _decode_ctx(microbatch=1)
    assert decode_qkv.is_gemv(ctx)
    assert decode_qkv.compute_efficiency_factor(ctx, A100, PAR) == A100.gemv_efficiency
    assert decode_qkv.memory_efficiency_factor(ctx, A100, PAR) == A100.gemv_memory_efficiency


def test_compute_bound_gemm_uses_compute_efficiency():
    """Prefill-shape GEMM (M=4·2048=8192, K=4096, N=12288) — neither GEMV nor small-dim."""
    qkv = Gemm(n=12288, k=4096, token_count=TokenCount.PREFILL, dtype_bytes=2)
    ctx = _prefill_ctx(microbatch=4)
    assert not qkv.is_gemv(ctx)
    assert not qkv.is_small_dim(ctx, PAR)
    assert qkv.compute_efficiency_factor(ctx, A100, PAR) == A100.compute_efficiency
    assert qkv.memory_efficiency_factor(ctx, A100, PAR) == A100.memory_efficiency


def test_small_dim_gemm_uses_small_gemm_efficiency():
    """A GEMM with one architectural dim < 1024 (M=512, K=512, N=2048) uses
    small_gemm_efficiency on the compute side, but its memory tier remains
    memory_efficiency (small_dim doesn't drop memory bandwidth like GEMV does).
    """
    op = Gemm(n=2048, k=512, token_count=TokenCount.PREFILL, dtype_bytes=2)
    ctx = WorkloadContext(
        workload_type=Workload.PREPARATION,
        batch_size=1, microbatch_size=1,
        prompt_len=512, response_len=0, num_microbatches=1,
    )
    # M = 1 * 512 = 512, which is < 1024 → small_dim. Note M < 16 would be
    # GEMV; M=512 is small_dim but not GEMV. (Pick a microbatch that keeps M
    # ≥ 16 but min(M,N,K) < 1024.)
    assert op.is_small_dim(ctx, PAR)
    assert not op.is_gemv(ctx)
    assert op.compute_efficiency_factor(ctx, A100, PAR) == A100.small_gemm_efficiency
    assert op.memory_efficiency_factor(ctx, A100, PAR) == A100.memory_efficiency


def test_gemv_takes_precedence_over_small_dim():
    """When M=1, the kernel is GEMV regardless of N and K. GEMV tier wins
    even if small_dim would also fire."""
    op = Gemm(n=512, k=512, token_count=TokenCount.DECODE, dtype_bytes=2)
    ctx = _decode_ctx(microbatch=1)
    assert op.is_gemv(ctx)
    assert op.is_small_dim(ctx, PAR)  # both true
    assert op.compute_efficiency_factor(ctx, A100, PAR) == A100.gemv_efficiency  # GEMV wins


def test_compute_time_uses_selected_efficiency():
    """End-to-end check: compute_time reflects the GEMV efficiency factor."""
    decode_op = Gemm(n=12288, k=4096, token_count=TokenCount.DECODE, dtype_bytes=2)
    prefill_op = Gemm(n=12288, k=4096, token_count=TokenCount.PREFILL, dtype_bytes=2)

    decode_ctx = _decode_ctx(1)
    prefill_ctx = _prefill_ctx(4)

    # Same flops/byte structure differs only by M; ratio of compute_times equals
    # M_ratio × (efficiency_compute / efficiency_gemv).
    flops_decode = decode_op.compute_flops(decode_ctx, PAR)
    flops_prefill = prefill_op.compute_flops(prefill_ctx, PAR)
    expected_decode_time = flops_decode / (A100.peak_compute_flops * A100.gemv_efficiency)
    expected_prefill_time = flops_prefill / (A100.peak_compute_flops * A100.compute_efficiency)
    assert decode_op.compute_time(decode_ctx, A100, PAR) == pytest.approx(expected_decode_time)
    assert prefill_op.compute_time(prefill_ctx, A100, PAR) == pytest.approx(expected_prefill_time)
