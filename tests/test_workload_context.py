"""WorkloadContext invariant + per-invocation operator sizing.

Pin the contract: `batch_size = microbatch_size × num_microbatches`,
and operators size off `microbatch_size` (per-invocation) rather than
`batch_size` (global iteration).
"""
from __future__ import annotations

import pytest

from topology.hardware import HardwareSpec
from operators.gemm import Gemm
from workload_context import (
    TokenCount,
    Workload,
    WorkloadContext,
)


A100 = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
)


def test_valid_workload_context_constructs():
    ctx = WorkloadContext(
        workload_type=Workload.TRAINING,
        batch_size=32,
        microbatch_size=4,
        prompt_len=2048,
        response_len=0,
        num_microbatches=8,
    )
    assert ctx.batch_size == ctx.microbatch_size * ctx.num_microbatches


def test_workload_context_invariant_violation_raises():
    with pytest.raises(ValueError, match="num_microbatches × microbatch_size must equal batch_size"):
        WorkloadContext(
            workload_type=Workload.TRAINING,
            batch_size=64,
            microbatch_size=4,
            prompt_len=2048,
            response_len=0,
            num_microbatches=8,  # 4 × 8 = 32, not 64
        )


def test_workload_context_zero_or_negative_fields_raise():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        WorkloadContext(
            workload_type=Workload.PREPARATION,
            batch_size=0, microbatch_size=1,
            prompt_len=128, response_len=0, num_microbatches=1,
        )
    with pytest.raises(ValueError, match="microbatch_size must be positive"):
        WorkloadContext(
            workload_type=Workload.PREPARATION,
            batch_size=4, microbatch_size=0,
            prompt_len=128, response_len=0, num_microbatches=1,
        )
    with pytest.raises(ValueError, match="num_microbatches must be positive"):
        WorkloadContext(
            workload_type=Workload.PREPARATION,
            batch_size=4, microbatch_size=4,
            prompt_len=128, response_len=0, num_microbatches=0,
        )


def test_token_count_resolves_from_microbatch_size():
    """PREFILL = microbatch × prompt_len, DECODE = microbatch."""
    ctx = WorkloadContext(
        workload_type=Workload.TRAINING,
        batch_size=32, microbatch_size=4,
        prompt_len=2048, response_len=0, num_microbatches=8,
    )
    assert TokenCount.PREFILL.resolve(ctx) == 4 * 2048
    assert TokenCount.DECODE.resolve(ctx) == 4


def test_operators_size_off_microbatch_size_not_batch_size():
    """Two contexts with different batch_size but same microbatch_size produce
    identical per-invocation costs."""
    ctx_small = WorkloadContext(
        workload_type=Workload.TRAINING,
        batch_size=4, microbatch_size=4,
        prompt_len=2048, response_len=0, num_microbatches=1,
    )
    ctx_big = WorkloadContext(
        workload_type=Workload.TRAINING,
        batch_size=64, microbatch_size=4,
        prompt_len=2048, response_len=0, num_microbatches=16,
    )

    qkv = Gemm(n=12288, k=4096, token_count=TokenCount.PREFILL, dtype_bytes=2)
    assert qkv.compute_flops(ctx_small) == qkv.compute_flops(ctx_big)
    assert qkv.memory_bytes(ctx_small) == qkv.memory_bytes(ctx_big)
    assert qkv.kernel_time(ctx_small, A100) == pytest.approx(
        qkv.kernel_time(ctx_big, A100)
    )
