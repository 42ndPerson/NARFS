from __future__ import annotations
from dataclasses import dataclass

import pytest

from hardware import HardwareSpec
from network_requirement import ParallelismConfig
from operator_base import Operator
from workload_context import Workload, WorkloadContext


@dataclass(frozen=True)
class ConcreteOp(Operator):
    flops: float = 0.0
    bytes_: float = 0.0

    def compute_flops(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        del ctx, par
        return self.flops

    def memory_bytes(self, ctx: WorkloadContext, par: ParallelismConfig) -> float:
        del ctx, par
        return self.bytes_

    def is_compute_bound(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> bool:
        del ctx, hw, par
        return self.flops >= self.bytes_

    def compute_efficiency_factor(
        self, ctx: WorkloadContext, hw: HardwareSpec, par: ParallelismConfig
    ) -> float:
        del ctx, par
        # Test op uses small_gemm_efficiency when small_dim=True so the legacy
        # Phase 1 small-dim test path keeps exercising the override mechanism.
        return hw.small_gemm_efficiency if self.small_dim else hw.compute_efficiency


@pytest.fixture
def ctx() -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.GENERATION,
        batch_size=8,
        microbatch_size=2,
        prompt_len=128,
        response_len=64,
        num_microbatches=4,
    )


@pytest.fixture
def hw() -> HardwareSpec:
    return HardwareSpec(
        peak_compute_flops=1e15,
        peak_memory_bandwidth=1e12,
        ridge_flops_per_byte=156.0,
        compute_efficiency=0.5,
        memory_efficiency=0.5,
        small_gemm_efficiency=0.25,
    )


@pytest.fixture
def par() -> ParallelismConfig:
    return ParallelismConfig()


def test_compute_bound_kernel_time(ctx, hw, par):
    op = ConcreteOp(flops=1e14, bytes_=1e9)

    expected_compute = 1e14 / (1e15 * 0.5)
    expected_memory = 1e9 / (1e12 * 0.5)

    assert op.compute_time(ctx, hw, par) == pytest.approx(expected_compute)
    assert op.memory_time(ctx, hw, par) == pytest.approx(expected_memory)
    assert op.kernel_time(ctx, hw, par) == pytest.approx(max(expected_compute, expected_memory))
    assert op.kernel_time(ctx, hw, par) == pytest.approx(expected_compute)


def test_memory_bound_kernel_time(ctx, hw, par):
    op = ConcreteOp(flops=1e10, bytes_=1e12)

    expected_compute = 1e10 / (1e15 * 0.5)
    expected_memory = 1e12 / (1e12 * 0.5)

    assert op.kernel_time(ctx, hw, par) == pytest.approx(max(expected_compute, expected_memory))
    assert op.kernel_time(ctx, hw, par) == pytest.approx(expected_memory)


def test_small_gemm_efficiency_path(ctx, hw, par):
    op_large = ConcreteOp(flops=1e14, bytes_=0.0, small_dim=False)
    op_small = ConcreteOp(flops=1e14, bytes_=0.0, small_dim=True)

    large_time = op_large.compute_time(ctx, hw, par)
    small_time = op_small.compute_time(ctx, hw, par)

    assert large_time == pytest.approx(1e14 / (1e15 * hw.compute_efficiency))
    assert small_time == pytest.approx(1e14 / (1e15 * hw.small_gemm_efficiency))
    assert small_time > large_time
