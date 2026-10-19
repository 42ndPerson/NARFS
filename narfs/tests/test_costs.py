"""Compute cost model: ceilings and waves, by hand."""
import numpy as np
import pytest

from narfs.blocks import Compute, OnDevices, Seq, Work
from narfs.costs import ComputeModel, CostTable, HostModel, HostStep, Kernel, elementwise, gemm, waves
from narfs.evaluate import evaluate
from narfs.hardware import H100_SXM, GpuProfile, load_gpu
from narfs.network.comm import load_nccl
from narfs.network.presets import rail_optimized
from narfs.hardware import DATA
from narfs.physics import Physics

# 4 SMs, 1 FLOP per clock per SM at 1 Hz, eta 0.5, 2x2 tiles, fast HBM
TOY = GpuProfile(name="toy", sms=4, clock_hz=1.0, tensor_fpc={"bf16": 1.0}, vector_flops=1.0, hbm_bw=1e6,
                 hbm_bytes=1e9, eta_gemm={"bf16": 0.5}, eta_hbm=0.5, tile=(2, 2))


def test_gemm_time_by_hand():
    cm = ComputeModel({"toy": TOY})
    # 4x4 output: 4 tiles on 4 SMs, one full wave; padded FLOPs 2*4*4*1 = 32 at 0.5 x 4 FLOP/s
    assert cm.time(gemm(4, 4, 1), "toy") == pytest.approx(16.0)
    # 5x4 output: 3 x 2 = 6 tiles, 2 waves at 6/8 of the SMs; padded FLOPs 2*6*4*1 = 48
    assert cm.terms(gemm(5, 4, 1), "toy")[2:] == (2, 0.75)
    assert cm.time(gemm(5, 4, 1), "toy") == pytest.approx(48 / (0.5 * 4 * 0.75))
    # 2x2 output, K = 256: one tile cannot fill 4 SMs, so K is split 4 ways (4 blocks of 64)
    c, m, nw, u = cm.terms(gemm(2, 2, 256), "toy")
    assert (nw, u) == (1, 1.0)
    assert c == pytest.approx(2 * 2 * 2 * 256 / (0.5 * 4))
    assert m == pytest.approx((2 * (2 * 256 + 256 * 2) + 2 * 4) / (0.5 * 1e6))


def test_the_design_examples_wave_count():
    # 70B QKV GEMM at tp = 8 and 4,096 tokens: 160 tiles of 128x256 on 132 SMs
    n = (64 + 2 * 8) * 128 // 8
    cm = ComputeModel({"h100-sxm": H100_SXM})
    _, _, nw, u = cm.terms(gemm(4096, n, 8192), "h100-sxm")
    assert (nw, round(u, 2)) == (2, 0.61)
    assert waves(264, 132) == (2, 1.0)


def test_no_cliff_at_old_size_tiers():
    cm = ComputeModel({"h100-sxm": H100_SXM})
    t = lambda m: cm.time(gemm(m, 8192, 8192), "h100-sxm")
    for a, b in ((15, 16), (1023, 1024), (100, 120)):            # same tiles and waves: no step
        assert t(b) / t(a) == pytest.approx(1.0, abs=0.01)
    sweep = [t(m) for m in range(1, 4097, 7)]
    assert all(y >= x * (1 - 1e-12) for x, y in zip(sweep, sweep[1:]))     # never faster with more tokens


def test_decode_gemm_is_memory_bound_and_fp8_has_its_own_ceiling():
    cm = ComputeModel({"h100-sxm": H100_SXM})
    c, m, _, _ = cm.terms(gemm(8, 8192, 8192), "h100-sxm")
    assert m > c
    assert cm.time(gemm(8, 8192, 8192), "h100-sxm") == pytest.approx(m)
    c16 = cm.terms(gemm(8192, 8192, 8192), "h100-sxm")[0]
    c8 = cm.terms(gemm(8192, 8192, 8192, "fp8"), "h100-sxm")[0]
    assert c16 / c8 == pytest.approx(2 * 0.65 / 0.73)


def test_vector_attention_and_table_override():
    cm = ComputeModel({"toy": TOY})
    assert cm.time(elementwise(0.0, 1e6), "toy") == pytest.approx(2.0)          # 1e6 B at 0.5 x 1e6 B/s
    att = Kernel("attention", flops=8.0, bytes=0.0, tiles=5)                    # 5 tiles: 2 waves, u = 5/8
    assert cm.time(att, "toy") == pytest.approx(8.0 / (0.5 * 4 * 5 / 8))
    table = CostTable({("toy", "gemm", (4, 4, 1), "bf16"): 0.125})
    assert ComputeModel({"toy": TOY}, table).time(gemm(4, 4, 1), "toy") == 0.125
    with pytest.raises(ValueError):
        Kernel("conv", 1.0, 1.0)


def test_profile_flags_and_mixed_device_classes():
    assert H100_SXM.flags == ("eta_hbm",)
    assert H100_SXM.with_(eta_hbm=0.9).flags == ()
    assert load_gpu(DATA / "h100-sxm.json") == H100_SXM
    slow = H100_SXM.with_(name="slow", clock_hz=1.0e9)
    cm = ComputeModel({"h100-sxm": H100_SXM, "slow": slow})
    assert cm.slowest(["h100-sxm", "slow"]) == "slow"


def test_compute_and_host_steps_through_physics():
    topo = rail_optimized(1, 2)
    p = Physics(topo, {"nccl": load_nccl(DATA / "nccl-2.32.3.json")}, ComputeModel({"h100-sxm": H100_SXM}),
                HostModel({"vllm": (2e-3, 1e-5)}))
    bind = {"devs": np.array([0, 1]), "host": topo.rid("host", [0])}
    k = gemm(4096, 4096, 4096)
    s = evaluate(OnDevices("devs", Seq((Compute(k), Work("host", HostStep("vllm", 100))))), p, bind)
    assert s.lo == pytest.approx(ComputeModel({"h100-sxm": H100_SXM}).time(k, "h100-sxm") + 3e-3)
