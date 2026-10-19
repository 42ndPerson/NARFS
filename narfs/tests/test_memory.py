"""Memory model: byte counts per term, feasibility and KV capacity."""
import numpy as np
import pytest

from narfs.expand import Batch
from narfs.hardware import H100_SXM
from narfs.memory import CallMemory, check, generate_memory, kv_problem, train_memory
from narfs.model.architecture import LLAMA3_70B, llama
from narfs.model.library import model_params
from narfs.parallel.mesh import Degrees, map_devices
from narfs.profiles import MEGATRON_DEFAULT, load_engine

EPROF, _ = load_engine("vllm-573f2f9-h100")
TINY = llama("tiny", 1024, 8, 8, 8, 4096, 32000)


def test_model_state_terms_by_hand():
    deg = Degrees(dp=2, tp=2)
    m = train_memory(TINY, deg, MEGATRON_DEFAULT.with_(reserve=0.0), Batch(8, 512), 1)
    p = model_params(TINY) / 2                                    # every tiny parameter is tp-sharded except norms
    assert m.terms["weights"] == pytest.approx(2 * p, rel=0.01)
    assert m.terms["gradients"] == pytest.approx(4 * p, rel=0.01)
    assert m.terms["optimizer"] == pytest.approx(12 * p, rel=0.01)
    z1 = train_memory(TINY, deg, MEGATRON_DEFAULT.with_(zero_stage=1), Batch(8, 512), 1)
    assert z1.terms["optimizer"] == pytest.approx(m.terms["optimizer"] / 2)       # sharded over dp = 2


def test_activations_follow_the_schedule_and_recompute():
    deg = Degrees(pp=4)
    m = train_memory(TINY, deg, MEGATRON_DEFAULT.with_(reserve=0.0), Batch(32, 512), 1)
    # 1F1B: stage s holds min(nmb, P - s) microbatches; the last stage also holds the logits
    assert m.transient[0] / m.transient[2] == pytest.approx(4 / 2)
    full = train_memory(TINY, deg, MEGATRON_DEFAULT.with_(reserve=0.0, recompute="full"), Batch(32, 512), 1)
    assert full.transient[0] < m.transient[0] / 3
    fwd = train_memory(TINY, deg, MEGATRON_DEFAULT, Batch(32, 512), 1, train=False)
    assert fwd.state[0] < m.state[0] / 5                         # weights only


def test_kv_capacity_matches_the_design_numbers():
    tp8 = generate_memory(LLAMA3_70B, Degrees(tp=8), EPROF, H100_SXM, 512, 4096)
    tp4 = generate_memory(LLAMA3_70B, Degrees(tp=4), EPROF, H100_SXM, 512, 4096)
    assert tp8.kv_capacity == pytest.approx(1.28e6, rel=0.06)
    assert tp4.kv_capacity == pytest.approx(0.42e6, rel=0.06)
    assert tp8.terms["kv bytes per token"] == 80 * 2 * 1 * 128 * 2
    small = generate_memory(LLAMA3_70B, Degrees(tp=2), EPROF, H100_SXM, 512, 4096)
    assert kv_problem("gen", small)                              # 70B weights at tp 2 leave no KV budget
    assert not kv_problem("gen", tp8)


def test_device_check_counts_each_model_once_and_names_the_calls():
    cap = np.full(8, 80e9)
    pl = map_devices(range(8), Degrees(tp=8))
    big = CallMemory((40e9,), (10e9,))
    other = CallMemory((30e9,), (20e9,))
    ok = check([("logprob", "actor", pl, big), ("train", "actor", pl, big)], cap)
    assert ok.feasible and ok.peak[0] == pytest.approx(50e9) and ok.headroom[0] == pytest.approx(30e9)
    bad = check([("train", "actor", pl, big), ("ref", "ref", pl, other)], cap)
    assert not bad.feasible and "train, ref" in bad.reason and "90.0 GB needed" in bad.reason


def test_the_70b_training_plans():
    b = Batch(512, 2048)
    fits = {}
    for deg in (Degrees(dp=2, pp=4, tp=8), Degrees(dp=4, pp=2, tp=8), Degrees(dp=8, tp=8), Degrees(dp=16, tp=4)):
        m = train_memory(LLAMA3_70B, deg, MEGATRON_DEFAULT, b, 1)
        fits[deg.shape()] = m.state[0] + m.transient[0] <= H100_SXM.hbm_bytes
    assert fits == {(2, 4, 1, 1, 8): True, (4, 2, 1, 1, 8): False, (8, 1, 1, 1, 8): False, (16, 1, 1, 1, 4): False}
