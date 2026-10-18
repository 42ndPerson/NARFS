"""Whole-system invariants on realistic hardware."""
import time
from dataclasses import replace

import numpy as np
import pytest

from narfs.blocks import Call, Collective, Stream, Transfers
from narfs.costs import CostModel, CostTable
from narfs.evaluate import EvalContext, evaluate
from narfs.expand import Workload, expand_generate, expand_train
from narfs.hardware import H100_80GB
from narfs.model.architecture import LLAMA3_8B, LLAMA3_70B
from narfs.model.hf_import import from_hf_config
from narfs.model.library import model_params
from narfs.network.presets import hgx_h100
from narfs.optimize import Simulator, anneal, group_plan, propose
from narfs.parallel.mesh import Degrees, map_devices
from narfs.parallel.rules import MEGATRON
from narfs.profiles import MEGATRON_DEFAULT, VLLM_DEFAULT
from narfs.rlhf import CallPlan, build, check, reshard_transfers
from narfs.blocks import Op
from narfs.tests.cases import ppo_case
from narfs.tests.walk import of_type

NET = hgx_h100(4)


def timed(block, placement, net=NET):
    return evaluate(Call("c", block), EvalContext(net, CostModel(), {"c": placement})).t


# ------------------------------------------------------------ invariants ---

def test_tp_allreduce_time_does_not_grow_with_pp():
    ar = Collective("allreduce", "tp", 64e6)
    one = timed(ar, map_devices(range(8), Degrees(tp=8)))
    from narfs.blocks import OnStage
    for pp in (2, 4):
        pl = map_devices(range(8 * pp), Degrees(pp=pp, tp=8))
        assert timed(OnStage(pp - 1, ar), pl) == pytest.approx(one)
        assert timed(ar, pl) == pytest.approx(one)    # stages on separate nodes: same burst


def test_training_time_scales_down_with_dp():
    wl = Workload(256, 1024, (1024,), (1.0,))
    t = [timed(expand_train(LLAMA3_8B, Degrees(dp=dp, tp=8), MEGATRON, MEGATRON_DEFAULT, wl, 1),
               map_devices(range(8 * dp), Degrees(dp=dp, tp=8))) for dp in (1, 2, 4)]
    assert t[0] > t[1] > t[2]
    assert t[0] / t[2] == pytest.approx(4, rel=0.15)


def test_moe_ep_shards_experts_and_data():
    moe = from_hf_config({
        "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 8,
        "vocab_size": 32000, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2})
    net = hgx_h100(8)
    wl = Workload(64, 1024, (1024,), (1.0,))
    # the same 64 devices and the same data replicas; ep = 8 adds alltoalls,
    # cuts expert weight reads, and keeps the expert grad sync on 8 replicas
    t_dp = timed(expand_train(moe, Degrees(dp=64), MEGATRON, MEGATRON_DEFAULT, wl, 1),
                 map_devices(range(64), Degrees(dp=64)), net)
    t_ep = timed(expand_train(moe, Degrees(dp=8, ep=8), MEGATRON, MEGATRON_DEFAULT, wl, 1),
                 map_devices(range(64), Degrees(dp=8, ep=8)), net)
    assert t_ep != pytest.approx(t_dp, rel=0.01)
    assert t_ep < 2 * t_dp and t_dp < 2 * t_ep          # same work per rank, not ep times more


# ----------------------------------------------------------------- memory ---

def test_memory_feasibility():
    prob, cand, sim = ppo_case(nodes=16)
    check(prob, cand)
    tiny = replace(cand.plan("actor_train"), degrees=Degrees(dp=8, tp=8))
    bad = cand.with_plan("actor_train", tiny)        # 70B with tp = 8, no pp, no ZeRO
    with pytest.raises(ValueError, match="out of memory"):
        check(prob, bad)
    assert sim.evaluate(bad) is None
    ok, why = sim.feasible(bad)
    assert not ok and "out of memory" in why


# ------------------------------------------------------------- generation ---

def gen_time(wl, kv_tokens=2e6):
    deg = Degrees(tp=8)
    gb = expand_generate(LLAMA3_8B, deg, MEGATRON, VLLM_DEFAULT, wl, kv_tokens)
    return timed(gb, map_devices(range(8), deg))


def test_genbatch_monotonic():
    base = Workload(256, 512, (256, 1024), (0.5, 0.5))
    t = gen_time(base)
    assert gen_time(replace(base, global_batch=1024)) > t
    assert gen_time(replace(base, lengths=(512, 2048))) > t
    assert gen_time(base, kv_tokens=5e4) >= t
    assert gen_time(replace(base, global_batch=1)) < t


# -------------------------------------------------------------- resharding ---

def moved_bytes(tr: Transfers):
    s, d, z = np.array(tr.src), np.array(tr.dst), np.array(tr.size)
    return float(z[s != d].sum())


def test_reshard_identical_placement_moves_nothing():
    pl = map_devices(range(16), Degrees(dp=2, tp=8))
    tr = reshard_transfers(LLAMA3_8B, pl, pl)
    assert moved_bytes(tr) == 0
    assert timed(tr, pl) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize("src_deg", [Degrees(tp=8), Degrees(pp=2, tp=4), Degrees(pp=4, tp=2)])
def test_reshard_delivers_every_parameter_once_per_replica(src_deg):
    src = map_devices(range(8), src_deg)
    dst = map_devices(range(8, 24), Degrees(dp=2, tp=8))
    tr = reshard_transfers(LLAMA3_8B, src, dst)
    full = model_params(LLAMA3_8B) * LLAMA3_8B.dtype_bytes
    dst0 = set(dst.dev_of[0].ravel().tolist())
    per_replica = sum(z for d, z in zip(tr.dst, tr.size) if d in dst0)
    # replicated parameters (norms) are copied to every TP rank, so slightly more
    assert per_replica == pytest.approx(full, rel=0.01)
    assert moved_bytes(tr) == pytest.approx(2 * per_replica)


# ------------------------------------------------------------------- RLHF ---

def test_ppo_iteration_bounds():
    prob, cand, sim = ppo_case(nodes=16)
    b = sim.bounds(cand)
    assert 0 < b.lo <= b.hi
    rows = sim.explain(cand)
    names = {r[1] for r in rows if r[0] == "Call"}
    assert {"generate", "actor_train", "critic_train"} <= names


def test_async_is_not_slower_than_sync():
    prob, cand, sim = ppo_case(nodes=16)
    sync = sim.evaluate(cand).time
    asim = Simulator(replace(prob, async_rl=True))
    assert asim.evaluate(cand).time <= sync


def test_offload_adds_streams():
    prob, cand, _ = ppo_case(nodes=16)
    tree, _ = build(prob, cand)
    assert not of_type(tree, Stream)
    tree, _ = build(replace(prob, offload_idle=True), cand)
    assert len(of_type(tree, Stream)) == 2 * len(prob.calls)


# -------------------------------------------------------------- optimizer ---

def test_rank_on_hi():
    prob, cand, _ = ppo_case(nodes=16)
    r = Simulator(prob, rank_on="hi").evaluate(cand)
    assert r.time == r.hi and r.lo <= r.hi
    with pytest.raises(ValueError):
        Simulator(prob, rank_on="mid")


def test_moves_keep_device_counts():
    import random
    prob, cand, _ = ppo_case(nodes=16)
    rng = random.Random(1)
    for _ in range(50):
        nxt = propose(prob, cand, rng)
        if nxt is not None:
            for _, p in nxt.plans:
                assert len(p.devices) == p.degrees.size


def test_anneal_never_returns_worse_than_start():
    prob, cand, sim = ppo_case(nodes=16)
    start = sim.evaluate(cand).time
    res = anneal(sim, cand, steps=30, seed=0)
    assert res.best_time <= start and res.evaluated >= 1
    assert sim.evaluate(res.best).time == pytest.approx(res.best_time)


def test_group_plan_rejects_too_many_devices():
    prob, _, _ = ppo_case(nodes=8)                    # 64 devices
    with pytest.raises(ValueError):
        group_plan(prob, [(128, ["generate"])], {"generate": Degrees(dp=16, tp=8)})


# ----------------------------------------------------------------- import ---

def test_hf_import_llama_matches_preset():
    cfg = {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
           "num_hidden_layers": 32, "vocab_size": 128256, "intermediate_size": 14336}
    arch = from_hf_config(cfg, name="llama3-8b")
    assert dict(arch.dims) == dict(LLAMA3_8B.dims) and arch.layers == LLAMA3_8B.layers


def test_hf_import_deepseek_style():
    cfg = {"hidden_size": 7168, "num_attention_heads": 128, "num_hidden_layers": 61, "vocab_size": 129280,
           "intermediate_size": 18432, "moe_intermediate_size": 2048, "n_routed_experts": 256,
           "n_shared_experts": 1, "num_experts_per_tok": 8, "first_k_dense_replace": 3,
           "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
           "v_head_dim": 128}
    arch = from_hf_config(cfg, name="dsv3")
    assert [(s.count, s.attention, s.ffn) for s in arch.layers] == [(3, "mla", "dense"), (58, "mla", "moe")]
    assert arch.dim("experts") == 256 and arch.dim("kv_lora") == 512


# ------------------------------------------------------------------ costs ---

def test_cost_model_tiers_and_table(tmp_path):
    cm = CostModel()
    big = Op("gemm", 2e12, 1e6, (4096, 4096, 4096))
    small = Op("gemm", 2e12, 1e6, (512, 4096, 4096))
    gemv = Op("gemm", 2e12, 1e6, (8, 4096, 4096))
    assert cm.op_time(big, H100_80GB) < cm.op_time(small, H100_80GB) < cm.op_time(gemv, H100_80GB)
    path = tmp_path / "table.json"
    path.write_text('[{"device": "H100-80GB", "kind": "gemm", "shape": [1024, 4096, 4096], "seconds": 0.002}]')
    table = CostModel(CostTable.load(path))
    assert table.op_time(Op("gemm", 0, 0, (1024, 4096, 4096)), H100_80GB) == pytest.approx(0.002)
    assert table.op_time(Op("gemm", 0, 0, (2048, 4096, 4096)), H100_80GB) == pytest.approx(0.004)


# ------------------------------------------------------------ performance ---

@pytest.mark.parametrize("nodes", [64])
def test_hot_loop_budget(nodes):
    """Build + check + evaluate of one candidate at 512 GPUs. The budget is
    loose on purpose (about 5x the time measured during development)."""
    prob, cand, sim = ppo_case(nodes=nodes)
    sim.evaluate(cand)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        sim.evaluate(cand)
        times.append(time.perf_counter() - t0)
    assert sorted(times)[2] < 0.05
