"""Flows and the simulator API: phases, weight sync, lagged and parallel
flows, determinism, the performance budget, infeasibility reasons and
problem errors."""
import math
import time

import numpy as np
import pytest

from narfs import rlhf
from narfs.hardware import DATA, H100_SXM
from narfs.model.architecture import LLAMA3_8B
from narfs.model.library import model_params
from narfs.network.comm import load_nccl
from narfs.network.presets import hgx_h100
from narfs.optimize import CallPlan, Candidate, colocated
from narfs.parallel.mesh import Degrees
from narfs.profiles import VLLM_H100
from narfs.sim import Problem, Simulator
from narfs.tests import cases


def test_grpo_phases_follow_the_flow():
    problem, cand = cases.grpo_8b()
    r = Simulator(problem).evaluate(cand)
    assert r.feasible, r.reason
    p = r.phases
    assert list(p) == ["grpo", "generate", "score", "ref_logprob", "old_logprob", "verifier", "policy_train",
                       "sync policy"]
    # seq adds; both forward calls run on node 1, and the verifier runs on its CPU pool at the same time
    assert r.time == pytest.approx(p["generate"] + p["score"] + p["policy_train"] + p["sync policy"])
    assert p["score"] == pytest.approx(p["ref_logprob"] + p["old_logprob"])
    assert p["verifier"] == pytest.approx(512 * 0.05 / 512)
    assert r.lo == r.time <= r.hi
    assert set(r.kv_capacity) == {"generate"} and r.kv_capacity["generate"] > 1e5
    assert r.memory_headroom.shape == (16,) and (r.memory_headroom >= 0).all()


def test_weight_sync_follows_the_layouts():
    problem, cand = cases.grpo_8b()
    r = Simulator(problem).evaluate(cand)
    # node 1 (dp 2, tp 4) to node 0 (dp 8): each source rank sends a quarter of the
    # model to each of 4 engine ranks, so every source NIC sends the whole model once
    weights = model_params(LLAMA3_8B) * 2
    assert r.phases["sync policy"] == pytest.approx(weights / H100_SXM.nic_rate, rel=0.01)
    # the same layout on the same devices moves nothing
    problem, _ = cases.grpo_8b(engine=VLLM_H100.with_(gpu_memory_utilization=0.4))
    names = ["generate", "ref_logprob", "old_logprob", "policy_train"]
    same = colocated([(8, names)], {n: cases.TRAIN_8B for n in names})
    r = Simulator(problem).evaluate(same)
    assert r.feasible, r.reason
    assert r.phases["sync policy"] == 0.0


def test_lagged_overlaps_generation_with_learning_only_on_separate_devices():
    problem, cand = cases.async_grpo_8b(colocate=False)
    r = Simulator(problem).evaluate(cand)
    assert r.feasible, r.reason
    p = r.phases
    assert r.time == pytest.approx(max(p["generate"], p["learn"]))
    problem, cand = cases.async_grpo_8b(colocate=True)
    r = Simulator(problem).evaluate(cand)
    assert r.feasible, r.reason
    p = r.phases
    assert r.time == pytest.approx(p["generate"] + p["learn"])            # the same GPUs: nothing overlaps


def test_ppo_scores_and_updates_on_two_groups_in_parallel():
    problem, cand = cases.ppo_8b()
    r = Simulator(problem).evaluate(cand)
    assert r.feasible, r.reason
    p = r.phases
    assert p["update"] == pytest.approx(max(p["actor_train"], p["critic_train"]))
    assert p["score"] == pytest.approx(max(p["actor_logprob"] + p["ref_logprob"], p["critic_value"] + p["reward"]))
    assert r.time == pytest.approx(p["generate"] + p["score"] + p["update"] + p["sync actor"])


def test_the_same_inputs_give_the_same_result():
    problem, cand = cases.grpo_8b()
    sim = Simulator(problem)
    first = sim.evaluate(cand)
    sim.evaluate(cases.remapped(cand, 8, 16))                     # caches filled by another candidate
    for r in (sim.evaluate(cand), Simulator(cases.grpo_8b()[0]).evaluate(cand)):
        assert (r.time, r.lo, r.hi) == (first.time, first.lo, first.hi)
        assert r.phases == first.phases and r.kv_capacity == first.kv_capacity
        assert np.array_equal(r.memory_headroom, first.memory_headroom)
    assert sim.explain(cand) == sim.explain(cand)


def test_performance_budget_at_512_gpus():
    problem, cand = cases.grpo_70b_512()
    sim = Simulator(problem)
    first = sim.evaluate(cand)
    assert first.feasible, first.reason
    times = []
    for k in range(1, 8):
        moved = cases.remapped(cand, 8 * k, 512)       # the same degrees on other nodes: new bindings, cached programs
        t0 = time.perf_counter()
        r = sim.evaluate(moved)
        times.append(time.perf_counter() - t0)
        assert r.feasible, r.reason
    assert sorted(times)[len(times) // 2] < 0.050


def test_infeasible_candidates_give_reasons_that_name_the_call():
    problem, cand = cases.grpo_8b()
    sim = Simulator(problem)
    plans = dict(cand.plans)

    def reason(**changes) -> str:
        p = {**plans, **changes}
        r = sim.evaluate(Candidate.of({n: v for n, v in p.items() if v is not None}))
        assert not r.feasible and r.time == math.inf
        return r.reason

    assert reason(old_logprob=None) == "call old_logprob has no placement"
    assert reason(critic=plans["generate"]) == "the candidate plans call 'critic', which the flow does not have"
    assert reason(policy_train=CallPlan(range(8, 12), cases.TRAIN_8B)).startswith("call policy_train: degrees")
    assert reason(generate=CallPlan(range(12, 20), Degrees(dp=8))) == \
        "call generate: device 16 is not in the cluster (16 GPUs)"
    assert reason(generate=CallPlan(range(8), Degrees(dp=4, pp=2))) == \
        "call generate: generation with pp > 1 is not modeled"
    split = reason(policy_train=CallPlan(range(8, 16), cases.TRAIN_8B, micro_batch=3))
    assert split.startswith("call policy_train: 128 sequences do not split")
    oom = reason(generate=CallPlan(range(8, 16), Degrees(dp=8)))         # the engine's 90% pool next to training state
    assert oom.startswith("out of memory on device 8:") and "generate, ref_logprob, old_logprob, policy_train" in oom
    small = Simulator(cases.grpo_8b(engine=VLLM_H100.with_(gpu_memory_utilization=0.25))[0]).evaluate(cand)
    assert not small.feasible and small.reason.startswith("call generate: the KV budget")
    assert small.memory_headroom is not None


def test_problem_errors_name_the_node():
    policy, ref = rlhf.Model("policy", LLAMA3_8B), rlhf.Model("ref", LLAMA3_8B)
    roll = rlhf.generate(policy, 64, 8, cases.LENGTHS)
    cluster = hgx_h100(1)
    with pytest.raises(ValueError, match="model 'ref' needs exactly one train call"):
        Simulator(Problem(cluster, rlhf.seq(roll, rlhf.sync(ref, roll))))
    with pytest.raises(ValueError, match="two calls are named 'generate'"):
        Simulator(Problem(cluster, rlhf.seq(roll, rlhf.generate(policy, 32, 8, cases.LENGTHS))))
    with pytest.raises(ValueError, match="CPU pool 'verifier' is declared with 64 and 32 workers"):
        Simulator(Problem(cluster, rlhf.seq(roll, rlhf.cpu("verifier", roll, 0.1, 64),
                                            rlhf.cpu("verifier", roll, 0.1, 32))))
    with pytest.raises(ValueError, match="communication library 'vllm' is not in the problem"):
        Simulator(Problem(cluster, roll, libraries={"nccl": load_nccl(DATA / "nccl-2.32.3.json")}))
    with pytest.raises(ValueError, match="512 sequences do not split into 3 minibatches"):
        Simulator(Problem(cluster, rlhf.train(policy, roll, minibatches=3)))


def test_explain_names_what_set_the_time_and_the_assumed_values():
    problem, cand = cases.grpo_8b()
    sim = Simulator(problem)
    text = sim.explain(cand)
    assert text.startswith("iteration:")
    assert "largest part: generate" in text and "decode steps" in text
    assert "assumed (unmeasured) values:" in text
    assert "h100-sxm: eta_hbm" in text and "call generate (vllm): host_step" in text
    missing = Candidate.of({n: p for n, p in cand.plans if n != "generate"})
    assert sim.explain(missing) == "infeasible: call generate has no placement"
