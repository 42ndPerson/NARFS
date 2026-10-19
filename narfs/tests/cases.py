"""Shared problem cases for system tests: (problem, candidate) pairs on HGX
H100 clusters, with a lognormal response-length distribution (mean 372
tokens, capped at 4,096)."""
from __future__ import annotations

import numpy as np

from narfs import rlhf
from narfs.model.architecture import LLAMA3_70B, LLAMA3_8B
from narfs.network.presets import hgx_h100
from narfs.optimize import CallPlan, Candidate, colocated
from narfs.parallel.mesh import Degrees
from narfs.profiles import MEGATRON_DEFAULT, MEGATRON_DISTOPT, VLLM_H100
from narfs.sim import Problem
from narfs.variation import Dist

LENGTHS = Dist.lognormal(5.6, 0.8, 4096)
TRAIN_8B = Degrees(dp=2, tp=4)


def grpo_8b(engine=None):
    """GRPO on Llama-3-8B, 2 nodes: generation on node 0, scoring and training on node 1."""
    policy, ref = rlhf.Model("policy", LLAMA3_8B), rlhf.Model("ref", LLAMA3_8B)
    flow = rlhf.grpo(policy, ref, LENGTHS, prompts=64, group=8, minibatches=4, engine=engine,
                     profile=MEGATRON_DISTOPT)
    cand = colocated([(8, ["generate"]), (8, ["ref_logprob", "old_logprob", "policy_train"])],
                     {"generate": Degrees(dp=8), "ref_logprob": TRAIN_8B, "old_logprob": TRAIN_8B,
                      "policy_train": TRAIN_8B})
    return Problem(hgx_h100(2), flow), cand


def async_grpo_8b(colocate: bool):
    """One-step off-policy GRPO on Llama-3-8B: on separate nodes, or all calls
    on node 0 (the engine then takes 40% of the GPU memory, not 90%)."""
    policy, ref = rlhf.Model("policy", LLAMA3_8B), rlhf.Model("ref", LLAMA3_8B)
    engine = VLLM_H100.with_(gpu_memory_utilization=0.4) if colocate else None
    flow = rlhf.async_grpo(policy, ref, LENGTHS, prompts=64, group=8, minibatches=4, engine=engine,
                           profile=MEGATRON_DISTOPT)
    learn = ["ref_logprob", "old_logprob", "policy_train"]
    groups = [(8, ["generate"] + learn)] if colocate else [(8, ["generate"]), (8, learn)]
    deg = {"generate": TRAIN_8B, **{n: TRAIN_8B for n in learn}}
    return Problem(hgx_h100(1 if colocate else 2), flow), colocated(groups, deg)


def ppo_8b():
    """PPO on Llama-3-8B, 3 nodes: generation; actor scoring and training; critic and reward."""
    m = {n: rlhf.Model(n, LLAMA3_8B) for n in ("actor", "critic", "ref", "reward")}
    flow = rlhf.ppo(m["actor"], m["critic"], m["ref"], m["reward"], LENGTHS, prompts=512, minibatches=4,
                    profile=MEGATRON_DISTOPT)
    cand = colocated([(8, ["generate"]), (8, ["actor_logprob", "ref_logprob", "actor_train"]),
                      (8, ["critic_value", "reward", "critic_train"])],
                     {"generate": Degrees(dp=8), **{n: TRAIN_8B for n in ("actor_logprob", "ref_logprob",
                      "actor_train", "critic_value", "reward", "critic_train")}})
    return Problem(hgx_h100(3), flow), cand


def grpo_70b_512():
    """GRPO on Llama-3-70B, 64 nodes: generation on nodes 0-31 (dp 32, tp 8),
    scoring and training on nodes 32-63 (dp 4, pp 8, tp 8)."""
    policy, ref = rlhf.Model("policy", LLAMA3_70B), rlhf.Model("ref", LLAMA3_70B)
    flow = rlhf.grpo(policy, ref, LENGTHS, prompts=512, group=8, minibatches=4, profile=MEGATRON_DEFAULT)
    train = Degrees(dp=4, pp=8, tp=8)
    cand = colocated([(256, ["generate"]), (256, ["ref_logprob", "old_logprob", "policy_train"])],
                     {"generate": Degrees(dp=32, tp=8), "ref_logprob": train, "old_logprob": train,
                      "policy_train": train})
    return Problem(hgx_h100(64), flow), cand


def remapped(cand: Candidate, shift: int, n_gpus: int) -> Candidate:
    """The same degrees on other devices: every device id moves by `shift` (mod the cluster)."""
    devs = np.roll(np.arange(n_gpus), shift)
    return Candidate.of({n: CallPlan(tuple(devs[list(p.devices)].tolist()), p.degrees, p.order, p.micro_batch)
                         for n, p in cand.plans})
