"""A standard PPO case on HGX H100 nodes: generation on half of the
cluster, every other call colocated on the other half."""
from __future__ import annotations

from narfs.expand import Workload
from narfs.model.architecture import LLAMA3_8B, LLAMA3_70B
from narfs.network.presets import hgx_h100
from narfs.optimize import Simulator, group_plan
from narfs.parallel.mesh import Degrees as D
from narfs.rlhf import ppo_problem

TRAIN_CALLS = ["actor_logprob", "ref_logprob", "critic_value", "reward", "actor_train", "critic_train"]


def ppo_case(nodes: int = 16, batch: int = 512, **kw):
    net = hgx_h100(nodes)
    half = net.n_devices // 2
    wl = Workload(batch, 1024, (256, 1024, 2048, 4096), (0.3, 0.4, 0.2, 0.1))
    prob = ppo_problem(net, LLAMA3_70B, wl, critic=LLAMA3_8B, **kw)
    big = D(dp=half // 32, pp=4, tp=8)
    small = D(dp=half // 8, tp=8)
    deg = {"generate": small, "actor_logprob": big, "ref_logprob": big, "critic_value": small,
           "reward": small, "actor_train": big, "critic_train": small}
    mb = {"actor_logprob": 2, "ref_logprob": 2, "critic_value": 2, "reward": 2}
    cand = group_plan(prob, [(half, ["generate"]), (half, TRAIN_CALLS)], deg, mb)
    return prob, cand, Simulator(prob)
