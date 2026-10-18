"""The RLHF level: the problem, the candidate encoding, feasibility, and the
iteration tree for one candidate.

Level 1 (calls on device groups), level 2 (degrees) and level 3 (device
mapping) are all inside a Candidate, because the optimizer searches them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from narfs.blocks import AsyncPipe, Call, Par, Seq, Stream, Transfers
from narfs.costs import CostModel
from narfs.evaluate import EvalContext
from narfs.expand import Workload, expand_forward, expand_generate, expand_train, microbatches
from narfs.memory import CallMemory, forward_memory, generate_memory, train_memory
from narfs.model import library as lib
from narfs.model.architecture import Architecture
from narfs.network.topology import Cluster
from narfs.parallel.mesh import AXES, Degrees, Placement, map_devices
from narfs.parallel.rules import MEGATRON, ShardingRules, local_dims, stage_layers
from narfs.profiles import MEGATRON_DEFAULT, FrameworkProfile


# ------------------------------------------------------------- problem ---

@dataclass(frozen=True)
class CallSpec:
    name: str
    model: str                       # key into Problem.models
    kind: str                        # "generate", "forward" or "train"
    profile: FrameworkProfile = MEGATRON_DEFAULT

    def __post_init__(self) -> None:
        if self.kind not in ("generate", "forward", "train"):
            raise ValueError(f"unknown call kind {self.kind!r}")


# Iteration structure: ("seq", ...), ("par", ...), ("call", name), ("reshard", src_call, dst_call)
PPO_PLAN = ("seq",
            ("call", "generate"),
            ("par", ("call", "actor_logprob"), ("call", "ref_logprob"), ("call", "critic_value"), ("call", "reward")),
            ("par", ("call", "actor_train"), ("call", "critic_train")),
            ("reshard", "actor_train", "generate"))


@dataclass(frozen=True)
class Problem:
    cluster: Cluster
    models: tuple                    # (name, Architecture) pairs
    calls: tuple                     # CallSpec, one per call
    workload: Workload
    plan: tuple = PPO_PLAN
    rules: ShardingRules = MEGATRON
    costs: CostModel = field(default_factory=CostModel)
    async_rl: bool = False           # generation of step k+1 overlaps training of step k
    offload_idle: bool = False       # weights of idle calls are moved to the host
    memory_margin: float = 0.05

    def model(self, name: str) -> Architecture:
        for n, a in self.models:
            if n == name:
                return a
        raise KeyError(name)

    def call(self, name: str) -> CallSpec:
        for c in self.calls:
            if c.name == name:
                return c
        raise KeyError(name)


# ----------------------------------------------------------- candidate ---

@dataclass(frozen=True)
class CallPlan:
    devices: tuple                   # device IDs, in the order the mapping uses
    degrees: Degrees
    order: tuple = AXES              # slowest to fastest varying axis
    micro_batch: int = 1


@dataclass(frozen=True)
class Candidate:
    plans: tuple                     # (call name, CallPlan) pairs

    def plan(self, name: str) -> CallPlan:
        for n, p in self.plans:
            if n == name:
                return p
        raise KeyError(name)

    def with_plan(self, name: str, plan: CallPlan) -> "Candidate":
        return Candidate(tuple((n, plan if n == name else p) for n, p in self.plans))


def placements(cand: Candidate) -> dict[str, Placement]:
    return {n: map_devices(p.devices, p.degrees, p.order) for n, p in cand.plans}


# -------------------------------------------------------------- memory ---

def call_memory(problem: Problem, name: str, plan: CallPlan) -> CallMemory:
    spec = problem.call(name)
    arch = problem.model(spec.model)
    deg = plan.degrees
    if spec.kind == "train":
        return train_memory(arch, deg, problem.rules, spec.profile, problem.workload, plan.micro_batch)
    if spec.kind == "forward":
        return forward_memory(arch, deg, problem.rules, spec.profile, problem.workload, plan.micro_batch)
    device = problem.cluster.device_spec(int(plan.devices[0]))
    return generate_memory(arch, deg, problem.rules, spec.profile, problem.workload, device)


def device_memory(problem: Problem, cand: Candidate, pls: dict, mems: dict):
    """Peak bytes per device, and each call's peak on its devices.

    Without offload, model state stays: per device, the state of every model
    is counted once (the largest layout of that model on the device), and the
    largest transient of any call on the device is added. With offload, only
    one call's state is on a device at a time."""
    G = problem.cluster.n_devices
    state_by_model: dict[str, np.ndarray] = {}
    per_call_state: dict[str, np.ndarray] = {}
    per_call_transient: dict[str, np.ndarray] = {}
    for name, _ in cand.plans:
        pl, m = pls[name], mems[name]
        state = np.zeros(G)
        transient = np.zeros(G)
        for s in range(pl.degrees.pp):
            devs = pl.stage_devices(s)
            state[devs] = m.state[s]
            transient[devs] = m.transient[s]
        per_call_state[name], per_call_transient[name] = state, transient
        model = problem.call(name).model
        state_by_model[model] = np.maximum(state_by_model.get(model, np.zeros(G)), state)
    if problem.offload_idle:
        per_call_peak = {n: per_call_state[n] + per_call_transient[n] for n in per_call_state}
    else:
        total_state = sum(state_by_model.values())
        per_call_peak = {n: total_state + per_call_transient[n] for n in per_call_state}
    peak = np.max(np.stack(list(per_call_peak.values())), axis=0)
    return peak, per_call_peak


# --------------------------------------------------------- feasibility ---

def check(problem: Problem, cand: Candidate):
    """Returns (placements, memories, per-call peak memory) or raises
    ValueError with the reason a candidate is not feasible. Cheap checks
    come first."""
    G = problem.cluster.n_devices
    names = {n for n, _ in cand.plans}
    missing = {c.name for c in problem.calls} - names
    if missing:
        raise ValueError(f"no plan for calls {sorted(missing)}")
    for name, plan in cand.plans:
        if len(plan.devices) != plan.degrees.size:
            raise ValueError(f"{name}: {len(plan.devices)} devices for degrees of size {plan.degrees.size}")
        if min(plan.devices) < 0 or max(plan.devices) >= G:
            raise ValueError(f"{name}: device ID outside the cluster")
        spec = problem.call(name)
        local_dims(problem.model(spec.model), plan.degrees, problem.rules)        # divisibility
        if spec.kind == "generate":
            if plan.degrees.pp != 1:
                raise ValueError(f"{name}: generation with pp > 1 is not implemented")
        else:
            microbatches(problem.workload, plan.degrees, plan.micro_batch)
    pls = placements(cand)
    mems = {n: call_memory(problem, n, p) for n, p in cand.plans}
    peak, per_call = device_memory(problem, cand, pls, mems)
    caps = np.array([d.hbm_bytes for d in problem.cluster.devices])[problem.cluster.class_of]
    over = peak > caps * (1.0 - problem.memory_margin)
    if over.any():
        d = int(np.argmax(peak - caps))
        raise ValueError(f"out of memory on device {d}: {peak[d] / 1e9:.1f} GB needed, "
                         f"{caps[d] / 1e9:.0f} GB available")
    return pls, mems, per_call


# ------------------------------------------------------------ resharding ---

def _param_split(arch: Architecture, spec) -> tuple[float, float, float]:
    """Full parameters of one layer, split into (TP-sharded, replicated, EP-sharded)."""
    h = arch.dim("hidden")
    rep = 2.0 * h
    if spec.attention == "gqa":
        hq, hkv, dh = arch.dim("heads"), arch.dim("kv_heads"), arch.dim("head_dim")
        tp = h * (hq + 2 * hkv) * dh + hq * dh * h
    else:
        hq, dh, dr, dv = arch.dim("heads"), arch.dim("head_dim"), arch.dim("rope_dim"), arch.dim("v_head_dim")
        ql, kl = arch.dim("q_lora"), arch.dim("kv_lora")
        tp = ql * hq * (dh + dr) + kl * hq * (dh + dv) + hq * dv * h
        rep += h * ql + h * (kl + dr)
    ep = 0.0
    if spec.ffn == "dense":
        tp += 3.0 * h * arch.dim("ffn")
    else:
        ef = arch.dim("expert_ffn")
        rep += h * arch.dim("experts")
        ep = arch.dim("experts") * 3.0 * h * ef
        tp += 3.0 * h * ef * arch.dim("shared_experts", 0)
    return tp, rep, ep


def _overlap(i: int, n: int, j: int, m: int) -> float:
    """Length of [i/n, (i+1)/n) ∩ [j/m, (j+1)/m)."""
    return max(0.0, min((i + 1) / n, (j + 1) / m) - max(i / n, j / m))


def reshard_transfers(arch: Architecture, src: Placement, dst: Placement) -> Transfers:
    """Generic layout change of every parameter, from one placement to another.

    Each destination rank needs its stage's layers, its TP slice of the
    TP-sharded parameters, its EP slice of the experts, and a full copy of the
    replicated parameters. It reads each piece from the source rank that holds
    the overlapping slice, in source DP replica (dst dp mod src dp)."""
    Ds, Dd = src.degrees, dst.degrees
    dt = arch.dtype_bytes
    s_layers = [set(range(a, a + len(x))) for a, x in _stage_starts(stage_layers(arch, Ds.pp))]
    d_layers = [set(range(a, a + len(x))) for a, x in _stage_starts(stage_layers(arch, Dd.pp))]
    layer_list = arch.layer_list()
    embed = arch.dim("vocab") * arch.dim("hidden")
    srcs, dsts, sizes = [], [], []
    dp_idx = np.arange(Dd.dp)[:, None, None]            # dst coordinates (dp, ep, cp) broadcast
    for sd in range(Dd.pp):
        for ss in range(Ds.pp):
            common = d_layers[sd] & s_layers[ss]
            p_tp = p_rep = p_ep = 0.0
            for i in common:
                a, b, c = _param_split(arch, layer_list[i])
                p_tp, p_rep, p_ep = p_tp + a, p_rep + b, p_ep + c
            if sd == 0 and ss == 0:
                p_tp += embed
            if sd == Dd.pp - 1 and ss == Ds.pp - 1:
                p_tp += embed
            if p_tp + p_rep + p_ep == 0:
                continue
            ep_idx = np.arange(Dd.ep)[None, :, None]
            for td in range(Dd.tp):
                d_dev = dst.dev_of[:, sd, :, :, td]                                # [dp, ep, cp]
                for ts in range(Ds.tp):
                    f = _overlap(td, Dd.tp, ts, Ds.tp)
                    if f > 0 and p_tp > 0:
                        s_dev = src.dev_of[dp_idx % Ds.dp, ss, ep_idx % Ds.ep, 0, ts]
                        _push(srcs, dsts, sizes, s_dev, d_dev, p_tp * f * dt)
                s_dev = src.dev_of[dp_idx % Ds.dp, ss, ep_idx % Ds.ep, 0, td % Ds.tp]
                if p_rep > 0:
                    _push(srcs, dsts, sizes, s_dev, d_dev, p_rep * dt)
            if p_ep > 0:
                for ed in range(Dd.ep):
                    d_dev = dst.dev_of[:, sd, ed, :, :]                            # [dp, cp, tp]
                    tp_idx = np.arange(Dd.tp)[None, None, :]
                    dpi = np.arange(Dd.dp)[:, None, None]
                    for es in range(Ds.ep):
                        f = _overlap(ed, Dd.ep, es, Ds.ep)
                        if f > 0:
                            s_dev = src.dev_of[dpi % Ds.dp, ss, es, 0, tp_idx % Ds.tp]
                            _push(srcs, dsts, sizes, s_dev, d_dev, p_ep * f * dt)
    if not srcs:
        return Transfers((), (), ())
    s, d, z = np.concatenate(srcs), np.concatenate(dsts), np.concatenate(sizes)
    return Transfers(tuple(s.tolist()), tuple(d.tolist()), tuple(z.tolist()), name="reshard")


def _push(srcs, dsts, sizes, s_dev, d_dev, size):
    s_dev, d_dev = np.broadcast_arrays(s_dev, d_dev)
    srcs.append(s_dev.ravel())
    dsts.append(d_dev.ravel())
    sizes.append(np.full(d_dev.size, float(size)))


def _stage_starts(stages):
    start = 0
    for layers in stages:
        yield start, layers
        start += len(layers)


# ------------------------------------------------------- iteration tree ---

def expand_call(problem: Problem, name: str, plan: CallPlan, mem: CallMemory):
    spec = problem.call(name)
    arch = problem.model(spec.model)
    wl = problem.workload
    if spec.kind == "train":
        body = expand_train(arch, plan.degrees, problem.rules, spec.profile, wl, plan.micro_batch)
    elif spec.kind == "forward":
        body = expand_forward(arch, plan.degrees, problem.rules, spec.profile, wl, plan.micro_batch)
    else:
        body = expand_generate(arch, plan.degrees, problem.rules, spec.profile, wl, mem.kv_capacity_tokens)
    if problem.offload_idle:
        weights = max(mem.state)
        body = Seq((Stream(weights, name="onload"), body, Stream(weights, name="offload")), name=name)
    return Call(name, body)


def build(problem: Problem, cand: Candidate):
    """The block tree and evaluation context for one candidate. Raises
    ValueError if the candidate is not feasible."""
    pls, mems, per_call_peak = check(problem, cand)

    def node(spec):
        tag = spec[0]
        if tag == "call":
            return expand_call(problem, spec[1], cand.plan(spec[1]), mems[spec[1]])
        if tag == "reshard":
            src, dst = spec[1], spec[2]
            arch = problem.model(problem.call(src).model)
            return reshard_transfers(arch, pls[src], pls[dst])
        children = tuple(node(c) for c in spec[1:])
        return Seq(children) if tag == "seq" else Par(children)

    tree = node(problem.plan)
    if problem.async_rl:
        if problem.plan[0] != "seq" or problem.plan[1][0] != "call":
            raise ValueError("async_rl needs a plan of the form ('seq', ('call', <generate>), ...)")
        producer = node(problem.plan[1])
        consumer = Seq(tuple(node(c) for c in problem.plan[2:]))
        tree = AsyncPipe(producer, consumer, name="async iteration")
    ctx = EvalContext(problem.cluster, problem.costs, pls, memory=per_call_peak)
    return tree, ctx


def ppo_problem(cluster: Cluster, actor: Architecture, workload: Workload,
                critic: Architecture | None = None, reward: Architecture | None = None,
                train_profile: FrameworkProfile = MEGATRON_DEFAULT,
                gen_profile: FrameworkProfile | None = None, **kw) -> Problem:
    """A standard PPO problem: actor, reference, critic and reward models."""
    from narfs.profiles import VLLM_DEFAULT
    critic = critic or actor
    reward = reward or critic
    gen_profile = gen_profile or VLLM_DEFAULT
    models = (("actor", actor), ("ref", actor), ("critic", critic), ("reward", reward))
    calls = (CallSpec("generate", "actor", "generate", gen_profile),
             CallSpec("actor_logprob", "actor", "forward", train_profile),
             CallSpec("ref_logprob", "ref", "forward", train_profile),
             CallSpec("critic_value", "critic", "forward", train_profile),
             CallSpec("reward", "reward", "forward", train_profile),
             CallSpec("actor_train", "actor", "train", train_profile),
             CallSpec("critic_train", "critic", "train", train_profile))
    return Problem(cluster, models, calls, workload, **kw)
