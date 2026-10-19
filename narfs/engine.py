"""L3: the generation engine lowering, in step space.

Generation needs no time to describe: each decode step adds one token to
every active sequence, so the trajectory (batch, context, admissions) is
time-free, and the evaluator turns steps into time.

    OnDevices(call, Seq(
      prefill of the first cohort, chunked by the token budget, once per prompt group,
      Repeat(n_k, step(b_k, c_k, prefill tokens of admitted requests))   per segment k))
    step(b, c) = Seq or Par(component steps for TokenMix(b, padded b, c, p), host step(b))

The call ends at the last token of the slowest replica, so the end is the
order statistic of (fluid start + length) over every request behind that
barrier (narfs.variation.trajectory). Engine policies are profile facts:
the slot count (max_num_seqs, and KV capacity by Little's law), CUDA-graph
padding, chunked-prefill mixing, prefix sharing, and synchronous or
overlapped host scheduling.
"""
from __future__ import annotations

import math

from narfs.blocks import OnDevices, Par, Repeat, Seq, Work
from narfs.costs import HostStep
from narfs.model.architecture import Architecture, runs
from narfs.model.library import TokenMix, get, make_ctx
from narfs.parallel.binding import Devices, Group, Hosts
from narfs.parallel.mesh import Degrees
from narfs.profiles import EngineProfile
from narfs.variation import Dist, trajectory


def slots(eprof: EngineProfile, samples: int, prompt: float, dist: Dist, kv_capacity: float) -> int:
    """Running sequences per replica: max_num_seqs, or fewer when the KV cache
    is the limit. A slot holds its share of the prompt (once per group with
    prefix caching) and, by Little's law, E[L^2] / (2 E[L]) generated tokens."""
    per_slot = prompt / (samples if eprof.prefix_caching else 1) + dist.second / (2.0 * dist.mean)
    by_kv = math.floor(kv_capacity / per_slot) if kv_capacity < math.inf else eprof.max_num_seqs
    return max(1, min(eprof.max_num_seqs, by_kv))


def lower_generate(call: str, arch: Architecture, deg: Degrees, eprof: EngineProfile, prompts: int, samples: int,
                   prompt: float, response: Dist, kv_capacity: float = math.inf, segments: int = 16):
    if deg.pp > 1:
        raise ValueError(f"call {call}: generation with pp > 1 is not modeled")
    replicas = deg.data
    requests = prompts * samples
    C = slots(eprof, samples, prompt, response, kv_capacity)
    traj = trajectory(response, requests, C, replicas, segments)
    c = make_ctx(arch, deg, eprof.rules, tp=Group(call, ("tp",)), ep=Group(call, ("ep",)), cp=Group(call, ("cp",)),
                 lib=eprof.lib)
    hosts = Hosts(call, "replica")
    prompt_compute = prompt / (samples if eprof.prefix_caching else 1)     # prefill tokens per admission

    def step(mix: TokenMix, name: str):
        parts = [get("embed").step(c, mix)]
        for layer, n in runs(arch.layers):
            parts.append(Repeat(n, Seq(tuple(get(comp).step(c, mix) for comp in layer)), name=f"{'+'.join(layer)} x{n}"))
        parts.append(get("head").step(c, mix))
        gpu = Seq(tuple(parts), name="gpu step")
        host = Work(hosts, HostStep(eprof.name, mix.batch), name="host step")
        return Par((gpu, host), name=name) if eprof.async_scheduling else Seq((host, gpu), name=name)

    # the first cohort's prompts, chunked by the token budget
    first = min(C, traj.per_group) * prompt_compute
    chunks = max(1, math.ceil(first / eprof.max_num_batched_tokens))
    prefill = Repeat(chunks, step(TokenMix(0.0, 1.0, 0.0, first / chunks, prompt / 2.0), "prefill chunk"),
                     name=f"prefill: {chunks} chunk{'s' if chunks > 1 else ''}")
    decode = []
    for i, s in enumerate(traj.segments):
        b = s.active
        mix = TokenMix(b, eprof.padded(b), prompt + s.age, s.admissions * prompt_compute, prompt / 2.0)
        decode.append(Repeat(s.steps, step(mix, "decode step"),
                             name=f"decode {i + 1}/{len(traj.segments)}: batch {b:.0f}, context {prompt + s.age:.0f}"))
    info = response.info
    tail = ""
    if info.get("tail") in ("gpd", "exp"):
        share = max(0.0, 1.0 - info["u"] / traj.end) if traj.end else 0.0
        tail = f"; fitted tail above u = {info['u']} (beta {info['beta']:.0f}, xi {info['xi']:.2f}) sets {share:.0%} of the steps"
    name = f"generate: {traj.end:.0f} decode steps, {traj.first} slots x {replicas} replicas{tail}"
    return OnDevices(Devices(call), Seq((prefill, Seq(tuple(decode), name="decode")), name=name), name=call)
