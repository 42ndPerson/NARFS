"""The memory model: per-device bytes, for feasibility and the KV capacity.

Memory is computed once per candidate, before lowering, from model facts,
placement, profile and workload. It never sees the block tree. It is the only
code that does memory arithmetic and the only check. Its outputs are
feasibility with a reason, headroom per device, and one number for the
engine lowering: the KV capacity.

Per call and PP stage, per rank:
    state      what stays on the device: weights; for training also gradients
               and optimizer state (2 + 4 + 12 = 18 bytes per parameter under
               mixed-precision Adam), the optimizer state divided by the ZeRO group
    transient  the call's working memory: saved activations (bytes per token
               per layer from the components x the schedule's in-flight
               microbatches), or the engine's KV pool; plus the framework's
               reserve (fragmentation, NCCL buffers, CUDA graphs, workspaces),
               read from its own memory report

Peak on a device = the state of every model on it (once per model) + the
largest transient of any call on it. An engine pre-allocates
gpu_memory_utilization of the GPU, so its transient is that pool minus its
weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from narfs.expand import Batch, microbatch_tokens, microbatches, stage_layers, stage_params
from narfs.hardware import DTYPE_BYTES, GpuProfile
from narfs.model.architecture import Architecture
from narfs.model.library import get, make_ctx
from narfs.parallel.mesh import Degrees
from narfs.profiles import EngineProfile, TrainProfile


@dataclass(frozen=True)
class CallMemory:
    state: tuple                     # bytes per rank, one value per PP stage
    transient: tuple                 # bytes per rank, one value per PP stage
    kv_capacity: float = 0.0         # KV tokens per replica (generation)
    terms: dict = field(default_factory=dict, compare=False)   # stage-0 bytes by term, for explain and tests


def _layer_act(c, arch: Architecture, layer: tuple) -> float:
    return sum(get(name).act_bytes_per_token(c) for name in layer)


def train_memory(arch: Architecture, deg: Degrees, prof: TrainProfile, batch: Batch, micro_batch: int,
                 train: bool = True) -> CallMemory:
    """A training call (or, with train=False, a forward-only call)."""
    nmb = microbatches(batch, deg, micro_batch)
    t = microbatch_tokens(batch, deg, micro_batch)
    P = deg.pp
    v = prof.pp_virtual if P > 1 else 1
    c = make_ctx(arch, deg, prof.rules, sp=prof.sequence_parallel)
    dt, gb = DTYPE_BYTES[arch.dtype], DTYPE_BYTES[prof.grad_dtype]
    params = stage_params(arch, deg, prof, v)
    layers = stage_layers(arch, P, v)
    logits = t * c.local_dim("vocab") * (dt + 4)                     # bf16 logits and fp32 loss terms
    state, transient, terms = [], [], {}
    for s in range(P):
        p = params[s]
        weights = p.total * dt
        if train:
            dense_group, expert_group = (deg.data, deg.dp) if prof.zero_stage >= 1 else (1, 1)
            grads = p.total * gb
            optim = 12.0 * (p.non_expert / dense_group + p.ep / expert_group)
            in_flight = min(nmb, P - s)
            if prof.recompute == "full":                             # layer inputs only, plus the one layer being recomputed
                acts = (in_flight * len(layers[s]) * t * dt * c.h_sp
                        + max(_layer_act(c, arch, arch.layers[i]) for i in layers[s]) * t)
            else:
                acts = in_flight * sum(_layer_act(c, arch, arch.layers[i]) for i in layers[s]) * t
        else:
            grads = optim = 0.0
            acts = max(_layer_act(c, arch, arch.layers[i]) for i in layers[s]) * t   # one layer's working set
        if s == P - 1:
            acts += logits
        state.append(weights + grads + optim)
        transient.append(acts + prof.reserve)
        if s == 0:
            terms = {"weights": weights, "gradients": grads, "optimizer": optim, "activations": acts,
                     "reserve": prof.reserve}
    return CallMemory(tuple(state), tuple(transient), 0.0, terms)


def generate_memory(arch: Architecture, deg: Degrees, eprof: EngineProfile, gpu: GpuProfile,
                    prompt: float = 0.0, longest: float = 0.0) -> CallMemory:
    """An inference engine: weights, the profile-run activation peak, the
    reserve, and the KV cache in the rest of its memory pool."""
    if deg.pp > 1:
        raise ValueError("generation with pp > 1 is not modeled")
    c = make_ctx(arch, deg, eprof.rules)
    dt = DTYPE_BYTES[arch.dtype]
    params = get("embed").params(c) + get("head").params(c)
    kv, state_seq = 0.0, 0.0
    for layer in arch.layers:
        for name in layer:
            comp = get(name)
            params = params + comp.params(c)
            kv += comp.kv_bytes_per_token(c)
            state_seq += comp.state_bytes(c)
    weights = params.total * dt
    act_peak = (eprof.max_num_batched_tokens * max(_layer_act(c, arch, layer) for layer in set(arch.layers))
                + eprof.max_num_seqs * arch.dim("vocab") * 4.0)
    pool = eprof.gpu_memory_utilization * gpu.hbm_bytes
    budget = pool - weights - act_peak - eprof.reserve - eprof.max_num_seqs * state_seq
    capacity = budget / kv if kv > 0 else (math.inf if budget > 0 else 0.0)
    terms = {"weights": weights, "activation peak": act_peak, "reserve": eprof.reserve,
             "kv budget": max(budget, 0.0), "kv bytes per token": kv, "state per sequence": state_seq,
             "fits one request": budget > 0 and (kv == 0 or capacity >= prompt + longest)}
    return CallMemory((weights,), (max(pool - weights, 0.0),), max(capacity, 0.0), terms)


@dataclass(frozen=True)
class MemoryResult:
    feasible: bool
    reason: str
    peak: np.ndarray                 # bytes per device
    headroom: np.ndarray             # capacity - peak, bytes per device


def check(uses, capacity: np.ndarray) -> MemoryResult:
    """`uses`: (call, model, Placement, CallMemory) for every call. A model's
    state counts once per device (its largest layout there)."""
    G = capacity.size
    state_by_model: dict = {}
    transient = np.zeros(G)
    owners: dict = {}
    for call, model, pl, mem in uses:
        st, tr = np.zeros(G), np.zeros(G)
        for s in range(pl.degrees.pp):
            devs = pl.stage_devices(s)
            st[devs] = mem.state[s]
            tr[devs] = mem.transient[s]
        state_by_model[model] = np.maximum(state_by_model.get(model, np.zeros(G)), st)
        transient = np.maximum(transient, tr)
        for d in pl.devices.tolist():
            owners.setdefault(d, []).append(call)
    peak = transient + (sum(state_by_model.values()) if state_by_model else 0.0)
    headroom = capacity - peak
    if (headroom >= 0).all():
        return MemoryResult(True, "", peak, headroom)
    d = int(np.argmin(headroom))
    calls = ", ".join(owners.get(d, []))
    return MemoryResult(False, f"out of memory on device {d}: {peak[d] / 1e9:.1f} GB needed, "
                               f"{capacity[d] / 1e9:.1f} GB available (calls: {calls})", peak, headroom)


def kv_problem(call: str, mem: CallMemory) -> str:
    """A reason when an engine's KV budget cannot hold one request."""
    if not mem.terms.get("fits one request", True):
        return (f"call {call}: the KV budget ({mem.terms['kv budget'] / 1e9:.1f} GB) cannot hold one request "
                f"after weights ({mem.terms['weights'] / 1e9:.1f} GB) and reserves")
    return ""
