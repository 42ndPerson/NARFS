"""Memory model: per-device bytes, for feasibility and for the KV capacity.

Two parts per call and stage:
  state      model state that stays on the device (weights; for training
             also gradients and optimizer state). Counted once per model
             per device, even if several calls use the same model there.
  transient  peak working memory of the call (activations, KV cache).

Peak on a device = sum of the state of every model on it
                 + max over the calls on it of their transient memory.
"""
from __future__ import annotations

from dataclasses import dataclass

from narfs.expand import (Workload, generation_rates, microbatches, replica_groups, stage_param_counts,
                          stage_param_split)
from narfs.hardware import DeviceSpec
from narfs.model.architecture import Architecture
from narfs.parallel.mesh import Degrees
from narfs.parallel.rules import ShardingRules, local_dims, stage_layers
from narfs.profiles import FrameworkProfile

WORKSPACE_BYTES = 2e9      # library workspaces, allocator slack


@dataclass(frozen=True)
class CallMemory:
    state: tuple           # bytes per rank, one value per PP stage
    transient: tuple       # bytes per rank, one value per PP stage
    kv_capacity_tokens: float = 0.0


def _activation_bytes_per_token_layer(arch: Architecture, deg: Degrees, profile: FrameworkProfile, seq: float) -> float:
    """Stored activations per token per layer per rank (Korthikanti et al.)."""
    h = arch.dim("hidden")
    if profile.recompute == "full":
        return h * arch.dtype_bytes
    per = 34.0 * h if profile.recompute == "selective" else h * (34.0 + 5.0 * arch.dim("heads") * seq / h)
    return per / deg.tp


def train_memory(arch, deg, rules, profile, wl: Workload, micro_batch: int) -> CallMemory:
    nmb = microbatches(wl, deg, micro_batch)
    split = stage_param_split(arch, deg, rules)
    layers = stage_layers(arch, deg.pp)
    tokens = micro_batch * wl.seq_len / deg.cp
    act = _activation_bytes_per_token_layer(arch, deg, profile, wl.seq_len)
    replicas = [n for _, n in replica_groups(deg)]           # (non-expert, expert)
    state, transient = [], []
    for s, parts in enumerate(split):
        state.append(0.0)
        for p, n in zip(parts, replicas):
            grads = p * profile.grad_dtype_bytes / (n if profile.zero_stage >= 2 else 1)
            optim = p * 12.0 / (n if profile.zero_stage >= 1 else 1)
            state[-1] += p * arch.dtype_bytes + grads + optim
        in_flight = min(nmb, deg.pp - s)
        transient.append(in_flight * len(layers[s]) * tokens * act + WORKSPACE_BYTES)
    return CallMemory(tuple(state), tuple(transient))


def forward_memory(arch, deg, rules, profile, wl: Workload, micro_batch: int) -> CallMemory:
    params = stage_param_counts(arch, deg, rules)
    local = local_dims(arch, deg, rules)
    tokens = micro_batch * wl.seq_len / deg.cp
    h, dt = arch.dim("hidden"), arch.dtype_bytes
    transient = []
    for s in range(deg.pp):
        logits = tokens * local["vocab"] * 4 if s == deg.pp - 1 else 0.0
        transient.append(8 * tokens * h * dt + logits + WORKSPACE_BYTES)
    return CallMemory(tuple(p * dt for p in params), tuple(transient))


def generate_memory(arch, deg, rules, profile, wl: Workload, spec: DeviceSpec) -> CallMemory:
    rates = generation_rates(arch, deg, rules)
    weights = rates["weight_bytes"]
    free = profile.gen_kv_fraction * spec.hbm_bytes - weights - WORKSPACE_BYTES
    capacity = max(0.0, free) / rates["kv_bytes_per_token"]
    per_replica = -(-wl.global_batch // deg.data)
    needed = per_replica * (wl.prompt_len + max(wl.lengths)) * rates["kv_bytes_per_token"]
    kv_used = min(needed, max(0.0, free))
    return CallMemory((weights,), (kv_used + WORKSPACE_BYTES,), capacity)
