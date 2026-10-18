"""Level 4: expansion of one call into a small block tree.

Expansion depends on the local shapes (from the degrees), the framework
profile and the workload. It does NOT depend on which devices hold the mesh
coordinates, so the same tree serves any device mapping.

Repeat is used only for consecutive identical layers inside one stage: same
shapes, same devices, same context, and no overlap across layer boundaries
(overlap inside a layer is a FlowShop2 inside the repeated body).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from narfs.blocks import Collective, FlowShop2, GenBatch, OnStage, Op, Pipeline, Repeat, Scale, Seq, StageP2P
from narfs.model import library as lib
from narfs.model.architecture import Architecture, LayerSpec
from narfs.parallel.mesh import Degrees
from narfs.parallel.rules import ShardingRules, local_dims, stage_layers
from narfs.profiles import FrameworkProfile


@dataclass(frozen=True)
class Workload:
    """One RLHF iteration's data: sequences per iteration and their lengths."""
    global_batch: int
    prompt_len: float
    lengths: tuple            # response length histogram: values ...
    probs: tuple              # ... and their probabilities

    @property
    def mean_response(self) -> float:
        total = sum(self.probs)
        return sum(l * p for l, p in zip(self.lengths, self.probs)) / total

    @property
    def seq_len(self) -> float:
        return self.prompt_len + self.mean_response


def microbatches(wl: Workload, deg: Degrees, micro_batch: int) -> int:
    """Microbatches per data replica (dp x ep replicas)."""
    per_replica, rest = divmod(wl.global_batch, deg.data)
    if rest or per_replica % micro_batch:
        raise ValueError(f"global batch {wl.global_batch} does not split into dp x ep = {deg.data} "
                         f"x micro-batches of {micro_batch}")
    return per_replica // micro_batch


def _runs(layers: list[LayerSpec]) -> list[tuple[LayerSpec, int]]:
    """Consecutive identical layer types, as (type, count)."""
    out: list = []
    for spec in layers:
        if out and out[-1][0] == spec:
            out[-1] = (spec, out[-1][1] + 1)
        else:
            out.append((spec, 1))
    return out


def _stage_blocks(c: lib.ShapeCtx, layers: list, first: bool, last: bool, train: bool):
    """Forward and backward blocks of one stage for one microbatch."""
    fwd: list = lib.embed_forward(c) if first else []
    bwd: list = []
    for spec, n in _runs(layers):
        body = lib.layer_forward(c, spec)
        fwd.append(Repeat(n, Seq(tuple(body)), name=f"{spec.attention}/{spec.ffn} layer"))
        if train:
            bwd.insert(0, Repeat(n, Seq(tuple(lib.layer_backward(c, body))), name=f"{spec.attention}/{spec.ffn} layer bwd"))
    if last:
        head = lib.head_forward(c, with_loss=train)
        fwd += head
        if train:
            bwd = [lib.backward(b) for b in reversed(head)] + bwd
    return Seq(tuple(fwd), name="forward"), Seq(tuple(bwd), name="backward")


def _shape_ctx(arch, deg, rules, profile, tokens, seq) -> lib.ShapeCtx:
    return lib.ShapeCtx(arch, local_dims(arch, deg, rules), deg, profile, tokens / deg.cp, seq, arch.dtype_bytes)


def stage_param_split(arch: Architecture, deg: Degrees, rules: ShardingRules) -> list[tuple[float, float]]:
    """Per-rank parameters of each PP stage, as (non-expert, routed expert).
    Non-expert replicas are the (dp, ep) groups; expert replicas the dp groups."""
    local = local_dims(arch, deg, rules)
    per_type = {spec: (lib.layer_params(arch, spec, local, deg), lib.layer_expert_params(arch, spec, local))
                for spec in set(arch.layer_list())}
    out = []
    stages = stage_layers(arch, deg.pp)
    for s, layers in enumerate(stages):
        expert = sum(per_type[spec][1] for spec in layers)
        dense = sum(per_type[spec][0] for spec in layers) - expert
        if s == 0:
            dense += lib.embed_params(arch, local)
        if s == len(stages) - 1:
            dense += lib.embed_params(arch, local)
        out.append((dense, expert))
    return out


def stage_param_counts(arch: Architecture, deg: Degrees, rules: ShardingRules) -> list[float]:
    """Per-rank parameters of each PP stage."""
    return [d + e for d, e in stage_param_split(arch, deg, rules)]


def replica_groups(deg: Degrees) -> tuple:
    """(axis, degree) of the replica groups of non-expert and expert parameters."""
    dense_axis = ("dp", "ep") if deg.ep > 1 else "dp"
    return (dense_axis, deg.of(dense_axis)), ("dp", deg.dp)


# ------------------------------------------------------------------ calls ---

def expand_train(arch: Architecture, deg: Degrees, rules: ShardingRules, profile: FrameworkProfile,
                 wl: Workload, micro_batch: int):
    nmb = microbatches(wl, deg, micro_batch)
    c = _shape_ctx(arch, deg, rules, profile, micro_batch * wl.seq_len, wl.seq_len)
    stages = stage_layers(arch, deg.pp)
    fb = [_stage_blocks(c, layers, s == 0, s == deg.pp - 1, train=True) for s, layers in enumerate(stages)]
    h = arch.dim("hidden")
    p2p = StageP2P(c.tokens * h * c.dt, count=2) if deg.pp > 1 else None
    pipe = Pipeline(tuple(Seq((f, b), name=f"stage {i}") for i, (f, b) in enumerate(fb)),
                    nmb, p2p, profile.pp_virtual, name="train pipeline")
    children: list = [pipe]

    # gradient sync of the largest stage: non-expert parameters over their
    # (dp, ep) replicas, routed experts over their dp replicas
    split = stage_param_split(arch, deg, rules)
    dense, expert = max(split, key=sum)
    groups = [(axis, n, p) for (axis, n), p in zip(replica_groups(deg), (dense, expert)) if n > 1 and p > 0]
    kind = "allreduce" if profile.zero_stage == 0 else "reducescatter"
    buckets = []                                   # (count, bucket) per replica group
    for axis, _, p in groups:
        grad_bytes = p * profile.grad_dtype_bytes
        k = max(1, math.ceil(grad_bytes / profile.dp_bucket_bytes))
        buckets.append((k, Collective(kind, axis, grad_bytes / k, name=f"dp grad {kind} bucket")))
    if buckets:
        k = sum(n for n, _ in buckets)
        if profile.dp_overlap:
            window = OnStage(0, Scale(1.0 / k, fb[0][1], name="backward chunk"))
            chunks = tuple(c for n, c in buckets for _ in range(n))
            children.append(FlowShop2((window,) * k, chunks, count_a=False, name="dp grad sync (exposed)"))
        else:
            children.append(Seq(tuple(Repeat(n, c) for n, c in buckets), name="dp grad sync"))
    sharded = {axis: n if profile.zero_stage >= 1 else 1 for axis, n, _ in groups}
    opt_params = sum(p / sharded.get(axis, 1) for (axis, _), p in zip(replica_groups(deg), (dense, expert)))
    children.append(Op("memory", 10.0 * opt_params, 30.0 * opt_params, name="adam"))
    if profile.zero_stage >= 1:
        for axis, _, p in groups:
            children.append(Collective("allgather", axis, p * arch.dtype_bytes, name="dp param allgather"))
    return Seq(tuple(children), name="train step")


def expand_forward(arch: Architecture, deg: Degrees, rules: ShardingRules, profile: FrameworkProfile,
                   wl: Workload, micro_batch: int):
    """Forward-only pass over prompt + response (logprobs, values, rewards)."""
    nmb = microbatches(wl, deg, micro_batch)
    c = _shape_ctx(arch, deg, rules, profile, micro_batch * wl.seq_len, wl.seq_len)
    stages = stage_layers(arch, deg.pp)
    fwd = [_stage_blocks(c, layers, s == 0, s == deg.pp - 1, train=False)[0] for s, layers in enumerate(stages)]
    p2p = StageP2P(c.tokens * arch.dim("hidden") * c.dt, count=1) if deg.pp > 1 else None
    return Pipeline(tuple(fwd), nmb, p2p, name="forward pipeline")


def generation_rates(arch: Architecture, deg: Degrees, rules: ShardingRules) -> dict:
    """Per-rank quantities for one decode step of one sequence."""
    local = local_dims(arch, deg, rules)
    h, dt = arch.dim("hidden"), arch.dtype_bytes
    weights = sum(lib.layer_params(arch, s, local, deg) * s.count for s in arch.layers) + 2 * lib.embed_params(arch, local)
    active = sum(lib.layer_active_params(arch, s, local, deg) * s.count for s in arch.layers) + lib.embed_params(arch, local)
    kv = 0.0
    for s in arch.layer_list():
        if s.attention == "gqa":
            kv += 2 * local["kv_heads"] * arch.dim("head_dim") * dt
        else:
            kv += (arch.dim("kv_lora") + arch.dim("rope_dim")) * dt
    return {"weight_bytes": weights * dt, "flops_per_token": 2.0 * active, "kv_bytes_per_token": kv}


def expand_generate(arch: Architecture, deg: Degrees, rules: ShardingRules, profile: FrameworkProfile,
                    wl: Workload, kv_capacity_tokens: float) -> GenBatch:
    if deg.pp != 1:
        raise ValueError("generation with pp > 1 is not implemented yet")
    rates = generation_rates(arch, deg, rules)
    h, dt = arch.dim("hidden"), arch.dtype_bytes
    comm = []
    if deg.tp > 1:
        comm.append(Collective("allreduce", "tp", h * dt, count=2 * arch.n_layers, name="decode tp allreduce"))
    n_moe = sum(s.count for s in arch.layers if s.ffn == "moe")
    if deg.ep > 1 and n_moe:
        comm.append(Collective("alltoall", "ep", arch.dim("top_k") * h * dt, count=2 * n_moe, name="decode ep alltoall"))
    per_replica = -(-wl.global_batch // deg.data)
    c = _shape_ctx(arch, deg, rules, profile.with_(sequence_parallel=False),
                   per_replica * wl.prompt_len, wl.prompt_len)
    prefill = _stage_blocks(c, arch.layer_list(), True, True, train=False)[0]
    return GenBatch(replica_requests=per_replica, prompt_len=wl.prompt_len, lengths=tuple(wl.lengths),
                    probs=tuple(wl.probs), kernels_per_step=arch.n_layers * profile.gen_kernels_per_layer,
                    max_seqs=profile.gen_max_seqs, kv_capacity_tokens=kv_capacity_tokens,
                    comm=tuple(comm), prefill=Seq((prefill,), name="prefill"), **rates)
