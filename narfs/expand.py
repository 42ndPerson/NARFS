"""L3: lowerings for training and forward-only calls.

A lowering turns a call (model, degrees, profile, batch) into core structure
with handles. It knows meaning (stages, microbatches, gradient buckets) and
never computes time. The same tree serves any device mapping of the same
degrees; only the binding changes.

    one_f_one_b   Seq(fill: s1 ... sP, Repeat(nmb v - 1, Par(s1 ... sP))), each
                  stage s_i = OnDevices(stage i, its work for one microbatch);
                  with v interleaved chunks every s is Repeat(1/v, s)
    overlap       a chunked producer and consumer on the same devices:
                  Seq(a1, Par(a2, b1), ..., Par(aK, b(K-1)), bK); runs of equal
                  pairs become Repeat. For equal chunks this is the two-machine
                  flow shop, and Par adds the shared-resource bound.

The 1F1B form is exact for identical microbatches in a forward-only pipeline
(a flow shop with identical jobs); with forward and backward lumped per stage
it is the standard 1F1B approximation. The DP gradient sync overlaps the real
last stage-0 backward through a tail hook: that backward is moved from the
fill to the end, where the buckets overlap it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from narfs import costs
from narfs.blocks import Collective, Compute, OnDevices, Par, Repeat, Seq, Transfer
from narfs.hardware import DTYPE_BYTES
from narfs.model.architecture import Architecture, runs
from narfs.model.library import Ctx, Params, backward, get, layer_params, make_ctx
from narfs.parallel.binding import Boundary, Devices, Group
from narfs.parallel.mesh import Degrees, stage_split
from narfs.profiles import TrainProfile
from narfs.variation import critical_instance


@dataclass(frozen=True)
class Batch:
    """The sequences one call processes per iteration."""
    sequences: int
    seq_mean: float                  # tokens per sequence (prompt + response)
    seq_sq: float = 0.0              # E[s^2] / E[s], the length-biased length (0: seq_mean)
    seq_std: float = 0.0             # standard deviation of tokens per sequence


# --------------------------------------------------------------- patterns ---

def overlap(a: list, b: list, name: str = "overlap"):
    """Chunk i of `b` starts after chunk i of `a`, on the same devices."""
    if len(a) != len(b) or not a:
        raise ValueError("overlap needs the same number (> 0) of producer and consumer chunks")
    parts: list = [a[0]]
    pairs = list(zip(a[1:], b[:-1]))
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] is pairs[i][0] and pairs[j + 1][1] is pairs[i][1]:
            j += 1
        body = Par(pairs[i])
        parts.append(body if j == i else Repeat(j - i + 1, body))
        i = j + 1
    parts.append(b[-1])
    return Seq(tuple(parts), name=name)


def one_f_one_b(stages: list, nmb: int, v: int = 1, fill: list | None = None, name: str = "pipeline"):
    """The 1F1B (v = 1) or interleaved schedule over per-stage microbatch work."""
    chunk = (lambda s: s) if v == 1 else (lambda s: Repeat(1.0 / v, s))
    parts = [chunk(s) for s in (fill or stages)]
    steady = nmb * v - 1
    if steady > 0:
        body = Par(tuple(chunk(s) for s in stages)) if len(stages) > 1 else chunk(stages[0])
        return Seq((Seq(tuple(parts), name="fill"), Repeat(steady, body, name="steady")), name=name)
    return Seq((Seq(tuple(parts), name="fill"),), name=name)


# ------------------------------------------------------------ call facts ---

def microbatches(batch: Batch, deg: Degrees, micro_batch: int) -> int:
    """Microbatches per data replica (dp x ep replicas)."""
    per_replica, rest = divmod(batch.sequences, deg.data)
    if rest or per_replica % micro_batch or per_replica == 0:
        raise ValueError(f"{batch.sequences} sequences do not split into dp x ep = {deg.data} replicas "
                         f"of micro-batches of {micro_batch}")
    return per_replica // micro_batch


def critical_factor(batch: Batch, deg: Degrees) -> float:
    """Token load of the slowest data replica over the mean (size variation:
    the replicas meet at the gradient sync, so the slowest sets the time)."""
    n = batch.sequences / deg.data
    mean = n * batch.seq_mean
    return critical_instance(mean, math.sqrt(n) * batch.seq_std, deg.data) / mean


def microbatch_tokens(batch: Batch, deg: Degrees, micro_batch: int) -> float:
    """Tokens per rank in one microbatch of the slowest replica, after the cp split."""
    return micro_batch * batch.seq_mean * critical_factor(batch, deg) / deg.cp


def stage_layers(arch: Architecture, pp: int, v: int = 1) -> list[list]:
    """Layer indices of each stage: the model is cut into pp x v chunks, and
    stage i holds chunks i, i + pp, ... (v = 1 is the contiguous split)."""
    chunks = stage_split(arch.n_layers, pp * v)
    return [[i for c in range(s, pp * v, pp) for i in chunks[c]] for s in range(pp)]


def stage_params(arch: Architecture, deg: Degrees, prof: TrainProfile, v: int = 1) -> list[Params]:
    """Per-rank parameters of each PP stage."""
    c = make_ctx(arch, deg, prof.rules)
    per_type = {t: layer_params(c, t) for t in set(arch.layers)}
    out = []
    for s, layers in enumerate(stage_layers(arch, deg.pp, v)):
        p = Params()
        for i in layers:
            p = p + per_type[arch.layers[i]]
        if s == 0:
            p = p + get("embed").params(c)
        if s == deg.pp - 1:
            p = p + get("head").params(c)
        out.append(p)
    return out


# ------------------------------------------------------------- lowering ---

def _ctx(call: str, arch, deg, prof: TrainProfile, stage: int, tokens: float, batch: Batch) -> Ctx:
    return make_ctx(arch, deg, prof.rules, tokens=tokens, seq=batch.seq_mean, seq_sq=batch.seq_sq,
                    tp=Group(call, ("tp",), stage), cp=Group(call, ("cp",), stage), ep=Group(call, ("ep",), stage),
                    lib=prof.lib, tp_comm=prof.tp_comm, tp_chunks=prof.tp_chunks, sp=prof.sequence_parallel,
                    vocab_parallel_loss=prof.vocab_parallel_loss)


def _stage_forward(c: Ctx, arch: Architecture, layers: list, first: bool, last: bool):
    parts = [get("embed").forward(c)] if first else []
    for layer, n in runs([arch.layers[i] for i in layers]):
        body = Seq(tuple(get(name).forward(c) for name in layer), name="+".join(layer))
        parts.append(Repeat(n, body, name=f"{'+'.join(layer)} x{n}"))
    if last:
        parts.append(get("head").forward(c))
    return Seq(tuple(parts), name="forward")


def _recompute(prof: TrainProfile, fwd):
    bwd = backward(fwd)
    if prof.recompute == "full":                  # each layer's forward runs again before its backward
        return Seq((fwd, bwd), name="backward (full recompute)")
    return bwd                                    # FlashAttention recomputes attention in its backward


def lower_call(call: str, arch: Architecture, deg: Degrees, prof: TrainProfile, batch: Batch, micro_batch: int,
               train: bool = True, minibatches: int = 1):
    """The program of a training call (`train`) or a forward-only call."""
    if minibatches > 1:
        sub = Batch(batch.sequences // minibatches, batch.seq_mean, batch.seq_sq, batch.seq_std)
        if sub.sequences * minibatches != batch.sequences:
            raise ValueError(f"{batch.sequences} sequences do not split into {minibatches} minibatches")
        step = _lower_step(call, arch, deg, prof, sub, micro_batch, train)
        return OnDevices(Devices(call), Repeat(minibatches, step, name="minibatches"), name=call)
    return OnDevices(Devices(call), _lower_step(call, arch, deg, prof, batch, micro_batch, train), name=call)


def _lower_step(call, arch, deg, prof: TrainProfile, batch: Batch, micro_batch: int, train: bool):
    nmb = microbatches(batch, deg, micro_batch)
    t = microbatch_tokens(batch, deg, micro_batch)
    P, v = deg.pp, prof.pp_virtual if deg.pp > 1 else 1
    layers = stage_layers(arch, P, v)
    h, dt = arch.dim("hidden"), DTYPE_BYTES[arch.dtype]
    split = deg.tp if (prof.scatter_gather_pp or prof.sequence_parallel) and deg.tp > 1 else 1
    act = t * h * dt / split                                   # one rank's share of a pipeline send

    heads, tails, stages = [], [], []
    for i in range(P):
        c = _ctx(call, arch, deg, prof, i, t, batch)
        fwd = _stage_forward(c, arch, layers[i], i == 0, i == P - 1)
        gather = lambda: (Collective("all_gather", Group(call, ("tp",), i), t * h * dt, prof.lib,
                                     "pp all_gather"),) if split > 1 and not prof.sequence_parallel else ()
        sends_f = [Transfer(Boundary(call, i, i + 1), act, name="pp send")] * (v if i < P - 1 else 0)
        sends_f += [Transfer(Boundary(call, P - 1, 0), act, name="pp send (wrap)")] * (v - 1 if i == P - 1 else 0)
        head = Seq((*(gather() if i > 0 else ()), fwd, *sends_f), name=f"stage {i} forward")
        tail = None
        if train:
            sends_b = [Transfer(Boundary(call, i - 1, i), act, reverse=True, name="pp grad send")] * (v if i > 0 else 0)
            sends_b += [Transfer(Boundary(call, P - 1, 0), act, reverse=True, name="pp grad send (wrap)")] \
                * (v - 1 if i == 0 else 0)
            tail = Seq((*(gather() if i < P - 1 else ()), _recompute(prof, fwd), *sends_b), name=f"stage {i} backward")
        heads.append(head)
        tails.append(tail)
        stages.append(OnDevices(Devices(call, i), Seq((head, tail)) if train else head, name=f"stage {i}"))

    if not train:
        return one_f_one_b(stages, nmb, v, name="forward pipeline")

    # tail hook: stage 0's last backward moves from the fill to the end, where the DP sync overlaps it
    fill = [OnDevices(Devices(call, 0), heads[0], name="stage 0 forward")] + stages[1:]
    pipe = one_f_one_b(stages, nmb, v, fill=fill, name="train pipeline")
    params = stage_params(arch, deg, prof, v)
    last_bwd = tails[0] if v == 1 else Repeat(1.0 / v, tails[0])
    syncs = [_dp_sync(call, i, deg, prof, params[i], last_bwd if i == 0 else None) for i in range(P)]
    end = Par(tuple(OnDevices(Devices(call, i), s, name=f"stage {i} grad sync") for i, s in enumerate(syncs)))
    opt = Par(tuple(OnDevices(Devices(call, i), _optimizer(call, i, deg, prof, params[i]), name=f"stage {i} optimizer")
                    for i in range(P)))
    return Seq((pipe, end, opt), name="train step")


def _sync_groups(call: str, i: int, deg: Degrees, p: Params):
    """(group handle, group size, per-rank parameters) of the gradient
    replicas: non-expert parameters over (dp, ep), routed experts over dp."""
    out = []
    if deg.data > 1 and p.non_expert > 0:
        out.append((Group(call, ("dp", "ep"), i), deg.data, p.non_expert))
    if deg.dp > 1 and p.ep > 0:
        out.append((Group(call, ("dp",), i), deg.dp, p.ep))
    return out


def _dp_sync(call: str, i: int, deg: Degrees, prof: TrainProfile, p: Params, last_bwd):
    gb = DTYPE_BYTES[prof.grad_dtype]
    kind = "all_reduce" if prof.zero_stage == 0 else "reduce_scatter"
    buckets = []
    for group, size, n in _sync_groups(call, i, deg, p):
        bucket = max(prof.dp_bucket_params, 1e6 * size)
        k = math.ceil(n / bucket)
        buckets += [Collective(kind, group, n * gb / k, prof.lib, f"dp grad {kind} bucket")] * k
    if last_bwd is None:
        return Seq(tuple(buckets), name="dp grad sync")
    if not buckets:
        return last_bwd
    if not prof.dp_overlap:
        return Seq((last_bwd, *buckets), name="last backward + dp grad sync")
    chunk = Repeat(1.0 / len(buckets), last_bwd, name="backward chunk")
    return overlap([chunk] * len(buckets), buckets, name="last backward + dp grad sync")


def _optimizer(call: str, i: int, deg: Degrees, prof: TrainProfile, p: Params):
    groups = _sync_groups(call, i, deg, p)
    shard = {g: size for g, size, _ in groups} if prof.zero_stage >= 1 else {}
    local = p.total
    for g, size, n in groups:
        local -= n - n / shard.get(g, 1)
    parts = [Compute(costs.elementwise(10.0 * local, 30.0 * local, "adam"), "adam")]
    if prof.zero_stage >= 1:
        dt = 2.0
        parts += [Collective("all_gather", g, n * dt, prof.lib, "dp param all_gather") for g, _, n in groups]
    return Seq(tuple(parts), name="optimizer")
