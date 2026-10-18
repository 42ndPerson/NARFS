"""The evaluator: one rule per block type, returning a Summary.

Contention uses the fluid model: each leaf is timed with only its own
contention, and a Parallel parent adds its children's demand on every
resource. The fluid model is a lower bound. With `serial=True` nothing
overlaps (Parallel, FlowShop2 and AsyncPipe are summed, and a Pipeline adds
its shared-resource time to every slot), which gives the upper bound.

All functions here are pure: they read the block and the context and return
new arrays. This keeps later caching and delta evaluation simple additions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from narfs.blocks import (AsyncPipe, Call, Collective, FlowShop2, GenBatch, Op, Par,
                          OnStage, Pipeline, Repeat, Scale, Seq, StageP2P, Stream, Transfers)
from narfs.costs import CostModel
from narfs.hardware import DeviceSpec
from narfs.network import kernels as K
from narfs.network.topology import Cluster
from narfs.parallel.mesh import Placement


@dataclass(frozen=True)
class Summary:
    """What every block returns.

    t         block time with its own contention only (seconds)
    demand    work on each resource of the global index over the block
              (bytes on links, seconds on `gpu`); None means all zero
    mem_peak  peak bytes per device during the block; None means all zero.
              Model state (weights, optimizer state) is added once per model
              by the memory model, so it is never double counted.
    """
    t: float
    demand: np.ndarray | None = None
    mem_peak: np.ndarray | None = None


@dataclass
class EvalContext:
    net: Cluster
    costs: CostModel
    placements: dict[str, Placement]
    memory: dict[str, np.ndarray] = field(default_factory=dict)   # call -> peak bytes per device
    serial: bool = False                     # True gives the upper bound
    record: list | None = None               # explain pass: rows are appended here
    call: str | None = None
    stage: int | None = None                 # set while a pipeline stage is evaluated
    _spec: dict = field(default_factory=dict)
    _collectives: dict = field(default_factory=dict)   # repeats inside ONE evaluation only

    @property
    def placement(self) -> Placement:
        if self.call is None:
            raise ValueError("a collective or op is outside any Call")
        return self.placements[self.call]

    def spec(self) -> DeviceSpec:
        """The slowest device class in the current call (it gates SPMD work)."""
        if self.call not in self._spec:
            classes = np.unique(self.net.class_of[self.placement.devices])
            self._spec[self.call] = min((self.net.devices[c] for c in classes), key=lambda d: d.peak_flops)
        return self._spec[self.call]


# ------------------------------------------------------------- helpers ---

def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _max(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return np.maximum(a, b)


def _scale(a, n):
    return None if a is None else a * n


def _shared_bound(demand, net: Cluster) -> float:
    return 0.0 if demand is None else K.burst_time(demand, net)


# ------------------------------------------------------------ evaluator ---

def evaluate(block, ctx: EvalContext) -> Summary:
    s = _evaluate(block, ctx)
    if ctx.record is not None:
        hot = None
        if s.demand is not None and s.demand.any():
            r = int(np.argmax(s.demand / ctx.net.cap))
            hot = ctx.net.resource_name(r)
        ctx.record.append((type(block).__name__, getattr(block, "name", ""), s.t, hot))
    return s


def _evaluate(b, ctx: EvalContext) -> Summary:
    net = ctx.net
    match b:
        case Op():
            return Summary(b.count * ctx.costs.op_time(b, ctx.spec()))

        case Collective():
            key = (ctx.call, ctx.stage, b.kind, b.axis, b.size)
            hit = ctx._collectives.get(key)
            if hit is None:
                groups = ctx.placement.groups(b.axis, ctx.stage)
                if groups.shape[1] == 1:
                    hit = (0.0, None)
                else:
                    demand, latency = K.collective(b.kind, groups, b.size, net)
                    hit = (K.burst_time(demand, net) + latency, demand)
                ctx._collectives[key] = hit
            t, demand = hit
            return Summary(b.count * t, _scale(demand, b.count))

        case StageP2P():
            src, dst = ctx.placement.pp_pairs()
            if src.size == 0:
                return Summary(0.0)
            demand = K.p2p_demand(src, dst, b.size, net)
            t = K.burst_time(demand, net) + net.nic_alpha
            return Summary(b.count * t, b.count * demand)

        case Stream():
            demand = K.stream_demand(ctx.placement.devices, b.size, net, b.kind)
            return Summary(b.fill + K.burst_time(demand, net), demand)

        case Transfers():
            if not b.src:
                return Summary(0.0)
            demand = K.p2p_demand(np.array(b.src), np.array(b.dst), np.array(b.size), net)
            return Summary(K.burst_time(demand, net) + net.nic_alpha, demand)

        case Seq():
            parts = [evaluate(c, ctx) for c in b.children]
            demand = mem = None
            for p in parts:
                demand, mem = _add(demand, p.demand), _max(mem, p.mem_peak)
            return Summary(sum(p.t for p in parts), demand, mem)

        case Par():
            parts = [evaluate(c, ctx) for c in b.children]
            demand = mem = None
            for p in parts:
                demand, mem = _add(demand, p.demand), _add(mem, p.mem_peak)
            if ctx.serial:
                t = sum(p.t for p in parts)
            else:
                t = max([p.t for p in parts] + [_shared_bound(demand, net)])
            return Summary(t, demand, mem)

        case Repeat():
            s = evaluate(b.body, ctx)
            return Summary(b.n * s.t, _scale(s.demand, b.n), s.mem_peak)

        case OnStage():
            outer, ctx.stage = ctx.stage, b.stage
            try:
                return evaluate(b.body, ctx)
            finally:
                ctx.stage = outer

        case Scale():
            s = evaluate(b.body, ctx)
            return Summary(b.factor * s.t, _scale(s.demand, b.factor), s.mem_peak)

        case Pipeline():
            return _pipeline(b, ctx)

        case FlowShop2():
            return _flowshop(b, ctx)

        case AsyncPipe():
            p, c = evaluate(b.producer, ctx), evaluate(b.consumer, ctx)
            demand = _add(p.demand, c.demand)
            if ctx.serial:
                t = p.t + c.t
            else:
                t = max(p.t, c.t, _shared_bound(demand, net))
            return Summary(t, demand, _add(p.mem_peak, c.mem_peak))

        case GenBatch():
            return _genbatch(b, ctx)

        case Call():
            outer, ctx.call = (ctx.call, ctx.stage), b.name
            ctx.stage = None
            try:
                s = evaluate(b.body, ctx)
                demand = s.demand.copy() if s.demand is not None else net.zeros()
                demand[net.GPU + ctx.placement.devices] += s.t     # devices are busy for the call
                mem = ctx.memory.get(b.name)
                return Summary(s.t, demand, _max(s.mem_peak, mem))
            finally:
                ctx.call, ctx.stage = outer

    raise TypeError(f"unknown block type {type(b).__name__}")


def _pipeline(b: Pipeline, ctx: EvalContext) -> Summary:
    outer = ctx.stage
    stages = []
    try:
        for i, s in enumerate(b.stages):
            ctx.stage = i
            stages.append(evaluate(s, ctx))
    finally:
        ctx.stage = outer
    p2p = evaluate(b.p2p, ctx) if b.p2p is not None else Summary(0.0)   # all stage pairs
    times = [s.t + p2p.t for s in stages]
    per_mb = p2p.demand
    for s in stages:
        per_mb = _add(per_mb, s.demand)
    v = b.virtual
    if ctx.serial:
        # stages run on different devices, so the schedule stays; contention
        # between stages is assumed not to overlap with their work at all
        slot = max(times) + _shared_bound(per_mb, ctx.net)
    else:
        slot = max(times + [_shared_bound(per_mb, ctx.net)])
    t = sum(times) / v + (b.nmb - 1.0 / v) * slot
    mem = None
    for s in stages:
        mem = _add(mem, s.mem_peak)
    return Summary(t, _scale(per_mb, b.nmb), mem)


def _flowshop(b: FlowShop2, ctx: EvalContext) -> Summary:
    if len(b.a) != len(b.b):
        raise ValueError("FlowShop2 needs the same number of chunks in a and b")
    seen: dict = {}

    def ev(x):
        key = id(x)
        if key not in seen:
            seen[key] = evaluate(x, ctx)
        return seen[key]

    sa = [ev(x) for x in b.a]
    sb = [ev(x) for x in b.b]
    a = np.array([s.t for s in sa])
    bb = np.array([s.t for s in sb])
    if ctx.serial:
        t = a.sum() + bb.sum()
    elif b.buffer is None:
        prefix_a = np.cumsum(a)
        suffix_b = np.cumsum(bb[::-1])[::-1]
        t = float((prefix_a + suffix_b).max())
    else:
        fa = np.zeros(len(a))
        fb = np.zeros(len(a))
        for i in range(len(a)):
            start_a = fa[i - 1] if i else 0.0
            if i >= b.buffer:
                start_a = max(start_a, fb[i - b.buffer])
            fa[i] = start_a + a[i]
            fb[i] = max(fa[i], fb[i - 1] if i else 0.0) + bb[i]
        t = float(fb[-1])
    demand = None
    for s in sb:
        demand = _add(demand, s.demand)
    if b.count_a:
        for s in sa:
            demand = _add(demand, s.demand)
    else:
        t -= float(a.sum())
        t = max(t, 0.0)
    return Summary(float(t), demand)


def _genbatch(b: GenBatch, ctx: EvalContext) -> Summary:
    """Closed form for continuous batching.

    1. Saturated phase: while requests wait, the batch stays at its cap B.
       Steps = (N - B) * E[len] / B, at the average context.
    2. Tail phase: B (or N) sequences run until the longest one ends. The
       batch at step s is n_tail * P(len > s), never below 1 while the
       expected longest sequence runs.
    Step time: roofline of one decode step at batch b and context c, plus the
    per-token collectives.
    """
    net, spec = ctx.net, ctx.spec()
    n = max(1, b.replica_requests)
    lengths = np.asarray(b.lengths, dtype=float)
    probs = np.asarray(b.probs, dtype=float) / float(np.sum(b.probs))
    mean_len = float((lengths * probs).sum())
    cap = max(1, int(min(b.max_seqs, b.kv_capacity_tokens // max(b.prompt_len + mean_len, 1.0))))

    # per-token collectives: time scales with the batch, latency does not
    comm_alpha, comm_per_seq, demand_per_seq = 0.0, 0.0, None
    for c in b.comm:
        s = evaluate(c, ctx)
        groups = ctx.placement.groups(c.axis)
        lat = K.steps_latency(c.kind, groups, net) * c.count if groups.shape[1] > 1 else 0.0
        comm_alpha += lat
        comm_per_seq += s.t - lat
        demand_per_seq = _add(demand_per_seq, s.demand)

    def step_time(batch, context):
        batch = np.maximum(batch, 1.0)
        gemv = batch < 16
        ce = np.where(gemv, spec.gemv_eff, spec.compute_eff)
        me = np.where(gemv, spec.gemv_memory_eff, spec.memory_eff)
        compute = batch * b.flops_per_token / (spec.peak_flops * ce)
        memory = (b.weight_bytes + batch * context * b.kv_bytes_per_token) / (spec.hbm_bw * me)
        return (np.maximum(compute, memory) + b.kernels_per_step * spec.launch_s
                + comm_alpha + batch * comm_per_seq)

    t = 0.0
    n_tail = min(n, cap)
    if n > cap:
        steps_sat = (n - cap) * mean_len / cap
        t += steps_sat * float(step_time(np.array(cap, dtype=float), b.prompt_len + mean_len / 2))
    # expected longest of n_tail draws: the (1 - 1/(n_tail+1)) quantile
    order = np.argsort(lengths)
    cdf = np.cumsum(probs[order])
    q = 1.0 - 1.0 / (n_tail + 1)
    l_max = float(lengths[order][min(np.searchsorted(cdf, q), len(cdf) - 1)])
    steps = np.arange(int(np.ceil(l_max)), dtype=float)
    survive = np.array([probs[lengths > s].sum() for s in lengths[order]])  # P(len > l_i)
    p_alive = np.interp(steps, lengths[order], survive, left=1.0, right=0.0)
    batch = np.maximum(n_tail * p_alive, 1.0)
    t += float(step_time(batch, b.prompt_len + steps).sum())

    tokens = n * mean_len
    demand = _scale(demand_per_seq, tokens)
    if b.prefill is not None:
        pre = evaluate(b.prefill, ctx)
        t += pre.t
        demand = _add(demand, pre.demand)
    return Summary(t, demand)


# ------------------------------------------------------------- outputs ---

@dataclass(frozen=True)
class Bounds:
    lo: float            # fluid model (optimistic)
    hi: float            # no overlap: Parallel, FlowShop2 and AsyncPipe serialized;
                         # Pipeline keeps its schedule but adds contention fully
    summary: Summary


def evaluate_bounds(block, ctx: EvalContext) -> Bounds:
    lo = evaluate(block, ctx)
    ctx.serial = True
    try:
        hi = evaluate(block, ctx).t
    finally:
        ctx.serial = False
    return Bounds(lo.t, hi, lo)


def explain(block, ctx: EvalContext) -> list[tuple]:
    """Rows (block type, name, time, hot resource) for every evaluated block,
    children before parents. For chosen plans only, not for the hot loop."""
    ctx.record = []
    try:
        evaluate(block, ctx)
        return ctx.record
    finally:
        ctx.record = None
