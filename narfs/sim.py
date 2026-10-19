"""The simulator API: a pure, deterministic function from a problem and a
candidate to a result. It holds no search code; the optimizer lives outside
and calls it.

    sim = Simulator(problem)            # cluster, flow, system profile
    r = sim.evaluate(candidate)         # never raises for an infeasible plan
    r.feasible, r.reason                # "out of memory on device 17: ..."
    r.time, r.lo, r.hi                  # seconds per iteration
    r.phases                            # seconds of each flow node
    r.memory_headroom                   # bytes per device
    print(sim.explain(candidate))       # bound-by rows and the assumed constants

For one candidate:

    1. The memory model checks feasibility and computes each engine's KV capacity.
    2. Each call's program is lowered, or reused when its degrees, micro-batch
       and KV capacity are unchanged.
    3. The flow becomes core structure: a call is its program, cpu is Work on
       its pool, sync is the reshard Transfer, seq is Seq, par and lagged are Par.
    4. The binder binds every handle to device arrays and the evaluator walks
       the tree.

`time` is the fluid estimate `lo`. The lowerings emit Par only where the
framework really overlaps work, so `lo` is the prediction; `hi` bounds it
from above and shows how much the prediction relies on perfect sharing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from narfs import rlhf
from narfs.blocks import Par, Seq, Transfer, Work
from narfs.costs import ComputeModel, CostTable, HostModel
from narfs.engine import lower_generate
from narfs.evaluate import Evaluator, render
from narfs.expand import Batch, lower_call
from narfs.hardware import DATA, DTYPE_BYTES, H100_SXM, GpuProfile
from narfs.memory import check, generate_memory, kv_problem, train_memory
from narfs.model.library import check_architecture, get, layer_params, make_ctx
from narfs.network.comm import EngineLibrary, load_nccl
from narfs.network.topology import Topology
from narfs.optimize import Candidate
from narfs.parallel.binding import Binder, Pool, Reshard, handles
from narfs.parallel.layouts import ParamLayout
from narfs.parallel.mesh import Degrees, map_devices
from narfs.physics import Physics
from narfs.profiles import GiB, VLLM_H100_LIBRARY, EngineProfile, TrainProfile

_CACHE_LIMIT = 4096


def default_libraries() -> dict:
    """NCCL 2.32.3 and the vLLM all-reduce paths at commit 573f2f9."""
    return {"nccl": load_nccl(DATA / "nccl-2.32.3.json"), VLLM_H100_LIBRARY.name: VLLM_H100_LIBRARY}


@dataclass(frozen=True, eq=False)
class Problem:
    """What stays fixed between candidates: the cluster, the flow (models,
    profiles and workload) and the system profile (GPU profiles by device
    class, and the communication libraries the profiles name)."""
    topology: Topology
    flow: rlhf.Flow
    gpus: dict = field(default_factory=lambda: {H100_SXM.name: H100_SXM})
    libraries: dict = field(default_factory=default_libraries)
    cost_table: CostTable | None = None


@dataclass(frozen=True, eq=False)
class Result:
    feasible: bool
    reason: str = ""
    time: float = math.inf           # seconds per iteration: the fluid estimate (lo)
    lo: float = math.inf
    hi: float = math.inf
    phases: dict = field(default_factory=dict)          # flow node name -> seconds (lo), in flow order
    memory_peak: np.ndarray | None = None               # bytes per device, when the memory check ran
    memory_headroom: np.ndarray | None = None           # capacity - peak, bytes per device
    kv_capacity: dict = field(default_factory=dict)     # generate call -> KV tokens per replica


@dataclass
class _Info:
    call: rlhf.Call
    batch: Batch | None = None       # forward and train: the whole batch
    step: Batch | None = None        # train: one minibatch (what memory holds at once)


@dataclass
class _Run:
    result: Result
    tree: object = None
    evaluator: Evaluator | None = None


def _param_layout(arch, rules, expert_tp: bool) -> ParamLayout:
    """A model's parameter bytes by sharding class, from the full model."""
    c = make_ctx(arch, Degrees(), rules)
    dt = DTYPE_BYTES[arch.dtype]
    per_type = {t: layer_params(c, t) for t in set(arch.layers)}
    layers = tuple((per_type[t].tp * dt, per_type[t].rep * dt, per_type[t].ep * dt) for t in arch.layers)
    return ParamLayout(layers, get("embed").params(c).total * dt, get("head").params(c).total * dt, expert_tp)


class Simulator:
    def __init__(self, problem: Problem):
        self.problem = problem
        calls, cpus, syncs = self._collect(problem.flow)
        topo = problem.topology
        missing = sorted(set(topo.gpu_class) - set(problem.gpus))
        if missing:
            raise ValueError(f"no GPU profile for device classes {missing}; the problem has {sorted(problem.gpus)}")

        self._calls: dict[str, _Info] = {}
        host_steps: dict = {}
        for c in calls:
            check_architecture(c.model.arch)
            self._check_library(c)
            info = _Info(c)
            if c.kind == "generate":
                prev = host_steps.setdefault(c.profile.name, c.profile.host_step)
                if prev != c.profile.host_step:
                    raise ValueError(f"engine {c.profile.name!r} has two host step models: {prev} and "
                                     f"{c.profile.host_step}")
            else:
                info.batch = c.data.batch
                info.step = info.batch
                if c.minibatches > 1:
                    n, rest = divmod(info.batch.sequences, c.minibatches)
                    if rest or n == 0:
                        raise ValueError(f"call {c.name}: {info.batch.sequences} sequences do not split into "
                                         f"{c.minibatches} minibatches")
                    b = info.batch
                    info.step = Batch(n, b.seq_mean, b.seq_sq, b.seq_std)
            self._calls[c.name] = info
        self.call_names = tuple(self._calls)

        pools: dict = {}
        for cp in cpus:
            if pools.setdefault(cp.pool, cp.workers) != cp.workers:
                raise ValueError(f"CPU pool {cp.pool!r} is declared with {pools[cp.pool]} and {cp.workers} workers")
        self.topology = topo.with_pools(pools) if pools else topo

        self._leaf: dict = {}                # id(flow node) -> its fixed tree node (cpu and sync)
        for cp in cpus:
            self._leaf[id(cp)] = Work(Pool(cp.pool), cp.worker_seconds, name=cp.name)
        for s in syncs:
            self._leaf[id(s)] = Transfer(self._reshard(s), 1.0, name=s.name)

        gpus = dict(problem.gpus)
        self.physics = Physics(self.topology, problem.libraries, ComputeModel(gpus, problem.cost_table),
                               HostModel(host_steps))
        self.capacity = np.array([gpus[c].hbm_bytes for c in self.topology.gpu_class])
        self._memories: dict = {}
        self._programs: dict = {}

    # ------------------------------------------------------- problem setup ---
    @staticmethod
    def _collect(flow):
        calls: dict = {}
        cpus, syncs = [], []

        def visit(n):
            match n:
                case rlhf.Call():
                    other = calls.setdefault(n.name, n)
                    if other is not n:
                        raise ValueError(f"two calls are named {n.name!r}")
                case rlhf.Cpu():
                    cpus.append(n)
                case rlhf.Sync():
                    syncs.append(n)
                case rlhf.Composite():
                    for c in n.children:
                        visit(c)
                case _:
                    raise TypeError(f"not a flow node: {n!r}")
        visit(flow)
        return list(calls.values()), cpus, syncs

    def _check_library(self, c: rlhf.Call) -> None:
        libs = self.problem.libraries
        lib = libs.get(c.profile.lib)
        if lib is None:
            raise ValueError(f"call {c.name}: communication library {c.profile.lib!r} is not in the problem "
                             f"(libraries: {sorted(libs)})")
        if isinstance(lib, EngineLibrary) and lib.nccl not in libs:
            raise ValueError(f"call {c.name}: library {lib.name!r} falls back to {lib.nccl!r}, which the problem "
                             f"does not define")

    def _reshard(self, s: rlhf.Sync) -> Reshard:
        dst = self._calls.get(s.to)
        if dst is None:
            raise ValueError(f"{s.name}: the flow has no call named {s.to!r}")
        src = [i.call for i in self._calls.values() if i.call.kind == "train" and i.call.model.name == s.model.name]
        if len(src) != 1:
            raise ValueError(f"{s.name}: model {s.model.name!r} needs exactly one train call to sync from, "
                             f"the flow has {len(src)}")
        expert_tp = dst.call.profile.rules.axis_of("expert_ffn") == "tp"
        return Reshard(src[0].name, s.to, _param_layout(s.model.arch, src[0].profile.rules, expert_tp))

    # ----------------------------------------------------------- evaluate ---
    def evaluate(self, candidate: Candidate) -> Result:
        """Time, bounds, phases and memory of one candidate. An infeasible
        candidate gives feasible=False and a reason that names the call."""
        return self._run(candidate).result

    def explain(self, candidate: Candidate, max_depth: int = 5) -> str:
        """Bound-by rows for people, and the assumed constants the result uses."""
        run = self._run(candidate)
        r = run.result
        if not r.feasible:
            return f"infeasible: {r.reason}"
        lines = [f"iteration: {r.time:.3f} s (lo {r.lo:.3f} s, hi {r.hi:.3f} s)", "",
                 render(run.evaluator.explain(run.tree, max_depth))]
        flags = self.assumed()
        if flags:
            lines += ["", "assumed (unmeasured) values:"] + [f"  {f}" for f in flags]
        return "\n".join(lines)

    def assumed(self) -> list[str]:
        """The values the result uses that have no trustworthy measurement."""
        out = []
        for name in sorted(set(self.topology.gpu_class)):
            g = self.problem.gpus[name]
            out += [f"{g.name}: {f}" for f in g.flags]
        for name, info in self._calls.items():
            p = info.call.profile
            if isinstance(p, EngineProfile):
                out += [f"call {name} ({p.name}): {f}" for f in p.flags]
            elif isinstance(p, TrainProfile) and not p.reserve_measured:
                out.append(f"call {name} ({p.name}): memory reserve {p.reserve / GiB:.1f} GiB")
        return out

    def _run(self, candidate: Candidate) -> _Run:
        self.physics.trim()
        reason = self._check_plans(candidate)
        if reason:
            return _Run(Result(False, reason))
        G = self.topology.n_gpus
        placements, mems, uses = {}, {}, []
        for name, info in self._calls.items():
            plan = candidate.plan(name)
            try:
                if plan.micro_batch < 1:
                    raise ValueError(f"micro_batch must be >= 1, got {plan.micro_batch}")
                bad = [d for d in plan.devices if not 0 <= d < G]
                if bad:
                    raise ValueError(f"device {bad[0]} is not in the cluster ({G} GPUs)")
                pl = map_devices(plan.devices, plan.degrees, plan.order)
                mem = self._memory(info, plan, pl)
            except ValueError as e:
                return _Run(Result(False, _named(name, e)))
            placements[name] = pl
            mems[name] = mem
            uses.append((name, info.call.model.name, pl, mem))
        kv = {n: m.kv_capacity for n, m in mems.items() if self._calls[n].call.kind == "generate"}
        mres = check(uses, self.capacity)
        memory = dict(memory_peak=mres.peak, memory_headroom=mres.headroom, kv_capacity=kv)
        if not mres.feasible:
            return _Run(Result(False, mres.reason, **memory))
        for n in kv:
            reason = kv_problem(n, mems[n])
            if reason:
                return _Run(Result(False, reason, **memory))

        programs, used = {}, set()
        for name, info in self._calls.items():
            try:
                programs[name], hs = self._program(info, candidate.plan(name), mems[name])
            except ValueError as e:
                return _Run(Result(False, _named(name, e), **memory))
            used |= hs
        nodes: list = []
        tree = self._lower_flow(self.problem.flow, programs, nodes)
        binder = Binder(placements, host_ids=lambda devs: self.topology.rid("host", devs))
        try:
            for h in used | {n.pairs for n in self._leaf.values() if isinstance(n, Transfer)}:
                binder[h]
        except (KeyError, ValueError) as e:
            return _Run(Result(False, e.args[0] if e.args else str(e), **memory))

        ev = Evaluator(self.physics, binder)
        s = ev(tree)
        phases: dict = {}
        for label, node in nodes:
            key, k = label, 2
            while key in phases:
                key, k = f"{label} ({k})", k + 1
            phases[key] = float(ev.summary(node).lo)
        return _Run(Result(True, "", float(s.lo), float(s.lo), float(s.hi), phases, **memory), tree, ev)

    def _check_plans(self, candidate: Candidate) -> str:
        names = [n for n, _ in candidate.plans]
        extra = [n for n in names if n not in self._calls]
        if extra:
            return f"the candidate plans call {extra[0]!r}, which the flow does not have"
        if len(set(names)) != len(names):
            return "the candidate plans a call twice"
        for name in self._calls:
            if candidate.plan(name) is None:
                return f"call {name} has no placement"
        return ""

    def _gpu(self, devices: np.ndarray) -> GpuProfile:
        """The GPU profile that limits a call's memory: the smallest HBM among its devices."""
        classes = {self.topology.gpu_class[d] for d in devices.tolist()}
        return min((self.problem.gpus[c] for c in classes), key=lambda g: g.hbm_bytes)

    def _memory(self, info: _Info, plan, pl):
        c = info.call
        if c.kind == "generate":
            gpu = self._gpu(pl.devices)
            key = (c.name, plan.degrees, gpu.name)
            build = lambda: generate_memory(c.model.arch, plan.degrees, c.profile, gpu, c.prompt, c.response.cap)
        else:
            key = (c.name, plan.degrees, plan.micro_batch)
            build = lambda: train_memory(c.model.arch, plan.degrees, c.profile, info.step, plan.micro_batch,
                                         train=c.kind == "train")
        return _cached(self._memories, key, build)

    def _program(self, info: _Info, plan, mem):
        """(the call's program, the handles it declares), cached by what the lowering reads."""
        c = info.call
        if c.kind == "generate":
            key = (c.name, plan.degrees, mem.kv_capacity)
            lower = lambda: lower_generate(c.name, c.model.arch, plan.degrees, c.profile, c.prompts, c.samples,
                                           c.prompt, c.response, mem.kv_capacity)
        else:
            key = (c.name, plan.degrees, plan.micro_batch)
            lower = lambda: lower_call(c.name, c.model.arch, plan.degrees, c.profile, info.batch, plan.micro_batch,
                                       train=c.kind == "train", minibatches=c.minibatches)

        def build():
            tree = lower()
            return tree, frozenset(handles(tree))
        return _cached(self._programs, key, build)

    def _lower_flow(self, node, programs: dict, nodes: list):
        """The core tree of a flow node; `nodes` gets (name, tree node) in flow order."""
        at = len(nodes)
        nodes.append(None)
        match node:
            case rlhf.Call():
                tree = programs[node.name]
            case rlhf.Cpu() | rlhf.Sync():
                tree = self._leaf[id(node)]
            case rlhf.Composite():
                kids = tuple(self._lower_flow(c, programs, nodes) for c in node.children)
                tree = Seq(kids, name=node.name) if node.kind == "seq" else Par(kids, name=node.name)
        nodes[at] = (node.name, tree)
        return tree


def _named(call: str, e: Exception) -> str:
    msg = e.args[0] if e.args else str(e)
    return msg if msg.startswith(f"call {call}") else f"call {call}: {msg}"


def _cached(cache: dict, key, build):
    hit = cache.get(key)
    if hit is None:
        if len(cache) > _CACHE_LIMIT:
            cache.clear()
        hit = cache[key] = build()
    return hit
