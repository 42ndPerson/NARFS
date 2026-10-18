"""The optimizer interface: evaluate, explain, start points, moves, search.

The search algorithm here (simulated annealing) is a simple default. The
stable part is the interface: a Candidate goes in, a feasibility verdict or
a Result comes out.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from narfs.evaluate import Summary, evaluate, evaluate_bounds, explain
from narfs.parallel.mesh import AXES, Degrees
from narfs.rlhf import Candidate, CallPlan, Problem, build


@dataclass(frozen=True)
class Result:
    time: float          # the ranking value (lo or hi, as the Simulator chooses)
    lo: float            # fluid model
    hi: float            # no overlap
    summary: Summary


class Simulator:
    """Evaluate candidates for one problem.

    rank_on="lo" ranks on the fluid model; "hi" ranks on the upper bound,
    which is safer when an optimistic error would mislead the search.
    """

    def __init__(self, problem: Problem, rank_on: str = "lo"):
        if rank_on not in ("lo", "hi"):
            raise ValueError("rank_on must be 'lo' or 'hi'")
        self.problem = problem
        self.rank_on = rank_on

    def feasible(self, cand: Candidate) -> tuple[bool, str]:
        try:
            build(self.problem, cand)
            return True, ""
        except ValueError as e:
            return False, str(e)

    def evaluate(self, cand: Candidate) -> Result | None:
        """None if the candidate is not feasible."""
        try:
            tree, ctx = build(self.problem, cand)
        except ValueError:
            return None
        if self.rank_on == "lo":
            s = evaluate(tree, ctx)
            return Result(s.t, s.t, float("nan"), s)
        b = evaluate_bounds(tree, ctx)
        return Result(b.hi, b.lo, b.hi, b.summary)

    def bounds(self, cand: Candidate) -> Result:
        tree, ctx = build(self.problem, cand)
        b = evaluate_bounds(tree, ctx)
        return Result(b.lo if self.rank_on == "lo" else b.hi, b.lo, b.hi, b.summary)

    def explain(self, cand: Candidate) -> list[tuple]:
        """Per-block rows (type, name, time, hot resource) for a chosen plan."""
        tree, ctx = build(self.problem, cand)
        return explain(tree, ctx)


# --------------------------------------------------------- start points ---

def group_plan(problem: Problem, groups: list[tuple[int, list[str]]],
               degrees: dict[str, Degrees], micro_batch: dict[str, int] | None = None) -> Candidate:
    """Contiguous device groups: `groups` is a list of (device count, [calls]).
    Calls in one group share (colocate on) the same devices."""
    micro_batch = micro_batch or {}
    plans, start = [], 0
    for count, calls in groups:
        devices = tuple(range(start, start + count))
        start += count
        for name in calls:
            plans.append((name, CallPlan(devices, degrees[name], AXES, micro_batch.get(name, 1))))
    if start > problem.cluster.n_devices:
        raise ValueError(f"groups use {start} devices, the cluster has {problem.cluster.n_devices}")
    return Candidate(tuple(plans))


# ----------------------------------------------------------------- moves ---

ORDERS = (AXES, ("pp", "dp", "ep", "cp", "tp"), ("dp", "pp", "ep", "tp", "cp"), ("tp", "pp", "ep", "cp", "dp"))


def _with_degrees(plan: CallPlan, tp: int | None = None, pp: int | None = None) -> CallPlan | None:
    d = plan.degrees
    tp = tp or d.tp
    pp = pp or d.pp
    rest = d.ep * d.cp * tp * pp
    if len(plan.devices) % rest:
        return None
    return replace(plan, degrees=Degrees(dp=len(plan.devices) // rest, pp=pp, ep=d.ep, cp=d.cp, tp=tp))


def _groups(cand: Candidate) -> dict[tuple, list[str]]:
    out: dict = {}
    for name, plan in cand.plans:
        out.setdefault(tuple(sorted(plan.devices)), []).append(name)
    return out


def propose(problem: Problem, cand: Candidate, rng: random.Random) -> Candidate | None:
    """One random move. Returns None if the move does not apply."""
    names = [n for n, _ in cand.plans]
    move = rng.choice(("tp", "pp", "micro_batch", "order", "resize", "swap"))
    name = rng.choice(names)
    plan = cand.plan(name)
    if move == "tp":
        new = _with_degrees(plan, tp=rng.choice((1, 2, 4, 8)))
    elif move == "pp":
        if problem.call(name).kind == "generate":
            return None
        new = _with_degrees(plan, pp=rng.choice((1, 2, 4, 8)))
    elif move == "micro_batch":
        new = replace(plan, micro_batch=max(1, plan.micro_batch * rng.choice((2, 0.5))))
        new = replace(new, micro_batch=int(new.micro_batch))
    elif move == "order":
        new = replace(plan, order=rng.choice(ORDERS))
    else:
        return _move_devices(problem, cand, rng, swap=(move == "swap"))
    return None if new is None else cand.with_plan(name, new)


def _move_devices(problem: Problem, cand: Candidate, rng: random.Random, swap: bool) -> Candidate | None:
    """Move one node of devices between two device groups (resize), or
    exchange random devices between them (swap, which allows irregular
    layouts). Every call of a group follows its group."""
    groups = list(_groups(cand).items())
    if len(groups) < 2:
        return None
    (ga, ca), (gb, cb) = rng.sample(groups, 2)
    ga, gb = list(ga), list(gb)
    if swap:
        k = rng.choice((1, 2, 4, 8))
        if k > min(len(ga), len(gb)):
            return None
        ia, ib = rng.sample(range(len(ga)), k), rng.sample(range(len(gb)), k)
        for i, j in zip(ia, ib):
            ga[i], gb[j] = gb[j], ga[i]
    else:
        step = problem.cluster.gpus_per_node
        if len(ga) <= step:
            return None
        moved, ga = ga[-step:], ga[:-step]
        gb = gb + moved
    out = cand
    for devices, calls in ((ga, ca), (gb, cb)):
        for name in calls:
            plan = replace(out.plan(name), devices=tuple(devices))
            plan = _with_degrees(plan)
            if plan is None:
                return None
            out = out.with_plan(name, plan)
    return out


# ---------------------------------------------------------------- search ---

@dataclass(frozen=True)
class SearchResult:
    best: Candidate
    best_time: float
    evaluated: int
    feasible: int


def anneal(sim: Simulator, start: Candidate, steps: int = 500, seed: int = 0,
           t0: float = 0.05, t1: float = 0.001) -> SearchResult:
    """Simulated annealing on the ranking value. Temperatures are relative
    to the current time (0.05 = 5%)."""
    rng = random.Random(seed)
    cur = sim.evaluate(start)
    if cur is None:
        raise ValueError("the start candidate is not feasible: " + sim.feasible(start)[1])
    cur_c, best_c, best_t = start, start, cur.time
    cur_t, evaluated, feasible = cur.time, 1, 1
    for i in range(steps):
        temp = t0 * (t1 / t0) ** (i / max(1, steps - 1))
        nxt = propose(sim.problem, cur_c, rng)
        if nxt is None:
            continue
        r = sim.evaluate(nxt)
        evaluated += 1
        if r is None:
            continue
        feasible += 1
        if r.time < cur_t or rng.random() < math.exp(-(r.time - cur_t) / (temp * cur_t)):
            cur_c, cur_t = nxt, r.time
            if cur_t < best_t:
                best_c, best_t = cur_c, cur_t
    return SearchResult(best_c, best_t, evaluated, feasible)
