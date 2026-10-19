"""L1: the core evaluator.

The core is a small algebra over time and resource demand. Every node
evaluates to a Summary with two times and their demands:

    lo   the fluid rule. Seq adds, Repeat scales, and Par takes
         max(max of t, max over r of the summed D[r] / C[r]). It is a lower
         bound: it ignores when bytes move inside a region and assumes
         perfect sharing.
    hi   no overlap inside one OnDevices scope (Par sums), and across scopes
         t = max of t + max over r of (sum of D[r] - largest D[r]) / C[r].
         It is exact for children on disjoint resources and for children that
         fully share one resource.

Demand is kept as {unit id: coefficient}. A unit is one demand vector over
the global resource index, in seconds of use per unit of a leaf's amount
(bytes / capacity on links, seconds on GPU and host resources). The L0 models
register each unit once per bound array, so Seq and Repeat only add and scale
coefficients, and vectors are built only at Par nodes.

Leaf costs come from the L0 models through the `Leaves` protocol, and handles
are resolved by the binding (L2). The core reads neither.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Protocol

import numpy as np

from narfs.blocks import Collective, Compute, OnDevices, Par, Repeat, Seq, Transfer, Work, label

_EMPTY: Mapping[int, float] = {}


@dataclass(frozen=True)
class Summary:
    lo: float
    hi: float
    d_lo: Mapping[int, float]
    d_hi: Mapping[int, float]


ZERO = Summary(0.0, 0.0, _EMPTY, _EMPTY)


class Leaves(Protocol):
    """What the core needs from L0. Units have unique resource ids."""

    n_resources: int

    def compute(self, work: Any, device_class: Hashable) -> float: ...
    def collective(self, node: Collective, binding: Any) -> tuple[float, dict]: ...
    def transfer(self, node: Transfer, binding: Any) -> tuple[float, dict]: ...
    def work(self, node: Work, binding: Any) -> tuple[float, dict]: ...
    def busy(self, devset: Any, binding: Any) -> int: ...
    def device_class(self, devset: Any, binding: Any) -> Hashable: ...
    def unit(self, uid: int) -> tuple[np.ndarray, np.ndarray]: ...
    def resource_name(self, r: int) -> str: ...
    def describe(self, node: Any, binding: Any, device_class: Hashable) -> str: ...


def _merge(ds: list) -> Mapping[int, float]:
    ds = [d for d in ds if d]
    if not ds:
        return _EMPTY
    if len(ds) == 1:
        return ds[0]
    out: dict = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return out


def _scale(d: Mapping[int, float], n: float) -> Mapping[int, float]:
    if not d or n == 1:
        return d
    return {k: v * n for k, v in d.items()}


@dataclass(frozen=True)
class Row:
    """One line of `explain`."""
    depth: int
    label: str
    lo: float
    hi: float
    bound_by: str      # the term that set the time: a child, a resource, or the leaf's cost law
    hot: str           # the most loaded resource
    util: float        # its seconds of use / lo


class Evaluator:
    """Evaluates trees for one binding. Nodes used twice are computed once
    (memo by node identity, device class and scope)."""

    def __init__(self, leaves: Leaves, binding: Any = None):
        self.leaves = leaves
        self.binding = binding
        self._memo: dict = {}
        self._busy: set[int] = set()
        self._unit_max: dict[int, float] = {}
        self._dev: Hashable = None
        self._scope: Any = None

    # ----------------------------------------------------------- vectors ---
    def vector(self, d: Mapping[int, float]) -> np.ndarray | None:
        """Seconds of use per resource for a demand."""
        if not d:
            return None
        ids, ws = [], []
        for uid, c in d.items():
            i, v = self.leaves.unit(uid)
            ids.append(i)
            ws.append(v * c)
        return np.bincount(np.concatenate(ids), weights=np.concatenate(ws), minlength=self.leaves.n_resources)

    def peak(self, d: Mapping[int, float]) -> float:
        """max over r of the demand's seconds of use."""
        if not d:
            return 0.0
        if len(d) == 1:
            (uid, c), = d.items()
            m = self._unit_max.get(uid)
            if m is None:
                v = self.leaves.unit(uid)[1]
                m = self._unit_max[uid] = float(v.max()) if v.size else 0.0
            return c * m
        return float(self.vector(d).max())

    # -------------------------------------------------------- evaluation ---
    def __call__(self, node) -> Summary:
        return self.summary(node)

    def summary(self, node) -> Summary:
        key = (id(node), self._dev, id(self._scope))
        s = self._memo.get(key)
        if s is None:
            s = self._eval(node)
            self._memo[key] = s
        return s

    def _eval(self, node) -> Summary:
        lv, b = self.leaves, self.binding
        match node:
            case Compute():
                if self._dev is None:
                    raise ValueError(f"Compute leaf {label(node)!r} is outside any OnDevices scope")
                t = lv.compute(node.work, self._dev)
                return Summary(t, t, _EMPTY, _EMPTY)
            case Collective():
                t, d = lv.collective(node, b)
                return Summary(t, t, d, d)
            case Transfer():
                t, d = lv.transfer(node, b)
                return Summary(t, t, d, d)
            case Work():
                t, d = lv.work(node, b)
                return Summary(t, t, d, d)
            case Seq():
                parts = [self.summary(c) for c in node.children]
                return Summary(sum(p.lo for p in parts), sum(p.hi for p in parts),
                               _merge([p.d_lo for p in parts]), _merge([p.d_hi for p in parts]))
            case Repeat():
                if node.n == 0:
                    return ZERO
                s = self.summary(node.body)
                return Summary(node.n * s.lo, node.n * s.hi, _scale(s.d_lo, node.n), _scale(s.d_hi, node.n))
            case Par():
                return self._par(node)
            case OnDevices():
                return self._on_devices(node)
        raise TypeError(f"unknown node type {type(node).__name__}")

    def _par(self, node: Par) -> Summary:
        parts = [self.summary(c) for c in node.children]
        if not parts:
            return ZERO
        d_lo = _merge([p.d_lo for p in parts])
        d_hi = _merge([p.d_hi for p in parts])
        lo = max(max(p.lo for p in parts), self.peak(d_lo))
        groups: dict = {}                      # scope -> (sum of hi, demands)
        for c, p in zip(node.children, parts):
            scope = c.devset if isinstance(c, OnDevices) else self._scope
            t, ds = groups.get(scope, (0.0, []))
            groups[scope] = (t + p.hi, ds + [p.d_hi])
        if len(groups) == 1:
            hi = next(iter(groups.values()))[0]
        else:
            vecs = [self.vector(_merge(ds)) for _, ds in groups.values()]
            vecs = [v for v in vecs if v is not None]
            leftover = 0.0
            if len(vecs) > 1:
                stack = np.vstack(vecs)
                leftover = float((stack.sum(axis=0) - stack.max(axis=0)).max())
            hi = max(t for t, _ in groups.values()) + leftover
        return Summary(lo, max(hi, lo), d_lo, d_hi)

    def _on_devices(self, node: OnDevices) -> Summary:
        outer = self._dev, self._scope
        self._dev = self.leaves.device_class(node.devset, self.binding)
        self._scope = node.devset
        try:
            s = self.summary(node.body)
        finally:
            self._dev, self._scope = outer
        busy = self.leaves.busy(node.devset, self.binding)
        self._busy.add(busy)
        # the devices are busy for the whole scope; inner scopes' busy time is inside it
        d_lo = {k: v for k, v in s.d_lo.items() if k not in self._busy}
        d_hi = {k: v for k, v in s.d_hi.items() if k not in self._busy}
        d_lo[busy] = s.lo
        d_hi[busy] = s.hi
        return Summary(s.lo, s.hi, d_lo, d_hi)

    # ----------------------------------------------------------- explain ---
    def hot(self, d: Mapping[int, float]) -> tuple[str, float]:
        """The most loaded resource of a demand and its seconds of use."""
        v = self.vector(d)
        if v is None or not v.any():
            return "", 0.0
        r = int(np.argmax(v))
        return self.leaves.resource_name(r), float(v[r])

    def explain(self, node, max_depth: int = 6) -> list[Row]:
        """Rows for `node` and its descendants down to `max_depth`, parents first."""
        rows: list[Row] = []
        self._explain(node, 0, max_depth, rows)
        return rows

    def _explain(self, node, depth: int, max_depth: int, rows: list) -> None:
        s = self.summary(node)
        hot, used = self.hot(s.d_lo)
        row = len(rows)
        rows.append(None)                      # placeholder, filled after the children
        children: list = []
        bound = ""
        match node:
            case Compute() | Collective() | Transfer() | Work():
                bound = self.leaves.describe(node, self.binding, self._dev)
            case Seq():
                children = list(node.children)
                if children:
                    parts = [self.summary(c) for c in children]
                    k = int(np.argmax([p.lo for p in parts]))
                    share = parts[k].lo / s.lo if s.lo > 0 else 0.0
                    bound = f"largest part: {label(children[k])} ({share:.0%})"
            case Repeat():
                children = [node.body]
                bound = f"{node.n:g} x {label(node.body)}"
            case Par():
                children = list(node.children)
                parts = [self.summary(c) for c in children]
                k = int(np.argmax([p.lo for p in parts])) if parts else 0
                if parts and parts[k].lo >= s.lo * (1 - 1e-9):
                    bound = f"branch: {label(children[k])}"
                else:
                    bound = f"shared resource: {hot}"
            case OnDevices():
                children = [node.body]
                bound = label(node.body)
        if depth < max_depth:
            if isinstance(node, OnDevices):
                outer = self._dev, self._scope
                self._dev = self.leaves.device_class(node.devset, self.binding)
                self._scope = node.devset
                try:
                    for c in children:
                        self._explain(c, depth + 1, max_depth, rows)
                finally:
                    self._dev, self._scope = outer
            else:
                for c in children:
                    self._explain(c, depth + 1, max_depth, rows)
        util = used / s.lo if s.lo > 0 else 0.0
        rows[row] = Row(depth, label(node), s.lo, s.hi, bound, hot, util)


def evaluate(node, leaves: Leaves, binding: Any = None) -> Summary:
    return Evaluator(leaves, binding)(node)


def render(rows: list[Row]) -> str:
    """Plain-text table of explain rows."""
    out = []
    for r in rows:
        name = "  " * r.depth + r.label
        hot = f"{r.hot} {r.util:.0%}" if r.hot else ""
        out.append(f"{name:<48s} {r.lo * 1e3:11.3f} ms {r.hi * 1e3:11.3f} ms  {r.bound_by:<44s} {hot}")
    return "\n".join(out)
