"""A toy L0 for core tests: numbers that are easy to check by hand.

Resources have names and capacities. A `Compute` leaf's work is its time in
seconds. A handle binds to a demand per unit amount, as {resource: units},
or for a device set to a list of GPU resource names (capacity 1 s/s).
"""
from __future__ import annotations

import numpy as np


class ToyLeaves:
    def __init__(self, caps: dict[str, float], latency: float = 0.0):
        self.names = list(caps)
        self.cap = np.array([float(caps[n]) for n in self.names])
        self.n_resources = len(self.names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.latency = latency
        self.compute_calls = 0
        self._units: list = []
        self._ids: dict = {}

    def _uid(self, key, demand: dict) -> int:
        uid = self._ids.get(key)
        if uid is None:
            idx = np.array([self.index[n] for n in demand], dtype=np.int64)
            val = np.array([demand[n] / self.cap[self.index[n]] for n in demand])
            uid = self._ids[key] = len(self._units)
            self._units.append((idx, val))
        return uid

    def _leaf(self, key, demand: dict, amount: float):
        uid = self._uid(key, demand)
        idx, val = self._units[uid]
        t = amount * float(val.max()) if val.size else 0.0
        return t + (self.latency if amount else 0.0), {uid: amount}

    def compute(self, work, device_class) -> float:
        self.compute_calls += 1
        return float(work)

    def collective(self, node, binding):
        return self._leaf(("coll", node.comm), binding[node.comm], node.bytes)

    def transfer(self, node, binding):
        return self._leaf(("xfer", node.pairs, node.reverse), binding[node.pairs], node.bytes)

    def work(self, node, binding):
        uid = self._uid(("work", node.resource), {binding[node.resource]: 1.0})
        idx, val = self._units[uid]
        return float(node.amount) * float(val.max()), {uid: float(node.amount)}

    def busy(self, devset, binding) -> int:
        return self._uid(("busy", devset), {g: 1.0 for g in binding[devset]})

    def device_class(self, devset, binding):
        return "unit"

    def unit(self, uid):
        return self._units[uid]

    def resource_name(self, r: int) -> str:
        return self.names[r]

    def describe(self, node, binding, device_class) -> str:
        return "toy leaf"
