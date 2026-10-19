"""L0 facade: the leaf costs the core evaluator asks for.

`Physics` implements the evaluator's `Leaves` protocol. It reads bound arrays
from the binding and hands them to the L0 models:

    Compute     the compute cost model, on the scope's device class
    Collective  the communication library model
    Transfer    the router: routes, then one bincount
    Work        a resource's capacity (a CPU pool, host threads), or an L0
                work item such as one engine host step

Demand vectors are interned in a UnitTable by content, so a route and a
bincount are paid once per distinct bound array, across candidates.

Bound values, by handle kind:
    comm group   int array [instances, members]
    pairs        (src, dst, weight) arrays
    device set   int array of device ids
    resources    int array of resource ids, or the name of a CPU pool
"""
from __future__ import annotations

import numpy as np

from narfs.network.comm import CommModel
from narfs.network.kernels import UnitTable
from narfs.network.topology import Topology


class Physics:
    def __init__(self, topo: Topology, libraries: dict, compute=None, host=None):
        self.topo = topo
        self.units = UnitTable(topo.cap)
        self.comm = CommModel(topo, self.units, libraries)
        self.compute_model = compute
        self.host_model = host
        self.n_resources = topo.n_resources
        self._last: dict = {}

    def trim(self) -> None:
        """Bound the unit table between evaluations; the communication model's
        group facts hold unit ids, so they go with it."""
        if self.units.trim():
            self.comm.clear()
        self._last.clear()

    def _bound(self, binding, handle):
        try:
            return binding[handle]
        except KeyError:
            raise KeyError(f"handle {handle!r} is not bound; the binding must supply every handle "
                           f"the program declares") from None

    # ------------------------------------------------------------ leaves ---
    def compute(self, work, device_class) -> float:
        return self.compute_model.time(work, device_class)

    def collective(self, node, binding):
        arr = self._bound(binding, node.comm)
        t, d, choice = self.comm.cost(node.kind, arr, node.bytes, node.lib)
        self._last[id(node)] = choice
        return t, d

    def transfer(self, node, binding):
        src, dst, w = self._bound(binding, node.pairs)
        if node.reverse:
            src, dst = dst, src
        if node.bytes == 0 or len(src) == 0:
            return 0.0, {}
        key = ("pairs", np.asarray(src).tobytes(), np.asarray(dst).tobytes(), np.asarray(w, dtype=float).tobytes())
        uid = self.units.intern(key, lambda: self.topo.pair_demand(src, dst, w))
        val = self.units.unit(uid)[1]
        if val.size == 0:
            return 0.0, {}
        return self.topo.route_latency(src, dst) + node.bytes * float(val.max()), {uid: node.bytes}

    def work(self, node, binding):
        target = self._bound(binding, node.resource)
        if isinstance(target, str):
            ids = np.array([self.topo.pool(target)])
        else:
            ids = np.asarray(target, dtype=np.int64)
        amount = node.amount
        if not isinstance(amount, (int, float)):
            amount = self.host_model.seconds(amount)
        if amount == 0 or ids.size == 0:
            return 0.0, {}
        uid = self.units.intern(("work", ids.tobytes()), lambda: (ids, np.ones(ids.size)))
        return float(amount) * float(self.units.unit(uid)[1].max()), {uid: float(amount)}

    def busy(self, devset, binding) -> int:
        devs = np.asarray(self._bound(binding, devset), dtype=np.int64)
        ids = self.topo.rid("gpu", devs)
        return self.units.intern(("busy", ids.tobytes()), lambda: (ids, np.ones(ids.size)))

    def device_class(self, devset, binding):
        devs = np.asarray(self._bound(binding, devset), dtype=np.int64)
        classes = sorted({self.topo.gpu_class[d] for d in devs.tolist()})
        if len(classes) == 1 or self.compute_model is None:
            return classes[0]
        return self.compute_model.slowest(classes)

    def unit(self, uid: int):
        return self.units.unit(uid)

    def resource_name(self, r: int) -> str:
        return self.topo.resource_name(r)

    def describe(self, node, binding, device_class) -> str:
        from narfs.blocks import Collective, Compute, Transfer, Work
        if isinstance(node, Collective):
            choice = self._last.get(id(node))
            return f"{node.kind} {choice}" if choice else node.kind
        if isinstance(node, Compute):
            return self.compute_model.describe(node.work, device_class)
        if isinstance(node, Transfer):
            return "point-to-point sends"
        if isinstance(node, Work):
            return "resource work"
        return ""
