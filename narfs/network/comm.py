"""L0: the communication library model, the cost function of a Collective leaf.

A collective is costed by a model of the library that runs it. The model
picks the path the library really runs for the operation, group and message
size, and costs it: the bytes per resource from the path's pattern
(narfs.network.kernels), plus one per-call latency. It builds no evaluator
nodes. The cost depends on what the operation is (kind, size, group, library),
never on why it exists.

NCCL: Ring with the LL, LL128 and Simple protocols, and NVLS (in-switch
reduction) for all-reduce inside one NVLS-capable domain. Latencies follow
NCCL's own cost model (src/tuning/ in NCCL 2.32.3): base + steps x hardware
latency. The library takes the option with the lowest time, as NCCL's tuner
does; here the bandwidth term comes from our own resource model.

An engine (for example vLLM) states its all-reduce paths per group size as an
ordered table of size limits; other operations, and sizes past the table, go
to its NCCL library.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from narfs.network import kernels as K
from narfs.network.topology import Topology

PROTOCOLS = ("LL", "LL128", "Simple")
PAYLOAD = {"LL": 0.5, "LL128": 120.0 / 128.0, "Simple": 1.0}      # exact by protocol design
ENGINE_PATHS = ("one_shot", "two_shot", "nvls", "nccl")


@dataclass(frozen=True)
class NcclProfile:
    """NCCL's latency tables (microseconds) and protocol limits for one version."""
    version: str
    base: dict                         # (algo, proto) -> us per call
    hw: dict                           # (hw, algo, proto) -> us per step; hw in NVLINK, PCI, NET
    net_overhead: float = 1.0          # us per step across nodes (1 on Intel hosts, 2 on AMD)
    plateau: float = 1.4               # multi-node Ring/Simple all-reduce latency factor (Ampere and later)
    ll_max_bw: tuple = (141e9, 39e9, 20.4e9)   # LL bus bandwidth cap for 1, 2 and >= 3 nodes
    source: str = ""


def load_nccl(path, cpu: str = "intel") -> NcclProfile:
    """An NcclProfile from a data file (see narfs/data/nccl-*.json)."""
    import json
    from pathlib import Path
    d = json.loads(Path(path).read_text())
    base = {(algo, p): float(v) for algo, row in d["base"].items() for p, v in zip(PROTOCOLS, row)}
    hw = {(h, algo, p): float(v) for h, table in d["hw"].items() for algo, row in table.items()
          for p, v in zip(PROTOCOLS, row)}
    ll = d["ll_max_bw_GBps"]
    return NcclProfile(version=d["version"], base=base, hw=hw, net_overhead=float(d["net_overhead"][cpu]),
                       plateau=float(d["plateau"]["ampere_and_later"]),
                       ll_max_bw=(ll["hopper_1_node"] * 1e9, ll[f"{cpu}_2_nodes"] * 1e9, ll[f"{cpu}_4_nodes"] * 1e9),
                       source=d["source"])


@dataclass(frozen=True)
class EnginePath:
    """One row of an engine's all-reduce table: `path` serves sizes below
    `max_bytes` (or up to it when `inclusive`)."""
    max_bytes: float
    path: str
    latency: float = 0.0               # seconds per call
    inclusive: bool = False

    def __post_init__(self) -> None:
        if self.path not in ENGINE_PATHS:
            raise ValueError(f"unknown engine path {self.path!r}; expected one of {ENGINE_PATHS}")


@dataclass(frozen=True)
class EngineLibrary:
    """An engine's own all-reduce paths, per group size, with NCCL behind them."""
    name: str
    paths: dict                        # group size -> tuple of EnginePath, in the order the engine tries them
    nccl: str = "nccl"
    source: str = ""


@dataclass(frozen=True)
class Choice:
    """What the library runs, for explain."""
    path: str
    proto: str
    latency: float
    seconds: float

    def __str__(self) -> str:
        return f"{self.path}/{self.proto}: {self.latency * 1e6:.1f} us + {(self.seconds - self.latency) * 1e6:.1f} us"


@dataclass
class _Group:
    """Facts about one bound group array, computed once."""
    arr: np.ndarray
    n: int
    domains: int                       # most scale-up domains in one instance
    ppn: int                           # most members of one instance in one domain
    single: bool                       # every instance inside one scale-up domain
    nvls: bool                         # ... and every one of those domains supports NVLS
    key: bytes
    units: dict = field(default_factory=dict)


class CommModel:
    def __init__(self, topo: Topology, units: K.UnitTable, libraries: dict):
        self.topo = topo
        self.units = units
        self.libraries = dict(libraries)
        self._groups: dict = {}

    def clear(self) -> None:
        """Forget the group facts (after the unit table is trimmed)."""
        self._groups.clear()

    def group(self, bound: np.ndarray) -> _Group:
        hit = self._groups.get(id(bound))
        if hit is not None and hit[0] is bound:
            return hit[1]
        arr = np.asarray(bound, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError(f"a communication group binds to [instances, members], got shape {arr.shape}")
        dom = self.topo.scaleup[arr]
        I, n = arr.shape
        inst = np.repeat(np.arange(I), n)
        pairs = np.unique(inst * self.topo.n_scaleup + dom.ravel(), return_counts=True)
        per_inst = np.bincount(pairs[0] // self.topo.n_scaleup, minlength=I)
        domains = int(per_inst.max())
        single = domains == 1
        g = _Group(arr, n, domains, int(pairs[1].max()), single,
                   bool(single and self.topo.nvls[dom].all()), arr.tobytes() + bytes(str(arr.shape), "ascii"))
        if len(self._groups) > 4096:
            self._groups.clear()
        self._groups[id(bound)] = (bound, g)              # keep `bound` alive so its id stays unique
        return g

    def _unit(self, g: _Group, pattern: str, factor: float = 1.0) -> int:
        key = (pattern, factor)
        uid = g.units.get(key)
        if uid is None:
            topo, arr = self.topo, g.arr
            build = {
                "ring": lambda: K.ring(topo, arr, factor),
                "one_shot": lambda: K.one_shot(topo, arr),
                "two_shot": lambda: K.two_shot(topo, arr),
                "nvls": lambda: K.nvls(topo, arr),
                "spread": lambda: K.spread(topo, arr),
            }[pattern]
            uid = g.units[key] = self.units.intern((pattern, factor, g.key), build)
        return uid

    def _peak(self, uid: int) -> float:
        val = self.units.unit(uid)[1]
        return float(val.max()) if val.size else 0.0

    # --------------------------------------------------------------- cost ---
    def cost(self, kind: str, arr: np.ndarray, nbytes: float, lib: str):
        """(seconds, {unit id: amount}, Choice) for one collective on every
        instance of `arr` at once."""
        g = self.group(arr)
        if g.n == 1 or nbytes <= 0:
            return 0.0, {}, Choice("local", "-", 0.0, 0.0)
        library = self.libraries[lib]
        if isinstance(library, EngineLibrary):
            if kind == "all_reduce":
                for p in library.paths.get(g.n, ()):
                    fits = nbytes <= p.max_bytes if p.inclusive else nbytes < p.max_bytes
                    if not fits:
                        continue
                    if p.path == "nccl":
                        break
                    if not g.single or (p.path == "nvls" and not g.nvls):
                        continue
                    uid = self._unit(g, p.path)
                    t = p.latency + nbytes * self._peak(uid)
                    return t, {uid: nbytes}, Choice(f"{library.name} {p.path}", "-", p.latency, t)
            library = self.libraries[library.nccl]
        return self._nccl(kind, g, nbytes, library)

    def _nccl(self, kind: str, g: _Group, S: float, nc: NcclProfile):
        n, k = g.n, g.domains
        if kind == "all_to_all":
            uid = self._unit(g, "spread")
            hw = "NVLINK" if k == 1 else "NET"
            lat = (nc.base[("Ring", "Simple")] + nc.hw[(hw, "Ring", "Simple")]) * 1e-6
            t = lat + S * self._peak(uid)
            return t, {uid: S}, Choice("p2p all-to-all", "Simple", lat, t)
        ar = kind == "all_reduce"
        factor = 2.0 * (n - 1) / n if ar else (n - 1) / n
        steps = 2 * (n - 1) if ar else n - 1
        inter_steps = 0 if k == 1 else (2 * (k - 1) if ar else k - 1)
        uid = self._unit(g, "ring", factor)
        peak = self._peak(uid)
        best = None
        for proto in PROTOCOLS:
            lat = self._ring_latency(nc, proto, steps, inter_steps, k, g.ppn)
            if ar and k > 1 and proto == "Simple":
                lat *= nc.plateau
            amount = S / PAYLOAD[proto]
            t = lat + amount * peak
            if proto == "LL":
                t = max(t, lat + factor * S / nc.ll_max_bw[min(k, 3) - 1])
            if best is None or t < best[0]:
                best = (t, {uid: amount}, Choice("ring", proto, lat, t))
        if ar and g.nvls:
            uid = self._unit(g, "nvls")
            lat = (nc.base[("NVLS", "Simple")] + nc.hw[("NVLINK", "NVLS", "Simple")]) * 1e-6
            t = lat + S * self._peak(uid)
            if t < best[0]:
                best = (t, {uid: S}, Choice("nvls", "Simple", lat, t))
        return best

    @staticmethod
    def _ring_latency(nc: NcclProfile, proto: str, steps: int, inter_steps: int, k: int, ppn: int) -> float:
        intra = nc.hw[("NVLINK", "Ring", proto)]
        if k == 1:
            inter = intra
        else:
            intra = max(intra, nc.net_overhead * (3.0 if proto == "Simple" else 1.0))
            inter = nc.hw[("NET", "Tree" if ppn == 1 else "Ring", proto)]
        return (nc.base[("Ring", proto)] + (steps - inter_steps) * intra + inter_steps * inter) * 1e-6

    def flows(self, kind: str, arr: np.ndarray, path: str):
        """Point-to-point flows of a path per byte of S (tests, validation tier)."""
        g = self.group(arr)
        n = g.n
        if path == "ring":
            factor = 2.0 * (n - 1) / n if kind == "all_reduce" else (n - 1) / n
            return K.ring_flows(self.topo, g.arr, factor)
        return {"one_shot": K.one_shot_flows, "two_shot": K.two_shot_flows, "spread": K.spread_flows}[path](self.topo, g.arr)
