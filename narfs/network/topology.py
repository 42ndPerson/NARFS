"""L0: the network, as a tiered router over directed resources.

The cluster is described in three planes:

    scale-up   one switched domain per node (or per rack for NVL72); GPUs in
               one domain talk over their NVLink ports
    scale-out  tiers (rail leaf, pod, core). Each tier has a domain key per GPU
               and an up/down capacity per domain, so trees may be unbalanced.
               There is no global spine pool.
    host       PCIe per GPU, a host thread per process (keyed by the process's
               lead GPU), and named CPU pools

Resources form one flat index of directed capacities, in units per second
(bytes for links, seconds for `gpu` and `host`, workers for pools):

    nvl_tx[G] nvl_rx[G] nic_tx[G] nic_rx[G] pcie[G] gpu[G] host[G]
    up[tier][domain] down[tier][domain] ... pool[name]

Route for a pair (a, b): in the same scale-up domain, nvl_tx[a] and nvl_rx[b].
Otherwise nic_tx[a]; then, at each tier where the keys differ, up[domain(a)]
and down[domain(b)], stopping at the first tier where they match; then
nic_rx[b]. Routes are arithmetic on the key arrays; no path is stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FAMILIES = ("nvl_tx", "nvl_rx", "nic_tx", "nic_rx", "pcie", "gpu", "host")


@dataclass(frozen=True)
class Tier:
    """One scale-out tier: a domain id per GPU (0..n-1) and the capacity of
    each domain's uplink, which is also its downlink capacity (bytes/s)."""
    name: str
    key: np.ndarray = field(repr=False)
    capacity: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        key = np.asarray(self.key, dtype=np.int64)
        cap = np.asarray(self.capacity, dtype=float)
        if key.min() < 0 or key.max() >= cap.size:
            raise ValueError(f"tier {self.name}: keys must index its {cap.size} domains")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "capacity", cap)


def _per_gpu(value, n: int, what: str) -> np.ndarray:
    arr = np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()
    if (arr <= 0).any():
        raise ValueError(f"{what} must be > 0")
    return arr


class Topology:
    """The global resource index and the router. Built once per cluster."""

    def __init__(self, *, gpu_class, scaleup, nvl_rate, nic_rate, pcie_rate, tiers=(),
                 nvls=False, nvls_level=1.0, pools=None, nvl_latency=0.0, net_latency=0.0,
                 node_of=None):
        self._args = dict(gpu_class=gpu_class, scaleup=scaleup, nvl_rate=nvl_rate, nic_rate=nic_rate,
                          pcie_rate=pcie_rate, tiers=tiers, nvls=nvls, nvls_level=nvls_level, pools=pools,
                          nvl_latency=nvl_latency, net_latency=net_latency, node_of=node_of)
        self.gpu_class = tuple(gpu_class)
        G = self.n_gpus = len(self.gpu_class)
        self.scaleup = np.asarray(scaleup, dtype=np.int64)
        if self.scaleup.shape != (G,):
            raise ValueError("scaleup needs one domain id per GPU")
        self.n_scaleup = int(self.scaleup.max()) + 1
        self.node_of = self.scaleup if node_of is None else np.asarray(node_of, dtype=np.int64)
        self.tiers = tuple(tiers)
        for t in self.tiers:
            if t.key.shape != (G,):
                raise ValueError(f"tier {t.name} needs one key per GPU")
        self.nvls = np.broadcast_to(np.asarray(nvls, dtype=bool), (self.n_scaleup,)).copy()
        self.nvls_level = np.broadcast_to(np.asarray(nvls_level, dtype=float), (self.n_scaleup,)).copy()
        self.nvl_latency = float(nvl_latency)
        self.net_latency = float(net_latency)

        # offsets of the resource index
        self.off = {f: i * G for i, f in enumerate(FAMILIES)}
        n = len(FAMILIES) * G
        self.up_off, self.down_off = [], []
        for t in self.tiers:
            self.up_off.append(n)
            self.down_off.append(n + t.capacity.size)
            n += 2 * t.capacity.size
        self.pool_names = tuple((pools or {}).keys())
        self.pool_off = n
        n += len(self.pool_names)
        self.n_resources = n

        cap = np.empty(n)
        cap[self.off["nvl_tx"]:self.off["nvl_tx"] + G] = _per_gpu(nvl_rate, G, "nvl_rate")
        cap[self.off["nvl_rx"]:self.off["nvl_rx"] + G] = _per_gpu(nvl_rate, G, "nvl_rate")
        cap[self.off["nic_tx"]:self.off["nic_tx"] + G] = _per_gpu(nic_rate, G, "nic_rate")
        cap[self.off["nic_rx"]:self.off["nic_rx"] + G] = _per_gpu(nic_rate, G, "nic_rate")
        cap[self.off["pcie"]:self.off["pcie"] + G] = _per_gpu(pcie_rate, G, "pcie_rate")
        cap[self.off["gpu"]:self.off["gpu"] + 2 * G] = 1.0                      # gpu and host
        for t, u, d in zip(self.tiers, self.up_off, self.down_off):
            cap[u:u + t.capacity.size] = t.capacity
            cap[d:d + t.capacity.size] = t.capacity
        for i, name in enumerate(self.pool_names):
            cap[self.pool_off + i] = float(pools[name])
        if (cap <= 0).any():
            raise ValueError("every capacity must be > 0")
        cap.setflags(write=False)
        self.cap = cap

    def with_pools(self, pools: dict) -> "Topology":
        """The same cluster with more CPU pools (name -> workers)."""
        merged = dict(self._args["pools"] or {})
        for name, workers in pools.items():
            if name in merged and merged[name] != workers:
                raise ValueError(f"CPU pool {name!r} declared with {merged[name]} and {workers} workers")
            merged[name] = workers
        return Topology(**{**self._args, "pools": merged})

    # ------------------------------------------------------------- ids ---
    def rid(self, family: str, gpus) -> np.ndarray:
        return self.off[family] + np.asarray(gpus, dtype=np.int64)

    def pool(self, name: str) -> int:
        try:
            return self.pool_off + self.pool_names.index(name)
        except ValueError:
            raise KeyError(f"no CPU pool named {name!r}; pools: {self.pool_names}") from None

    def resource_name(self, r: int) -> str:
        G = self.n_gpus
        if r < len(FAMILIES) * G:
            f, g = divmod(r, G)
            return f"{FAMILIES[f]}[gpu {g}]"
        for t, u, d in zip(self.tiers, self.up_off, self.down_off):
            n = t.capacity.size
            if u <= r < u + n:
                return f"{t.name}_up[{r - u}]"
            if d <= r < d + n:
                return f"{t.name}_down[{r - d}]"
        return f"pool[{self.pool_names[r - self.pool_off]}]"

    # ------------------------------------------------------------ routes ---
    def same_domain(self, a, b) -> np.ndarray:
        return self.scaleup[np.asarray(a)] == self.scaleup[np.asarray(b)]

    def locality(self, a, b) -> np.ndarray:
        """First shared level: 0 = scale-up domain, t + 1 = scale-out tier t.
        Read only by the communication model."""
        a, b = np.asarray(a), np.asarray(b)
        out = np.full(np.broadcast(a, b).shape, len(self.tiers) + 1, dtype=np.int64)
        for i in reversed(range(len(self.tiers))):
            k = self.tiers[i].key
            out = np.where(k[a] == k[b], i + 1, out)
        return np.where(self.same_domain(a, b), 0, out)

    def pair_demand(self, src, dst, weight) -> tuple[np.ndarray, np.ndarray]:
        """Bytes per resource for sends src[i] -> dst[i] of weight[i] bytes,
        aggregated: (unique resource ids, bytes)."""
        src = np.asarray(src, dtype=np.int64).ravel()
        dst = np.asarray(dst, dtype=np.int64).ravel()
        w = np.broadcast_to(np.asarray(weight, dtype=float), src.shape).ravel()
        move = src != dst
        src, dst, w = src[move], dst[move], w[move]
        if src.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        near = self.same_domain(src, dst)
        far = ~near
        ids = [self.off["nvl_tx"] + src[near], self.off["nvl_rx"] + dst[near],
               self.off["nic_tx"] + src[far], self.off["nic_rx"] + dst[far]]
        ws = [w[near], w[near], w[far], w[far]]
        climbing = far.copy()
        for t, u, d in zip(self.tiers, self.up_off, self.down_off):
            ka, kb = t.key[src], t.key[dst]
            climbing &= ka != kb
            if not climbing.any():
                break
            ids += [u + ka[climbing], d + kb[climbing]]
            ws += [w[climbing], w[climbing]]
        return aggregate(np.concatenate(ids), np.concatenate(ws), self.n_resources)

    def route_latency(self, src, dst) -> float:
        """Latency of the slowest route among the sends."""
        src, dst = np.asarray(src).ravel(), np.asarray(dst).ravel()
        move = src != dst
        if not move.any():
            return 0.0
        far = ~self.same_domain(src[move], dst[move])
        return self.net_latency if far.any() else self.nvl_latency


def aggregate(ids: np.ndarray, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum values by resource id: (unique ids, sums)."""
    if ids.size == 0:
        return ids.astype(np.int64), values.astype(float)
    total = np.bincount(ids, weights=values, minlength=n)
    nz = np.flatnonzero(total)
    return nz, total[nz]
