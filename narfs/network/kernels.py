"""Array kernels for the network: the only array-style code in the project.

Each pattern function takes a topology and a group array of shape
[instances, members] (all instances run at once), and returns the bytes on
each resource per byte of the collective's size S, aggregated as
(unique resource ids, bytes). Costs grow with members x tiers, never with
device pairs.

Each pattern also has a `*_flows` twin that lists the point-to-point flows
(src, dst, bytes per byte of S). Tests check that routing the flows gives the
pattern, and the flow-level validation tier uses the flows.
"""
from __future__ import annotations

import numpy as np

from narfs.network.topology import Topology, aggregate


# ------------------------------------------------------------ helpers ---

def _runs(sorted_keys: np.ndarray):
    """Run starts, run ids per element, local index and run size per element,
    for a flat array where equal keys are contiguous."""
    n = sorted_keys.size
    new = np.ones(n, dtype=bool)
    new[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = np.flatnonzero(new)
    run_id = np.cumsum(new) - 1
    size = np.diff(np.append(starts, n))
    local = np.arange(n) - starts[run_id]
    return starts, run_id, local, size


def ring_order(topo: Topology, groups: np.ndarray) -> np.ndarray:
    """Members of each instance in ring order: members of one scale-up domain
    are contiguous, and domains follow their lowest member's scale-out keys
    (top tier first), so a ring crosses each tier as few times as it can."""
    I, n = groups.shape
    flat = groups.ravel()
    inst = np.repeat(np.arange(I), n)
    dom = topo.scaleup[flat]
    key = inst * topo.n_scaleup + dom
    order = np.lexsort((flat, key))
    starts, run_id, _, _ = _runs(key[order])
    low = np.empty_like(flat)
    low[order] = flat[order][starts][run_id]               # lowest member of each (instance, domain)
    keys = [flat, dom] + [t.key[low] for t in topo.tiers] + [inst]
    return flat[np.lexsort(tuple(keys))].reshape(I, n)


def _ring_layout(topo: Topology, groups: np.ndarray):
    ring = ring_order(topo, groups)
    I, n = ring.shape
    flat = ring.ravel()
    inst = np.repeat(np.arange(I), n)
    key = inst * topo.n_scaleup + topo.scaleup[flat]
    starts, run_id, local, size = _runs(key)
    run_inst = inst[starts]
    runs_per_inst = np.bincount(run_inst, minlength=I)
    channels = int(size.max())
    return ring, flat, starts, run_id, local, size, run_inst, runs_per_inst, channels


# ------------------------------------------------------------ patterns ---

def ring(topo: Topology, groups: np.ndarray, factor: float):
    """A ring collective that moves `factor` x S over every ring edge
    (2(n-1)/n for all-reduce, (n-1)/n for all-gather and reduce-scatter).

    Each ring channel uses one rail. With C channels, channel c enters and
    leaves every domain through the domain's member at local index c mod g, and
    goes around the domain's members over NVLink in between. So the NIC bytes
    of a domain are spread over its members, and on a rail-optimized fabric a
    channel's inter-node bytes stay on one rail, as NCCL's rings do."""
    ring_, flat, starts, run_id, local, size, run_inst, k, C = _ring_layout(topo, groups)
    g = size[run_id]
    nvl = np.where(g > 1, float(factor), 0.0)
    ids = [topo.off["nvl_tx"] + flat, topo.off["nvl_rx"] + flat]
    ws = [nvl, nvl]
    src, dst = _crossings(starts, size, run_inst, k, flat, C)
    if src.size:
        cid, cw = topo.pair_demand(src, dst, factor / C)
        ids.append(cid)
        ws.append(cw)
    return aggregate(np.concatenate(ids), np.concatenate(ws), topo.n_resources)


def _next_runs(run_inst: np.ndarray) -> np.ndarray:
    R = run_inst.size
    nxt = np.arange(1, R + 1)
    first = np.r_[0, np.flatnonzero(np.diff(run_inst)) + 1]            # first run of each instance
    last = np.r_[first[1:] - 1, R - 1]
    nxt[last] = first
    return nxt


def _crossings(starts, size, run_inst, k, flat, C):
    """(member leaving a domain, member entering the next one) for every
    channel of every multi-domain instance."""
    multi = k[run_inst] > 1
    if not multi.any():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    nxt = _next_runs(run_inst)
    sel = np.flatnonzero(multi)
    c = np.arange(C)
    src = starts[sel, None] + c[None, :] % size[sel, None]
    nx = nxt[sel]
    dst = starts[nx, None] + c[None, :] % size[nx, None]
    return flat[src.ravel()], flat[dst.ravel()]


def ring_flows(topo: Topology, groups: np.ndarray, factor: float):
    """The ring's flows, channel by channel (for tests and validation)."""
    ring_, flat, starts, run_id, local, size, run_inst, k, C = _ring_layout(topo, groups)
    per = factor / C
    srcs, dsts = [], []
    for i in range(ring_.shape[0]):
        runs = np.flatnonzero(run_inst == i)
        for c in range(C):
            for a, r in enumerate(runs):
                m = flat[starts[r]:starts[r] + size[r]]
                g = size[r]
                if g > 1:                                                # once around the domain over NVLink
                    srcs += list(m)
                    dsts += list(np.roll(m, -1))
                if k[i] > 1:                                             # then on to the next domain, one rail
                    r2 = runs[(a + 1) % len(runs)]
                    m2 = flat[starts[r2]:starts[r2] + size[r2]]
                    srcs.append(m[c % g])
                    dsts.append(m2[c % size[r2]])
    return np.array(srcs), np.array(dsts), np.full(len(srcs), per)


def _members(topo: Topology, groups: np.ndarray, per_member: float):
    flat = groups.ravel()
    ids = np.concatenate([topo.off["nvl_tx"] + flat, topo.off["nvl_rx"] + flat])
    return aggregate(ids, np.full(ids.size, float(per_member)), topo.n_resources)


def one_shot(topo: Topology, groups: np.ndarray):
    """Each member reads the whole buffer from every peer: (n-1) x S each way.
    Inside one scale-up domain only."""
    return _members(topo, groups, groups.shape[1] - 1)


def one_shot_flows(topo: Topology, groups: np.ndarray):
    return _all_pairs(groups, 1.0)


def two_shot(topo: Topology, groups: np.ndarray):
    """Reduce-scatter then all-gather by direct loads: 2(n-1)/n x S each way."""
    n = groups.shape[1]
    return _members(topo, groups, 2.0 * (n - 1) / n)


def two_shot_flows(topo: Topology, groups: np.ndarray):
    return _all_pairs(groups, 2.0 / groups.shape[1])


def nvls(topo: Topology, groups: np.ndarray):
    """In-switch reduction: each port carries S(1 + 1/n) each way, divided by
    the domain's NVLS level (applied like a protocol payload factor)."""
    n = groups.shape[1]
    flat = groups.ravel()
    level = topo.nvls_level[topo.scaleup[flat]]
    per = (1.0 + 1.0 / n) / level
    ids = np.concatenate([topo.off["nvl_tx"] + flat, topo.off["nvl_rx"] + flat])
    return aggregate(ids, np.concatenate([per, per]), topo.n_resources)


def spread(topo: Topology, groups: np.ndarray):
    """Uniform all-to-all: each member sends S/n to every member (its own share
    stays local). Uses domain counts, so the cost is O(members x tiers).
    Assumes nested scale-out tiers."""
    I, n = groups.shape
    flat = groups.ravel()
    inst = np.repeat(np.arange(I), n)

    def count(key):
        _, inv, cnt = np.unique(inst * (key.max() + 1) + key, return_inverse=True, return_counts=True)
        return cnt[inv]

    dom = topo.scaleup[flat]
    g = count(dom)
    ids = [topo.off["nvl_tx"] + flat, topo.off["nvl_rx"] + flat,
           topo.off["nic_tx"] + flat, topo.off["nic_rx"] + flat]
    near, far = (g - 1) / n, (n - g) / n
    ws = [near, near, far, far]
    for t, u, d in zip(topo.tiers, topo.up_off, topo.down_off):
        tk = t.key[flat]
        c = count(tk)
        both = count(dom * (tk.max() + 1) + tk)
        out = (n - g - c + both) / n                       # peers outside both the domain and the tier domain
        ids += [u + tk, d + tk]
        ws += [out, out]
    ws = [np.broadcast_to(w, flat.shape) for w in ws]
    return aggregate(np.concatenate(ids), np.concatenate(ws), topo.n_resources)


def spread_flows(topo: Topology, groups: np.ndarray):
    return _all_pairs(groups, 1.0 / groups.shape[1])


def _all_pairs(groups: np.ndarray, per: float):
    I, n = groups.shape
    a = np.repeat(groups, n, axis=1).ravel()
    b = np.tile(groups, (1, n)).ravel()
    keep = a != b
    return a[keep], b[keep], np.full(int(keep.sum()), per)


# ------------------------------------------------------------- units ---

class UnitTable:
    """Interned demand vectors. A unit is (resource ids, seconds of use per
    unit amount); its id is what the evaluator's demand maps hold. Units are
    keyed by content, so they are shared across candidates."""

    def __init__(self, cap: np.ndarray, limit: int = 200_000):
        self.cap = cap
        self.limit = limit
        self._ids: dict = {}
        self._units: list = []

    def __len__(self) -> int:
        return len(self._units)

    def intern(self, key, build) -> int:
        uid = self._ids.get(key)
        if uid is None:
            ids, amount = build()
            uid = self._ids[key] = len(self._units)
            self._units.append((np.asarray(ids, dtype=np.int64), np.asarray(amount, dtype=float) / self.cap[ids]))
        return uid

    def unit(self, uid: int):
        return self._units[uid]

    def trim(self) -> None:
        """Forget every unit (call between evaluations when the table is large)."""
        if len(self._units) > self.limit:
            self._ids.clear()
            self._units.clear()
