"""Array kernels for the network: the only array-style code in the project.

Every kernel is a pure function. It takes integer device arrays and a byte
size, and returns a new demand vector over the cluster's resource index.
A kernel handles all instances of one collective class in one call, so the
cost grows with the number of member devices, never with device pairs.

`groups` is always an int array of shape [n_instances, group_size]: row i
holds the device IDs of collective instance i.
"""
from __future__ import annotations

import numpy as np

from narfs.network.topology import Cluster


# ---------------------------------------------------------------- helpers ---

def instance_shape(groups: np.ndarray, net: Cluster):
    """Shape of every instance on the hierarchy, in one pass.

    Returns (g, k, spans):
      g      [n_inst, size]  members of the same instance on each member's node
      k      [n_inst, 1]     distinct nodes in each instance
      spans  [n_inst, 1]     True if the instance uses more than one leaf switch
    """
    n_inst, size = groups.shape
    inst = np.repeat(np.arange(n_inst), size)
    node_key = inst * net.nodes + net.node_of[groups.ravel()]
    per_node = np.bincount(node_key, minlength=n_inst * net.nodes)
    g = per_node[node_key].reshape(n_inst, size)
    k = np.count_nonzero(per_node.reshape(n_inst, net.nodes), axis=1).reshape(n_inst, 1)
    leaf_key = inst * net.n_leaves + net.leaf_of[groups.ravel()]
    per_leaf = np.bincount(leaf_key, minlength=n_inst * net.n_leaves).reshape(n_inst, net.n_leaves)
    spans = (np.count_nonzero(per_leaf, axis=1) > 1).reshape(n_inst, 1)
    return g, k, spans


def members_per_node(groups: np.ndarray, net: Cluster) -> np.ndarray:
    """For each member: how many members of its instance are on its node."""
    return instance_shape(groups, net)[0]


def nodes_per_instance(groups: np.ndarray, net: Cluster) -> np.ndarray:
    """For each instance: how many distinct nodes it spans. Shape [n_inst, 1]."""
    return instance_shape(groups, net)[1]


def _member_demand(groups: np.ndarray, intra: np.ndarray, inter: np.ndarray,
                   spans: np.ndarray, net: Cluster) -> np.ndarray:
    """One accumulation for all member traffic of a collective class:
    `intra` bytes each way over NVLink, `inter` bytes each way over the NIC
    and the member's leaf, and the spine for instances that span leaves."""
    leaf = net.leaf_of[groups]
    ids = np.concatenate([(net.NVL_TX + groups).ravel(), (net.NVL_RX + groups).ravel(),
                          (net.NIC_TX + groups).ravel(), (net.NIC_RX + groups).ravel(),
                          (net.LEAF_UP + leaf).ravel(), (net.LEAF_DOWN + leaf).ravel()])
    intra = np.broadcast_to(intra, groups.shape).ravel()
    inter = np.broadcast_to(inter, groups.shape).ravel()
    vals = np.concatenate([intra, intra, inter, inter, inter, inter])
    demand = np.bincount(ids, weights=vals, minlength=net.n_resources)
    demand[net.SPINE] += float((np.broadcast_to(inter.reshape(groups.shape) * spans, groups.shape)).sum())
    return demand


def add_to(demand: np.ndarray, ids: np.ndarray, values: np.ndarray | float) -> None:
    """demand[ids] += values, with repeated ids accumulated."""
    np.add.at(demand, np.asarray(ids).ravel(), np.broadcast_to(values, np.shape(ids)).ravel())


# ---------------------------------------------------------- collectives ---

def allreduce_demand(groups: np.ndarray, size: float, net: Cluster) -> np.ndarray:
    """Hierarchical all-reduce of `size` bytes per rank, for every instance.

    Per member, with g = members on its node and k = nodes in its instance:
      NVLink (each direction): 2(g-1)/g * size     intra-node reduce-scatter + all-gather
      NIC    (each direction): 2(k-1)/k * size/g   inter-node ring on a 1/g shard
    """
    g, k, spans = instance_shape(groups, net)
    intra = 2.0 * (g - 1) / g * size
    inter = np.where(k > 1, 2.0 * (k - 1) / k * size / g, 0.0)
    return _member_demand(groups, intra, inter, spans, net)


def allgather_demand(groups: np.ndarray, size: float, net: Cluster) -> np.ndarray:
    """Hierarchical all-gather (or reduce-scatter) with `size` = full tensor
    bytes per rank. Half the traffic of an all-reduce of the same size."""
    g, k, spans = instance_shape(groups, net)
    intra = (g - 1) / g * size
    inter = np.where(k > 1, (k - 1) / k * size / g, 0.0)
    return _member_demand(groups, intra, inter, spans, net)


reducescatter_demand = allgather_demand


def alltoall_demand(groups: np.ndarray, size: float, net: Cluster) -> np.ndarray:
    """All-to-all where each member sends `size` bytes in total, split evenly
    over all members (its own share stays local).

      to members on the same node:   (g_node - 1)/G * size over NVLink
      to members on other nodes:     (G - g_node)/G * size over NIC and leaf
    """
    G = groups.shape[1]
    g, _, spans = instance_shape(groups, net)
    intra = (g - 1) / G * size
    inter = (G - g) / G * size
    return _member_demand(groups, intra, inter, spans, net)


def p2p_demand(src: np.ndarray, dst: np.ndarray, size: np.ndarray | float, net: Cluster) -> np.ndarray:
    """Point-to-point transfers src[i] -> dst[i] of size[i] bytes."""
    demand = net.zeros()
    src = np.asarray(src).ravel()
    dst = np.asarray(dst).ravel()
    size = np.broadcast_to(np.asarray(size, dtype=float), src.shape)
    moving = src != dst
    same_node = (net.node_of[src] == net.node_of[dst]) & moving
    cross = moving & ~same_node
    add_to(demand, net.NVL_TX + src[same_node], size[same_node])
    add_to(demand, net.NVL_RX + dst[same_node], size[same_node])
    s, d, b = src[cross], dst[cross], size[cross]
    add_to(demand, net.NIC_TX + s, b)
    add_to(demand, net.NIC_RX + d, b)
    add_to(demand, net.LEAF_UP + net.leaf_of[s], b)
    add_to(demand, net.LEAF_DOWN + net.leaf_of[d], b)
    demand[net.SPINE] += float(b[net.leaf_of[s] != net.leaf_of[d]].sum())
    return demand


def stream_demand(devices: np.ndarray, size: float, net: Cluster, kind: str = "pcie") -> np.ndarray:
    """A steady stream of `size` bytes per device over a per-device resource."""
    if kind != "pcie":
        raise ValueError(f"unknown stream kind {kind!r}")
    demand = net.zeros()
    add_to(demand, net.PCIE + np.asarray(devices), size)
    return demand


# ------------------------------------------------------------------ times ---

def burst_time(demand: np.ndarray, net: Cluster) -> float:
    """Time for a lockstep burst: the most loaded resource sets the time.
    All instances of a class run at the same instant, so their bytes share
    each resource."""
    return float((demand / net.cap).max())


def steps_latency(kind: str, groups: np.ndarray, net: Cluster) -> float:
    """Latency term: number of algorithm steps times the per-step latency of
    the slowest level used."""
    g, k, _ = instance_shape(groups, net)
    g_max, k_max = int(g.max()), int(k.max())
    if kind == "allreduce":
        intra, inter = 2 * (g_max - 1), 2 * (k_max - 1)
    elif kind in ("allgather", "reducescatter"):
        intra, inter = g_max - 1, k_max - 1
    elif kind == "alltoall":
        intra, inter = (1 if g_max > 1 else 0), (1 if k_max > 1 else 0)
    else:
        raise ValueError(f"unknown collective kind {kind!r}")
    return intra * net.nvlink_alpha + inter * net.nic_alpha


def collective(kind: str, groups: np.ndarray, size: float, net: Cluster) -> tuple[np.ndarray, float]:
    """Demand vector and latency term of one collective class, computing the
    instance shapes once. This is the entry point the evaluator uses."""
    g, k, spans = instance_shape(groups, net)
    G = groups.shape[1]
    if kind == "allreduce":
        intra = 2.0 * (g - 1) / g * size
        inter = np.where(k > 1, 2.0 * (k - 1) / k * size / g, 0.0)
        steps_intra, steps_inter = 2 * (g.max() - 1), 2 * (k.max() - 1)
    elif kind in ("allgather", "reducescatter"):
        intra = (g - 1) / g * size
        inter = np.where(k > 1, (k - 1) / k * size / g, 0.0)
        steps_intra, steps_inter = g.max() - 1, k.max() - 1
    elif kind == "alltoall":
        intra = (g - 1) / G * size
        inter = (G - g) / G * size
        steps_intra, steps_inter = int(g.max() > 1), int(k.max() > 1)
    else:
        raise ValueError(f"unknown collective kind {kind!r}")
    latency = float(steps_intra * net.nvlink_alpha + steps_inter * net.nic_alpha)
    return _member_demand(groups, intra, inter, spans, net), latency


DEMAND_KERNELS = {
    "allreduce": allreduce_demand,
    "allgather": allgather_demand,
    "reducescatter": reducescatter_demand,
    "alltoall": alltoall_demand,
}
