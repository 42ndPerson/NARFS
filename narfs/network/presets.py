"""Topology builders for common clusters.

`rail_optimized` builds the usual GPU fabric: an NVSwitch domain per node, one
NIC per GPU, rail r of each group of nodes on its own leaf, and optional pod
tiers above. Explicit, unbalanced trees can be built with `Topology` directly.
"""
from __future__ import annotations

import numpy as np

from narfs.network.topology import Tier, Topology

# H100 SXM public values (see the system profile data files for sources)
H100_NVLINK = 375e9            # NVLink payload rate per direction, measured (nvbandwidth class tools)
H100_NVLS_LEVEL = 0.82         # NVLS level: measured 8-GPU NVLS bus bandwidth / port bound
NIC_400G = 50e9 * 0.98         # 400 Gb/s ConnectX-7 payload, theory
PCIE_GEN5_X16 = 63e9 * 0.88    # PCIe Gen5 x16 payload, theory
NVLINK_STEP = 3.4e-6           # NCCL 2.32.3 hardware latency per step, NVLink, Simple
NET_STEP = 14e-6               # NCCL 2.32.3 hardware latency per step, network, Simple


def rail_optimized(nodes: int, gpus_per_node: int = 8, *, gpu: str = "h100-sxm",
                   nvl_rate: float = H100_NVLINK, nic_rate: float = NIC_400G, pcie_rate: float = PCIE_GEN5_X16,
                   nodes_per_leaf: int = 32, leaves_per_pod: int | None = None,
                   leaf_oversub: float = 1.0, pod_oversub: float = 1.0,
                   nvls: bool = True, nvls_level: float = H100_NVLS_LEVEL,
                   pools: dict | None = None, nvl_latency: float = NVLINK_STEP,
                   net_latency: float = NET_STEP) -> Topology:
    """`nodes` nodes of `gpus_per_node` GPUs. Rail r of every `nodes_per_leaf`
    nodes shares a leaf; `leaves_per_pod` leaves share a pod switch tier (None:
    the leaves meet at one spine). Uplink capacity = ports below x NIC rate /
    oversubscription."""
    G = nodes * gpus_per_node
    g = np.arange(G)
    node = g // gpus_per_node
    rail = g % gpus_per_node
    group = node // nodes_per_leaf
    leaf = group * gpus_per_node + rail
    n_leaves = int(leaf.max()) + 1
    below = np.bincount(leaf, minlength=n_leaves)
    tiers = [Tier("leaf", leaf, below * nic_rate / leaf_oversub)]
    if leaves_per_pod is not None:
        pod = leaf // leaves_per_pod
        below_pod = np.bincount(pod, minlength=int(pod.max()) + 1)
        tiers.append(Tier("pod", pod, below_pod * nic_rate / (leaf_oversub * pod_oversub)))
    return Topology(gpu_class=(gpu,) * G, scaleup=node, nvl_rate=nvl_rate, nic_rate=nic_rate,
                    pcie_rate=pcie_rate, tiers=tiers, nvls=nvls, nvls_level=nvls_level, pools=pools,
                    nvl_latency=nvl_latency, net_latency=net_latency)


def hgx_h100(nodes: int, **kw) -> Topology:
    """HGX H100 nodes (8 GPUs on NVSwitch) with 400 Gb/s rail-optimized NICs."""
    return rail_optimized(nodes, 8, **kw)
