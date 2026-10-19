"""Topology builders for common clusters.

`rail_optimized` builds the usual GPU fabric: an NVSwitch domain per node, one
NIC per GPU, rail r of each group of nodes on its own leaf, and optional pod
tiers above. Explicit, unbalanced trees can be built with `Topology` directly.

Rates come from the GPU profile, and the point-to-point latencies from NCCL's
per-step Simple-protocol latencies (data files in narfs/data).
"""
from __future__ import annotations

import numpy as np

from narfs.hardware import DATA, H100_SXM, GpuProfile
from narfs.network.comm import load_nccl
from narfs.network.topology import Tier, Topology

_NCCL = load_nccl(DATA / "nccl-2.32.3.json")
NVLINK_STEP = _NCCL.hw[("NVLINK", "Ring", "Simple")] * 1e-6
NET_STEP = _NCCL.hw[("NET", "Ring", "Simple")] * 1e-6


def rail_optimized(nodes: int, gpus_per_node: int = 8, *, gpu: GpuProfile = H100_SXM,
                   nvl_rate: float | None = None, nic_rate: float | None = None, pcie_rate: float | None = None,
                   nodes_per_leaf: int = 32, leaves_per_pod: int | None = None,
                   leaf_oversub: float = 1.0, pod_oversub: float = 1.0,
                   nvls: bool = True, nvls_level: float | None = None, pools: dict | None = None,
                   nvl_latency: float = NVLINK_STEP, net_latency: float = NET_STEP) -> Topology:
    """`nodes` nodes of `gpus_per_node` GPUs. Rail r of every `nodes_per_leaf`
    nodes shares a leaf; `leaves_per_pod` leaves share a pod tier (None: the
    leaves meet at one spine). Uplink capacity = ports below x NIC rate /
    oversubscription. Rates default to the GPU profile's values."""
    nvl_rate = gpu.nvlink_rate if nvl_rate is None else nvl_rate
    nic_rate = gpu.nic_rate if nic_rate is None else nic_rate
    pcie_rate = gpu.pcie_rate if pcie_rate is None else pcie_rate
    nvls_level = gpu.nvls_level if nvls_level is None else nvls_level
    G = nodes * gpus_per_node
    g = np.arange(G)
    node = g // gpus_per_node
    rail = g % gpus_per_node
    group = node // nodes_per_leaf
    leaf = group * gpus_per_node + rail
    below = np.bincount(leaf, minlength=int(leaf.max()) + 1)
    tiers = [Tier("leaf", leaf, below * nic_rate / leaf_oversub)]
    if leaves_per_pod is not None:
        pod = leaf // leaves_per_pod
        below_pod = np.bincount(pod, minlength=int(pod.max()) + 1)
        tiers.append(Tier("pod", pod, below_pod * nic_rate / (leaf_oversub * pod_oversub)))
    return Topology(gpu_class=(gpu.name,) * G, scaleup=node, nvl_rate=nvl_rate, nic_rate=nic_rate,
                    pcie_rate=pcie_rate, tiers=tiers, nvls=nvls, nvls_level=nvls_level, pools=pools,
                    nvl_latency=nvl_latency, net_latency=net_latency)


def hgx_h100(nodes: int, **kw) -> Topology:
    """HGX H100 nodes (8 GPUs on NVSwitch) with 400 Gb/s rail-optimized NICs."""
    return rail_optimized(nodes, 8, gpu=H100_SXM, **kw)
