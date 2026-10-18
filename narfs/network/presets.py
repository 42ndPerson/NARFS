"""Topology presets for common clusters."""
from __future__ import annotations

from narfs.hardware import A100_80GB, H100_80GB
from narfs.network.topology import Cluster


def hgx_h100(nodes: int, spine_oversub: float = 1.0) -> Cluster:
    """HGX H100 nodes (8 GPUs, NVSwitch) with 400 Gb/s rail-optimized NICs."""
    return Cluster(nodes=nodes, gpus_per_node=8, devices=(H100_80GB,), nvlink_bw=450e9,
                   nic_bw=50e9, spine_oversub=spine_oversub)


def dgx_a100(nodes: int, spine_oversub: float = 1.0) -> Cluster:
    """DGX A100 nodes (8 GPUs, NVSwitch) with 200 Gb/s rail-optimized NICs."""
    return Cluster(nodes=nodes, gpus_per_node=8, devices=(A100_80GB,), nvlink_bw=300e9,
                   nic_bw=25e9, nvlink_alpha=4e-6, spine_oversub=spine_oversub)
