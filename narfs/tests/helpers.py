"""Small fixtures with numbers that are easy to check by hand.

The unit device does 1 flop/s and moves 1 byte/s at 100% efficiency with no
launch time, so an Op("memory", 0, b) takes exactly b seconds.
"""
from __future__ import annotations

import numpy as np

from narfs.costs import CostModel
from narfs.evaluate import EvalContext
from narfs.hardware import DeviceSpec
from narfs.network.topology import Cluster
from narfs.parallel.mesh import Degrees, Placement, map_devices

UNIT = DeviceSpec("unit", peak_flops=1.0, hbm_bw=1.0, hbm_bytes=1e12, compute_eff=1.0,
                  small_gemm_eff=1.0, gemv_eff=1.0, memory_eff=1.0, gemv_memory_eff=1.0,
                  launch_s=0.0, pcie_bw=1.0)


def tiny_cluster(nodes: int = 2, gpus_per_node: int = 2, **kw) -> Cluster:
    args = dict(nvlink_bw=100.0, nic_bw=10.0, nvlink_alpha=0.0, nic_alpha=0.0, nodes_per_rail_group=2)
    args.update(kw)
    return Cluster(nodes=nodes, gpus_per_node=gpus_per_node, devices=(UNIT,), **args)


def one_device(d: int = 0) -> Placement:
    return Placement(Degrees(), np.array([d]).reshape(1, 1, 1, 1, 1))


def ctx_for(net: Cluster, placements: dict, serial: bool = False) -> EvalContext:
    return EvalContext(net, CostModel(), placements, serial=serial)


def mem(seconds: float, name: str = "") -> "Op":
    from narfs.blocks import Op
    return Op("memory", 0.0, float(seconds), name=name)


__all__ = ["UNIT", "tiny_cluster", "one_device", "ctx_for", "mem", "map_devices", "Degrees"]
