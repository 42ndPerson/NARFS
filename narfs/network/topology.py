"""Cluster description and the global resource index.

A resource is one directed capacity with an explicit integer ID. Devices
and resources are integers; routes are arithmetic on device coordinates.
Nothing here stores a path for every device pair.

Resource classes (each one a contiguous block of the index):

    nvl_tx[G], nvl_rx[G]   NVLink port of each GPU, per direction
    nic_tx[G], nic_rx[G]   NIC of each GPU (rail-optimized: one NIC per GPU)
    pcie[G]                host <-> device link of each GPU
    gpu[G]                 device busy time (capacity 1 second per second)
    leaf_up[L], leaf_down[L]  uplink and downlink of each leaf switch
    spine[1]               pooled spine bisection

Units: bytes for links, seconds for `gpu`. Capacities are per second, so
demand / capacity is always in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from narfs.hardware import DeviceSpec


@dataclass(frozen=True)
class Cluster:
    """A rail-optimized cluster: NVSwitch inside a node, one NIC per GPU,
    rail r of each rail group goes to leaf (group, r), and leaves meet at a
    spine with optional oversubscription.

    `node_class` gives the device class of each node, so mixed clusters are
    allowed. Per-resource capacity overrides go in `cap_overrides`.
    """

    nodes: int
    gpus_per_node: int
    devices: tuple[DeviceSpec, ...]
    node_class: tuple[int, ...] = ()
    nvlink_bw: float = 450e9          # per GPU per direction, bytes/s
    nic_bw: float = 50e9              # per GPU per direction, bytes/s
    nodes_per_rail_group: int = 32    # nodes that share one leaf per rail
    spine_oversub: float = 1.0        # 2.0 means a 2:1 spine
    nvlink_alpha: float = 3e-6        # latency per intra-node step
    nic_alpha: float = 10e-6          # latency per inter-node step
    cap_overrides: tuple[tuple[int, float], ...] = ()

    # derived arrays (built in __post_init__)
    node_of: np.ndarray = field(init=False, repr=False, compare=False)
    gpu_of: np.ndarray = field(init=False, repr=False, compare=False)
    leaf_of: np.ndarray = field(init=False, repr=False, compare=False)
    class_of: np.ndarray = field(init=False, repr=False, compare=False)
    cap: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.node_class:
            object.__setattr__(self, "node_class", (0,) * self.nodes)
        if len(self.node_class) != self.nodes:
            raise ValueError("node_class needs one entry per node")
        G = self.n_devices
        node_of = np.arange(G) // self.gpus_per_node
        gpu_of = np.arange(G) % self.gpus_per_node
        group_of = node_of // self.nodes_per_rail_group
        object.__setattr__(self, "node_of", node_of)
        object.__setattr__(self, "gpu_of", gpu_of)
        object.__setattr__(self, "leaf_of", group_of * self.gpus_per_node + gpu_of)
        object.__setattr__(self, "class_of", np.asarray(self.node_class)[node_of])
        cap = np.empty(self.n_resources)
        cap[self.NVL_TX:self.NIC_TX] = self.nvlink_bw
        cap[self.NIC_TX:self.PCIE] = self.nic_bw
        pcie = np.array([d.pcie_bw for d in self.devices])[self.class_of]
        cap[self.PCIE:self.GPU] = pcie
        cap[self.GPU:self.LEAF_UP] = 1.0
        nodes_per_leaf = min(self.nodes_per_rail_group, self.nodes)
        leaf_bw = self.nic_bw * nodes_per_leaf
        cap[self.LEAF_UP:self.SPINE] = leaf_bw
        cap[self.SPINE] = self.nic_bw * G / self.spine_oversub
        for rid, value in self.cap_overrides:
            cap[rid] = value
        cap.setflags(write=False)
        object.__setattr__(self, "cap", cap)

    # ---- sizes and offsets of the resource index ----
    @property
    def n_devices(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def n_leaves(self) -> int:
        groups = -(-self.nodes // self.nodes_per_rail_group)
        return groups * self.gpus_per_node

    @property
    def NVL_TX(self) -> int: return 0
    @property
    def NVL_RX(self) -> int: return self.n_devices
    @property
    def NIC_TX(self) -> int: return 2 * self.n_devices
    @property
    def NIC_RX(self) -> int: return 3 * self.n_devices
    @property
    def PCIE(self) -> int: return 4 * self.n_devices
    @property
    def GPU(self) -> int: return 5 * self.n_devices
    @property
    def LEAF_UP(self) -> int: return 6 * self.n_devices
    @property
    def LEAF_DOWN(self) -> int: return 6 * self.n_devices + self.n_leaves
    @property
    def SPINE(self) -> int: return 6 * self.n_devices + 2 * self.n_leaves

    @property
    def n_resources(self) -> int:
        return self.SPINE + 1

    def zeros(self) -> np.ndarray:
        return np.zeros(self.n_resources)

    def device_spec(self, device: int) -> DeviceSpec:
        return self.devices[int(self.class_of[device])]

    # ---- routing: arithmetic on coordinates ----
    def route(self, src: int, dst: int) -> list[int]:
        """Resource IDs used by a transfer from `src` to `dst`."""
        if src == dst:
            return []
        if self.node_of[src] == self.node_of[dst]:
            return [self.NVL_TX + src, self.NVL_RX + dst]
        out = [self.NIC_TX + src, self.LEAF_UP + int(self.leaf_of[src])]
        if self.leaf_of[src] != self.leaf_of[dst]:
            out.append(self.SPINE)
        out += [self.LEAF_DOWN + int(self.leaf_of[dst]), self.NIC_RX + dst]
        return out

    def resource_name(self, rid: int) -> str:
        G = self.n_devices
        for name, start, size in (("nvl_tx", self.NVL_TX, G), ("nvl_rx", self.NVL_RX, G),
                                  ("nic_tx", self.NIC_TX, G), ("nic_rx", self.NIC_RX, G),
                                  ("pcie", self.PCIE, G), ("gpu", self.GPU, G)):
            if start <= rid < start + size:
                d = rid - start
                return f"{name}[node {self.node_of[d]} gpu {self.gpu_of[d]}]"
        if self.LEAF_UP <= rid < self.LEAF_DOWN:
            return f"leaf_up[{rid - self.LEAF_UP}]"
        if self.LEAF_DOWN <= rid < self.SPINE:
            return f"leaf_down[{rid - self.LEAF_DOWN}]"
        return "spine"
