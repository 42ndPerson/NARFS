from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from network_op import LogicalTransfer


@dataclass(frozen=True)
class Device:
    """A specific GPU on a specific host.

    Pairs (host_id, gpu_id) so transfers can be modeled at GPU granularity —
    required for rail-optimized topologies where two GPUs on the same host
    have distinct uplinks and are not interchangeable.
    """
    host_id: int
    gpu_id: int


@dataclass(frozen=True)
class Link:
    # Endpoints are either int network nodes (switches, fabric IDs) or
    # Device objects (GPU endpoints). Devices may only appear at the outer
    # ends of a Path; Path.__init__ enforces that.
    node_a_id: int | Device
    node_b_id: int | Device
    bandwidth: float  # Gbps
    latency: float    # ms


@dataclass
class Path:
    """A sequence of Links forming a route between two endpoints.

    The outer endpoints (node_a of the first link, node_b of the last) may
    be int network nodes or Devices; Devices appear only at endpoints,
    never as intermediate hops. An optional per-endpoint device bandwidth
    cap is enforced by inserting a phantom 0-latency self-loop Link at
    that end with the cap as its bandwidth — the phantom participates in
    shared-bandwidth contention identically to a real link, so multiple
    paths sharing the same device endpoint at the same cap will hash-equal
    on that phantom and accumulate load on it.
    """
    path: list[Link]
    bandwidth: float
    latency: float

    def __init__(
        self,
        links: list[Link],
        src_device_bandwidth: Optional[float] = None,
        dst_device_bandwidth: Optional[float] = None,
    ):
        if not links:
            raise ValueError("Path requires at least one link")

        for i in range(len(links) - 1):
            if links[i].node_b_id != links[i + 1].node_a_id:
                raise ValueError(
                    f"Invalid path: link {i} ends at {links[i].node_b_id} but "
                    f"link {i + 1} starts at {links[i + 1].node_a_id}"
                )

        n = len(links)
        for i, link in enumerate(links):
            if isinstance(link.node_a_id, Device) and i != 0:
                raise ValueError(
                    f"Device {link.node_a_id} found mid-path at link {i}; "
                    "devices may only appear at path endpoints"
                )
            if isinstance(link.node_b_id, Device) and i != n - 1:
                raise ValueError(
                    f"Device {link.node_b_id} found mid-path at link {i}; "
                    "devices may only appear at path endpoints"
                )

        full_path: list[Link] = list(links)

        if src_device_bandwidth is not None:
            src = links[0].node_a_id
            if not isinstance(src, Device):
                raise ValueError(
                    "src_device_bandwidth requires the path's source endpoint "
                    "to be a Device"
                )
            full_path.insert(
                0,
                Link(node_a_id=src, node_b_id=src, bandwidth=src_device_bandwidth, latency=0.0),
            )

        if dst_device_bandwidth is not None:
            dst = links[-1].node_b_id
            if not isinstance(dst, Device):
                raise ValueError(
                    "dst_device_bandwidth requires the path's destination "
                    "endpoint to be a Device"
                )
            full_path.append(
                Link(node_a_id=dst, node_b_id=dst, bandwidth=dst_device_bandwidth, latency=0.0),
            )

        self.path = full_path
        self.bandwidth = min(link.bandwidth for link in full_path)
        self.latency = sum(link.latency for link in full_path)


@dataclass
class HostTopo:
    """A directed graph of Devices connected by Paths.

    Intra-host links (e.g. NVLink between two GPUs on the same host) are
    represented as ordinary Paths in `device_connections` — there is no
    longer a separate intra_host_bandwidth/latency knob. A device-level
    bandwidth cap (e.g. NIC egress) is expressed by passing
    `src_device_bandwidth` / `dst_device_bandwidth` to the Path
    constructor, which enforces it via a phantom 0-latency self-loop link.
    """
    device_connections: dict[tuple[Device, Device], Path]

    def get_simultaneous_system_flow_time(
        self, transfers: list[LogicalTransfer]
    ) -> dict[tuple[Device, Device], float]:
        """Time for each (src_device, dst_device) pair to clear all of its
        transfers under flow-time contention (each link's completion time =
        total bytes on that link / link bandwidth). Used for boundary
        one-shot ops where the per-link queue drains once.
        """
        link_loads: defaultdict[Link, float] = defaultdict(float)
        for transfer in transfers:
            path = self.device_connections[(transfer.src_device, transfer.dst_device)]
            for link in path.path:
                link_loads[link] += transfer.data_size_GB

        link_times: dict[Link, float] = {}
        for link, load in link_loads.items():
            link_times[link] = (load * 8) / link.bandwidth

        connection_times: dict[tuple[Device, Device], float] = {}
        for transfer in transfers:
            path = self.device_connections[(transfer.src_device, transfer.dst_device)]
            time = max(link_times[link] for link in path.path)
            time += path.latency / 1000.0
            connection_times[(transfer.src_device, transfer.dst_device)] = time

        return connection_times

    def get_simultaneous_system_shared_bandwidth(
        self, transfers: list[LogicalTransfer]
    ) -> dict[LogicalTransfer, float]:
        """Effective per-transfer bandwidth assuming every transfer streams
        simultaneously and shares each traversed link's bandwidth fairly.
        Used for steady-state ops where the contention picture is constant
        for the duration of the op.
        """
        link_loads: defaultdict[Link, int] = defaultdict(int)
        for transfer in transfers:
            path = self.device_connections[(transfer.src_device, transfer.dst_device)]
            for link in path.path:
                link_loads[link] += 1

        connection_bandwidths: dict[LogicalTransfer, float] = {}
        for transfer in transfers:
            path = self.device_connections[(transfer.src_device, transfer.dst_device)]
            connection_bandwidths[transfer] = min(
                link.bandwidth / link_loads[link] for link in path.path
            )

        return connection_bandwidths

    def get_simultaneous_logical_transfer_times(
        self, transfers: list[LogicalTransfer]
    ) -> dict[LogicalTransfer, float]:
        """Per-transfer wall time under shared-bandwidth contention,
        including the path's accumulated latency.
        """
        connection_bandwidths = self.get_simultaneous_system_shared_bandwidth(transfers)

        transfer_times: dict[LogicalTransfer, float] = {}
        for t in transfers:
            bw = connection_bandwidths[t]
            latency_s = self.device_connections[(t.src_device, t.dst_device)].latency / 1000.0
            transfer_times[t] = (t.data_GB * 8) / bw + latency_s

        return transfer_times
