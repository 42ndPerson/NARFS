"""Intra-host transfers go through ordinary Paths in the new device-keyed
HostTopo. The previous intra_host_bandwidth/intra_host_latency short-circuit
is gone, so both contention modes share bandwidth via the same link-load
divisor — no more asymmetry between flow-time and shared-bandwidth modes.
"""
from __future__ import annotations

from network_op import LogicalTransfer
from topo import Device, HostTopo, Link, Path


def _make_topo(bandwidth: float = 600.0, latency_ms: float = 0.0) -> tuple[HostTopo, list[Device]]:
    """Two GPUs on host 0 plus two GPUs on host 1, with intra-host paths
    sharing one link per host and inter-host paths sharing one cross-host
    link. Returns the topo and the list [d00, d01, d10, d11].
    """
    d00 = Device(host_id=0, gpu_id=0)
    d01 = Device(host_id=0, gpu_id=1)
    d10 = Device(host_id=1, gpu_id=0)
    d11 = Device(host_id=1, gpu_id=1)

    # One intra-host link per host (id 100/101) and one inter-host link (id 200).
    intra0 = Link(node_a_id=100, node_b_id=100, bandwidth=bandwidth, latency=latency_ms)
    intra1 = Link(node_a_id=101, node_b_id=101, bandwidth=bandwidth, latency=latency_ms)
    inter = Link(node_a_id=200, node_b_id=200, bandwidth=bandwidth, latency=latency_ms)

    # Each path enters the shared link via a 0-latency / huge-bandwidth
    # device-attach link so the shared link is the contention point.
    huge = 1e9

    def _path(src: Device, dst: Device, shared: Link) -> Path:
        attach_src = Link(node_a_id=src, node_b_id=shared.node_a_id, bandwidth=huge, latency=0.0)
        attach_dst = Link(node_a_id=shared.node_b_id, node_b_id=dst, bandwidth=huge, latency=0.0)
        return Path([attach_src, shared, attach_dst])

    connections = {
        (d00, d01): _path(d00, d01, intra0),
        (d01, d00): _path(d01, d00, intra0),
        (d10, d11): _path(d10, d11, intra1),
        (d11, d10): _path(d11, d10, intra1),
        (d00, d10): _path(d00, d10, inter),
        (d10, d00): _path(d10, d00, inter),
    }
    return HostTopo(device_connections=connections), [d00, d01, d10, d11]


def _intra_host_transfers(devices: list[Device], n: int, base_data_gb: float = 1.0) -> list[LogicalTransfer]:
    src, dst = devices[0], devices[1]
    return [
        LogicalTransfer(src_device=src, dst_device=dst, data_GB=base_data_gb + i * 1e-3)
        for i in range(n)
    ]


def test_shared_bandwidth_divides_evenly_across_concurrent_intra_host_flows():
    """With the intra-host short-circuit gone, n concurrent intra-host
    transfers share the link's bandwidth like any other contended link."""
    topo, devices = _make_topo(bandwidth=600.0)
    transfers = _intra_host_transfers(devices, n=8)
    result = topo.get_simultaneous_system_shared_bandwidth(transfers)
    assert len(result) == 8
    for bw in result.values():
        assert abs(bw - 600.0 / 8) < 1e-6


def test_shared_bandwidth_solo_flow_gets_full_bandwidth():
    topo, devices = _make_topo(bandwidth=600.0)
    transfers = _intra_host_transfers(devices, n=1)
    result = topo.get_simultaneous_system_shared_bandwidth(transfers)
    bw = next(iter(result.values()))
    assert abs(bw - 600.0) < 1e-6


def test_flow_time_shares_intra_host_bandwidth_across_concurrent_loads():
    topo, devices = _make_topo(bandwidth=600.0)
    transfers = _intra_host_transfers(devices, n=8, base_data_gb=1.0)
    flow_times = topo.get_simultaneous_system_flow_time(transfers)
    # 8 concurrent flows of ~1 GB each → ~8 GB on the shared link.
    # time = (8 GB * 8 bits/byte) / 600 Gbps ≈ 0.107 s.
    t = next(iter(flow_times.values()))
    assert 0.10 < t < 0.12, f"expected ~0.107s, got {t}"


def test_intra_host_loads_partitioned_per_host_in_flow_time():
    """Loads on different hosts' intra-host links share bandwidth
    independently — distinct Link objects, distinct contention pools."""
    topo, devices = _make_topo(bandwidth=600.0)
    d00, d01, d10, d11 = devices
    transfers = (
        [LogicalTransfer(src_device=d00, dst_device=d01, data_GB=1.0 + i * 1e-3) for i in range(4)]
        + [LogicalTransfer(src_device=d10, dst_device=d11, data_GB=2.0 + i * 1e-3) for i in range(2)]
    )
    flow_times = topo.get_simultaneous_system_flow_time(transfers)
    t_host_0 = flow_times[(d00, d01)]
    t_host_1 = flow_times[(d10, d11)]
    # host 0: 4 transfers, base 1.0 → ~4.006 GB → 4.006*8/600 ≈ 0.0534s
    # host 1: 2 transfers, base 2.0 → ~4.001 GB → ~0.0533s
    assert 0.05 < t_host_0 < 0.06
    assert 0.05 < t_host_1 < 0.06
