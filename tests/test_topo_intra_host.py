"""Intra-host bandwidth behavior in HostTopo (current state).

The two contention paths handle intra-host loads differently — pinning the
behavior here so changes are intentional and visible:

- `get_simultaneous_system_flow_time` aggregates intra-host data into
  `intra_host_loads[host]` as bytes, then computes a per-host time of
  `(total_data * 8) / intra_host_bandwidth`. Concurrent intra-host
  transfers DO share bandwidth in this path.

- `get_simultaneous_system_shared_bandwidth` returns the full
  `self.intra_host_bandwidth` for every intra-host transfer, regardless of
  how many other intra-host transfers are concurrent. The pre-existing
  TODO at the assignment site flags this as the expected enhancement
  (mirror the cross-host link-load divisor). The two paths are therefore
  asymmetric for intra-host traffic.

These tests document the current state and would surface any change.
"""
from __future__ import annotations

from network_op import LogicalTransfer
from topo import HostTopo


def _make_topo(intra_host_bandwidth: float = 600.0, latency_ms: float = 0.0) -> HostTopo:
    return HostTopo(
        intra_host_bandwidth=intra_host_bandwidth,
        intra_host_latency=latency_ms,
        hosts_connections={},
    )


def _intra_host_transfers(n: int, host: int = 0, base_data_gb: float = 1.0) -> list[LogicalTransfer]:
    # Distinct frozen-dataclass instances so the result dict keyed by
    # LogicalTransfer has n entries.
    return [
        LogicalTransfer(src_host_id=host, dst_host_id=host, data_GB=base_data_gb + i * 1e-3)
        for i in range(n)
    ]


def test_shared_bandwidth_returns_full_intra_host_bandwidth_per_flow():
    """Every concurrent intra-host transfer sees the full `intra_host_bandwidth`
    (no sharing). This is the pre-existing TODO behavior on
    `get_simultaneous_system_shared_bandwidth`."""
    topo = _make_topo(intra_host_bandwidth=600.0)
    transfers = _intra_host_transfers(n=8, host=0)
    result = topo.get_simultaneous_system_shared_bandwidth(transfers)
    assert len(result) == 8
    for bw in result.values():
        assert abs(bw - 600.0) < 1e-6


def test_shared_bandwidth_solo_flow_gets_full_bandwidth():
    topo = _make_topo(intra_host_bandwidth=600.0)
    transfers = _intra_host_transfers(n=1, host=0)
    result = topo.get_simultaneous_system_shared_bandwidth(transfers)
    bw = next(iter(result.values()))
    assert abs(bw - 600.0) < 1e-6


def test_flow_time_does_share_intra_host_bandwidth_across_concurrent_loads():
    """The flow-time path DOES share `intra_host_bandwidth` across concurrent
    intra-host loads — opposite of `get_simultaneous_system_shared_bandwidth`'s
    current behavior. Documents the asymmetry between the two methods."""
    topo = _make_topo(intra_host_bandwidth=600.0)
    transfers = _intra_host_transfers(n=8, host=0, base_data_gb=1.0)
    flow_times = topo.get_simultaneous_system_flow_time(transfers)
    # 8 concurrent flows of ~1 GB each → total ~8 GB on host 0.
    # time = (8 GB * 8 bits/byte) / 600 Gbps ≈ 0.107 s.
    t = next(iter(flow_times.values()))
    assert 0.10 < t < 0.12, f"expected ~0.107s, got {t}"


def test_intra_host_loads_partitioned_per_host_in_flow_time():
    """Concurrent intra-host flows on different hosts share bandwidth
    independently in the flow-time path."""
    topo = _make_topo(intra_host_bandwidth=600.0)
    transfers = (
        _intra_host_transfers(n=4, host=0, base_data_gb=1.0)
        + _intra_host_transfers(n=2, host=1, base_data_gb=2.0)
    )
    flow_times = topo.get_simultaneous_system_flow_time(transfers)
    # host 0: 4 transfers totaling ~4 GB → 4*8/600 ≈ 0.0533s
    # host 1: 2 transfers totaling ~4 GB (1.001 + 1.002... hmm actually 4.003 with the 2.0 base)
    # Wait: base 2.0 + i*1e-3 for i in 0..1 → 2.0 + 2.001 = 4.001
    # → 4.001*8/600 ≈ 0.0533s
    t_host_0 = flow_times[(0, 0)]
    t_host_1 = flow_times[(1, 1)]
    # host 0: 4 transfers, base 1.0, so ~4.006 GB → 4.006*8/600 ≈ 0.0534s
    assert 0.05 < t_host_0 < 0.06
    assert 0.05 < t_host_1 < 0.06
