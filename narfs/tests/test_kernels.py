"""Network kernels, checked by hand on a 2-node x 2-GPU cluster.

Leaves: device d is on leaf (d % 2) (rail = GPU index), so devices 0 and 2
share leaf 0 and devices 1 and 3 share leaf 1. NVLink 100 B/s, NIC 10 B/s,
leaf 20 B/s, spine 40 B/s.
"""
import numpy as np
import pytest

from narfs.network import kernels as K
from narfs.tests.helpers import tiny_cluster


@pytest.fixture
def net():
    return tiny_cluster()


def test_capacities_and_offsets(net):
    assert net.n_devices == 4 and net.n_leaves == 2
    assert net.cap[net.NVL_TX] == 100 and net.cap[net.NIC_RX] == 10
    assert net.cap[net.LEAF_UP] == 20 and net.cap[net.SPINE] == 40
    assert net.cap[net.GPU] == 1.0
    assert net.n_resources == 6 * 4 + 2 * 2 + 1


def test_route_is_arithmetic(net):
    assert net.route(0, 0) == []
    assert net.route(0, 1) == [net.NVL_TX + 0, net.NVL_RX + 1]
    assert net.route(0, 2) == [net.NIC_TX + 0, net.LEAF_UP + 0, net.LEAF_DOWN + 0, net.NIC_RX + 2]
    assert net.SPINE in net.route(0, 3)


def test_instance_shape(net):
    g, k, spans = K.instance_shape(np.array([[0, 1, 2, 3], [0, 1, 2, 3]]), net)
    assert (g == 2).all() and (k == 2).all() and spans.all()
    g, k, spans = K.instance_shape(np.array([[0, 2]]), net)
    assert (g == 1).all() and k[0, 0] == 2 and not spans.any()


def test_allreduce_hand_numbers(net):
    d = K.allreduce_demand(np.array([[0, 1, 2, 3]]), 8.0, net)
    # g = 2, k = 2: NVLink 2(g-1)/g * S = 8; NIC 2(k-1)/k * S/g = 4
    assert np.allclose(d[net.NVL_TX:net.NVL_RX], 8) and np.allclose(d[net.NVL_RX:net.NIC_TX], 8)
    assert np.allclose(d[net.NIC_TX:net.NIC_RX], 4) and np.allclose(d[net.NIC_RX:net.PCIE], 4)
    assert np.allclose(d[net.LEAF_UP:net.LEAF_DOWN], 8)       # two members per leaf
    assert d[net.SPINE] == pytest.approx(16)                   # the instance spans both leaves
    assert K.burst_time(d, net) == pytest.approx(0.4)


def test_allgather_is_half_of_allreduce(net):
    groups = np.array([[0, 1, 2, 3]])
    assert np.allclose(2 * K.allgather_demand(groups, 8.0, net), K.allreduce_demand(groups, 8.0, net))
    assert K.reducescatter_demand is K.allgather_demand


def test_intra_node_collective_uses_no_nic(net):
    d = K.allreduce_demand(np.array([[0, 1], [2, 3]]), 10.0, net)
    assert d[net.NIC_TX:net.PCIE].sum() == 0 and d[net.SPINE] == 0
    assert np.allclose(d[net.NVL_TX:net.NIC_TX], 10)


def test_same_rail_needs_no_spine(net):
    d = K.allreduce_demand(np.array([[0, 2], [1, 3]]), 10.0, net)
    assert d[net.SPINE] == 0
    assert K.burst_time(d, net) == pytest.approx(1.0)          # 10 B per NIC at 10 B/s


def test_alltoall_hand_numbers(net):
    d = K.alltoall_demand(np.array([[0, 1, 2, 3]]), 8.0, net)
    # G = 4, g = 2: (g-1)/G * S = 2 over NVLink, (G-g)/G * S = 4 over the NIC
    assert np.allclose(d[net.NVL_TX:net.NVL_RX], 2)
    assert np.allclose(d[net.NIC_TX:net.NIC_RX], 4)


def test_p2p(net):
    d = K.p2p_demand(np.array([0, 0, 1]), np.array([2, 3, 1]), np.array([10.0, 5.0, 99.0]), net)
    assert d[net.NIC_TX + 0] == 15 and d[net.NIC_RX + 2] == 10 and d[net.NIC_RX + 3] == 5
    assert d[net.SPINE] == 5                                   # only 0 -> 3 changes leaf
    assert d[net.NVL_TX:net.NIC_TX].sum() == 0                 # 1 -> 1 moves nothing


def test_collective_entry_matches_kernels(net):
    groups = np.array([[0, 1, 2, 3]])
    for kind, fn in K.DEMAND_KERNELS.items():
        demand, latency = K.collective(kind, groups, 8.0, net)
        assert np.allclose(demand, fn(groups, 8.0, net))
        assert latency == K.steps_latency(kind, groups, net)


def test_latency_counts_steps():
    net = tiny_cluster(nvlink_alpha=1.0, nic_alpha=10.0)
    groups = np.array([[0, 1, 2, 3]])
    assert K.steps_latency("allreduce", groups, net) == 2 * 1 + 2 * 10
    assert K.steps_latency("allgather", groups, net) == 1 + 10
    assert K.steps_latency("alltoall", groups, net) == 1 + 10
    with pytest.raises(ValueError):
        K.steps_latency("broadcast", groups, net)


def test_stream_uses_pcie(net):
    d = K.stream_demand(np.array([0, 1]), 3.0, net)
    assert d[net.PCIE + 0] == 3 and d[net.PCIE + 1] == 3 and d.sum() == 6
