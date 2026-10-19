"""Topology, router and communication model, checked by hand.

Small fabric: 4 nodes x 2 GPUs (nvl 100 B/s, nic 10 B/s). Rail r of nodes
{0, 1} is leaf r and of nodes {2, 3} is leaf 2 + r:

    node:  0      1      2      3
    gpu:   0 1    2 3    4 5    6 7
    leaf:  0 1    0 1    2 3    2 3
"""
from pathlib import Path

import numpy as np
import pytest

from narfs.blocks import Collective, OnDevices, Par, Transfer, Work
from narfs.evaluate import evaluate
from narfs.network import kernels as K
from narfs.network.comm import EngineLibrary, EnginePath, load_nccl
from narfs.network.presets import hgx_h100, rail_optimized
from narfs.network.topology import Tier, Topology
from narfs.physics import Physics

NCCL = load_nccl(Path(__file__).resolve().parents[1] / "data" / "nccl-2.32.3.json")


def small(**kw):
    args = dict(nvl_rate=100.0, nic_rate=10.0, pcie_rate=5.0, nodes_per_leaf=2, nvls=False,
                nvl_latency=0.0, net_latency=0.0)
    args.update(kw)
    return rail_optimized(4, 2, **args)


def as_dict(topo, pattern):
    ids, val = pattern
    return {topo.resource_name(int(i)): float(v) for i, v in zip(ids, val)}


def routed(topo, flows):
    src, dst, w = flows
    return as_dict(topo, topo.pair_demand(src, dst, w))


def test_routes_by_hand():
    t = small()
    assert as_dict(t, t.pair_demand([0], [1], [1.0])) == {"nvl_tx[gpu 0]": 1.0, "nvl_rx[gpu 1]": 1.0}
    assert as_dict(t, t.pair_demand([0], [2], [1.0])) == {"nic_tx[gpu 0]": 1.0, "nic_rx[gpu 2]": 1.0}
    assert as_dict(t, t.pair_demand([0], [3], [1.0])) == {
        "nic_tx[gpu 0]": 1.0, "nic_rx[gpu 3]": 1.0, "leaf_up[0]": 1.0, "leaf_down[1]": 1.0}
    assert as_dict(t, t.pair_demand([0], [4], [2.0])) == {
        "nic_tx[gpu 0]": 2.0, "nic_rx[gpu 4]": 2.0, "leaf_up[0]": 2.0, "leaf_down[2]": 2.0}
    assert t.locality([0, 0, 0, 0], [1, 2, 3, 4]).tolist() == [0, 1, 2, 2]


def test_unbalanced_tree_with_explicit_keys():
    # three nodes of 2, 3 and 1 GPUs; two leaves with unequal uplinks; a pod tier on top
    t = Topology(gpu_class=("x",) * 6, scaleup=[0, 0, 1, 1, 1, 2], nvl_rate=100.0, nic_rate=10.0,
                 pcie_rate=5.0, tiers=[Tier("leaf", [0, 0, 1, 1, 1, 2], [7.0, 3.0, 5.0]),
                                       Tier("pod", [0, 0, 0, 0, 0, 1], [9.0, 9.0])])
    assert as_dict(t, t.pair_demand([1], [5], [1.0])) == {
        "nic_tx[gpu 1]": 1.0, "nic_rx[gpu 5]": 1.0, "leaf_up[0]": 1.0, "leaf_down[2]": 1.0,
        "pod_up[0]": 1.0, "pod_down[1]": 1.0}
    assert as_dict(t, t.pair_demand([0], [3], [1.0])) == {
        "nic_tx[gpu 0]": 1.0, "nic_rx[gpu 3]": 1.0, "leaf_up[0]": 1.0, "leaf_down[1]": 1.0}
    for groups in (np.array([[0, 1, 2, 3, 4, 5]]), np.array([[1, 4, 5]]), np.array([[0, 2], [1, 3]])):
        for factor in (1.0, 0.5):
            assert as_dict(t, K.ring(t, groups, factor)) == pytest.approx(routed(t, K.ring_flows(t, groups, factor)))


def test_ring_inside_one_node():
    t = small()
    assert as_dict(t, K.ring(t, np.array([[0, 1]]), 1.0)) == {
        "nvl_tx[gpu 0]": 1.0, "nvl_tx[gpu 1]": 1.0, "nvl_rx[gpu 0]": 1.0, "nvl_rx[gpu 1]": 1.0}


def test_ring_over_two_nodes_spreads_nic_bytes_and_stays_on_rails():
    t = small()
    d = as_dict(t, K.ring(t, np.array([[0, 1, 2, 3]]), 1.5))        # all-reduce of 4 ranks: 2(n-1)/n = 1.5
    for g in range(4):
        assert d[f"nvl_tx[gpu {g}]"] == pytest.approx(1.5)
        assert d[f"nic_tx[gpu {g}]"] == pytest.approx(0.75)       # two channels, one per rail
        assert d[f"nic_rx[gpu {g}]"] == pytest.approx(0.75)
    assert not any(k.startswith("leaf") for k in d)


def test_ring_across_leaf_groups_uses_uplinks():
    t = small()
    d = as_dict(t, K.ring(t, np.array([[6, 2, 4, 0]]), 1.5))        # rail 0 of every node, any order
    assert d == pytest.approx({**{f"nic_tx[gpu {g}]": 1.5 for g in (0, 2, 4, 6)},
                               **{f"nic_rx[gpu {g}]": 1.5 for g in (0, 2, 4, 6)},
                               "leaf_up[0]": 1.5, "leaf_down[2]": 1.5, "leaf_up[2]": 1.5, "leaf_down[0]": 1.5})


@pytest.mark.parametrize("groups", [np.array([[0, 1]]), np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
                                    np.array([[0, 2, 4, 6], [1, 3, 5, 7]]), np.array([[0, 1, 2, 4, 5]])])
def test_flows_give_the_patterns(groups):
    t = small()
    for factor in (2.0 * (groups.shape[1] - 1) / groups.shape[1], 0.5):
        assert as_dict(t, K.ring(t, groups, factor)) == pytest.approx(routed(t, K.ring_flows(t, groups, factor)))
    assert as_dict(t, K.spread(t, groups)) == pytest.approx(routed(t, K.spread_flows(t, groups)))
    one = np.array([[0, 1]])
    assert as_dict(t, K.one_shot(t, one)) == pytest.approx(routed(t, K.one_shot_flows(t, one)))
    assert as_dict(t, K.two_shot(t, one)) == pytest.approx(routed(t, K.two_shot_flows(t, one)))


def test_nvls_pattern_by_hand():
    t = rail_optimized(1, 4, nvl_rate=100.0, nic_rate=10.0, pcie_rate=5.0, nvls=True, nvls_level=0.8)
    d = as_dict(t, K.nvls(t, np.array([[0, 1, 2, 3]])))
    assert d == pytest.approx({**{f"nvl_tx[gpu {g}]": 1.25 / 0.8 for g in range(4)},
                               **{f"nvl_rx[gpu {g}]": 1.25 / 0.8 for g in range(4)}})


def physics(topo, **libs):
    return Physics(topo, {"nccl": NCCL, **libs})


def test_nccl_small_message_latency_matches_its_cost_model():
    p = physics(hgx_h100(2))
    t, _, c = p.comm.cost("all_reduce", np.arange(8).reshape(1, 8), 8.0, "nccl")
    assert (c.path, c.proto) == ("ring", "LL")
    assert c.latency == pytest.approx((6.6 + 14 * 0.6) * 1e-6)       # base + 2(n-1) NVLink steps
    t, _, c = p.comm.cost("all_reduce", np.arange(16).reshape(1, 16), 8.0, "nccl")
    # two nodes: 28 intra steps at max(0.6, 1.0 network overhead) + 2 network steps at 2.7
    assert c.latency == pytest.approx((6.6 + 28 * 1.0 + 2 * 2.7) * 1e-6)
    t, _, c = p.comm.cost("all_gather", np.arange(8).reshape(1, 8), 8.0, "nccl")
    assert c.latency == pytest.approx((6.6 + 7 * 0.6) * 1e-6)


def test_nccl_takes_nvls_for_large_all_reduce_in_one_node():
    p = physics(hgx_h100(1))
    g = np.arange(8).reshape(1, 8)
    big = 256 * 2**20
    t, d, c = p.comm.cost("all_reduce", g, big, "nccl")
    assert c.path == "nvls"
    assert t == pytest.approx(25e-6 + big * (1 + 1 / 8) / 0.82 / 375e9)
    t, d, c = p.comm.cost("all_reduce", g, 64 * 2**10, "nccl")
    assert c.path == "ring"
    t, d, c = physics(hgx_h100(1, nvls=False)).comm.cost("all_reduce", g, big, "nccl")
    assert (c.path, c.proto) == ("ring", "Simple")
    assert t == pytest.approx((8.4 + 14 * 3.4) * 1e-6 + big * 1.75 / 375e9)


def test_engine_paths_by_size_and_group():
    vllm = EngineLibrary("vllm", {8: (EnginePath(256 * 2**10, "one_shot", 5e-6),
                                      EnginePath(64 * 2**20, "nvls", 8e-6, inclusive=True))})
    p = physics(hgx_h100(2), vllm=vllm)
    g = np.arange(8).reshape(1, 8)
    t, _, c = p.comm.cost("all_reduce", g, 64 * 2**10, "vllm")
    assert c.path == "vllm one_shot" and t == pytest.approx(5e-6 + 64 * 2**10 * 7 / 375e9)
    assert p.comm.cost("all_reduce", g, 256 * 2**10, "vllm")[2].path == "vllm nvls"
    assert p.comm.cost("all_reduce", g, 64 * 2**20, "vllm")[2].path == "vllm nvls"
    assert p.comm.cost("all_reduce", g, 65 * 2**20, "vllm")[2].path == "nvls"          # NCCL's NVLS
    assert p.comm.cost("all_gather", g, 1024.0, "vllm")[2].path == "ring"
    across = np.array([[0, 1, 2, 3, 8, 9, 10, 11]])                                 # two nodes: NCCL
    assert p.comm.cost("all_reduce", across, 1024.0, "vllm")[2].path == "ring"


def test_units_are_shared_by_content():
    p = physics(small())
    a = p.comm.cost("all_reduce", np.array([[0, 1, 2, 3]]), 100.0, "nccl")[1]
    b = p.comm.cost("all_reduce", np.array([[0, 1, 2, 3]]), 300.0, "nccl")[1]
    assert a.keys() == b.keys()
    assert p.comm.cost("all_reduce", np.array([[5]]), 100.0, "nccl")[:2] == (0.0, {})


def test_leaves_in_the_evaluator():
    topo = small(pools={"verifier": 4.0})
    p = physics(topo)
    bind = {"tp_a": np.array([[0, 1]]), "tp_b": np.array([[2, 3]]), "dp": np.array([[0, 2], [1, 3]]),
            "fwd": (np.array([0]), np.array([2]), np.array([1.0])), "pool": "verifier",
            "devs": np.array([0, 1, 2, 3])}
    # TP all-reduces in two nodes use disjoint NVLink ports: they overlap fully
    one = evaluate(Collective("all_reduce", "tp_a", 100.0), p, bind).lo
    both = evaluate(Par((Collective("all_reduce", "tp_a", 100.0), Collective("all_reduce", "tp_b", 100.0))), p, bind)
    assert both.lo == pytest.approx(one)
    # a send and its reverse use opposite directed resources
    fwd = Transfer("fwd", 50.0)
    pair = Par((fwd, Transfer("fwd", 50.0, reverse=True)))
    assert evaluate(pair, p, bind).lo == pytest.approx(evaluate(fwd, p, bind).lo) == pytest.approx(5.0)
    # two sends the same way share the NIC
    assert evaluate(Par((fwd, Transfer("fwd", 50.0))), p, bind).lo == pytest.approx(10.0)
    assert evaluate(Work("pool", 8.0), p, bind).lo == pytest.approx(2.0)
    s = evaluate(OnDevices("devs", Collective("all_reduce", "dp", 40.0)), p, bind)
    assert s.lo > 0 and s.d_lo
