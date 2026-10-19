"""Layout layer: degrees, mapping, handles and layout transfers."""
import numpy as np
import pytest

from narfs.blocks import Collective, OnDevices, Seq, Transfer
from narfs.parallel.binding import Binder, Boundary, Devices, Group, Hosts, Pool, Reshard, handles
from narfs.parallel.layouts import ParamLayout, reshard_pairs
from narfs.parallel.mesh import AXES, Degrees, map_devices, stage_split


def test_degrees():
    d = Degrees(dp=2, pp=3, ep=2, tp=4)
    assert d.size == 48 and d.data == 4 and d.of(("dp", "ep")) == 4
    with pytest.raises(ValueError):
        Degrees(tp=0)


def test_group_arrays_with_tp_innermost():
    pl = map_devices(range(16), Degrees(dp=2, pp=2, tp=4))
    assert pl.groups("tp").tolist()[:2] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert pl.groups("tp", stage=1).tolist() == [[4, 5, 6, 7], [12, 13, 14, 15]]
    assert pl.groups("dp", stage=0).tolist() == [[0, 8], [1, 9], [2, 10], [3, 11]]
    assert pl.groups(("pp", "tp")).shape == (2, 8)
    assert pl.stage_devices(1).tolist() == [4, 5, 6, 7, 12, 13, 14, 15]


def test_axis_order_and_irregular_device_lists():
    devs = [5, 9, 2, 30, 7, 11, 1, 0]
    pl = map_devices(devs, Degrees(dp=2, tp=4), order=("tp", "pp", "ep", "cp", "dp"))   # dp fastest
    assert pl.groups("dp").tolist() == [[5, 9], [2, 30], [7, 11], [1, 0]]
    with pytest.raises(ValueError, match="need 8 devices"):
        map_devices(range(7), Degrees(dp=2, tp=4))
    with pytest.raises(ValueError, match="twice"):
        map_devices([0, 0], Degrees(dp=2))


def test_dense_and_expert_groups_with_ep():
    pl = map_devices(range(8), Degrees(dp=2, ep=2, tp=2))
    assert pl.groups(("dp", "ep")).tolist() == [[0, 2, 4, 6], [1, 3, 5, 7]]     # non-expert gradient replicas
    assert pl.groups("dp").tolist() == [[0, 4], [1, 5], [2, 6], [3, 7]]          # expert gradient replicas


def test_stage_split():
    assert [list(r) for r in stage_split(10, 4)] == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
    with pytest.raises(ValueError):
        stage_split(3, 4)


def test_stage_boundaries_pair_the_same_coordinate():
    pl = map_devices(range(12), Degrees(dp=2, pp=3, tp=2))
    b = Binder({"t": pl})
    src, dst, w = b[Boundary("t", 0, 1)]
    assert list(zip(src.tolist(), dst.tolist())) == [(0, 2), (1, 3), (6, 8), (7, 9)]
    src, dst, _ = b[Boundary("t", 2, 0)]                      # interleaved wrap-around
    assert list(zip(src.tolist(), dst.tolist())) == [(4, 0), (5, 1), (10, 6), (11, 7)]


def _received(pairs):
    src, dst, size = pairs
    out = {}
    for d, z in zip(dst.tolist(), size.tolist()):
        out[d] = out.get(d, 0.0) + z
    return out


def test_reshard_identical_layouts_moves_nothing_across_devices():
    lay = ParamLayout(layers=((8.0, 2.0, 0.0),) * 4, embed=4.0, head=4.0)
    pl = map_devices(range(8), Degrees(dp=2, pp=2, tp=2))
    src, dst, size = reshard_pairs(lay, pl, pl)
    assert (src == dst).all()


def test_reshard_bytes_by_hand():
    # 4 layers of (8 tp-sharded, 2 replicated) bytes, embed 4 and head 4 (tp-sharded)
    lay = ParamLayout(layers=((8.0, 2.0, 0.0),) * 4, embed=4.0, head=4.0)
    train = map_devices(range(8), Degrees(dp=2, pp=2, tp=2))      # stage s holds 2 layers
    gen = map_devices(range(8, 12), Degrees(dp=1, tp=4))           # one stage, tp 4
    got = _received(reshard_pairs(lay, train, gen))
    # each generation rank needs a quarter of the tp-sharded bytes and all replicated bytes
    need = (4 * 8.0 + 4.0 + 4.0) / 4 + 4 * 2.0
    assert got == pytest.approx({d: need for d in range(8, 12)})
    src, dst, size = reshard_pairs(lay, train, gen)
    assert set(src.tolist()) <= set(train.dev_of[0].ravel().tolist())     # dst dp 0 reads source replica 0


def test_reshard_experts_over_ep():
    lay = ParamLayout(layers=((0.0, 0.0, 16.0),) * 2)
    a = map_devices(range(4), Degrees(ep=4))
    b = map_devices(range(4, 6), Degrees(ep=2))
    got = _received(reshard_pairs(lay, a, b))
    assert got == pytest.approx({4: 16.0, 5: 16.0})              # half of 2 x 16 expert bytes each


def test_binder_contract_and_hosts():
    pl = map_devices(range(16), Degrees(dp=2, tp=8))
    b = Binder({"gen": pl}, host_ids=lambda d: 1000 + d)
    assert b[Hosts("gen", per="replica")].tolist() == [1000, 1008]
    assert b[Hosts("gen")].size == 16
    assert b[Pool("verifier")] == "verifier"
    assert b[Group("gen", "tp")] is b[Group("gen", ("tp",))]           # one array per handle
    tree = OnDevices(Devices("gen"), Seq((Collective("all_reduce", Group("gen", "tp"), 1.0),
                                          Transfer(Boundary("gen", 0, 0)))))
    assert handles(tree) == {Devices("gen"), Group("gen", ("tp",)), Boundary("gen", 0, 0)}
    b.check(tree)
    with pytest.raises(KeyError, match="no placement"):
        Binder({}).check(tree)
    with pytest.raises(KeyError, match="no PP stage 3"):
        b[Devices("gen", 3)]
    with pytest.raises(ValueError):
        Group("gen", ("xp",))
