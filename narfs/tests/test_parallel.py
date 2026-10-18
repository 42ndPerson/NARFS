"""Degrees, placements, sharding rules."""
import numpy as np
import pytest

from narfs.model.architecture import DEEPSEEK_V3, LLAMA3_8B, LLAMA3_70B
from narfs.parallel.mesh import Degrees, Placement, map_devices
from narfs.parallel.rules import MEGATRON, local_dims, stage_layers


def test_degrees():
    d = Degrees(dp=2, pp=3, tp=4)
    assert d.size == 24 and d.shape() == (2, 3, 1, 1, 4) and d.of("pp") == 3
    with pytest.raises(ValueError):
        Degrees(tp=0)


def test_map_devices_default_order_puts_tp_innermost():
    pl = map_devices(range(8), Degrees(dp=2, tp=4))
    assert pl.groups("tp").tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert pl.groups("dp").tolist() == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_map_devices_other_order_and_irregular_list():
    pl = map_devices([7, 2, 5, 0], Degrees(dp=2, tp=2), order=("tp", "pp", "ep", "cp", "dp"))
    assert pl.groups("dp").tolist() == [[7, 2], [5, 0]]
    assert pl.groups("tp").tolist() == [[7, 5], [2, 0]]


def test_stage_groups_and_pairs():
    pl = map_devices(range(8), Degrees(dp=2, pp=2, tp=2))
    assert pl.groups("tp", stage=1).tolist() == [[2, 3], [6, 7]]
    assert pl.stage_devices(0).tolist() == [0, 1, 4, 5]
    src, dst = pl.pp_pairs()
    assert src.tolist() == [0, 1, 4, 5] and dst.tolist() == [2, 3, 6, 7]


def test_multi_axis_groups():
    pl = map_devices(range(8), Degrees(dp=2, ep=2, tp=2))
    assert pl.groups(("dp", "ep")).tolist() == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert Degrees(dp=2, ep=3).of(("dp", "ep")) == 6 == Degrees(dp=2, ep=3).data


def test_placement_validation():
    with pytest.raises(ValueError):
        Placement(Degrees(tp=2), np.array([0, 0]).reshape(1, 1, 1, 1, 2))
    with pytest.raises(ValueError):
        map_devices(range(3), Degrees(tp=2))


def test_local_dims_divisibility_and_kv_replication():
    local = local_dims(LLAMA3_70B, Degrees(tp=8), MEGATRON)
    assert local["heads"] == 8 and local["kv_heads"] == 1 and local["ffn"] == 3584
    assert local["hidden"] == 8192                                     # not sharded
    local = local_dims(LLAMA3_8B, Degrees(tp=16), MEGATRON)
    assert local["kv_heads"] == 1                                       # 8 kv heads < tp: replicated
    with pytest.raises(ValueError):
        local_dims(LLAMA3_8B, Degrees(tp=3), MEGATRON)
    with pytest.raises(ValueError):
        local_dims(LLAMA3_8B, Degrees(pp=64), MEGATRON)
    assert local_dims(DEEPSEEK_V3, Degrees(ep=8), MEGATRON)["experts"] == 32


def test_stage_layers_even_split():
    stages = stage_layers(DEEPSEEK_V3, 4)
    assert [len(s) for s in stages] == [16, 15, 15, 15]
    assert stages[0][0].ffn == "dense" and stages[-1][-1].ffn == "moe"
