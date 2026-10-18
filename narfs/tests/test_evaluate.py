"""Evaluator rules on hand-checkable trees (unit device: memory op b = b seconds)."""
import numpy as np
import pytest

from narfs.blocks import (AsyncPipe, Call, Collective, FlowShop2, OnStage, Par, Pipeline, Repeat,
                          Scale, Seq, Transfers)
from narfs.evaluate import evaluate, evaluate_bounds, explain
from narfs.parallel.mesh import Degrees, map_devices
from narfs.tests.helpers import ctx_for, mem, one_device, tiny_cluster


@pytest.fixture
def net():
    return tiny_cluster()


def run(net, block, placement=None, serial=False):
    ctx = ctx_for(net, {"c": placement or one_device()}, serial=serial)
    return evaluate(Call("c", block), ctx)


def xfer(src, dst, size):
    return Transfers((src,), (dst,), (float(size),))


def test_seq_sums(net):
    assert run(net, Seq((mem(1), mem(2), mem(3)))).t == pytest.approx(6)


def test_par_fluid_shared_link(net):
    # each transfer alone takes 1 s on NIC 0; together they share it: 2 s
    both = Par((xfer(0, 2, 10), xfer(0, 2, 10)))
    assert run(net, both).t == pytest.approx(2.0)
    # disjoint NICs and leaves: no contention
    assert run(net, Par((xfer(0, 2, 10), xfer(1, 3, 10)))).t == pytest.approx(1.0)


def test_par_compute_and_comm_are_parallel_constraints(net):
    block = Par((mem(1.5), xfer(0, 2, 10)))
    assert run(net, block).t == pytest.approx(1.5)
    assert run(net, block, serial=True).t == pytest.approx(2.5)


def test_two_way_equals_one_way(net):
    # resources are directed: 0 -> 2 and 2 -> 0 do not share a direction
    assert run(net, Par((xfer(0, 2, 10), xfer(2, 0, 10)))).t == pytest.approx(run(net, xfer(0, 2, 10)).t)


def test_repeat_equals_unroll(net):
    body = Seq((mem(1), xfer(0, 2, 10)))
    a = run(net, Repeat(3, body))
    b = run(net, Seq((body,) * 3))
    assert a.t == pytest.approx(b.t)
    assert np.allclose(a.demand, b.demand)


def test_scale(net):
    s = run(net, Scale(0.25, xfer(0, 2, 10)))
    assert s.t == pytest.approx(0.25)
    assert s.demand[net.NIC_TX] == pytest.approx(2.5)


def test_call_marks_devices_busy(net):
    pl = map_devices([1, 3], Degrees(dp=2))
    s = run(net, mem(2), pl)
    assert s.demand[net.GPU + 1] == 2 and s.demand[net.GPU + 3] == 2 and s.demand[net.GPU + 0] == 0


def chunks(*ts):
    return tuple(mem(t) for t in ts)


def test_flowshop_closed_form(net):
    # a ends at 1, 3, 6; b chunks run 1-3, 3-5, 6-8
    fs = FlowShop2(chunks(1, 2, 3), chunks(2, 2, 2))
    assert run(net, fs).t == pytest.approx(8)
    assert run(net, fs, serial=True).t == pytest.approx(12)


def test_flowshop_exposed(net):
    fs = FlowShop2(chunks(1, 2, 3), chunks(2, 2, 2), count_a=False)
    assert run(net, fs).t == pytest.approx(2)       # only the part of b after a ends


def test_flowshop_buffer(net):
    a, b = chunks(1, 1, 1), chunks(3, 3, 3)
    assert run(net, FlowShop2(a, b)).t == pytest.approx(10)
    assert run(net, FlowShop2(a, b, buffer=3)).t == pytest.approx(10)    # deep buffer = unlimited
    # buffer 1: chunk i of a waits for chunk i-1 of b
    assert run(net, FlowShop2(a, b, buffer=1)).t == pytest.approx(12)


def test_flowshop_needs_equal_chunks(net):
    with pytest.raises(ValueError):
        run(net, FlowShop2(chunks(1), chunks(1, 1)))


def test_pipeline_1f1b_and_interleaved(net):
    pl = map_devices([0, 1], Degrees(pp=2))
    stages = (mem(2), mem(3))
    # T = sum t / v + (nmb - 1/v) * slot
    assert run(net, Pipeline(stages, nmb=4), pl).t == pytest.approx(5 + 3 * 3)
    assert run(net, Pipeline(stages, nmb=4, virtual=2), pl).t == pytest.approx(2.5 + 3.5 * 3)


def test_pipeline_stage_collectives_are_stage_scoped(net):
    # pp = 2 over the two nodes, tp = 2 inside each node
    pl = map_devices([0, 1, 2, 3], Degrees(pp=2, tp=2))
    ar = Collective("allreduce", "tp", 10.0)
    t1 = run(net, ar, map_devices([0, 1], Degrees(tp=2))).t
    assert t1 == pytest.approx(0.1)
    pipe = run(net, Pipeline((ar, ar), nmb=3), pl)
    assert pipe.t == pytest.approx(2 * t1 + 2 * t1)              # sum t + (nmb - 1) * slot, slot = t1
    s0 = run(net, OnStage(0, ar), pl)
    assert s0.t == pytest.approx(t1)
    assert s0.demand[net.NVL_TX + 2] == 0                        # stage 1 devices untouched


def test_async_pipe(net):
    assert run(net, AsyncPipe(mem(3), mem(5))).t == pytest.approx(5)
    assert run(net, AsyncPipe(mem(3), mem(5)), serial=True).t == pytest.approx(8)


def test_bounds_and_explain(net):
    tree = Call("c", Par((mem(1), xfer(0, 2, 10), Seq((mem(1), mem(1))))))
    b = evaluate_bounds(tree, ctx_for(net, {"c": one_device()}))
    assert b.lo == pytest.approx(2) and b.hi == pytest.approx(4)
    rows = explain(tree, ctx_for(net, {"c": one_device()}))
    assert rows[-1][0] == "Call"
    assert any(r[0] == "Transfers" and r[3].startswith("nic_tx") for r in rows)


def test_hot_link_in_irregular_placement():
    # 4:1 oversubscribed spine: 10 B/s
    net = tiny_cluster(spine_oversub=4.0)
    ar = Collective("allreduce", "dp", 10.0, name="dp ar")
    regular = map_devices([0, 2, 1, 3], Degrees(dp=2, tp=2), order=("tp", "pp", "ep", "cp", "dp"))
    irregular = map_devices([0, 3, 1, 2], Degrees(dp=2, tp=2), order=("tp", "pp", "ep", "cp", "dp"))
    t_reg = run(net, ar, regular).t
    t_irr = run(net, ar, irregular).t
    assert t_reg == pytest.approx(1.0) and t_irr == pytest.approx(4.0)
    rows = explain(Call("c", ar), ctx_for(net, {"c": irregular}))
    assert rows[0][1] == "dp ar" and rows[0][3] == "spine"


def test_op_outside_call_raises(net):
    with pytest.raises(ValueError):
        evaluate(mem(1), ctx_for(net, {}))
