"""Core evaluator rules on hand-computed trees (toy L0, unit device)."""
import pytest

from narfs.blocks import Collective, Compute, OnDevices, Par, Repeat, Seq, Transfer, Work
from narfs.evaluate import Evaluator, evaluate, render
from narfs.tests.helpers import ToyLeaves

# two GPUs, one shared link (capacity 2 units/s), a CPU pool of 4 workers
CAPS = {"gpu0": 1.0, "gpu1": 1.0, "link": 2.0, "pool": 4.0}
BIND = {"g0": ["gpu0"], "g1": ["gpu1"], "g01": ["gpu0", "gpu1"],
        "comm": {"link": 1.0}, "pairs": {"link": 1.0}, "cpu": "pool"}


def run(node, caps=CAPS, bind=BIND):
    return evaluate(node, ToyLeaves(caps), bind)


def on(dev, *children):
    return OnDevices(dev, Seq(children))


def test_seq_adds_and_repeat_scales():
    s = run(on("g0", Compute(1.0), Repeat(3, Compute(2.0)), Repeat(0.5, Compute(4.0))))
    assert s.lo == s.hi == pytest.approx(1 + 6 + 2)


def test_par_in_one_scope_is_max_lo_and_sum_hi():
    s = run(OnDevices("g0", Par((Compute(3.0), Compute(2.0)))))
    assert (s.lo, s.hi) == (3.0, 5.0)


def test_par_of_scopes_on_disjoint_gpus_overlaps_fully():
    s = run(Par((on("g0", Compute(3.0)), on("g1", Compute(2.0)))))
    assert (s.lo, s.hi) == (3.0, 3.0)


def test_par_of_scopes_on_one_gpu_time_shares():
    # the GPU is busy for both scopes: the fluid bound adds their busy time
    s = run(Par((on("g0", Compute(3.0)), on("g0", Compute(2.0)))))
    assert (s.lo, s.hi) == (5.0, 5.0)


def test_shared_link_sets_the_fluid_bound():
    # 4 bytes and 2 bytes on a 2 B/s link: 2 s and 1 s alone, 3 s together
    s = run(Par((Collective("all_reduce", "comm", 4.0), Transfer("pairs", 2.0))))
    assert s.lo == pytest.approx(3.0)


def test_hi_across_scopes_adds_the_smaller_shared_use():
    # scope A: 3 s, 4 bytes on the link (2 s); scope B: 2 s, 4 bytes on the link (2 s)
    a = on("g0", Par((Compute(3.0), Collective("all_reduce", "comm", 4.0))))
    b = on("g1", Par((Compute(2.0), Collective("all_reduce", "comm", 4.0))))
    s = run(Par((a, b)))
    assert s.lo == pytest.approx(4.0)                  # the link: 2 s + 2 s
    # hi: inside a scope nothing overlaps (5 s and 4 s), then max + (sum - max) of the link
    assert s.hi == pytest.approx(5.0 + 2.0)


def test_outer_scope_replaces_inner_busy_time():
    inner = OnDevices("g01", Seq((on("g0", Compute(2.0)), on("g1", Compute(3.0)))))
    s = run(Par((inner, on("g0", Compute(1.0)))))
    # gpu0 is held by the outer scope for 5 s, so the second scope waits: 5 + 1
    assert s.lo == pytest.approx(6.0)


def test_work_on_a_cpu_pool():
    s = run(Work("cpu", 8.0))                          # 8 worker-seconds on 4 workers
    assert s.lo == pytest.approx(2.0)
    s = run(Par((Work("cpu", 8.0), Work("cpu", 4.0))))
    assert s.lo == pytest.approx(3.0)


def test_shared_node_is_computed_once():
    leaves = ToyLeaves(CAPS)
    stage = Compute(1.0)
    tree = OnDevices("g0", Seq((stage, Repeat(5, Par((stage, stage))))))
    s = Evaluator(leaves, BIND)(tree)
    assert s.lo == pytest.approx(6.0)
    assert leaves.compute_calls == 1


def test_compute_outside_a_scope_is_an_error():
    with pytest.raises(ValueError, match="outside any OnDevices"):
        run(Compute(1.0, name="stray"))


def test_bad_nodes_are_rejected():
    with pytest.raises(ValueError):
        Collective("allreduce", "comm", 1.0)
    with pytest.raises(ValueError):
        Repeat(-1, Compute(1.0))


def test_explain_names_the_branch_or_the_shared_resource():
    leaves = ToyLeaves(CAPS)
    ev = Evaluator(leaves, BIND)
    tree = Par((Collective("all_reduce", "comm", 4.0, name="ar"), Transfer("pairs", 2.0, name="send")), name="both")
    rows = ev.explain(tree)
    assert rows[0].label == "both" and rows[0].bound_by == "shared resource: link"
    assert rows[0].hot == "link" and rows[0].util == pytest.approx(1.0)
    tree = OnDevices("g0", Par((Compute(3.0, name="big"), Compute(1.0)), name="p"))
    rows = Evaluator(leaves, BIND).explain(tree)
    assert rows[1].label == "p" and rows[1].bound_by == "branch: big"
    assert "big" in render(rows)
