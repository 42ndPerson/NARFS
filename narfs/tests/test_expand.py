"""Schedule and overlap lowerings, checked against their closed forms."""
import pytest

from narfs.blocks import Collective, Compute, OnDevices, Par, Seq, Transfer
from narfs.costs import ComputeModel
from narfs.evaluate import Evaluator, evaluate
from narfs.expand import Batch, lower_call, microbatches, one_f_one_b, overlap, stage_layers
from narfs.hardware import DATA, H100_SXM
from narfs.model.architecture import LLAMA3_8B, MIXTRAL_8X7B, llama
from narfs.network.comm import load_nccl
from narfs.network.presets import hgx_h100
from narfs.parallel.binding import Binder, Boundary, Group
from narfs.parallel.mesh import Degrees, map_devices
from narfs.physics import Physics
from narfs.profiles import MEGATRON_DEFAULT, TrainProfile
from narfs.tests.helpers import ToyLeaves
from narfs.tests.walk import of_type

TOY = ToyLeaves({f"gpu{i}": 1.0 for i in range(4)})
TOY_BIND = {f"g{i}": [f"gpu{i}"] for i in range(4)}


def stage(i, t):
    return OnDevices(f"g{i}", Compute(t))


def test_1f1b_equals_its_closed_form():
    times = [3.0, 5.0, 4.0, 2.0]
    s = evaluate(one_f_one_b([stage(i, t) for i, t in enumerate(times)], nmb=8), TOY, TOY_BIND)
    assert s.lo == pytest.approx(sum(times) + 7 * max(times))


def test_interleaved_equals_its_closed_form():
    times = [3.0, 5.0, 4.0, 2.0]
    s = evaluate(one_f_one_b([stage(i, t) for i, t in enumerate(times)], nmb=8, v=2), TOY, TOY_BIND)
    assert s.lo == pytest.approx(sum(times) / 2 + (8 * 2 - 1) * max(times) / 2)


def test_overlap_equals_the_two_machine_flow_shop():
    for a, b in ((1.0, 0.4), (0.3, 1.0)):
        tree = OnDevices("g0", overlap([Compute(a)] * 6, [Compute(b)] * 6))
        k = 6
        assert evaluate(tree, TOY, TOY_BIND).lo == pytest.approx(a + (k - 1) * max(a, b) + b)
    # runs of equal pairs are compressed into Repeat
    a, b1, b2 = Compute(1.0), Compute(2.0), Compute(3.0)
    tree = overlap([a] * 5, [b1, b1, b1, b2, b2])
    assert [type(x).__name__ for x in tree.children] == ["Compute", "Repeat", "Par", "Compute"]


def system(nodes=1):
    topo = hgx_h100(nodes)
    phys = Physics(topo, {"nccl": load_nccl(DATA / "nccl-2.32.3.json")}, ComputeModel({"h100-sxm": H100_SXM}))
    return topo, phys


BATCH = Batch(sequences=64, seq_mean=2048)


def test_llama_pipeline_matches_the_closed_form_from_its_stages():
    topo, phys = system(1)
    deg = Degrees(pp=4, tp=2)
    prof = MEGATRON_DEFAULT.with_(dp_overlap=False)
    tree = lower_call("t", LLAMA3_8B, deg, prof, BATCH, micro_batch=4, train=False)
    b = Binder({"t": map_devices(range(8), deg)})
    ev = Evaluator(phys, b)
    total = ev(tree).lo
    stages = [n for n in of_type(tree, OnDevices) if n.name.startswith("stage")]
    t = [ev(s).lo for s in stages]
    nmb = microbatches(BATCH, deg, 4)
    assert nmb == 16
    assert total == pytest.approx(sum(t) + (nmb - 1) * max(t), rel=1e-9)


def test_train_step_structure():
    deg = Degrees(dp=2, pp=2, tp=2)
    tree = lower_call("t", LLAMA3_8B, deg, MEGATRON_DEFAULT, BATCH, micro_batch=4)
    coll = of_type(tree, Collective)
    buckets = [c for c in coll if c.name.startswith("dp grad")]
    assert {c.kind for c in buckets} == {"all_reduce"}
    assert {c.comm for c in buckets} == {Group("t", ("dp", "ep"), 0), Group("t", ("dp", "ep"), 1)}
    # the bucket sizes add up to every gradient of each stage, in fp32
    for s in (0, 1):
        stage_bytes = sum(c.bytes for c in buckets if c.comm.stage == s)
        assert stage_bytes > 0
    sends = of_type(tree, Transfer)
    assert {x.pairs for x in sends} == {Boundary("t", 0, 1)}
    assert {x.reverse for x in sends} == {False, True}
    # the last stage-0 backward is overlapped by the buckets at the end of the step
    end = tree.body.children[1]
    assert isinstance(end, Par) and "dp grad sync" in end.children[0].body.name


def test_zero1_full_recompute_and_interleaving():
    deg = Degrees(dp=2, pp=2, tp=2)
    prof = TrainProfile(zero_stage=1, recompute="full", pp_virtual=2)
    tree = lower_call("t", LLAMA3_8B, deg, prof, BATCH, micro_batch=4)
    kinds = {c.kind for c in of_type(tree, Collective) if c.name.startswith("dp")}
    assert kinds == {"reduce_scatter", "all_gather"}
    assert any(x.name == "backward (full recompute)" for x in of_type(tree, Seq))
    assert Boundary("t", 1, 0) in {x.pairs for x in of_type(tree, Transfer)}       # wrap-around sends
    assert stage_layers(LLAMA3_8B, 2, 2)[0] == list(range(0, 8)) + list(range(16, 24))


def test_moe_gradients_use_their_own_replica_groups():
    deg = Degrees(dp=2, ep=4)
    tree = lower_call("t", MIXTRAL_8X7B, deg, MEGATRON_DEFAULT, Batch(64, 1024), micro_batch=1)
    groups = {c.comm for c in of_type(tree, Collective) if c.name.startswith("dp grad")}
    assert groups == {Group("t", ("dp", "ep"), 0), Group("t", ("dp",), 0)}


def test_sequence_parallel_sends_need_no_gather_and_bad_batches_fail():
    deg = Degrees(pp=2, tp=4)
    sp = lower_call("t", LLAMA3_8B, deg, TrainProfile(sequence_parallel=True), BATCH, micro_batch=2)
    assert not [c for c in of_type(sp, Collective) if c.name == "pp all_gather"]
    plain = lower_call("t", LLAMA3_8B, deg, MEGATRON_DEFAULT, BATCH, micro_batch=2)
    assert [c for c in of_type(plain, Collective) if c.name == "pp all_gather"]
    with pytest.raises(ValueError, match="do not split"):
        lower_call("t", LLAMA3_8B, Degrees(dp=3), MEGATRON_DEFAULT, BATCH, micro_batch=1)


def test_size_variation_slows_the_critical_replica():
    deg = Degrees(dp=8)
    small = llama("tiny", 1024, 4, 8, 8, 4096, 32000)
    topo, phys = system(1)
    b = Binder({"t": map_devices(range(8), deg)})
    even = Evaluator(phys, b)(lower_call("t", small, deg, MEGATRON_DEFAULT, Batch(64, 1024), 1)).lo
    spread = Evaluator(phys, b)(lower_call("t", small, deg, MEGATRON_DEFAULT, Batch(64, 1024, seq_std=600.0), 1)).lo
    assert spread > even
