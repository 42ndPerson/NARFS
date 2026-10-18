"""Expansion: tree structure per framework profile, DP sync, MoE, generation."""
import pytest

from narfs.blocks import Collective, FlowShop2, GenBatch, OnStage, Op, Pipeline, Repeat, Seq, Stream
from narfs.expand import Workload, expand_forward, expand_generate, expand_train, microbatches
from narfs.model.architecture import LLAMA3_8B
from narfs.model.hf_import import from_hf_config
from narfs.parallel.mesh import Degrees
from narfs.parallel.rules import MEGATRON
from narfs.profiles import (MEGATRON_ASYNC_TP, MEGATRON_DEFAULT, MEGATRON_DISTOPT, VLLM_DEFAULT,
                            FrameworkProfile)
from narfs.tests.walk import iter_blocks, of_type

WL = Workload(64, 512, (128, 512), (0.5, 0.5))

SMALL_MOE = from_hf_config({
    "hidden_size": 1024, "num_attention_heads": 8, "num_key_value_heads": 8, "num_hidden_layers": 4,
    "vocab_size": 32000, "intermediate_size": 4096, "num_local_experts": 8, "num_experts_per_tok": 2,
    "moe_intermediate_size": 1024}, name="small-moe")


def kinds(tree, axis):
    return sorted({c.kind for c in of_type(tree, Collective) if c.axis == axis})


def test_microbatches():
    assert microbatches(WL, Degrees(dp=4), 2) == 8
    with pytest.raises(ValueError):
        microbatches(WL, Degrees(dp=3), 1)


def test_blocking_tp_uses_allreduce():
    tree = expand_train(LLAMA3_8B, Degrees(dp=2, tp=2), MEGATRON, MEGATRON_DEFAULT, WL, 1)
    assert kinds(tree, "tp") == ["allreduce"]
    layer_flowshops = [f for f in of_type(tree, FlowShop2) if "tp comm" in f.name]
    assert not layer_flowshops


def test_overlapped_sp_uses_flowshop_with_rs_and_ag():
    tree = expand_train(LLAMA3_8B, Degrees(dp=2, tp=2), MEGATRON, MEGATRON_ASYNC_TP, WL, 1)
    assert kinds(tree, "tp") == ["allgather", "reducescatter"]
    fs = [f for f in of_type(tree, FlowShop2) if "tp comm" in f.name]
    assert fs and all(len(f.a) == MEGATRON_ASYNC_TP.tp_chunks for f in fs)


def test_no_tp_collectives_at_tp1():
    tree = expand_train(LLAMA3_8B, Degrees(dp=4), MEGATRON, MEGATRON_ASYNC_TP, WL, 1)
    assert kinds(tree, "tp") == []


def test_layers_are_repeated_not_unrolled():
    tree = expand_train(LLAMA3_8B, Degrees(pp=2, tp=2), MEGATRON, MEGATRON_DEFAULT, WL, 1)
    pipe = of_type(tree, Pipeline)[0]
    assert len(pipe.stages) == 2
    reps = of_type(pipe, Repeat)
    assert {r.n for r in reps} == {16}                      # 32 layers over 2 stages, fwd and bwd


def test_dp_sync_occurs_once_and_outside_the_pipeline():
    tree = expand_train(LLAMA3_8B, Degrees(dp=4, pp=2, tp=2), MEGATRON, MEGATRON_DEFAULT, WL, 1)
    pipe = of_type(tree, Pipeline)[0]
    assert not [c for c in of_type(pipe, Collective) if c.axis == "dp"]
    syncs = [b for b in tree.children if isinstance(b, FlowShop2)]
    assert len(syncs) == 1 and not syncs[0].count_a
    assert all(isinstance(a, OnStage) and a.stage == 0 for a in syncs[0].a)


def test_dp_sync_blocking_and_zero():
    prof = MEGATRON_DEFAULT.with_(dp_overlap=False)
    tree = expand_train(LLAMA3_8B, Degrees(dp=4, tp=2), MEGATRON, prof, WL, 1)
    sync = [b for b in tree.children if isinstance(b, Seq) and b.name == "dp grad sync"]
    assert len(sync) == 1 and all(isinstance(r, Repeat) for r in sync[0].children)
    assert kinds(tree, "dp") == ["allreduce"]
    tree = expand_train(LLAMA3_8B, Degrees(dp=4, tp=2), MEGATRON, MEGATRON_DISTOPT, WL, 1)
    assert kinds(tree, "dp") == ["allgather", "reducescatter"]


def test_recompute_policies_change_backward():
    def bwd_ops(profile):
        tree = expand_train(LLAMA3_8B, Degrees(dp=2), MEGATRON, profile, WL, 1)
        return sum(1 for b in iter_blocks(tree) if isinstance(b, Op))
    none, sel, full = (bwd_ops(MEGATRON_DEFAULT.with_(recompute=r)) for r in ("none", "selective", "full"))
    assert none < sel < full


def test_moe_ep_adds_alltoall_and_shrinks_expert_weights():
    def expert_op(tree):
        return next(b for b in iter_blocks(tree) if isinstance(b, Op) and b.name == "experts gate+up")
    t1 = expand_forward(SMALL_MOE, Degrees(dp=2), MEGATRON, MEGATRON_DEFAULT, WL, 1)
    t2 = expand_forward(SMALL_MOE, Degrees(dp=2, ep=2), MEGATRON, MEGATRON_DEFAULT, WL, 1)
    assert kinds(t1, "ep") == [] and kinds(t2, "ep") == ["alltoall"]
    assert expert_op(t2).bytes < expert_op(t1).bytes


def test_ep_ranks_hold_different_data():
    # dp x ep data replicas: ep = 2 halves the microbatches of each rank
    assert microbatches(WL, Degrees(dp=2, ep=2), 1) == 16
    nmb = lambda deg: of_type(expand_forward(SMALL_MOE, deg, MEGATRON, MEGATRON_DEFAULT, WL, 1), Pipeline)[0].nmb
    assert nmb(Degrees(dp=2, ep=2)) == nmb(Degrees(dp=4))


def test_moe_grad_sync_uses_both_replica_groups():
    tree = expand_train(SMALL_MOE, Degrees(dp=2, ep=2), MEGATRON, MEGATRON_DEFAULT, WL, 1)
    sync = next(b for b in tree.children if isinstance(b, FlowShop2))
    assert {c.axis for c in sync.b} == {("dp", "ep"), "dp"}
    dense = expand_train(LLAMA3_8B, Degrees(dp=2), MEGATRON, MEGATRON_DEFAULT, WL, 1)
    assert {c.axis for c in next(b for b in dense.children if isinstance(b, FlowShop2)).b} == {"dp"}


def test_generation_expansion():
    gb = expand_generate(LLAMA3_8B, Degrees(dp=2, tp=4), MEGATRON, VLLM_DEFAULT, WL, 1e6)
    assert isinstance(gb, GenBatch) and gb.replica_requests == 32      # 64 requests over dp = 2
    assert expand_generate(SMALL_MOE, Degrees(dp=2, ep=2), MEGATRON, VLLM_DEFAULT, WL, 1e6).replica_requests == 16
    assert [c.axis for c in gb.comm] == ["tp"] and gb.comm[0].count == 2 * 32
    with pytest.raises(ValueError):
        expand_generate(LLAMA3_8B, Degrees(pp=2), MEGATRON, VLLM_DEFAULT, WL, 1e6)


def test_profile_validation():
    with pytest.raises(ValueError):
        FrameworkProfile("x", zero_stage=3)
    with pytest.raises(ValueError):
        FrameworkProfile("x", tp_comm="sometimes")
