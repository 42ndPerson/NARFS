from __future__ import annotations

import pytest

from hardware import HardwareSpec
from model_mapping import ModelMapping
from network_op import AllReduceRing
from network_requirement import CollectiveKind
from parallelism import ParallelismConfig, ParallelismGroup
from operators.backward import BackwardOp
from operators.llama_builder import LlamaModel
from operators.llama_config import LlamaConfig
from operators.optimizer import AdamOptimizerOp
from strategies import MegatronScheme
from topo import Device
from workload_context import Workload, WorkloadContext


def _dev(host: int) -> Device:
    return Device(host_id=host, gpu_id=0)


SMALL_LLAMA = LlamaConfig(
    h=512,
    n_layers=4,
    n_q=8,
    n_kv=2,
    head_size=64,
    m=2048,
    rope_dim=64,
    vocab=32000,
    flash_block_size=64,
    dtype_bytes=2,
)


A100 = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
)


def _make_model() -> LlamaModel:
    return LlamaModel(
        name="small_llama",
        role="actor",
        architecture=SMALL_LLAMA,
    )


def _make_mapping(workload: Workload, parallelism: ParallelismConfig) -> ModelMapping:
    """Constructs a 1-host-per-shard mesh sized to (dp * pp * tp).

    Shards laid out by (dp, pp, tp) → host_id deterministically.
    """
    model = _make_model()
    num_devs = parallelism.dp * parallelism.pp * parallelism.ep * parallelism.tp
    return MegatronScheme(parallelism).build_mm(
        model=model,
        workload=workload,
        devices=[_dev(i) for i in range(num_devs)],
    )


def _ctx(batch: int = 2, prompt_len: int = 64, response_len: int = 8) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.PREPARATION,
        batch_size=batch,
        microbatch_size=batch,
        prompt_len=prompt_len,
        response_len=response_len,
        num_microbatches=1,
    )


def test_tp2_emits_all_reduce_network_ops_on_correct_shards():
    mm = _make_mapping(Workload.PREPARATION, ParallelismConfig(tp=2, pp=1, dp=1, ep=1))

    # Find the single dedup'd AR requirement actually in the dict.
    ar_keys = [
        k for k in mm.network_ops.keys()
        if k.kind is CollectiveKind.ALL_REDUCE and k.group is ParallelismGroup.TP
    ]
    assert len(ar_keys) == 1, f"expected 1 unique TP-AR requirement, got {len(ar_keys)}"
    ar_op = mm.network_ops[ar_keys[0]]
    assert isinstance(ar_op, AllReduceRing)
    # Ring on 2 TP shards → 2 logical transfers (host 0 → 1 → 0).
    assert len(ar_op.endpoint_pairs) == 2
    # Verify the transfers reference the TP-group shards (host 0 and host 1).
    src_hosts = {src for src, _ in ar_op.endpoint_pairs}
    dst_hosts = {dst for _, dst in ar_op.endpoint_pairs}
    assert src_hosts == {_dev(0), _dev(1)}
    assert dst_hosts == {_dev(0), _dev(1)}

    # Each layer emits 2 AR requirements (attn_out + ffn_down) × 4 layers = 8.
    total_ar_uses = sum(
        1
        for reqs in mm.per_op_network_requirements
        for r in reqs
        if r.kind is CollectiveKind.ALL_REDUCE
    )
    assert total_ar_uses == 2 * SMALL_LLAMA.n_layers


def test_simulate_pp1_returns_positive_total_with_empty_transfer_times():
    mm = _make_mapping(Workload.PREPARATION, ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    result = mm.simulate(_ctx(), A100, op_transfers={}, transfer_times_steady={}, transfer_times_boundary={})
    assert result.total_time > 0.0
    assert len(result.pp_stage_times) == 1


def test_simulate_pp2_partition_and_stage_times():
    mm = _make_mapping(Workload.PREPARATION, ParallelismConfig(tp=2, pp=2, dp=1, ep=1))
    # Pipeline partition: roughly even split across 2 stages.
    assert len(mm.pipeline_partition) == 2
    n_ops = len(mm.operator_pattern)
    sizes = [len(stage) for stage in mm.pipeline_partition]
    assert sum(sizes) == n_ops
    # Stages should differ by at most 1 op (even-by-count partition).
    assert max(sizes) - min(sizes) <= 1

    result = mm.simulate(_ctx(), A100, op_transfers={}, transfer_times_steady={}, transfer_times_boundary={})
    assert len(result.pp_stage_times) == 2
    assert result.total_time > 0.0


def test_training_includes_backward_and_per_layer_optimizer():
    mm = _make_mapping(Workload.TRAINING, ParallelismConfig(tp=2, pp=1, dp=1, ep=1))

    backward_ops = [op for op in mm.operator_pattern if isinstance(op, BackwardOp)]
    assert backward_ops, "expected BackwardOp instances in training pattern"

    # Spot-check 3x scaling on a backward op: pick one and compare to its forward.
    sample = backward_ops[0]
    ctx = _ctx()
    fwd_flops = sample.forward.compute_flops(ctx)
    bwd_flops = sample.compute_flops(ctx)
    if fwd_flops > 0:
        assert bwd_flops == pytest.approx(3.0 * fwd_flops)

    optimizers = [op for op in mm.operator_pattern if isinstance(op, AdamOptimizerOp)]
    # One per LayerTag with non-zero parameters: n_layers numeric tags +
    # boundary tags (first=embed, last=final_norm+lm_head).
    expected_min = SMALL_LLAMA.n_layers
    assert len(optimizers) >= expected_min, (
        f"expected at least {expected_min} per-layer optimizers, got {len(optimizers)}"
    )
    assert all(op.num_parameters > 0 for op in optimizers)


def test_pp_partition_co_locates_forward_and_backward_per_layer():
    """Each layer's forward and backward operators must land in the same stage.

    Even-by-op-count partitioning (the bug this replaces) put forward of
    layer N on stage 0 and backward of layer N on stage 1, which violates
    real PP — backward needs the activations and weights stored locally
    after the corresponding forward.
    """
    mm = _make_mapping(Workload.TRAINING, ParallelismConfig(tp=2, pp=2, dp=1, ep=1))

    # For each op-index, find its layer tag, then its assigned stage.
    op_to_stage: dict[int, int] = {}
    for stage_idx, op_indices in enumerate(mm.pipeline_partition):
        for i in op_indices:
            op_to_stage[i] = stage_idx

    # Group op indices by (layer_index, anchor) tag.
    by_tag: dict[tuple, list[int]] = {}
    for i, tag in enumerate(mm.per_op_layer_tags):
        key = (tag.layer_index, tag.anchor)
        by_tag.setdefault(key, []).append(i)

    # Every op in a tag group must share a stage.
    for key, indices in by_tag.items():
        stages = {op_to_stage[i] for i in indices}
        assert len(stages) == 1, (
            f"layer tag {key} spans multiple stages {stages}; forward and "
            f"backward of the same layer must co-locate"
        )


def test_pp_partition_v3_balances_cost():
    """V3 with 3 dense + 58 MoE layers and PP=4 should produce roughly balanced
    stage costs (per-layer kernel_time at the reference workload). Dense layers
    are much cheaper than MoE; cost-balanced partition naturally clusters
    several dense layers on one stage while spreading MoE across the rest."""
    from operators.v3_block import V3Model
    from operators.v3_config import V3Config

    arch = V3Config()
    model = V3Model(
        name="v3_small",
        role="actor",
        architecture=arch,
    )
    parallelism = ParallelismConfig(tp=1, pp=4, dp=1, ep=1)
    num_devs = parallelism.dp * parallelism.pp * parallelism.ep * parallelism.tp
    mm = MegatronScheme(parallelism).build_mm(
        model=model,
        workload=Workload.PREPARATION,
        devices=[_dev(i) for i in range(num_devs)],
    )

    # Compute per-stage cost at the reference workload the partitioner used.
    # Use the strategy-baked compute_time_fn on each CombinedOp (per-rank).
    from model_mapping import _PARTITION_REF_HW
    ref_ctx = mm._reference_partition_ctx()
    stage_costs = [
        sum(
            mm.forward_seq[i].compute_time_fn(ref_ctx, _PARTITION_REF_HW)
            for i in op_indices
        )
        for op_indices in mm.pipeline_partition
    ]

    assert len(stage_costs) == 4
    # All stages should have non-zero work (no empty stage with 4 stages and
    # 61 layers + boundary ops).
    assert all(c > 0 for c in stage_costs)

    # Balance check: the maximum stage cost should not exceed the minimum by
    # more than ~1.5×. Compare to the even-by-count partition the old code
    # produced, which on V3 had max/min much worse than this.
    max_cost = max(stage_costs)
    min_cost = min(stage_costs)
    assert max_cost / min_cost < 1.5, (
        f"PP partition imbalanced: stage_costs={stage_costs}, max/min={max_cost/min_cost:.2f}"
    )


def test_generation_simulate_returns_prefill_and_decode():
    mm = _make_mapping(Workload.GENERATION, ParallelismConfig(tp=2, pp=1, dp=1, ep=1))

    # Both patterns should be populated.
    assert mm.operator_pattern, "prefill pattern empty"
    assert mm.decode_operator_pattern, "decode pattern empty"

    ctx = _ctx(batch=1, prompt_len=64, response_len=8)
    result = mm.simulate(ctx, A100, op_transfers={}, transfer_times_steady={}, transfer_times_boundary={})

    assert result.prefill_time is not None
    assert result.decode_time is not None
    assert result.prefill_time > 0.0
    assert result.decode_time > 0.0
    assert result.total_time == pytest.approx(result.prefill_time + result.decode_time)
