from __future__ import annotations

import math

import pytest

from topology.hardware import HardwareSpec
from mapping.model_mapping import ModelMapping
from mapping.model_mapping_group import ModelMappingGroup
from network_op import (
    LogicalTransfer,
    NetworkOp,
    P2P,
)
from parallelism import ParallelismConfig
from operators.llama_builder import LlamaModel
from operators.llama_config import LlamaConfig
from network_op import NetworkPhase
from schemes.llama_resharding import (
    MegatronToMegatronP2P,
    llama_total_parameter_bytes,
)
from simulation.simulator import simulate_rlhf_iteration, simulate_stage
from simulation.stages import RLHFTimeline, StageBoundary
from schemes.strategies import MegatronScheme
from topology.topo import Device, HostTopo, Link, Path
from simulation.transition import StageTransition
from workload_context import (
    Workload,
    WorkloadContext,
)


def _dev(host: int) -> Device:
    return Device(host_id=host, gpu_id=0)


# ---- shared fixtures ---------------------------------------------------------

# n_kv=4 (not 2 from Phase 4a's spec) so it divides both tp=2 and tp=4 cleanly.
SMALL_LLAMA = LlamaConfig(
    h=512,
    n_layers=4,
    n_q=8,
    n_kv=4,
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


def _make_topo(num_hosts: int, spine_bw_gbps: float = 100.0, link_latency_ms: float = 0.1) -> HostTopo:
    """All inter-device traffic shares one logical spine link → contention is global."""
    spine_link = Link(node_a_id=-1, node_b_id=-2, bandwidth=spine_bw_gbps, latency=link_latency_ms)
    huge = 1e9

    def _path(src: Device, dst: Device) -> Path:
        attach_src = Link(node_a_id=src, node_b_id=-1, bandwidth=huge, latency=0.0)
        attach_dst = Link(node_a_id=-2, node_b_id=dst, bandwidth=huge, latency=0.0)
        return Path([attach_src, spine_link, attach_dst])

    connections = {
        (_dev(i), _dev(j)): _path(_dev(i), _dev(j))
        for i in range(num_hosts)
        for j in range(num_hosts)
        if i != j
    }
    return HostTopo(device_connections=connections)


def _make_mapping(
    model: LlamaModel,
    parallelism: ParallelismConfig,
    workload: Workload,
    host_ids: list[int],
) -> ModelMapping:
    expected = parallelism.dp * parallelism.pp * parallelism.tp
    if len(host_ids) != expected:
        raise ValueError(f"need {expected} hosts for {parallelism}, got {len(host_ids)}")

    return MegatronScheme(parallelism).build_mm(
        model=model,
        workload=workload,
        devices=[_dev(h) for h in host_ids],
    )


def _ctx(batch: int = 1, prompt_len: int = 64, response_len: int = 0) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.PREPARATION,
        batch_size=batch,
        microbatch_size=batch,
        prompt_len=prompt_len,
        response_len=response_len,
        num_microbatches=1,
    )


# ---- 1. Resharding: P2P, BOUNDARY, total bytes ------------------------------


def test_naive_p2p_resharding_byte_count_matches_full_model():
    model = LlamaModel(
        name="llama_small",
        role="actor",
        architecture=SMALL_LLAMA,
    )
    train_mm = _make_mapping(
        model,
        ParallelismConfig(tp=2, pp=2, dp=2, ep=1),
        Workload.TRAINING,
        host_ids=list(range(8)),
    )
    gen_mm = _make_mapping(
        model,
        ParallelismConfig(tp=4, pp=1, dp=4, ep=1),
        Workload.GENERATION,
        host_ids=list(range(8, 24)),
    )

    transition = StageTransition(source=train_mm, dest=gen_mm)

    assert transition.network_ops, "expected non-empty resharding ops"
    for op in transition.network_ops:
        assert isinstance(op, P2P), f"resharding op should be P2P, got {type(op).__name__}"
        assert NetworkOp.phase_of(op) is NetworkPhase.BOUNDARY

    # Resharding ops bake `total_data_GB` at construction (sizes are known
    # up-front from parameter byte counts). Sum directly.
    total_bytes = sum(op.total_data_GB * 1e9 for op in transition.network_ops)
    expected = llama_total_parameter_bytes(SMALL_LLAMA)
    assert math.isclose(total_bytes, expected, rel_tol=1e-6), (
        f"total resharding bytes {total_bytes:.0f} != full model bytes {expected}"
    )


# ---- 2. RLHFTimeline + simulate_rlhf_iteration --------------------------------


def test_rlhf_iteration_sums_stages_and_boundary():
    model = LlamaModel(
        name="llama_small",
        role="actor",
        architecture=SMALL_LLAMA,
    )
    train_mm = _make_mapping(
        model,
        ParallelismConfig(tp=2, pp=1, dp=1, ep=1),
        Workload.TRAINING,
        host_ids=[0, 1],
    )
    gen_mm = _make_mapping(
        model,
        ParallelismConfig(tp=2, pp=1, dp=1, ep=1),
        Workload.GENERATION,
        host_ids=[2, 3],
    )
    train_sm = ModelMappingGroup(model_mappings=[train_mm])
    gen_sm = ModelMappingGroup(model_mappings=[gen_mm])
    transition = StageTransition(source=train_mm, dest=gen_mm, scheme=MegatronToMegatronP2P())
    timeline = RLHFTimeline(
        stages=[[train_sm], [gen_sm]],
        boundaries=[StageBoundary(transitions=[transition])],
    )
    topo = _make_topo(num_hosts=4, spine_bw_gbps=100.0)

    result = simulate_rlhf_iteration(timeline, _ctx(prompt_len=128), A100, topo)

    assert result.total_time > 0.0
    assert len(result.per_stage_times) == 2
    assert all(t > 0.0 for t in result.per_stage_times), result.per_stage_times
    assert len(result.per_boundary_times) == 1
    assert result.per_boundary_times[0] > 0.0


# ---- 3. simulate_isolated sums sequential models in one group ---------------


def test_simulate_isolated_sums_actor_and_critic_in_same_group():
    actor = LlamaModel(
        name="actor",
        role="actor",
        architecture=SMALL_LLAMA,
    )
    critic = LlamaModel(
        name="critic",
        role="critic",
        architecture=SMALL_LLAMA,
    )
    actor_mm = _make_mapping(actor, ParallelismConfig(tp=2, pp=1, dp=1, ep=1), Workload.PREPARATION, [0, 1])
    critic_mm = _make_mapping(critic, ParallelismConfig(tp=2, pp=1, dp=1, ep=1), Workload.PREPARATION, [0, 1])

    group = ModelMappingGroup(model_mappings=[actor_mm, critic_mm])
    topo = _make_topo(num_hosts=2)
    ctx = _ctx()

    isolated_time = group.simulate_isolated(ctx, A100, topo)

    actor_alone = ModelMappingGroup(model_mappings=[actor_mm])
    critic_alone = ModelMappingGroup(model_mappings=[critic_mm])
    actor_time = actor_alone.simulate_isolated(ctx, A100, topo)
    critic_time = critic_alone.simulate_isolated(ctx, A100, topo)

    # Sequential within the group: total ≥ actor + critic. Actor and critic
    # are on the same hosts and their AR transfers contend in the joint
    # snapshot, so each per-MM component is bigger than its alone-time. The
    # structural property under test is sum-not-max, which still holds.
    assert isolated_time >= actor_time + critic_time
    # And it's not double-counting: it's bounded by 2× each component (rough sanity).
    assert isolated_time <= 2 * (actor_time + critic_time)


# ---- 4. simulate_stage takes max over disjoint groups -----------------------


def test_simulate_stage_takes_max_over_disjoint_groups():
    # Two groups on disjoint hosts. Different host counts → different
    # parallelism shapes → different stage times → max is well-defined.
    big = LlamaModel(
        name="big",
        role="actor",
        architecture=SMALL_LLAMA,
    )
    small = LlamaModel(
        name="small",
        role="critic",
        architecture=LlamaConfig(
            h=256, n_layers=2, n_q=4, n_kv=2, head_size=64, m=1024,
            rope_dim=64, vocab=8000, dtype_bytes=2,
        ),
    )
    big_mm = _make_mapping(big, ParallelismConfig(tp=2, pp=1, dp=1, ep=1), Workload.PREPARATION, [0, 1])
    small_mm = _make_mapping(small, ParallelismConfig(tp=2, pp=1, dp=1, ep=1), Workload.PREPARATION, [2, 3])

    topo = _make_topo(num_hosts=4)
    ctx = _ctx(prompt_len=128)

    big_only = simulate_stage([ModelMappingGroup([big_mm])], ctx, A100, topo)
    small_only = simulate_stage([ModelMappingGroup([small_mm])], ctx, A100, topo)
    both = simulate_stage(
        [
            ModelMappingGroup([big_mm]),
            ModelMappingGroup([small_mm]),
        ],
        ctx, A100, topo,
    )

    # Stage time uses max-not-sum across groups. `_make_topo` puts every
    # host pair on the same shared spine, so disjoint *hosts* don't imply
    # disjoint *links* — every collective (TP-AR, TP-AllGather on the LM
    # head) from big and small contends through the spine. The structural
    # property under test is the lower bound: `both` is at least the slowest
    # group alone. Sanity-cap loose enough to catch gross double-counting.
    assert max(big_only, small_only) <= both, (
        f"simulate_stage(both)={both:.6f} should be at least max(alone)={max(big_only, small_only):.6f}"
    )
    assert both <= 2 * (big_only + small_only), (
        f"simulate_stage(both)={both:.6f} exceeds 2x sum(alone)={2*(big_only + small_only):.6f}"
    )


# ---- 5. Steady-op contention reduces effective bandwidth --------------------


def test_steady_contention_two_concurrent_groups_share_spine():
    # Both groups do TP-AR with the same data_GB through one shared spine.
    # When both run, the AR transfers contend; per-group wall time grows.
    arch = SMALL_LLAMA
    model_a = LlamaModel(name="a", role="actor", architecture=arch)
    model_b = LlamaModel(name="b", role="critic", architecture=arch)

    mm_a = _make_mapping(model_a, ParallelismConfig(tp=2, pp=1, dp=1, ep=1), Workload.PREPARATION, [0, 1])
    mm_b = _make_mapping(model_b, ParallelismConfig(tp=2, pp=1, dp=1, ep=1), Workload.PREPARATION, [2, 3])

    topo = _make_topo(num_hosts=4, spine_bw_gbps=1.0, link_latency_ms=0.0)
    ctx = _ctx(prompt_len=64)

    g_a = ModelMappingGroup([mm_a])
    g_b = ModelMappingGroup([mm_b])

    alone = simulate_stage([g_a], ctx, A100, topo)
    together = simulate_stage([g_a, g_b], ctx, A100, topo)

    # With 2 concurrent groups, per-group AR transfers see ~half the spine
    # bandwidth → AR network time roughly doubles, total stage time (max
    # over groups) grows. Disjoint topology costs would let both finish in
    # `alone` time; shared-spine contention is what we're testing.
    assert together > alone, f"expected together>alone, got together={together:.6f} alone={alone:.6f}"


# ---- 6. Direct topo-level contention sanity (foundation of test 5) ----------


def test_topo_shared_bandwidth_reflects_contention():
    """Direct verification of the contention model the simulator depends on."""
    topo = _make_topo(num_hosts=4, spine_bw_gbps=10.0, link_latency_ms=0.0)
    t1 = LogicalTransfer(src_device=_dev(0), dst_device=_dev(1), data_GB=1.0)
    t2 = LogicalTransfer(src_device=_dev(2), dst_device=_dev(3), data_GB=1.0)

    bw_alone = topo.get_simultaneous_system_shared_bandwidth([t1])
    bw_together = topo.get_simultaneous_system_shared_bandwidth([t1, t2])

    assert bw_together[t1] < bw_alone[t1]
    # With one shared link and two transfers, each gets half the bandwidth.
    assert bw_together[t1] == pytest.approx(bw_alone[t1] / 2)
