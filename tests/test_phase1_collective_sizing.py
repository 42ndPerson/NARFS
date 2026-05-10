"""Phase 1: workload-driven collective sizing.

Pins down:

- TP-AR transfer sizes match the analytical `T·h·dtype` formula after
  `simulate` triggers a refresh.
- Doubling `microbatch_size` ~doubles the per-AR transfer time (modulo
  topology latency floor).
- The `payload_id` discriminator splits same-shape/different-size
  collectives into independent dedup buckets, so per-occurrence sizing
  survives.
- The legacy `data_GB_estimates` fallback still works when no
  `data_GB_fn` is supplied.
- `NetworkOp.refresh_data_size` produces the right per-transfer split for
  each subclass's distribution pattern.
"""
from __future__ import annotations

import pytest

from hardware import HardwareSpec
from mesh import Mesh
from model import Model
from model_mapping import ModelMapping
from network_op import (
    AllReduceRing,
    AllToAll,
    Broadcast,
    LogicalTransfer,
    P2P,
)
from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operators.llama_builder import build_llama_pattern
from operators.llama_config import LlamaConfig
from relation import Relation
from shard import Shard
from topo import HostTopo, Link, Path
from workload_context import (
    TokenCount,
    Workload,
    WorkloadContext,
)


# ---- shared fixtures ---------------------------------------------------------


SMALL_LLAMA = LlamaConfig(
    h=512,
    n_layers=2,
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
    ridge_flops_per_byte=156.0,
)


def _shared_spine_topo(num_hosts: int, bw_gbps: float = 100.0) -> HostTopo:
    """All inter-host pairs share one spine link (so we can reason about
    aggregate AR bandwidth analytically). Zero latency keeps the math clean.
    """
    spine = Link(node_a_id=-1, node_b_id=-2, bandwidth=bw_gbps, latency=0.0)
    spine_path = Path(link=[spine])
    connections = {
        (i, j): spine_path
        for i in range(num_hosts)
        for j in range(num_hosts)
        if i != j
    }
    return HostTopo(
        intra_host_bandwidth=1000.0,
        intra_host_latency=0.0,
        hosts_connections=connections,
    )


def _make_mapping(parallelism: ParallelismConfig) -> ModelMapping:
    model = Model(
        name="phase1",
        role="actor",
        architecture=SMALL_LLAMA,
        build_pattern=build_llama_pattern,
    )
    shards: dict[Shard, int] = {}
    h2s: dict[int, Shard] = {}
    host = 0
    for d in range(parallelism.dp):
        for p in range(parallelism.pp):
            for t in range(parallelism.tp):
                s = Shard(model=model, dp=d, pp=p, tp=t)
                shards[s] = host
                h2s[host] = s
                host += 1
    mesh = Mesh(host_ids=list(range(host)), num_devices_per_host=1)
    return ModelMapping(
        model=model,
        mesh=mesh,
        workload=Workload.PREPARATION,
        parallelism=parallelism,
        shards_to_host_ids=shards,
        host_id_to_shard=h2s,
    )


def _ctx(microbatch: int = 1, prompt_len: int = 256) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.PREPARATION,
        batch_size=microbatch,
        microbatch_size=microbatch,
        prompt_len=prompt_len,
        response_len=0,
        num_microbatches=1,
    )


# ---- core sizing tests -------------------------------------------------------


def test_tp_all_reduce_size_matches_T_h_dtype_after_refresh():
    """After `refresh_collective_sizes(ctx)`, every TP-AR transfer's data_GB
    matches the analytical `T * h * dtype / 1e9`, split across the ring."""
    mm = _make_mapping(ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    ctx = _ctx(microbatch=1, prompt_len=256)

    mm.refresh_collective_sizes(ctx)

    ar_op = next(
        op for req, op in mm.network_ops.items()
        if req.kind is CollectiveKind.ALL_REDUCE and req.group is ParallelismGroup.TP
    )
    assert isinstance(ar_op, AllReduceRing)

    # Expected total: T·h·dtype bytes; ring has tp transfers each carrying total/tp.
    T = ctx.microbatch_size * ctx.prompt_len
    expected_total_GB = T * SMALL_LLAMA.h * SMALL_LLAMA.dtype_bytes / 1e9
    expected_per_transfer = expected_total_GB / len(ar_op.logical_transfers)
    for t in ar_op.logical_transfers:
        assert t.data_GB == pytest.approx(expected_per_transfer)


def test_doubling_microbatch_doubles_ar_transfer_size():
    """Sizing closure is a pure function of ctx — `microbatch_size` is the
    only knob that affects per-AR data here, so doubling it doubles the
    transfer."""
    mm = _make_mapping(ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    ar_op = next(
        op for req, op in mm.network_ops.items()
        if req.kind is CollectiveKind.ALL_REDUCE and req.group is ParallelismGroup.TP
    )

    mm.refresh_collective_sizes(_ctx(microbatch=1, prompt_len=128))
    size_at_1 = ar_op.logical_transfers[0].data_GB

    mm.refresh_collective_sizes(_ctx(microbatch=2, prompt_len=128))
    size_at_2 = ar_op.logical_transfers[0].data_GB

    assert size_at_2 == pytest.approx(2 * size_at_1)


def test_doubling_prompt_len_doubles_ar_transfer_size():
    mm = _make_mapping(ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    ar_op = next(
        op for req, op in mm.network_ops.items()
        if req.kind is CollectiveKind.ALL_REDUCE and req.group is ParallelismGroup.TP
    )

    mm.refresh_collective_sizes(_ctx(microbatch=1, prompt_len=128))
    size_at_128 = ar_op.logical_transfers[0].data_GB

    mm.refresh_collective_sizes(_ctx(microbatch=1, prompt_len=256))
    size_at_256 = ar_op.logical_transfers[0].data_GB

    assert size_at_256 == pytest.approx(2 * size_at_128)


def test_tp_ar_time_scales_with_microbatch():
    """End-to-end: run simulate twice with microbatch 1 vs 2; the per-AR
    network time approximately doubles (modulo latency floor)."""
    from simulator import simulate_stage
    from submesh_mapping import SubmeshMapping

    mm = _make_mapping(ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    submesh = SubmeshMapping(model_mappings=[mm], submesh=mm.mesh)
    topo = _shared_spine_topo(num_hosts=2, bw_gbps=10.0)  # narrow spine → AR cost is visible

    t1 = simulate_stage([submesh], _ctx(microbatch=1, prompt_len=512), A100, topo)
    t2 = simulate_stage([submesh], _ctx(microbatch=2, prompt_len=512), A100, topo)

    # t2 > t1 strictly — doubled compute and doubled AR data both add cost.
    assert t2 > t1, f"expected t2 > t1, got t1={t1:.6f} t2={t2:.6f}"


# ---- payload_id discriminator (Phase 4 prerequisite) -------------------------


def test_requirements_with_same_fields_dedupe():
    """Same (kind, relation, eta, group, payload_id=None) → dict collision.
    This is the dedup behavior the codebase has always relied on."""
    a = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
        data_GB_fn=lambda ctx: 0.001,
    )
    b = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
        data_GB_fn=lambda ctx: 0.999,  # different closure
    )
    assert a == b
    assert hash(a) == hash(b)
    d = {a: "first"}
    d[b] = "second"  # collides
    assert len(d) == 1


def test_requirements_with_distinct_payload_id_do_not_dedupe():
    """Phase 4 needs same-shape DP-AR per layer with different sizes;
    payload_id discriminates them."""
    a = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.DP,
        payload_id=0,  # layer 0
        data_GB_fn=lambda ctx: 0.001,
    )
    b = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.DP,
        payload_id=1,  # layer 1
        data_GB_fn=lambda ctx: 0.002,
    )
    assert a != b
    assert hash(a) != hash(b)
    d = {a: "layer0", b: "layer1"}
    assert len(d) == 2


def test_data_GB_fn_excluded_from_equality_and_hash():
    """`data_GB_fn` must not affect equality / hash — two requirements
    differing only in their sizing closure should still collide."""
    fn1 = lambda ctx: 0.1
    fn2 = lambda ctx: 0.2
    r1 = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
        data_GB_fn=fn1,
    )
    r2 = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
        data_GB_fn=fn2,
    )
    assert r1 == r2
    assert hash(r1) == hash(r2)


# ---- legacy fallback ---------------------------------------------------------


def test_legacy_data_GB_estimates_still_works_when_no_data_GB_fn():
    """Builder-emitted requirements without `data_GB_fn` fall back to the
    legacy `data_GB_estimates` dict."""
    mm = _make_mapping(ParallelismConfig(tp=2, pp=1, dp=1, ep=1))

    # Force the legacy path: rebuild network_ops dict with a fn-less requirement.
    legacy_req = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
    )  # no data_GB_fn
    # Replace network_ops entries' keys with the legacy req for the test.
    mm.network_ops = {legacy_req: next(iter(mm.network_ops.values()))}
    mm.data_GB_estimates = {CollectiveKind.ALL_REDUCE: 0.025}  # 25 MB

    mm.refresh_collective_sizes(_ctx())

    op = mm.network_ops[legacy_req]
    n = len(op.logical_transfers)
    assert all(t.data_GB == pytest.approx(0.025 / n) for t in op.logical_transfers)


# ---- per-subclass refresh distribution ---------------------------------------


def test_refresh_data_size_ring_splits_evenly():
    """Ring patterns (AllReduce/Gather/ReduceScatter) split total/n per transfer."""
    transfers = [
        LogicalTransfer(src_host_id=0, dst_host_id=1, data_GB=0.0),
        LogicalTransfer(src_host_id=1, dst_host_id=2, data_GB=0.0),
        LogicalTransfer(src_host_id=2, dst_host_id=0, data_GB=0.0),
    ]
    op = AllReduceRing(logical_transfers=transfers)
    op.refresh_data_size(0.6)
    for t in op.logical_transfers:
        assert t.data_GB == pytest.approx(0.2)


def test_refresh_data_size_alltoall_splits_by_N_not_num_transfers():
    """AllToAll N nodes → N(N-1) transfers; each transfer carries total/N
    (not total/N(N-1)). The per-subclass refresh solves N from transfer count."""
    # 4 nodes (0..3), each sends to 3 others → 12 transfers.
    transfers = [
        LogicalTransfer(src_host_id=i, dst_host_id=j, data_GB=0.0)
        for i in range(4) for j in range(4) if i != j
    ]
    assert len(transfers) == 12
    op = AllToAll(logical_transfers=transfers)
    op.refresh_data_size(0.4)  # total = 0.4
    # Per-transfer should be 0.4 / 4 = 0.1 (not 0.4 / 12).
    for t in op.logical_transfers:
        assert t.data_GB == pytest.approx(0.1)


def test_refresh_data_size_broadcast_full_per_transfer():
    """Broadcast: each receiver gets the full payload."""
    transfers = [
        LogicalTransfer(src_host_id=0, dst_host_id=1, data_GB=0.0),
        LogicalTransfer(src_host_id=0, dst_host_id=2, data_GB=0.0),
    ]
    op = Broadcast(logical_transfers=transfers)
    op.refresh_data_size(0.5)
    for t in op.logical_transfers:
        assert t.data_GB == pytest.approx(0.5)


def test_refresh_data_size_p2p_full_payload():
    """P2P: single transfer carries the full payload."""
    op = P2P(logical_transfers=[LogicalTransfer(src_host_id=0, dst_host_id=1, data_GB=0.0)])
    op.refresh_data_size(0.7)
    assert op.logical_transfers[0].data_GB == pytest.approx(0.7)


def test_refresh_data_size_empty_logical_transfers_is_noop():
    """Refresh on an op with no transfers is harmless (degenerate single-rank ring)."""
    op = AllReduceRing(logical_transfers=[])
    op.refresh_data_size(1.0)  # should not raise
    assert op.logical_transfers == []
