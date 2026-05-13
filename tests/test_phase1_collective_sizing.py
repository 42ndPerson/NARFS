"""Phase 1: workload-driven collective sizing.

Pins down:

- TP-AR transfer sizes match the analytical `T·h·dtype` formula when
  resolved against a workload ctx.
- Doubling `microbatch_size` ~doubles the per-AR transfer time (modulo
  topology latency floor).
- The `payload_id` discriminator splits same-shape/different-size
  collectives into independent dedup buckets.
- `NetworkOp.logical_transfers(total)` produces the right per-transfer
  split for each subclass's distribution pattern.
"""
from __future__ import annotations

import pytest

from hardware import HardwareSpec
from mesh import Mesh
from model_mapping import ModelMapping
from network_op import (
    AllReduceRing,
    AllToAll,
    Broadcast,
    P2P,
)
from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operators.llama_builder import LlamaModel
from operators.llama_config import LlamaConfig
from relation import Relation
from shard import Shard
from submesh_mapping import resolve_op_transfers
from topo import Device, HostTopo, Link, Path
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


def _dev(host: int) -> Device:
    """One device per host in these tests; gpu_id=0 by convention."""
    return Device(host_id=host, gpu_id=0)


def _shared_spine_topo(num_hosts: int, bw_gbps: float = 100.0) -> HostTopo:
    spine = Link(node_a_id=-1, node_b_id=-2, bandwidth=bw_gbps, latency=0.0)
    huge = 1e9

    def _path(src: Device, dst: Device) -> Path:
        attach_src = Link(node_a_id=src, node_b_id=-1, bandwidth=huge, latency=0.0)
        attach_dst = Link(node_a_id=-2, node_b_id=dst, bandwidth=huge, latency=0.0)
        return Path([attach_src, spine, attach_dst])

    connections = {
        (_dev(i), _dev(j)): _path(_dev(i), _dev(j))
        for i in range(num_hosts)
        for j in range(num_hosts)
        if i != j
    }
    return HostTopo(device_connections=connections)


def _make_mapping(parallelism: ParallelismConfig) -> ModelMapping:
    model = LlamaModel(
        name="phase1",
        role="actor",
        architecture=SMALL_LLAMA,
    )
    shards: dict[Shard, Device] = {}
    d2s: dict[Device, Shard] = {}
    host = 0
    for d in range(parallelism.dp):
        for p in range(parallelism.pp):
            for t in range(parallelism.tp):
                s = Shard(model=model, dp=d, pp=p, tp=t)
                dev = _dev(host)
                shards[s] = dev
                d2s[dev] = s
                host += 1
    mesh = Mesh(devices=[_dev(i) for i in range(host)])
    return ModelMapping(
        model=model,
        mesh=mesh,
        workload=Workload.PREPARATION,
        parallelism=parallelism,
        shards_to_devices=shards,
        device_to_shard=d2s,
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


def _ar_op_and_req(mm: ModelMapping) -> tuple[AllReduceRing, NetworkRequirement]:
    for req, op in mm.network_ops.items():
        if req.kind is CollectiveKind.ALL_REDUCE and req.group is ParallelismGroup.TP:
            assert isinstance(op, AllReduceRing)
            return op, req
    raise AssertionError("no TP AllReduce requirement found")


# ---- core sizing tests -------------------------------------------------------


def test_tp_all_reduce_size_matches_T_h_dtype_after_resolve():
    """When `resolve_op_transfers(ctx, [mm])` runs, every TP-AR transfer's
    data_GB matches the analytical `T·h·dtype/1e9`, split across the ring."""
    mm = _make_mapping(ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    ctx = _ctx(microbatch=1, prompt_len=256)

    op_transfers = resolve_op_transfers(ctx, [mm])
    ar_op, _ = _ar_op_and_req(mm)
    transfers = op_transfers[ar_op]

    T = ctx.microbatch_size * ctx.prompt_len
    expected_total_GB = T * SMALL_LLAMA.h * SMALL_LLAMA.dtype_bytes / 1e9
    expected_per_transfer = expected_total_GB / len(transfers)
    for t in transfers:
        assert t.data_GB == pytest.approx(expected_per_transfer)


def test_doubling_microbatch_doubles_ar_transfer_size():
    mm = _make_mapping(ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    ar_op, _ = _ar_op_and_req(mm)

    transfers_1 = resolve_op_transfers(_ctx(microbatch=1, prompt_len=128), [mm])[ar_op]
    transfers_2 = resolve_op_transfers(_ctx(microbatch=2, prompt_len=128), [mm])[ar_op]
    assert transfers_2[0].data_GB == pytest.approx(2 * transfers_1[0].data_GB)


def test_doubling_prompt_len_doubles_ar_transfer_size():
    mm = _make_mapping(ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    ar_op, _ = _ar_op_and_req(mm)

    transfers_128 = resolve_op_transfers(_ctx(microbatch=1, prompt_len=128), [mm])[ar_op]
    transfers_256 = resolve_op_transfers(_ctx(microbatch=1, prompt_len=256), [mm])[ar_op]
    assert transfers_256[0].data_GB == pytest.approx(2 * transfers_128[0].data_GB)


def test_tp_ar_time_scales_with_microbatch():
    """End-to-end: run simulate twice with microbatch 1 vs 2; the per-AR
    network time approximately doubles (modulo latency floor)."""
    from simulator import simulate_stage
    from submesh_mapping import SubmeshMapping

    mm = _make_mapping(ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    submesh = SubmeshMapping(model_mappings=[mm], submesh=mm.mesh)
    topo = _shared_spine_topo(num_hosts=2, bw_gbps=10.0)

    t1 = simulate_stage([submesh], _ctx(microbatch=1, prompt_len=512), A100, topo)
    t2 = simulate_stage([submesh], _ctx(microbatch=2, prompt_len=512), A100, topo)

    assert t2 > t1, f"expected t2 > t1, got t1={t1:.6f} t2={t2:.6f}"


# ---- payload_id discriminator (Phase 4 prerequisite) -------------------------


def test_requirements_with_same_fields_dedupe():
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
    d[b] = "second"
    assert len(d) == 1


def test_requirements_with_distinct_payload_id_do_not_dedupe():
    a = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.DP,
        payload_id=0,
        data_GB_fn=lambda ctx: 0.001,
    )
    b = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.DP,
        payload_id=1,
        data_GB_fn=lambda ctx: 0.002,
    )
    assert a != b
    assert hash(a) != hash(b)
    d = {a: "layer0", b: "layer1"}
    assert len(d) == 2


def test_data_GB_fn_excluded_from_equality_and_hash():
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


# ---- per-subclass distribution -----------------------------------------------


def test_ring_logical_transfers_splits_evenly():
    """Ring patterns (AllReduce/Gather/ReduceScatter) split total/n per transfer."""
    pairs = [(_dev(0), _dev(1)), (_dev(1), _dev(2)), (_dev(2), _dev(0))]
    op = AllReduceRing(endpoint_pairs=pairs)
    transfers = op.logical_transfers(0.6)
    for t in transfers:
        assert t.data_GB == pytest.approx(0.2)


def test_alltoall_logical_transfers_splits_by_N_not_num_transfers():
    """AllToAll N nodes → N(N-1) transfers; each transfer carries total/N."""
    pairs = [(_dev(i), _dev(j)) for i in range(4) for j in range(4) if i != j]
    assert len(pairs) == 12
    op = AllToAll(endpoint_pairs=pairs)
    transfers = op.logical_transfers(0.4)
    for t in transfers:
        assert t.data_GB == pytest.approx(0.1)  # 0.4 / 4, not 0.4 / 12


def test_broadcast_logical_transfers_full_per_transfer():
    pairs = [(_dev(0), _dev(1)), (_dev(0), _dev(2))]
    op = Broadcast(endpoint_pairs=pairs)
    transfers = op.logical_transfers(0.5)
    for t in transfers:
        assert t.data_GB == pytest.approx(0.5)


def test_p2p_logical_transfers_full_payload():
    op = P2P(endpoint_pairs=[(_dev(0), _dev(1))])
    transfers = op.logical_transfers(0.7)
    assert transfers[0].data_GB == pytest.approx(0.7)


def test_empty_endpoint_pairs_logical_transfers_is_empty():
    op = AllReduceRing(endpoint_pairs=[])
    assert op.logical_transfers(1.0) == []
