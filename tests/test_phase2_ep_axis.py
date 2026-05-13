"""Phase 2: `Shard.ep` as an independent parallelism axis.

Pre-Phase-2 the simulator aliased EP to the TP axis, so configurations
like `TP=2, EP=4` couldn't coexist on the same mesh. This module pins the
new behavior: TP and EP vary independently, and `_shards_in_group` walks
each axis with the *other three* held fixed.

Also pins the Phase 2 axis-divisibility validator: a `ParallelismConfig`
that promises e.g. `ep=4` but is given a shard set that only spans 2 EP
ranks raises a clear error rather than silently shrinking the collective.
"""
from __future__ import annotations

import pytest

from hardware import HardwareSpec
from mesh import Mesh
from model_mapping import ModelMapping
from network_op import AllToAll, AllReduceRing
from network_requirement import (
    CollectiveKind,
    ParallelismConfig,
    ParallelismGroup,
)
from operators.llama_builder import LlamaModel
from operators.llama_config import LlamaConfig
from operators.v3_block import V3Model
from operators.v3_config import V3Config
from shard import Shard
from topo import Device
from workload_context import Workload


def _dev(host: int) -> Device:
    return Device(host_id=host, gpu_id=0)


SMALL_LLAMA = LlamaConfig(
    h=512, n_layers=2, n_q=8, n_kv=4, head_size=64,
    m=2048, rope_dim=64, vocab=32000, dtype_bytes=2,
)


def _llama_model() -> LlamaModel:
    return LlamaModel(
        name="phase2",
        role="actor",
        architecture=SMALL_LLAMA,
    )


def _v3_model() -> V3Model:
    return V3Model(
        name="phase2_v3",
        role="actor",
        architecture=V3Config(),
    )


def test_shard_ep_default_zero_keeps_legacy_constructions_valid():
    """Existing fixtures pass `dp/pp/tp` only; `ep=0` is a clean default."""
    s = Shard(model=_llama_model(), dp=0, pp=0, tp=0)
    assert s.ep == 0


def test_shard_ep_independent_axis():
    """Two shards differing only in `ep` are distinct (frozen-dataclass eq)."""
    m = _llama_model()
    s0 = Shard(model=m, dp=0, pp=0, tp=0, ep=0)
    s1 = Shard(model=m, dp=0, pp=0, tp=0, ep=1)
    assert s0 != s1
    assert hash(s0) != hash(s1)


def test_tp_and_ep_axes_coexist_on_same_mesh():
    """Construct a TP=2, EP=4 mesh. _shards_in_group(TP) returns 2 shards
    along the TP axis (with EP fixed); _shards_in_group(EP) returns 4
    shards along the EP axis (with TP fixed)."""
    model = _v3_model()
    parallelism = ParallelismConfig(tp=2, pp=1, dp=1, ep=4)
    shards: dict[Shard, Device] = {}
    d2s: dict[Device, Shard] = {}
    host = 0
    for ep in range(4):
        for tp in range(2):
            s = Shard(model=model, dp=0, pp=0, tp=tp, ep=ep)
            dev = _dev(host)
            shards[s] = dev
            d2s[dev] = s
            host += 1
    mesh = Mesh(devices=[_dev(i) for i in range(host)])
    mm = ModelMapping(
        model=model, mesh=mesh,
        workload=Workload.PREPARATION,
        parallelism=parallelism,
        shards_to_devices=shards, device_to_shard=d2s,
    )

    tp_group = mm._shards_in_group(ParallelismGroup.TP)
    ep_group = mm._shards_in_group(ParallelismGroup.EP)
    assert len(tp_group) == 2
    assert len(ep_group) == 4

    # All TP-group shards share the reference shard's `ep` (and dp/pp).
    ref_ep = tp_group[0].ep
    assert all(s.ep == ref_ep for s in tp_group)
    # All EP-group shards share the reference shard's `tp`.
    ref_tp = ep_group[0].tp
    assert all(s.tp == ref_tp for s in ep_group)


def test_ep_alltoall_walks_ep_ranks_not_tp_ranks():
    """With TP=2, EP=4 and an MoE workload that emits AllToAll on EP, the
    resulting AllToAll NetworkOp spans 4 ranks (the EP axis), not 2 (the
    TP axis it used to alias). 4-rank AllToAll has 4*3 = 12 transfers."""
    model = _v3_model()
    parallelism = ParallelismConfig(tp=2, pp=1, dp=1, ep=4)
    shards: dict[Shard, Device] = {}
    d2s: dict[Device, Shard] = {}
    host = 0
    for ep in range(4):
        for tp in range(2):
            s = Shard(model=model, dp=0, pp=0, tp=tp, ep=ep)
            dev = _dev(host)
            shards[s] = dev
            d2s[dev] = s
            host += 1
    mesh = Mesh(devices=[_dev(i) for i in range(host)])
    mm = ModelMapping(
        model=model, mesh=mesh,
        workload=Workload.PREPARATION,
        parallelism=parallelism,
        shards_to_devices=shards, device_to_shard=d2s,
    )

    a2a_op = next(
        op for req, op in mm.network_ops.items()
        if req.kind is CollectiveKind.ALL_TO_ALL_DISPATCH
    )
    assert isinstance(a2a_op, AllToAll)
    # 4 EP ranks → 4*3 = 12 directed transfers.
    assert len(a2a_op.endpoint_pairs) == 12


def test_tp_ar_with_ep_active_walks_only_tp_ranks():
    """The TP AR (e.g. after attn_out, ffn_down, shared expert down) must
    span exactly TP ranks even when EP > 1 — i.e., 2 ranks not 8."""
    model = _v3_model()
    parallelism = ParallelismConfig(tp=2, pp=1, dp=1, ep=4)
    shards: dict[Shard, Device] = {}
    d2s: dict[Device, Shard] = {}
    host = 0
    for ep in range(4):
        for tp in range(2):
            s = Shard(model=model, dp=0, pp=0, tp=tp, ep=ep)
            dev = _dev(host)
            shards[s] = dev
            d2s[dev] = s
            host += 1
    mesh = Mesh(devices=[_dev(i) for i in range(host)])
    mm = ModelMapping(
        model=model, mesh=mesh,
        workload=Workload.PREPARATION,
        parallelism=parallelism,
        shards_to_devices=shards, device_to_shard=d2s,
    )

    ar_op = next(
        op for req, op in mm.network_ops.items()
        if req.kind is CollectiveKind.ALL_REDUCE and req.group is ParallelismGroup.TP
    )
    assert isinstance(ar_op, AllReduceRing)
    # Ring of TP=2 → 2 transfers (i→i+1 around the ring).
    assert len(ar_op.endpoint_pairs) == 2


def test_axis_validator_rejects_underprovisioned_ep_layout():
    """parallelism.ep=4 but only 2 EP ranks in the shard layout → ValueError."""
    model = _v3_model()
    parallelism = ParallelismConfig(tp=1, pp=1, dp=1, ep=4)
    # Bad layout: only 2 EP ranks, parallelism promises 4.
    shards: dict[Shard, Device] = {}
    d2s: dict[Device, Shard] = {}
    for ep in range(2):  # under-provisioned
        s = Shard(model=model, dp=0, pp=0, tp=0, ep=ep)
        dev = _dev(ep)
        shards[s] = dev
        d2s[dev] = s
    mesh = Mesh(devices=[_dev(0), _dev(1)])
    with pytest.raises(ValueError, match="EP"):
        ModelMapping(
            model=model, mesh=mesh,
            workload=Workload.PREPARATION,
            parallelism=parallelism,
            shards_to_devices=shards, device_to_shard=d2s,
        )


def test_axis_validator_rejects_underprovisioned_tp_layout():
    model = _llama_model()
    parallelism = ParallelismConfig(tp=4, pp=1, dp=1, ep=1)
    # Only 2 TP shards, parallelism wants 4.
    shards = {Shard(model=model, dp=0, pp=0, tp=t): _dev(t) for t in range(2)}
    d2s = {dev: s for s, dev in shards.items()}
    mesh = Mesh(devices=[_dev(0), _dev(1)])
    with pytest.raises(ValueError, match="TP"):
        ModelMapping(
            model=model, mesh=mesh,
            workload=Workload.PREPARATION,
            parallelism=parallelism,
            shards_to_devices=shards, device_to_shard=d2s,
        )


def test_axis_validator_silent_when_degree_is_one():
    """`parallelism.ep == 1` means no EP collective; no validation needed."""
    model = _llama_model()
    # Construct a single-shard layout with ep=0 (default) and parallelism ep=1.
    s = Shard(model=model, dp=0, pp=0, tp=0)
    dev = _dev(0)
    mesh = Mesh(devices=[dev])
    # Should construct without raising (parallelism degrees all 1).
    ModelMapping(
        model=model, mesh=mesh,
        workload=Workload.PREPARATION,
        parallelism=ParallelismConfig(tp=1, pp=1, dp=1, ep=1),
        shards_to_devices={s: dev}, device_to_shard={dev: s},
    )
