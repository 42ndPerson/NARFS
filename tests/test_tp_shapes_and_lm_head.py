"""Strategy-applied TP sharding + LM-head AllGather.

Operators are parallelism-pure: they carry no sharding role and expose no
role-aware shape helpers. The `ParallelismScheme` stores its sharding
decisions externally on `OperatorWithReqs` — `compute_time_fn` (the
per-rank kernel-time closure) and `per_rank_parameter_bytes` (the per-rank
weight footprint). These tests pin the *outcome* of sharding — that the
per-rank footprint and kernel time at TP=4 are 1/4 of TP=1 for sharded
operators — and pin the LM-head AllGather requirement.

Direction-of-shard distinctions (column vs row vs heads) are not visible
at the operator level; that's the acknowledged ±30% approximation in the
per-rank baking (per-rank tile-shape asymmetry isn't modeled).
"""
from __future__ import annotations

import pytest

from topology.hardware import HardwareSpec
from network_requirement import CollectiveKind, NetworkRequirement
from parallelism import ParallelismConfig, ParallelismGroup
from operators.attention import DecodeAttention
from operators.gemm import Gemm
from operators.llama_builder import LlamaModel
from operators.llama_config import LlamaConfig
from operators.pattern import LabeledOp, expand_tagged_pattern
from operators.v3_block import V3Model
from operators.v3_config import V3Config
from schemes.strategies import MegatronScheme
from topology.topo import Device
from workload_context import (
    Workload,
    WorkloadContext,
)


SMALL_LLAMA = LlamaConfig(
    h=512, n_layers=2, n_q=8, n_kv=4, head_size=64,
    m=2048, rope_dim=64, vocab=32000, dtype_bytes=2,
)


_HW = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
)


def _dev(i: int) -> Device:
    return Device(host_id=i, gpu_id=0)


def _ctx(prompt_len: int = 128) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.GENERATION,
        batch_size=2, microbatch_size=2,
        prompt_len=prompt_len, response_len=0, num_microbatches=1,
    )


def _llama_entries(par: ParallelismConfig, phase: str = "prefill"):
    model = LlamaModel(name="t", role="actor", architecture=SMALL_LLAMA)
    pattern = MegatronScheme(par).decorated_pattern(model, phase)
    return [e for e in expand_tagged_pattern(pattern) if not isinstance(e, LabeledOp)]


def _v3_entries(cfg: V3Config, par: ParallelismConfig, phase: str = "prefill"):
    model = V3Model(name="t", role="actor", architecture=cfg)
    pattern = MegatronScheme(par).decorated_pattern(model, phase)
    return [e for e in expand_tagged_pattern(pattern) if not isinstance(e, LabeledOp)]


# ---- MLA TP-sharding (now via strategy, observed through per_rank_parameter_bytes) -----


def test_mla_q_up_kv_up_attn_out_shrink_under_tp():
    """q_up / kv_up (COL) and attn_out (ROW) all shrink per-rank param bytes
    by tp under the strategy's role assignment."""
    cfg = V3Config()
    tp = 4
    entries = _v3_entries(cfg, ParallelismConfig(tp=tp))

    expected_q_up_n = cfg.n_h * (cfg.head_size + cfg.d_rope)
    q_up_entry = next(
        e for e in entries
        if isinstance(e.operator, Gemm) and e.operator.n == expected_q_up_n
    )
    assert q_up_entry.per_rank_parameter_bytes == q_up_entry.operator.parameter_bytes() // tp

    expected_kv_up_n = cfg.n_h * 2 * cfg.head_size
    kv_up_entry = next(
        e for e in entries
        if isinstance(e.operator, Gemm) and e.operator.n == expected_kv_up_n
    )
    assert kv_up_entry.per_rank_parameter_bytes == kv_up_entry.operator.parameter_bytes() // tp

    expected_attn_out_k = cfg.n_h * cfg.head_size
    attn_out_entry = next(
        e for e in entries
        if isinstance(e.operator, Gemm)
        and e.operator.k == expected_attn_out_k and e.operator.n == cfg.h
    )
    assert attn_out_entry.per_rank_parameter_bytes == attn_out_entry.operator.parameter_bytes() // tp


def test_mla_attention_per_rank_kernel_time_shrinks_under_tp():
    """HEADS_PARALLEL attention: per-rank kernel time at tp=2 is half of tp=1."""
    cfg = V3Config()
    ctx = _ctx(prompt_len=128)

    tp1 = next(
        e for e in _v3_entries(cfg, ParallelismConfig(tp=1), phase="decode")
        if isinstance(e.operator, DecodeAttention)
    )
    tp2 = next(
        e for e in _v3_entries(cfg, ParallelismConfig(tp=2), phase="decode")
        if isinstance(e.operator, DecodeAttention)
    )
    assert tp2.operator.num_heads == cfg.n_h  # operator stores full count
    assert tp2.compute_time_fn(ctx, _HW) == pytest.approx(tp1.compute_time_fn(ctx, _HW) / 2)


def test_strategy_rejects_nondivisible_n_h():
    """Strategy validates n_h % tp at build_mm time."""
    cfg = V3Config()  # n_h=128
    par = ParallelismConfig(tp=5)
    model = V3Model(name="t", role="actor", architecture=cfg)
    with pytest.raises(ValueError, match="n_h"):
        MegatronScheme(par).build_mm(
            model=model,
            workload=Workload.PREPARATION,
            devices=[_dev(i) for i in range(5)],
        )


# ---- V3 dense FFN intermediate -----------------------------------------------


def test_v3_dense_ffn_intermediate_shrinks_under_tp():
    cfg = V3Config()
    tp = 4
    entries = _v3_entries(cfg, ParallelismConfig(tp=tp))
    ffn_down_entry = next(
        e for e in entries
        if isinstance(e.operator, Gemm)
        and e.operator.n == cfg.h and e.operator.k == cfg.m_dense
    )
    assert ffn_down_entry.per_rank_parameter_bytes == ffn_down_entry.operator.parameter_bytes() // tp


def test_strategy_rejects_nondivisible_m_dense():
    cfg = V3Config(m_dense=99)
    par = ParallelismConfig(tp=2)
    model = V3Model(name="t", role="actor", architecture=cfg)
    with pytest.raises(ValueError, match="m_dense"):
        MegatronScheme(par).build_mm(
            model=model,
            workload=Workload.PREPARATION,
            devices=[_dev(i) for i in range(2)],
        )


# ---- V3 MoE shared expert ----------------------------------------------------


def test_v3_moe_shared_expert_shrinks_under_tp_routed_ep_only():
    """Routed experts shard along EP only — TP role is REPLICATED so per-rank
    parameter bytes do NOT divide by tp. Shared experts use COL/ROW under TP
    so they DO shrink by tp.
    """
    cfg = V3Config()
    tp = 4
    par = ParallelismConfig(tp=tp)
    entries = _v3_entries(cfg, par)

    routed = [
        e for e in entries
        if isinstance(e.operator, Gemm) and e.operator.n_replicas == cfg.n_routed_experts
    ]
    for r in routed:
        # No TP sharding on routed (EP only, ep=1 here means no shrink at all).
        assert r.per_rank_parameter_bytes == r.operator.parameter_bytes()

    def _is_shared_expert(op: Gemm) -> bool:
        if op.n_replicas != 1:
            return False
        return (op.n == cfg.m_expert and op.k == cfg.h) or (op.n == cfg.h and op.k == cfg.m_expert)

    shared = [
        e for e in entries
        if isinstance(e.operator, Gemm) and _is_shared_expert(e.operator)
    ]
    n_moe_layers = cfg.n_layers - cfg.n_dense_layers
    assert len(shared) == 3 * n_moe_layers  # gate, up, down per layer
    for s in shared:
        assert s.per_rank_parameter_bytes == s.operator.parameter_bytes() // tp


def test_strategy_rejects_nondivisible_m_expert():
    cfg = V3Config(m_expert=99)
    par = ParallelismConfig(tp=2)
    model = V3Model(name="t", role="actor", architecture=cfg)
    with pytest.raises(ValueError, match="m_expert"):
        MegatronScheme(par).build_mm(
            model=model,
            workload=Workload.PREPARATION,
            devices=[_dev(i) for i in range(2)],
        )


# ---- LM head TP-shard + AllGather --------------------------------------------


def test_v3_lm_head_shrinks_and_emits_all_gather_under_tp():
    cfg = V3Config()
    tp = 4
    entries = _v3_entries(cfg, ParallelismConfig(tp=tp))
    lm_head_entry = entries[-1]
    op = lm_head_entry.operator
    assert isinstance(op, Gemm)
    assert op.n == cfg.vocab
    assert op.k == cfg.h
    assert lm_head_entry.per_rank_parameter_bytes == op.parameter_bytes() // tp
    assert any(
        isinstance(r, NetworkRequirement)
        and r.kind is CollectiveKind.ALL_GATHER
        and r.group is ParallelismGroup.TP
        for r in lm_head_entry.requirements
    ), lm_head_entry.requirements


def test_v3_lm_head_no_all_gather_when_tp_eq_one():
    cfg = V3Config()
    entries = _v3_entries(cfg, ParallelismConfig(tp=1))
    assert entries[-1].requirements == []


def test_llama_lm_head_shrinks_and_emits_all_gather_under_tp():
    tp = 4
    entries = _llama_entries(ParallelismConfig(tp=tp), phase="training")
    lm_head_entry = entries[-1]
    op = lm_head_entry.operator
    assert isinstance(op, Gemm)
    assert op.n == SMALL_LLAMA.vocab
    assert op.k == SMALL_LLAMA.h
    assert lm_head_entry.per_rank_parameter_bytes == op.parameter_bytes() // tp
    assert any(
        isinstance(r, NetworkRequirement)
        and r.kind is CollectiveKind.ALL_GATHER
        and r.group is ParallelismGroup.TP
        for r in lm_head_entry.requirements
    ), lm_head_entry.requirements


def test_llama_lm_head_no_all_gather_when_tp_eq_one():
    entries = _llama_entries(ParallelismConfig(tp=1), phase="training")
    assert entries[-1].requirements == []


def test_lm_head_all_gather_data_GB_matches_T_vocab_dtype():
    """The LM head AllGather sizes off the *full* vocab logits T*vocab*dtype."""
    cfg = V3Config(vocab=32000, h=512, dtype_bytes=2, n_layers=1, n_dense_layers=1)
    entries = _v3_entries(cfg, ParallelismConfig(tp=2))
    lm_head_reqs = entries[-1].requirements
    ag_req = next(r for r in lm_head_reqs if r.kind is CollectiveKind.ALL_GATHER)

    ctx = _ctx(prompt_len=128)
    T = ctx.microbatch_size * ctx.prompt_len
    expected_GB = T * cfg.vocab * cfg.dtype_bytes / 1e9
    actual_GB = ag_req.data_GB_fn(ctx)
    assert abs(actual_GB - expected_GB) < 1e-12, (actual_GB, expected_GB)


def test_v3_strategy_rejects_nondivisible_vocab():
    cfg = V3Config(vocab=129281)
    par = ParallelismConfig(tp=4)
    model = V3Model(name="t", role="actor", architecture=cfg)
    with pytest.raises(ValueError, match="vocab"):
        MegatronScheme(par).build_mm(
            model=model,
            workload=Workload.PREPARATION,
            devices=[_dev(i) for i in range(4)],
        )


def test_llama_strategy_rejects_nondivisible_vocab():
    arch = LlamaConfig(
        h=512, n_layers=2, n_q=8, n_kv=4, head_size=64, m=2048,
        rope_dim=64, vocab=32001, dtype_bytes=2,
    )
    par = ParallelismConfig(tp=4)
    model = LlamaModel(name="t", role="actor", architecture=arch)
    with pytest.raises(ValueError, match="vocab"):
        MegatronScheme(par).build_mm(
            model=model,
            workload=Workload.PREPARATION,
            devices=[_dev(i) for i in range(4)],
        )
