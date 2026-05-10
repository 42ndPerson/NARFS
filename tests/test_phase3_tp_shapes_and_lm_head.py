"""Phase 3: TP-sharded operator shapes + LM-head AllGather.

Pre-Phase-3 the V3 operator shapes (MLA heads, dense FFN intermediate, MoE
shared expert) and both Llama/V3 LM heads were built at full size regardless
of TP degree, with only the TP AllReduce communication scaling. This module
pins the new behavior: shapes shrink under tp, divisibility is enforced,
and the LM head emits a TP AllGather sized to the full vocab logits.

These tests do not exercise the timing model end-to-end; they verify the
operator graph structure and the network requirement attached to the LM head.
"""
from __future__ import annotations

import pytest

from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operators.attention import DecodeAttention, PrefillAttention
from operators.builders import expand_tagged_pattern, split_pattern
from operators.elementwise import RoPE
from operators.gemm import Gemm
from operators.llama_builder import build_llama_pattern
from operators.llama_config import LlamaConfig
from operators.mla import build_mla_decode, build_mla_prefill
from operators.v3_block import (
    build_v3_dense_layer,
    build_v3_model,
    build_v3_moe_layer,
)
from operators.v3_config import V3Config
from workload_context import (
    TokenCount,
    Workload,
    WorkloadContext,
)


SMALL_LLAMA = LlamaConfig(
    h=512, n_layers=2, n_q=8, n_kv=4, head_size=64,
    m=2048, rope_dim=64, vocab=32000, dtype_bytes=2,
)


def _ctx(prompt_len: int = 128) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.GENERATION,
        batch_size=2,
        microbatch_size=2,
        prompt_len=prompt_len,
        response_len=0,
        num_microbatches=1,
    )


# ---- MLA TP-sharding ---------------------------------------------------------


def test_mla_prefill_q_up_kv_up_attn_out_shrink_under_tp():
    cfg = V3Config()
    tp = 4
    ops_full = build_mla_prefill(cfg)
    ops_sharded = build_mla_prefill(cfg, tp=tp)

    # Layout: q_down, q_up, kv_down, kv_up, rope, attention, attn_out
    q_up_full = ops_full[1]
    q_up_sharded = ops_sharded[1]
    assert isinstance(q_up_full, Gemm) and isinstance(q_up_sharded, Gemm)
    assert q_up_sharded.n == q_up_full.n // tp

    kv_up_full = ops_full[3]
    kv_up_sharded = ops_sharded[3]
    assert isinstance(kv_up_full, Gemm) and isinstance(kv_up_sharded, Gemm)
    assert kv_up_sharded.n == kv_up_full.n // tp

    rope_full = ops_full[4]
    rope_sharded = ops_sharded[4]
    assert isinstance(rope_full, RoPE) and isinstance(rope_sharded, RoPE)
    assert rope_sharded.num_heads == rope_full.num_heads // tp

    attn_full = ops_full[5]
    attn_sharded = ops_sharded[5]
    assert isinstance(attn_full, PrefillAttention) and isinstance(attn_sharded, PrefillAttention)
    assert attn_sharded.num_heads == attn_full.num_heads // tp
    assert attn_sharded.num_kv_heads == attn_full.num_kv_heads // tp

    attn_out_full = ops_full[6]
    attn_out_sharded = ops_sharded[6]
    assert isinstance(attn_out_full, Gemm) and isinstance(attn_out_sharded, Gemm)
    assert attn_out_sharded.k == attn_out_full.k // tp


def test_mla_prefill_q_down_kv_down_unchanged_under_tp():
    """q_down/kv_down feed the latent and are replicated across TP ranks —
    their shapes don't depend on tp."""
    cfg = V3Config()
    full = build_mla_prefill(cfg)
    sharded = build_mla_prefill(cfg, tp=4)
    # q_down and kv_down are positions 0 and 2.
    for i in (0, 2):
        assert isinstance(full[i], Gemm) and isinstance(sharded[i], Gemm)
        assert full[i].n == sharded[i].n
        assert full[i].k == sharded[i].k


def test_mla_decode_attn_uses_local_head_count():
    cfg = V3Config()
    ops = build_mla_decode(cfg, tp=2)
    attn = next(op for op in ops if isinstance(op, DecodeAttention))
    assert attn.num_heads == cfg.n_h // 2


def test_mla_rejects_nondivisible_tp():
    cfg = V3Config()
    # n_h=128; tp=5 doesn't divide it.
    with pytest.raises(ValueError, match="n_h"):
        build_mla_prefill(cfg, tp=5)
    with pytest.raises(ValueError, match="n_h"):
        build_mla_decode(cfg, tp=5)


# ---- V3 dense FFN intermediate ----------------------------------------------


def test_v3_dense_ffn_intermediate_shrinks_under_tp():
    cfg = V3Config()
    parallelism = ParallelismConfig(tp=4)
    ops_with_reqs = build_v3_dense_layer(cfg, TokenCount.PREFILL, parallelism)
    ffn_gemms = [op for op, _ in ops_with_reqs if isinstance(op, Gemm)]
    # The dense FFN appears at the end of the layer; the last 4 entries are
    # the FFN gates, ups, swiglu, down. The 3 GEMMs (gate, up, down) bracket
    # the swiglu. Inspect the down (last) — its k should be m_dense // tp.
    ffn_down = ffn_gemms[-1]
    assert ffn_down.n == cfg.h
    assert ffn_down.k == cfg.m_dense // 4


def test_v3_dense_layer_rejects_nondivisible_m_dense():
    cfg = V3Config()
    # m_dense=18432; tp=5 doesn't divide.
    with pytest.raises(ValueError, match="m_dense"):
        build_v3_dense_layer(cfg, TokenCount.PREFILL, ParallelismConfig(tp=5))


# ---- V3 MoE shared expert ----------------------------------------------------


def test_v3_moe_shared_expert_shrinks_under_tp_routed_unchanged():
    cfg = V3Config()
    tp = 4
    ops_with_reqs_full = build_v3_moe_layer(cfg, TokenCount.PREFILL, ParallelismConfig(tp=1))
    ops_with_reqs_sharded = build_v3_moe_layer(cfg, TokenCount.PREFILL, ParallelismConfig(tp=tp))

    # Routed experts have n_replicas=n_routed_experts; shared expert has 1.
    routed_full = [op for op, _ in ops_with_reqs_full if isinstance(op, Gemm) and op.n_replicas == cfg.n_routed_experts]
    routed_sharded = [op for op, _ in ops_with_reqs_sharded if isinstance(op, Gemm) and op.n_replicas == cfg.n_routed_experts]
    # Routed ops keep full m_expert (they're EP-sharded, not TP-sharded).
    assert all(op.n == cfg.m_expert or op.k == cfg.m_expert for op in routed_full)
    assert all(op.n == cfg.m_expert or op.k == cfg.m_expert for op in routed_sharded)

    # Shared-expert gemms have the (m_expert, h) dimension pair: gate/up are
    # (n=m_expert, k=h); down is (n=h, k=m_expert). Other Gemms in the layer
    # have at least one dimension that is neither h nor m_expert.
    def _is_shared_expert(op, m_shared):
        if not isinstance(op, Gemm) or op.n_replicas != 1:
            return False
        return (op.n == m_shared and op.k == cfg.h) or (op.n == cfg.h and op.k == m_shared)

    shared_full = [op for op, _ in ops_with_reqs_full if _is_shared_expert(op, cfg.m_expert)]
    shared_sharded = [op for op, _ in ops_with_reqs_sharded if _is_shared_expert(op, cfg.m_expert // tp)]
    assert len(shared_full) == 3  # gate, up, down
    assert len(shared_sharded) == 3


def test_v3_moe_layer_rejects_nondivisible_m_expert():
    cfg = V3Config()
    # m_expert=2048; tp=3 doesn't divide.
    with pytest.raises(ValueError, match="m_expert"):
        build_v3_moe_layer(cfg, TokenCount.PREFILL, ParallelismConfig(tp=3))


# ---- LM head TP-shard + AllGather --------------------------------------------


def test_v3_lm_head_shrinks_and_emits_all_gather_under_tp():
    cfg = V3Config()
    tp = 4
    pattern = build_v3_model(cfg, TokenCount.PREFILL, ParallelismConfig(tp=tp))
    ops, reqs = split_pattern(expand_tagged_pattern(pattern))
    lm_head = ops[-1]
    lm_head_reqs = reqs[-1]
    assert isinstance(lm_head, Gemm)
    assert lm_head.n == cfg.vocab // tp
    assert lm_head.k == cfg.h
    # AllGather requirement on TP group.
    assert any(
        isinstance(r, NetworkRequirement)
        and r.kind is CollectiveKind.ALL_GATHER
        and r.group is ParallelismGroup.TP
        for r in lm_head_reqs
    ), lm_head_reqs


def test_v3_lm_head_no_all_gather_when_tp_eq_one():
    cfg = V3Config()
    pattern = build_v3_model(cfg, TokenCount.PREFILL, ParallelismConfig(tp=1))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    lm_head_reqs = reqs[-1]
    assert lm_head_reqs == [], lm_head_reqs


def test_llama_lm_head_shrinks_and_emits_all_gather_under_tp():
    arch = SMALL_LLAMA
    tp = 4
    pattern = build_llama_pattern(arch, "training", ParallelismConfig(tp=tp))
    ops, reqs = split_pattern(expand_tagged_pattern(pattern))
    lm_head = ops[-1]
    lm_head_reqs = reqs[-1]
    assert isinstance(lm_head, Gemm)
    assert lm_head.n == arch.vocab // tp
    assert lm_head.k == arch.h
    assert any(
        isinstance(r, NetworkRequirement)
        and r.kind is CollectiveKind.ALL_GATHER
        and r.group is ParallelismGroup.TP
        for r in lm_head_reqs
    ), lm_head_reqs


def test_llama_lm_head_no_all_gather_when_tp_eq_one():
    pattern = build_llama_pattern(SMALL_LLAMA, "training", ParallelismConfig(tp=1))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    assert reqs[-1] == []


def test_lm_head_all_gather_data_GB_matches_T_vocab_dtype():
    """The LM head AllGather sizes off the *full* vocab logits T*vocab*dtype."""
    cfg = V3Config(vocab=32000, h=512, dtype_bytes=2, n_layers=1, n_dense_layers=1)
    pattern = build_v3_model(cfg, TokenCount.PREFILL, ParallelismConfig(tp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    lm_head_reqs = reqs[-1]
    ag_req = next(r for r in lm_head_reqs if r.kind is CollectiveKind.ALL_GATHER)

    ctx = _ctx(prompt_len=128)
    T = ctx.microbatch_size * ctx.prompt_len
    expected_GB = T * cfg.vocab * cfg.dtype_bytes / 1e9
    actual_GB = ag_req.data_GB_fn(ctx)
    assert abs(actual_GB - expected_GB) < 1e-12, (actual_GB, expected_GB)


def test_v3_lm_head_rejects_nondivisible_vocab():
    cfg = V3Config(vocab=129281)  # prime-ish, not divisible by 4
    with pytest.raises(ValueError, match="vocab"):
        build_v3_model(cfg, TokenCount.PREFILL, ParallelismConfig(tp=4))


def test_llama_lm_head_rejects_nondivisible_vocab():
    arch = LlamaConfig(
        h=512, n_layers=2, n_q=8, n_kv=4, head_size=64, m=2048,
        rope_dim=64, vocab=32001, dtype_bytes=2,
    )
    with pytest.raises(ValueError, match="vocab"):
        build_llama_pattern(arch, "training", ParallelismConfig(tp=4))
