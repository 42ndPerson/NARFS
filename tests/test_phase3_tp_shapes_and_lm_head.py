"""Phase 3 (updated for the strategy-based architecture):
TP-sharded operator shapes + LM-head AllGather.

The architecture's `forward_sequence` emits labeled operators with NO
sharding decisions. The strategy decorates each operator with its
`parallelism_role` and attaches collectives (TP-AR, TP-AG, EP-AllToAll)
based on labels. These tests inspect the *strategy-decorated* pattern,
which is what ModelMapping ultimately consumes.
"""
from __future__ import annotations

import pytest

from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operators.attention import DecodeAttention
from operators.gemm import Gemm
from operators.llama_builder import LlamaModel
from operators.llama_config import LlamaConfig
from operators.pattern import expand_tagged_pattern, split_pattern
from operators.v3_block import V3Model
from operators.v3_config import V3Config
from strategies import MegatronStrategy
from topo import Device
from workload_context import (
    Workload,
    WorkloadContext,
)


SMALL_LLAMA = LlamaConfig(
    h=512, n_layers=2, n_q=8, n_kv=4, head_size=64,
    m=2048, rope_dim=64, vocab=32000, dtype_bytes=2,
)


def _dev(i: int) -> Device:
    return Device(host_id=i, gpu_id=0)


def _ctx(prompt_len: int = 128) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.GENERATION,
        batch_size=2, microbatch_size=2,
        prompt_len=prompt_len, response_len=0, num_microbatches=1,
    )


def _llama_decorated(par: ParallelismConfig, phase: str = "prefill"):
    """Strategy-decorated pattern for SMALL_LLAMA at given parallelism."""
    model = LlamaModel(name="t", role="actor", architecture=SMALL_LLAMA)
    return MegatronStrategy(par).decorated_pattern(model, phase)


def _v3_decorated(cfg: V3Config, par: ParallelismConfig, phase: str = "prefill"):
    """Strategy-decorated pattern for given V3Config + parallelism."""
    model = V3Model(name="t", role="actor", architecture=cfg)
    return MegatronStrategy(par).decorated_pattern(model, phase)


# ---- MLA TP-sharding (now via strategy) -------------------------------------


def test_mla_q_up_kv_up_attn_out_shrink_under_tp():
    """The strategy applies COLUMN_PARALLEL to q_up/kv_up and ROW_PARALLEL
    to attn_out, so their per-rank shapes shrink by tp.
    """
    cfg = V3Config()
    tp = 4
    par = ParallelismConfig(tp=tp)
    pattern = _v3_decorated(cfg, par)
    # Expand the pattern and find the relevant ops by their architectural shape.
    expanded = expand_tagged_pattern(pattern)
    ops, _ = split_pattern(expanded)

    # Find q_up: Gemm with n = n_h * (head_size + d_rope)
    expected_q_up_n = cfg.n_h * (cfg.head_size + cfg.d_rope)
    q_up = next(o for o in ops if isinstance(o, Gemm) and o.n == expected_q_up_n)
    n_eff, _ = q_up._shape(par)
    assert n_eff == expected_q_up_n // tp

    # kv_up: Gemm with n = n_h * 2 * head_size
    expected_kv_up_n = cfg.n_h * 2 * cfg.head_size
    kv_up = next(o for o in ops if isinstance(o, Gemm) and o.n == expected_kv_up_n)
    n_eff, _ = kv_up._shape(par)
    assert n_eff == expected_kv_up_n // tp

    # attn_out: Gemm with k = n_h * head_size, n = cfg.h, ROW_PARALLEL
    expected_attn_out_k = cfg.n_h * cfg.head_size
    attn_out = next(
        o for o in ops
        if isinstance(o, Gemm) and o.k == expected_attn_out_k and o.n == cfg.h
    )
    _, k_eff = attn_out._shape(par)
    assert k_eff == expected_attn_out_k // tp


def test_mla_attention_uses_local_head_count():
    """Strategy applies HEADS_PARALLEL to attention; per-rank head count = n_h / tp."""
    cfg = V3Config()
    par = ParallelismConfig(tp=2)
    pattern = _v3_decorated(cfg, par, phase="decode")
    ops, _ = split_pattern(expand_tagged_pattern(pattern))
    attn = next(o for o in ops if isinstance(o, DecodeAttention))
    assert attn.num_heads == cfg.n_h
    assert attn._heads(par) == cfg.n_h // 2


def test_strategy_rejects_nondivisible_n_h():
    """Strategy validates n_h % tp at build_mm time."""
    cfg = V3Config()  # n_h=128
    par = ParallelismConfig(tp=5)
    model = V3Model(name="t", role="actor", architecture=cfg)
    with pytest.raises(ValueError, match="n_h"):
        MegatronStrategy(par).build_mm(
            model=model,
            workload=Workload.PREPARATION,
            devices=[_dev(i) for i in range(5)],
        )


# ---- V3 dense FFN intermediate -----------------------------------------------


def test_v3_dense_ffn_intermediate_shrinks_under_tp():
    cfg = V3Config()
    tp = 4
    par = ParallelismConfig(tp=tp)
    pattern = _v3_decorated(cfg, par, phase="prefill")
    ops, _ = split_pattern(expand_tagged_pattern(pattern))
    # ffn_down: Gemm with n = cfg.h, k = cfg.m_dense (full shape).
    ffn_down = next(
        o for o in ops
        if isinstance(o, Gemm) and o.n == cfg.h and o.k == cfg.m_dense
    )
    _, k_eff = ffn_down._shape(par)
    assert k_eff == cfg.m_dense // tp


def test_strategy_rejects_nondivisible_m_dense():
    # Construct V3 where only m_dense fails the tp=2 check (everything else
    # divides). m_dense=99 isn't divisible by 2; vocab/n_h/m_expert defaults
    # are powers-of-2-compatible.
    cfg = V3Config(m_dense=99)
    par = ParallelismConfig(tp=2)
    model = V3Model(name="t", role="actor", architecture=cfg)
    with pytest.raises(ValueError, match="m_dense"):
        MegatronStrategy(par).build_mm(
            model=model,
            workload=Workload.PREPARATION,
            devices=[_dev(i) for i in range(2)],
        )


# ---- V3 MoE shared expert ----------------------------------------------------


def test_v3_moe_shared_expert_shrinks_under_tp_routed_unchanged():
    cfg = V3Config()
    tp = 4
    par = ParallelismConfig(tp=tp)
    pattern = _v3_decorated(cfg, par, phase="prefill")
    ops, _ = split_pattern(expand_tagged_pattern(pattern))

    routed: list[Gemm] = [
        o for o in ops
        if isinstance(o, Gemm) and o.n_replicas == cfg.n_routed_experts
    ]
    # Routed experts: EP-sharded only, TP role REPLICATED.
    for r in routed:
        n_eff, k_eff = r._shape(par)
        assert n_eff == r.n and k_eff == r.k
        assert r.n == cfg.m_expert or r.k == cfg.m_expert

    def _is_shared_expert(op: Gemm) -> bool:
        if op.n_replicas != 1:
            return False
        return (op.n == cfg.m_expert and op.k == cfg.h) or (op.n == cfg.h and op.k == cfg.m_expert)

    shared: list[Gemm] = [
        o for o in ops if isinstance(o, Gemm) and _is_shared_expert(o)
    ]
    n_moe_layers = cfg.n_layers - cfg.n_dense_layers
    assert len(shared) == 3 * n_moe_layers  # gate, up, down per layer
    for s in shared:
        n_eff, k_eff = s._shape(par)
        assert (n_eff == s.n // tp and k_eff == s.k) or (n_eff == s.n and k_eff == s.k // tp)


def test_strategy_rejects_nondivisible_m_expert():
    cfg = V3Config(m_expert=99)
    par = ParallelismConfig(tp=2)
    model = V3Model(name="t", role="actor", architecture=cfg)
    with pytest.raises(ValueError, match="m_expert"):
        MegatronStrategy(par).build_mm(
            model=model,
            workload=Workload.PREPARATION,
            devices=[_dev(i) for i in range(2)],
        )


# ---- LM head TP-shard + AllGather --------------------------------------------


def test_v3_lm_head_shrinks_and_emits_all_gather_under_tp():
    cfg = V3Config()
    tp = 4
    par = ParallelismConfig(tp=tp)
    pattern = _v3_decorated(cfg, par, phase="prefill")
    ops, reqs = split_pattern(expand_tagged_pattern(pattern))
    lm_head = ops[-1]
    lm_head_reqs = reqs[-1]
    assert isinstance(lm_head, Gemm)
    assert lm_head.n == cfg.vocab
    assert lm_head.k == cfg.h
    n_eff, _ = lm_head._shape(par)
    assert n_eff == cfg.vocab // tp
    assert any(
        isinstance(r, NetworkRequirement)
        and r.kind is CollectiveKind.ALL_GATHER
        and r.group is ParallelismGroup.TP
        for r in lm_head_reqs
    ), lm_head_reqs


def test_v3_lm_head_no_all_gather_when_tp_eq_one():
    cfg = V3Config()
    pattern = _v3_decorated(cfg, ParallelismConfig(tp=1), phase="prefill")
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    assert reqs[-1] == [], reqs[-1]


def test_llama_lm_head_shrinks_and_emits_all_gather_under_tp():
    tp = 4
    par = ParallelismConfig(tp=tp)
    pattern = _llama_decorated(par, phase="training")
    ops, reqs = split_pattern(expand_tagged_pattern(pattern))
    lm_head = ops[-1]
    lm_head_reqs = reqs[-1]
    assert isinstance(lm_head, Gemm)
    assert lm_head.n == SMALL_LLAMA.vocab
    assert lm_head.k == SMALL_LLAMA.h
    n_eff, k_eff = lm_head._shape(par)
    assert n_eff == SMALL_LLAMA.vocab // tp
    assert k_eff == SMALL_LLAMA.h
    assert any(
        isinstance(r, NetworkRequirement)
        and r.kind is CollectiveKind.ALL_GATHER
        and r.group is ParallelismGroup.TP
        for r in lm_head_reqs
    ), lm_head_reqs


def test_llama_lm_head_no_all_gather_when_tp_eq_one():
    pattern = _llama_decorated(ParallelismConfig(tp=1), phase="training")
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    assert reqs[-1] == []


def test_lm_head_all_gather_data_GB_matches_T_vocab_dtype():
    """The LM head AllGather sizes off the *full* vocab logits T*vocab*dtype."""
    cfg = V3Config(vocab=32000, h=512, dtype_bytes=2, n_layers=1, n_dense_layers=1)
    pattern = _v3_decorated(cfg, ParallelismConfig(tp=2), phase="prefill")
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    lm_head_reqs = reqs[-1]
    ag_req = next(r for r in lm_head_reqs if r.kind is CollectiveKind.ALL_GATHER)

    ctx = _ctx(prompt_len=128)
    T = ctx.microbatch_size * ctx.prompt_len
    expected_GB = T * cfg.vocab * cfg.dtype_bytes / 1e9
    actual_GB = ag_req.data_GB_fn(ctx)
    assert abs(actual_GB - expected_GB) < 1e-12, (actual_GB, expected_GB)


def test_v3_strategy_rejects_nondivisible_vocab():
    cfg = V3Config(vocab=129281)  # prime-ish, not divisible by 4
    par = ParallelismConfig(tp=4)
    model = V3Model(name="t", role="actor", architecture=cfg)
    with pytest.raises(ValueError, match="vocab"):
        MegatronStrategy(par).build_mm(
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
        MegatronStrategy(par).build_mm(
            model=model,
            workload=Workload.PREPARATION,
            devices=[_dev(i) for i in range(4)],
        )
