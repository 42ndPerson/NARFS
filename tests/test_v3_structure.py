from __future__ import annotations

from operators.attention import DecodeAttention
from operators.pattern import expand_tagged_pattern, split_pattern
from operators.mla import build_mla_decode
from operators.v3_block import (
    build_v3_dense_layer,
    build_v3_model,
    build_v3_moe_layer,
)
from operators.v3_config import V3Config
from workload_context import TokenCount, Workload, WorkloadContext


def _prefill_ctx(batch: int, prompt_len: int) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.GENERATION,
        batch_size=batch,
        microbatch_size=batch,
        prompt_len=prompt_len,
        response_len=0,
        num_microbatches=1,
    )


def _expanded_ops(cfg: V3Config, phase: TokenCount):
    expanded = expand_tagged_pattern(build_v3_model(cfg, phase))
    return split_pattern(expanded)


def test_v3_prefill_operator_count():
    cfg = V3Config()
    ops, _ = _expanded_ops(cfg, TokenCount.PREFILL)

    dense_ops = build_v3_dense_layer(cfg, TokenCount.PREFILL)
    moe_ops = build_v3_moe_layer(cfg, TokenCount.PREFILL)

    expected = (
        2  # embed + final_rmsnorm
        + cfg.n_dense_layers * len(dense_ops)
        + (cfg.n_layers - cfg.n_dense_layers) * len(moe_ops)
        + 1  # lm_head
    )
    assert len(ops) == expected, f"got {len(ops)} ops, expected {expected}"


def test_v3_total_parameter_bytes():
    cfg = V3Config()
    ops, _ = _expanded_ops(cfg, TokenCount.PREFILL)

    total_bytes = sum(op.parameter_bytes() for op in ops)

    expected_params = 671e9
    expected_bytes = expected_params * cfg.dtype_bytes
    rel_err = abs(total_bytes - expected_bytes) / expected_bytes
    assert rel_err < 0.05, (
        f"total_bytes={total_bytes:.3e} (={total_bytes/cfg.dtype_bytes:.3e} params) "
        f"expected≈{expected_bytes:.3e} rel_err={rel_err:.3f}"
    )


def test_mla_decode_kv_cache_per_token():
    cfg = V3Config()
    decode_ops = build_mla_decode(cfg)
    # build_mla_decode emits LabeledOps; unwrap to find the attention op.
    decode_attn = next(
        lop.operator for lop in decode_ops if isinstance(lop.operator, DecodeAttention)
    )

    expected_mla_kv = (cfg.d_kv_compress + cfg.d_rope) * cfg.dtype_bytes
    assert expected_mla_kv == 1152
    assert decode_attn.kv_bytes_per_token == expected_mla_kv

    vanilla_mha_kv = 2 * cfg.n_h * cfg.head_size * cfg.dtype_bytes
    assert vanilla_mha_kv == 65536

    ratio = vanilla_mha_kv / decode_attn.kv_bytes_per_token
    assert 50 <= ratio <= 60, f"MLA KV-cache savings ratio {ratio:.1f}x outside [50, 60]"


def test_v3_activated_parameters_per_token():
    cfg = V3Config()
    ops, _ = _expanded_ops(cfg, TokenCount.PREFILL)

    activated_bytes = sum(op.activated_parameter_bytes() for op in ops)

    expected_activated_params = 37e9
    expected_bytes = expected_activated_params * cfg.dtype_bytes
    rel_err = abs(activated_bytes - expected_bytes) / expected_bytes
    assert rel_err < 0.10, (
        f"activated_bytes={activated_bytes:.3e} "
        f"(={activated_bytes/cfg.dtype_bytes:.3e} params) "
        f"expected≈{expected_bytes:.3e} rel_err={rel_err:.3f}"
    )


def test_v3_prefill_graph_builds():
    cfg = V3Config()
    ctx = _prefill_ctx(batch=1, prompt_len=4096)
    ops, _ = _expanded_ops(cfg, TokenCount.PREFILL)
    for op in ops:
        assert op.compute_flops(ctx) >= 0.0
        assert op.memory_bytes(ctx) >= 0.0
