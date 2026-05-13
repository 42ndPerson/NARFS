"""Phase 4: per-layer DP gradient AllReduce in training.

Pre-Phase-4 the simulator emitted no DP-gradient communication, so any
training-time analysis with dp > 1 was missing the per-layer AllReduce
that synchronizes gradients at the end of each layer's backward pass.

This module pins the new behavior:
- Each layer template (Llama, V3 dense, V3 MoE) emits one DP-AR.
- The AR is sized to that layer template's per-rank parameter byte count.
- `payload_id` discriminates dense vs MoE templates so dedup doesn't
  collapse their distinct sizes — Phase 1 already ships the discriminator.
- DP-AR fires only when phase == "training" and parallelism.dp > 1; not
  in prefill, decode, or generation. Not when dp == 1.
- The same template's DP-AR dedupes across all repetitions (e.g. one
  NetworkRequirement instance covers all 58 V3 MoE layers); the simulator
  counts that resolved NetworkOp's time once per attaching operator,
  reflecting the per-layer firing in the wall-clock total.
"""
from __future__ import annotations

from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operators.pattern import expand_tagged_pattern, split_pattern
from operators.llama_builder import LlamaModel
from operators.llama_config import LlamaConfig
from operators.v3_block import V3Model
from operators.v3_config import V3Config
from strategies import MegatronScheme


SMALL_LLAMA = LlamaConfig(
    h=512, n_layers=4, n_q=8, n_kv=4, head_size=64,
    m=2048, rope_dim=64, vocab=32000, dtype_bytes=2,
)


def _llama_pattern(phase: str, par: ParallelismConfig):
    """Return the MegatronScheme-decorated pattern (with requirements)
    for a SMALL_LLAMA at the given parallelism."""
    model = LlamaModel(name="t", role="actor", architecture=SMALL_LLAMA)
    return MegatronScheme(par).decorated_pattern(model, phase)


def _v3_pattern(cfg: V3Config, phase: str, par: ParallelismConfig):
    model = V3Model(name="t", role="actor", architecture=cfg)
    return MegatronScheme(par).decorated_pattern(model, phase)


def _dp_ar_reqs(reqs_list: list[list[NetworkRequirement]]) -> list[NetworkRequirement]:
    """Flatten and filter to DP AllReduce requirements."""
    flat = [r for rs in reqs_list for r in rs]
    return [r for r in flat if r.kind is CollectiveKind.ALL_REDUCE and r.group is ParallelismGroup.DP]


# ---- Llama: one template, all layers share its DP-AR --------------------------


def test_llama_training_emits_dp_ar_with_dp_gt_1():
    pattern = _llama_pattern("training", ParallelismConfig(tp=1, pp=1, dp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    dp_ars = _dp_ar_reqs(reqs)
    assert len(dp_ars) > 0, "training+dp=2 should emit DP-AR per layer instance"


def test_llama_training_no_dp_ar_when_dp_eq_one():
    pattern = _llama_pattern("training", ParallelismConfig(tp=1, pp=1, dp=1))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    assert _dp_ar_reqs(reqs) == []


def test_llama_prefill_no_dp_ar_even_with_dp_gt_1():
    pattern = _llama_pattern("prefill", ParallelismConfig(tp=1, pp=1, dp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    assert _dp_ar_reqs(reqs) == [], "prefill must not emit DP-AR; backward doesn't run"


def test_llama_decode_no_dp_ar_even_with_dp_gt_1():
    pattern = _llama_pattern("decode", ParallelismConfig(tp=1, pp=1, dp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    assert _dp_ar_reqs(reqs) == []


def test_llama_dp_ar_attached_to_last_op_of_each_layer():
    """Layer template's DP-AR should hang on ffn_down (the last in-layer op),
    so it appears once per layer instance in the expanded list."""
    pattern = _llama_pattern("training", ParallelismConfig(tp=1, pp=1, dp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    # n_layers = 4 — DP-AR appears 4 times in the flat req list (one per
    # repetition of the template).
    dp_ars = _dp_ar_reqs(reqs)
    assert len(dp_ars) == SMALL_LLAMA.n_layers


def test_llama_dp_ar_dedupes_to_single_requirement():
    """Same template repeats → all DP-ARs are equal NetworkRequirement
    objects → the dedup hash collapses them to one NetworkOp instance.
    We pin the equality here so changes to dedup keys don't silently break
    the model_mapping cache."""
    pattern = _llama_pattern("training", ParallelismConfig(tp=1, pp=1, dp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    dp_ars = _dp_ar_reqs(reqs)
    assert len(set(dp_ars)) == 1


def test_llama_dp_ar_size_matches_layer_parameter_bytes():
    """data_GB_fn returns total layer param bytes in GB, ignoring ctx."""
    pattern = _llama_pattern("training", ParallelismConfig(tp=1, pp=1, dp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    dp_ar = _dp_ar_reqs(reqs)[0]

    from operators.llama_builder import build_llama_layer
    one_layer = build_llama_layer(SMALL_LLAMA, "training")
    # build_llama_layer emits LabeledOps; sum their underlying operators'
    # parameter bytes. At tp=1/dp=2, per-rank == full bytes.
    expected_GB = sum(lop.operator.parameter_bytes() for lop in one_layer) / 1e9
    actual_GB = dp_ar.data_GB_fn(None)
    assert abs(actual_GB - expected_GB) < 1e-12


# ---- V3: dense and MoE templates have distinct payload_ids ------------------


def _v3_small() -> V3Config:
    # Tiny V3 to keep tests fast; preserves the "dense + MoE" template split.
    return V3Config(
        h=512, n_h=8, head_size=64, d_q_compress=128, d_kv_compress=64,
        d_rope=32, m_dense=1024, m_expert=256, n_routed_experts=8,
        n_shared_experts=1, top_k=2, vocab=8000, n_layers=4,
        n_dense_layers=2, dtype_bytes=2,
    )


def test_v3_training_emits_dp_ar_per_layer_template():
    cfg = _v3_small()
    pattern = _v3_pattern(cfg,"training", ParallelismConfig(tp=1, pp=1, dp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    dp_ars = _dp_ar_reqs(reqs)
    # n_dense_layers = 2, n_moe_layers = 2 → 4 attachments total.
    assert len(dp_ars) == cfg.n_layers


def test_v3_dense_and_moe_dp_ar_have_distinct_payload_ids():
    cfg = _v3_small()
    pattern = _v3_pattern(cfg,"training", ParallelismConfig(tp=1, pp=1, dp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    dp_ars = _dp_ar_reqs(reqs)
    payload_ids = {r.payload_id for r in dp_ars}
    assert len(payload_ids) == 2, f"expected 2 distinct payload_ids, got {payload_ids}"


def test_v3_moe_and_dense_dp_ar_have_distinct_sizes():
    """Pin the dedup discriminator: dense and MoE templates carry different
    parameter byte counts. If both collapsed to the same NetworkRequirement,
    sizes would silently merge to whichever inserted first.

    Note: the absolute MoE/dense ratio depends on n_routed_experts and the
    other V3 hyperparams; for the small fixture used here it's modest, so
    we assert the structural inequality rather than a magnitude bound.
    """
    cfg = _v3_small()
    pattern = _v3_pattern(cfg,"training", ParallelismConfig(tp=1, pp=1, dp=2))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    dp_ars = _dp_ar_reqs(reqs)
    by_payload: dict = {}
    for r in dp_ars:
        by_payload.setdefault(r.payload_id, r)
    assert len(by_payload) == 2
    sizes = sorted(r.data_GB_fn(None) for r in by_payload.values())
    dense_size, moe_size = sizes  # smaller is dense, larger is MoE
    assert moe_size > dense_size, (dense_size, moe_size)
    assert moe_size != dense_size


def test_v3_training_no_dp_ar_when_dp_eq_one():
    cfg = _v3_small()
    pattern = _v3_pattern(cfg,"training", ParallelismConfig(tp=1, pp=1, dp=1))
    _, reqs = split_pattern(expand_tagged_pattern(pattern))
    assert _dp_ar_reqs(reqs) == []
