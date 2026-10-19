"""Model definition and components: parameter counts against Hugging Face
configs, FLOP rules, per-rank shapes, and backward rules."""
import pytest

from narfs.blocks import Collective, Compute, OnDevices, Par, Repeat, Seq
from narfs.model.architecture import (DEEPSEEK_V3, LLAMA3_70B, LLAMA3_8B, MIXTRAL_8X7B, Architecture, pattern,
                                      runs)
from narfs.model.hf_import import from_hf_config
from narfs.model.library import (Ctx, TokenMix, backward, check_architecture, get, layer_params, make_ctx,
                                 model_params)
from narfs.parallel.mesh import Degrees
from narfs.parallel.rules import MEGATRON

HF = {
    "llama3-8b": dict(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, num_hidden_layers=32,
                      intermediate_size=14336, vocab_size=128256, tie_word_embeddings=False),
    "mixtral": dict(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, num_hidden_layers=32,
                    intermediate_size=14336, vocab_size=32000, num_local_experts=8, num_experts_per_tok=2),
    "deepseek-v3": dict(hidden_size=7168, num_attention_heads=128, num_hidden_layers=61, intermediate_size=18432,
                        vocab_size=129280, q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
                        qk_rope_head_dim=64, v_head_dim=128, n_routed_experts=256, n_shared_experts=1,
                        num_experts_per_tok=8, moe_intermediate_size=2048, first_k_dense_replace=3),
}

HYBRID = Architecture(
    "hybrid-moe",
    dict(hidden=4096, heads=32, kv_heads=8, head_dim=128, vocab=128_000, ssm_inner=8192, ssm_heads=128,
         ssm_state=128, ssm_groups=8, experts=64, top_k=6, expert_ffn=1408, shared_experts=0),
    pattern([("mamba2", "moe")] * 3 + [("gqa", "moe")], repeat=12))


def test_parameter_counts_match_hugging_face():
    assert model_params(LLAMA3_8B) == 8_030_261_248
    assert model_params(LLAMA3_70B) == 70_553_706_496
    assert model_params(MIXTRAL_8X7B) == 46_702_792_704
    # DeepSeek-V3 without the multi-token-prediction module; the router bias (14,848) is not modeled
    assert model_params(DEEPSEEK_V3) == pytest.approx(671_026_419_200, rel=1e-6)


def test_hf_import_matches_the_presets():
    assert model_params(from_hf_config(HF["llama3-8b"])) == model_params(LLAMA3_8B)
    assert model_params(from_hf_config(HF["mixtral"])) == model_params(MIXTRAL_8X7B)
    ds = from_hf_config(HF["deepseek-v3"])
    assert runs(ds.layers) == [(("mla", "dense"), 3), (("mla", "moe"), 58)]
    assert model_params(ds) == model_params(DEEPSEEK_V3)


def _flops(node, kind=None) -> float:
    match node:
        case Compute():
            return node.work.flops if kind in (None, node.work.kind) else 0.0
        case Seq() | Par():
            return sum(_flops(c, kind) for c in node.children)
        case Repeat():
            return node.n * _flops(node.body, kind)
        case OnDevices():
            return _flops(node.body, kind)
    return 0.0


def _layer(c, layer):
    return Seq(tuple(get(name).forward(c) for name in layer))


def test_gemm_flops_are_twice_params_times_tokens():
    t = 1024
    c = make_ctx(LLAMA3_8B, Degrees(), MEGATRON, tokens=t, seq=t)
    layer = _layer(c, LLAMA3_8B.layers[0])
    p = layer_params(c, LLAMA3_8B.layers[0])
    assert _flops(layer, "gemm") == pytest.approx(2 * (p.total - 2 * 4096) * t)          # norms are not GEMMs
    # attention: QK^T and PV over half the sequence on average
    assert _flops(layer, "attention") == pytest.approx(2 * t * (t / 2) * 32 * 2 * 128)
    # backward GEMMs are dgrad + wgrad: twice the forward
    assert _flops(backward(layer), "gemm") == pytest.approx(2 * _flops(layer, "gemm"))


def test_per_rank_shapes_rebuild_the_full_model():
    deg = Degrees(tp=8, ep=4)
    for arch in (LLAMA3_70B, MIXTRAL_8X7B, DEEPSEEK_V3):
        rules = MEGATRON
        if arch is MIXTRAL_8X7B:
            deg = Degrees(tp=8, ep=8)
        c = make_ctx(arch, deg, rules)
        total = get("embed").params(c) + get("head").params(c)
        for layer in arch.layers:
            total = total + layer_params(c, layer)
        assert total.tp * deg.tp + total.rep + total.ep * deg.ep == pytest.approx(model_params(arch))


def test_kv_and_state_bytes():
    c = make_ctx(LLAMA3_70B, Degrees(tp=8), MEGATRON)
    assert get("gqa").kv_bytes_per_token(c) == 2 * 1 * 128 * 2                  # one KV head per rank
    c = make_ctx(LLAMA3_8B, Degrees(tp=16), MEGATRON)                           # 8 KV heads replicated
    assert c.local["kv_heads"] == 1
    c = make_ctx(DEEPSEEK_V3, Degrees(tp=8), MEGATRON)
    assert get("mla").kv_bytes_per_token(c) == (512 + 64) * 2                    # latent, replicated over tp
    c = make_ctx(HYBRID, Degrees(tp=2), MEGATRON)
    assert get("mamba2").kv_bytes_per_token(c) == 0
    assert get("mamba2").state_bytes(c) == 4096 * 128 * 4 + (4096 + 2 * 4 * 128) * 3 * 2


def test_every_component_builds_forward_and_step():
    check_architecture(HYBRID)
    check_architecture(DEEPSEEK_V3)
    mix = TokenMix(batch=64, padded=64, context=1500, prefill=256, prefill_context=128)
    for arch, deg in ((HYBRID, Degrees(tp=2, ep=2)), (DEEPSEEK_V3, Degrees(tp=8, ep=8)), (LLAMA3_8B, Degrees(tp=2))):
        c = make_ctx(arch, deg, MEGATRON, tokens=2048, seq=2048, tp="tp", ep="ep", cp="cp")
        for name in sorted(arch.components() | {"embed", "head"}):
            comp = get(name)
            assert _flops(comp.forward(c)) > 0 or name == "embed"
            assert _flops(comp.step(c, mix)) > 0 or name == "embed"
            assert comp.params(c).total > 0
            backward(comp.forward(c))


def test_overlapped_tp_communication_is_the_chunked_pattern():
    c = make_ctx(LLAMA3_8B, Degrees(tp=4), MEGATRON, tokens=4096, seq=4096, tp="tp", tp_comm="overlapped",
                 tp_chunks=4)
    out = get("dense").forward(c).children[-1]
    first, rep, last = out.children
    assert isinstance(rep, Repeat) and rep.n == 3 and isinstance(rep.body, Par)
    assert first.work.shape == (1024, 4096, 14336 // 4)                           # a smaller GEMM per chunk
    assert isinstance(last, Collective) and last.bytes == pytest.approx(4096 * 4096 * 2 / 4)


def test_sequence_parallel_collectives_and_their_adjoints():
    c = make_ctx(LLAMA3_8B, Degrees(tp=4), MEGATRON, tokens=1024, seq=1024, tp="tp", sp=True)
    fwd = get("gqa").forward(c)
    kinds = [n.kind for n in fwd.children if isinstance(n, Collective)]
    assert kinds == ["all_gather", "reduce_scatter"]
    bwd = [n.kind for n in backward(fwd).children if isinstance(n, Collective)]
    assert bwd == ["all_gather", "reduce_scatter"]                                # reversed and adjoint


def test_moe_busiest_rank_and_active_experts():
    c = make_ctx(MIXTRAL_8X7B, Degrees(ep=8), MEGATRON, tokens=4096, seq=4096, ep="ep")
    tl, rows, _ = get("moe")._rows(c, 4096)
    assert tl == 4096 and rows > 4096 * 2                                       # the busiest of 8 EP ranks
    one = make_ctx(MIXTRAL_8X7B, Degrees(), MEGATRON)
    assert get("moe")._rows(one, 4096)[1] == 4096 * 2


def test_bad_definitions_are_rejected():
    with pytest.raises(KeyError, match="no component"):
        get("conv3d")
    bad = Architecture("bad", dict(hidden=8, vocab=8), pattern([("gqa", "dense")], 1))
    with pytest.raises(ValueError, match="needs dimensions"):
        check_architecture(bad)
    with pytest.raises(ValueError, match="not divisible"):
        make_ctx(LLAMA3_8B, Degrees(tp=3), MEGATRON)
