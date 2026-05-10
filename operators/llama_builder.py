from __future__ import annotations

from model import ArchitectureConfig
from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operator_base import Operator
from operators.attention import DecodeAttention, PrefillAttention
from operators.builders import (
    OperatorWithReqs,
    PatternEntry,
    TaggedRepeat,
)
from operators.elementwise import RMSNorm, RoPE, SwiGLUActivation
from operators.embed import TokenEmbedding
from operators.gemm import Gemm
from operators.llama_config import LlamaConfig
from relation import Relation
from workload_context import TokenCount


def _phase_to_token_count(phase: str) -> TokenCount:
    if phase in ("prefill", "training"):
        return TokenCount.PREFILL
    if phase == "decode":
        return TokenCount.DECODE
    raise ValueError(f"Unknown phase: {phase!r}")


def _activation_size_fn(h: int, dtype_bytes: int, tc: TokenCount):
    """Closure for `data_GB_fn`: T * h * dtype_bytes / 1e9, where T is the
    per-invocation token count for the chosen TokenCount.

    Captures `h` and `dtype_bytes` at construction so the closure stays a
    pure function of `WorkloadContext`.
    """
    def _fn(ctx) -> float:
        return tc.resolve(ctx) * h * dtype_bytes / 1e9
    return _fn


def _lm_head_gather_size_fn(vocab: int, dtype_bytes: int, tc: TokenCount):
    """Closure for LM-head AllGather: T * vocab * dtype_bytes / 1e9."""
    def _fn(ctx) -> float:
        return tc.resolve(ctx) * vocab * dtype_bytes / 1e9
    return _fn


def _tp_all_reduce(arch: LlamaConfig, tc: TokenCount) -> NetworkRequirement:
    return NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
        # Reduces hidden-state activation [batch, seq, h] across TP ranks.
        data_GB_fn=_activation_size_fn(arch.h, arch.dtype_bytes, tc),
    )


def _lm_head_all_gather(arch: LlamaConfig, tc: TokenCount) -> NetworkRequirement:
    return NetworkRequirement(
        kind=CollectiveKind.ALL_GATHER,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
        data_GB_fn=_lm_head_gather_size_fn(arch.vocab, arch.dtype_bytes, tc),
    )


def _dp_grad_all_reduce(ops: list[Operator], payload_tag: str) -> NetworkRequirement:
    """DP gradient AllReduce sized to the per-rank parameter byte count.

    `payload_tag` is a discriminator mounted into `payload_id` so the
    NetworkRequirement-dedup machinery doesn't collapse template-level
    differences in parameter size (e.g. V3 dense layer vs MoE layer). The
    same-tag requirement still dedupes across all repetitions of the same
    layer template — which is the right behavior, since `_simulate_one`
    counts the resolved NetworkOp's time once per attaching operator.

    `parameter_bytes()` is static; the closure just returns the precomputed
    total. Wrapped through `data_GB_fn` for symmetry with the rest of the
    Phase-1 sizing machinery.
    """
    total_GB = sum(op.parameter_bytes() for op in ops) / 1e9
    return NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.DP,
        payload_id=("dp_grad", payload_tag),
        data_GB_fn=lambda _ctx, _v=total_GB: _v,
    )


def build_llama_layer(
    arch: LlamaConfig,
    phase: str,
    parallelism: ParallelismConfig,
) -> list[OperatorWithReqs]:
    """One Llama decoder layer with TP-sharded shapes and AR requirements.

    Megatron-style TP: heads are split across tp ranks for QKV/attn-output;
    intermediate (m) is split across tp ranks for the FFN. AllReduce after
    attn_out and ffn_down sums the partial outputs.
    """
    tp = parallelism.tp
    if arch.n_q % tp != 0:
        raise ValueError(f"n_q ({arch.n_q}) must be divisible by tp ({tp})")
    if arch.n_kv % tp != 0:
        raise ValueError(f"n_kv ({arch.n_kv}) must be divisible by tp ({tp})")
    if arch.m % tp != 0:
        raise ValueError(f"m ({arch.m}) must be divisible by tp ({tp})")

    n_q_local = arch.n_q // tp
    n_kv_local = arch.n_kv // tp
    m_local = arch.m // tp

    tc = _phase_to_token_count(phase)
    q_dim = n_q_local * arch.head_size
    kv_dim = n_kv_local * arch.head_size

    if tc is TokenCount.PREFILL:
        attention: Operator = PrefillAttention(
            num_heads=n_q_local,
            num_kv_heads=n_kv_local,
            head_size=arch.head_size,
            flash_block_size=arch.flash_block_size,
            dtype_bytes=arch.dtype_bytes,
        )
    else:
        attention = DecodeAttention(
            num_heads=n_q_local,
            num_kv_heads=n_kv_local,
            head_size=arch.head_size,
            dtype_bytes=arch.dtype_bytes,
        )

    no_req: list[NetworkRequirement] = []
    ar_req: list[NetworkRequirement] = [_tp_all_reduce(arch, tc)] if tp > 1 else []

    layer: list[OperatorWithReqs] = [
        (RMSNorm(arch.h, tc, arch.dtype_bytes), no_req),
        (Gemm(n=q_dim + 2 * kv_dim, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), no_req),
        (
            RoPE(
                num_heads=n_q_local + n_kv_local,
                head_size=arch.head_size,
                rope_dim=arch.rope_dim,
                token_count=tc,
                dtype_bytes=arch.dtype_bytes,
            ),
            no_req,
        ),
        (attention, no_req),
        # attn_out: emits TP AllReduce when tp > 1.
        (Gemm(n=arch.h, k=q_dim, token_count=tc, dtype_bytes=arch.dtype_bytes), ar_req),
        (RMSNorm(arch.h, tc, arch.dtype_bytes), no_req),
        (Gemm(n=m_local, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), no_req),
        (Gemm(n=m_local, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), no_req),
        (SwiGLUActivation(intermediate_size=m_local, token_count=tc, dtype_bytes=arch.dtype_bytes), no_req),
        # ffn_down: emits TP AllReduce when tp > 1.
        (Gemm(n=arch.h, k=m_local, token_count=tc, dtype_bytes=arch.dtype_bytes), ar_req),
    ]

    # Phase 4: per-layer DP gradient AllReduce in training when dp > 1.
    if phase == "training" and parallelism.dp > 1:
        dp_ar = _dp_grad_all_reduce([op for op, _ in layer], "llama_layer")
        last_op, last_reqs = layer[-1]
        layer[-1] = (last_op, last_reqs + [dp_ar])

    return layer


def build_llama_pattern(
    arch: ArchitectureConfig,
    phase: str,
    parallelism: ParallelismConfig,
) -> list[PatternEntry]:
    """Full Llama model graph: embed, n_layers transformer blocks, final norm, LM head.

    LM head is TP-sharded along vocab; AllGather collects logits across TP
    ranks when tp > 1. Signature matches `BuildPatternFn`. Cast to LlamaConfig
    at runtime.
    """
    if not isinstance(arch, LlamaConfig):
        raise TypeError(f"build_llama_pattern requires LlamaConfig, got {type(arch).__name__}")

    tp = parallelism.tp
    if arch.vocab % tp != 0:
        raise ValueError(f"vocab ({arch.vocab}) must be divisible by tp ({tp})")
    vocab_local = arch.vocab // tp

    layer = build_llama_layer(arch, phase, parallelism)
    tc = _phase_to_token_count(phase)
    lm_head_reqs: list[NetworkRequirement] = (
        [_lm_head_all_gather(arch, tc)] if tp > 1 else []
    )
    return [
        (TokenEmbedding(vocab=arch.vocab, h=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), []),
        TaggedRepeat(arch.n_layers, layer),
        (RMSNorm(arch.h, tc, arch.dtype_bytes), []),
        (Gemm(n=vocab_local, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), lm_head_reqs),
    ]
