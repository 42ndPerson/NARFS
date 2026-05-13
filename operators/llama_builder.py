from __future__ import annotations
from dataclasses import dataclass

from model import Model
from network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from operator_base import Operator
from operators.attention import DecodeAttention, PrefillAttention
from operators.pattern import (
    OperatorWithReqs,
    PatternEntry,
    LayerRepeat,
)
from operators.elementwise import RMSNorm, RoPE, SwiGLUActivation
from operators.embed import TokenEmbedding
from operators.gemm import Gemm
from operators.llama_config import LlamaConfig
from parallelism_role import ParallelismRole, TPMode
from relation import Relation
from workload_context import TokenCount


# Canonical Megatron-style sharding roles for a Llama layer.
_COL = ParallelismRole(tp=TPMode.COLUMN_PARALLEL)
_ROW = ParallelismRole(tp=TPMode.ROW_PARALLEL)
_HEADS = ParallelismRole(tp=TPMode.HEADS_PARALLEL)
_VOCAB = ParallelismRole(tp=TPMode.VOCAB_PARALLEL)


def _phase_to_token_count(phase: str) -> TokenCount:
    if phase in ("prefill", "training"):
        return TokenCount.PREFILL
    if phase == "decode":
        return TokenCount.DECODE
    raise ValueError(f"Unknown phase: {phase!r}")


def _activation_size_fn(h: int, dtype_bytes: int, tc: TokenCount):
    """Closure for `data_GB_fn`: T * h * dtype_bytes / 1e9, where T is the
    per-invocation token count for the chosen TokenCount.
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


def _dp_grad_all_reduce(
    ops: list[Operator], parallelism: ParallelismConfig, payload_tag: str
) -> NetworkRequirement:
    """DP gradient AllReduce sized to the per-rank parameter byte count.

    `payload_tag` is a discriminator mounted into `payload_id` so the
    NetworkRequirement-dedup machinery doesn't collapse template-level
    differences in parameter size (e.g. V3 dense layer vs MoE layer).
    """
    total_GB = sum(op.parameter_bytes_per_rank(parallelism) for op in ops) / 1e9
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
    """One Llama decoder layer with full-model shapes + Megatron-style
    parallelism roles.

    Operators carry full-model shapes; per-rank division happens inside the
    operator based on its `parallelism_role` and the runtime `parallelism`.
    AllReduce after attn_out and ffn_down sums the row-parallel partial outputs.
    """
    tc = _phase_to_token_count(phase)
    qkv_n = (arch.n_q + 2 * arch.n_kv) * arch.head_size
    q_dim = arch.n_q * arch.head_size

    if tc is TokenCount.PREFILL:
        attention: Operator = PrefillAttention(
            num_heads=arch.n_q,
            num_kv_heads=arch.n_kv,
            head_size=arch.head_size,
            flash_block_size=arch.flash_block_size,
            dtype_bytes=arch.dtype_bytes,
            parallelism_role=_HEADS,
        )
    else:
        attention = DecodeAttention(
            num_heads=arch.n_q,
            num_kv_heads=arch.n_kv,
            head_size=arch.head_size,
            dtype_bytes=arch.dtype_bytes,
            parallelism_role=_HEADS,
        )

    no_req: list[NetworkRequirement] = []
    ar_req: list[NetworkRequirement] = [_tp_all_reduce(arch, tc)] if parallelism.tp > 1 else []

    layer: list[OperatorWithReqs] = [
        (RMSNorm(arch.h, tc, arch.dtype_bytes), no_req),
        # QKV: column-parallel along the combined head dim.
        (Gemm(n=qkv_n, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes,
              parallelism_role=_COL), no_req),
        (
            RoPE(
                num_heads=arch.n_q + arch.n_kv,
                head_size=arch.head_size,
                rope_dim=arch.rope_dim,
                token_count=tc,
                dtype_bytes=arch.dtype_bytes,
                parallelism_role=_HEADS,
            ),
            no_req,
        ),
        (attention, no_req),
        # attn_out: row-parallel — input dim (q_dim) is sharded; emits TP-AR.
        (Gemm(n=arch.h, k=q_dim, token_count=tc, dtype_bytes=arch.dtype_bytes,
              parallelism_role=_ROW), ar_req),
        (RMSNorm(arch.h, tc, arch.dtype_bytes), no_req),
        # gate / up: column-parallel along the intermediate dim.
        (Gemm(n=arch.m, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes,
              parallelism_role=_COL), no_req),
        (Gemm(n=arch.m, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes,
              parallelism_role=_COL), no_req),
        (SwiGLUActivation(intermediate_size=arch.m, token_count=tc,
                          dtype_bytes=arch.dtype_bytes, parallelism_role=_COL), no_req),
        # ffn_down: row-parallel — input dim (m) is sharded; emits TP-AR.
        (Gemm(n=arch.h, k=arch.m, token_count=tc, dtype_bytes=arch.dtype_bytes,
              parallelism_role=_ROW), ar_req),
    ]

    # Phase 4: per-layer DP gradient AllReduce in training when dp > 1.
    if phase == "training" and parallelism.dp > 1:
        dp_ar = _dp_grad_all_reduce(
            [op for op, _ in layer], parallelism, "llama_layer"
        )
        last_op, last_reqs = layer[-1]
        layer[-1] = (last_op, last_reqs + [dp_ar])

    return layer


@dataclass(frozen=True)
class LlamaModel(Model):
    """Llama-style decoder model. Architecture must be a `LlamaConfig`.

    Full graph: embed, n_layers transformer blocks, final norm, LM head.
    LM head is TP-sharded along vocab; AllGather collects logits across TP
    ranks when tp > 1.
    """

    def forward_sequence(
        self, phase: str, parallelism: ParallelismConfig
    ) -> list[PatternEntry]:
        arch = self.architecture
        if not isinstance(arch, LlamaConfig):
            raise TypeError(
                f"LlamaModel requires LlamaConfig, got {type(arch).__name__}"
            )

        # Early divisibility checks for clearer build-time errors. Operators
        # would re-raise these at compute time, but catching them here
        # surfaces the failure at mapping construction.
        tp = parallelism.tp
        if tp > 1:
            if arch.vocab % tp != 0:
                raise ValueError(f"vocab ({arch.vocab}) must be divisible by tp ({tp})")
            if arch.n_q % tp != 0:
                raise ValueError(f"n_q ({arch.n_q}) must be divisible by tp ({tp})")
            if arch.n_kv % tp != 0:
                raise ValueError(f"n_kv ({arch.n_kv}) must be divisible by tp ({tp})")
            if arch.m % tp != 0:
                raise ValueError(f"m ({arch.m}) must be divisible by tp ({tp})")

        layer = build_llama_layer(arch, phase, parallelism)
        tc = _phase_to_token_count(phase)
        lm_head_reqs: list[NetworkRequirement] = (
            [_lm_head_all_gather(arch, tc)] if parallelism.tp > 1 else []
        )
        return [
            # Embedding is replicated across TP in Llama (full vocab table on every rank).
            (TokenEmbedding(vocab=arch.vocab, h=arch.h, token_count=tc,
                            dtype_bytes=arch.dtype_bytes), []),
            LayerRepeat(arch.n_layers, layer),
            (RMSNorm(arch.h, tc, arch.dtype_bytes), []),
            # LM head: vocab-parallel — splits N along the vocab dim, gathers logits.
            (Gemm(n=arch.vocab, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes,
                  parallelism_role=_VOCAB), lm_head_reqs),
        ]
