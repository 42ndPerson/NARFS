from __future__ import annotations
from dataclasses import dataclass, field

from operator_base import Operator
from operators.attention import DecodeAttention, PrefillAttention
from operators.elementwise import RMSNorm, RoPE, SwiGLUActivation
from operators.gemm import Gemm
from workload_context import TokenCount


@dataclass(frozen=True)
class TransformerBlockConfig:
    hidden_size: int
    num_heads: int          # query heads
    num_kv_heads: int       # KV heads (GQA)
    head_size: int
    intermediate_size: int  # FFN inner dim
    rope_dim: int
    flash_block_size: int = 64
    dtype_bytes: int = 2


@dataclass(frozen=True)
class TransformerBlock:
    """Helper that materializes one decoder layer's operator sequence.

    Sequence (residual adds folded in / negligible):
      pre-attn RMSNorm -> fused QKV gemm -> RoPE(Q+K) ->
      Prefill|Decode attention -> attn-output gemm ->
      pre-ffn RMSNorm -> gate gemm -> up gemm -> SwiGLU -> down gemm
    """

    config: TransformerBlockConfig
    token_count: TokenCount  # PREFILL or DECODE — selects attention variant + GEMM M

    def operators(self) -> list[Operator]:
        c = self.config
        tc = self.token_count
        q_dim = c.num_heads * c.head_size
        kv_dim = c.num_kv_heads * c.head_size

        if tc is TokenCount.PREFILL:
            attention: Operator = PrefillAttention(
                num_heads=c.num_heads,
                num_kv_heads=c.num_kv_heads,
                head_size=c.head_size,
                flash_block_size=c.flash_block_size,
                dtype_bytes=c.dtype_bytes,
            )
        else:
            attention = DecodeAttention(
                num_heads=c.num_heads,
                num_kv_heads=c.num_kv_heads,
                head_size=c.head_size,
                dtype_bytes=c.dtype_bytes,
            )

        return [
            RMSNorm(c.hidden_size, tc, c.dtype_bytes),
            Gemm(n=q_dim + 2 * kv_dim, k=c.hidden_size, token_count=tc, dtype_bytes=c.dtype_bytes),
            RoPE(
                num_heads=c.num_heads + c.num_kv_heads,
                head_size=c.head_size,
                rope_dim=c.rope_dim,
                token_count=tc,
                dtype_bytes=c.dtype_bytes,
            ),
            attention,
            Gemm(n=c.hidden_size, k=q_dim, token_count=tc, dtype_bytes=c.dtype_bytes),
            RMSNorm(c.hidden_size, tc, c.dtype_bytes),
            Gemm(n=c.intermediate_size, k=c.hidden_size, token_count=tc, dtype_bytes=c.dtype_bytes),
            Gemm(n=c.intermediate_size, k=c.hidden_size, token_count=tc, dtype_bytes=c.dtype_bytes),
            SwiGLUActivation(c.intermediate_size, tc, c.dtype_bytes),
            Gemm(n=c.hidden_size, k=c.intermediate_size, token_count=tc, dtype_bytes=c.dtype_bytes),
        ]


@dataclass(frozen=True)
class Repeat:
    """A run of identical operator groups: `count` copies of `operators` in sequence.

    Use a list of Repeats to describe heterogeneous model structure
    (e.g. [Repeat(32, block), Repeat(1, [final_norm, lm_head])]).
    """

    count: int
    operators: list[Operator] = field(default_factory=list)

    def expand(self) -> list[Operator]:
        return list(self.operators) * self.count


def expand_layers(layers: list[Operator | Repeat]) -> list[Operator]:
    out: list[Operator] = []
    for item in layers:
        if isinstance(item, Repeat):
            out.extend(item.expand())
        else:
            out.append(item)
    return out
