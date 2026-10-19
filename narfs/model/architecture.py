"""Model definitions as data.

A model is named dimensions plus a list of layers. A layer is a tuple of
component names, such as ("gqa", "dense") or ("mamba2", "moe"); each name is
a registered component class (narfs.model.library). A new kind of layer is one
registered class; nothing else changes.

Dimension names used by the built-in components:
    hidden, heads, kv_heads, head_dim, ffn, vocab
    MLA:     q_lora, kv_lora, rope_dim, v_head_dim
    MoE:     experts, expert_ffn, top_k, shared_experts
    Mamba-2: ssm_inner, ssm_heads, ssm_state, ssm_groups, conv_kernel
"""
from __future__ import annotations

from dataclasses import dataclass


def pattern(layers, repeat: int = 1) -> tuple:
    """A layer list: `layers` (layer types) repeated `repeat` times."""
    out = []
    for t in layers:
        out.append(tuple(t) if not isinstance(t, str) else (t,))
    return tuple(out) * repeat


@dataclass(frozen=True)
class Architecture:
    name: str
    dims: tuple                        # (name, int) pairs, so the object is hashable
    layers: tuple                      # one tuple of component names per layer, in model order
    dtype: str = "bf16"
    tied_embeddings: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.dims, dict):
            object.__setattr__(self, "dims", tuple(self.dims.items()))
        names = [n for n, _ in self.dims]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate dimension name")
        object.__setattr__(self, "layers", pattern(self.layers))
        if not self.layers:
            raise ValueError(f"{self.name}: no layers")

    def dim(self, name: str, default: int | None = None) -> int:
        for n, v in self.dims:
            if n == name:
                return v
        if default is not None:
            return default
        raise KeyError(f"{self.name} has no dimension {name!r}")

    def has(self, name: str) -> bool:
        return any(n == name for n, _ in self.dims)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def components(self) -> set:
        return {c for layer in self.layers for c in layer}


def runs(layers) -> list[tuple[tuple, int]]:
    """Consecutive identical layer types, as (layer type, count)."""
    out: list = []
    for t in layers:
        if out and out[-1][0] == t:
            out[-1] = (t, out[-1][1] + 1)
        else:
            out.append((t, 1))
    return out


def llama(name: str, hidden: int, layers: int, heads: int, kv_heads: int, ffn: int, vocab: int,
          head_dim: int | None = None, tied: bool = False) -> Architecture:
    return Architecture(name, dict(hidden=hidden, heads=heads, kv_heads=kv_heads, head_dim=head_dim or hidden // heads,
                                   ffn=ffn, vocab=vocab),
                        pattern([("gqa", "dense")], layers), tied_embeddings=tied)


LLAMA3_8B = llama("llama3-8b", 4096, 32, 32, 8, 14336, 128256)
LLAMA3_70B = llama("llama3-70b", 8192, 80, 64, 8, 28672, 128256)

MIXTRAL_8X7B = Architecture(
    "mixtral-8x7b",
    dict(hidden=4096, heads=32, kv_heads=8, head_dim=128, vocab=32000, experts=8, top_k=2, expert_ffn=14336,
         shared_experts=0),
    pattern([("gqa", "moe")], 32))

DEEPSEEK_V3 = Architecture(
    "deepseek-v3",
    dict(hidden=7168, heads=128, head_dim=128, v_head_dim=128, q_lora=1536, kv_lora=512, rope_dim=64, ffn=18432,
         experts=256, expert_ffn=2048, top_k=8, shared_experts=1, vocab=129280),
    pattern([("mla", "dense")], 3) + pattern([("mla", "moe")], 58))
