"""Architecture descriptions: named dimensions plus a list of layer types.

A description is data. All operator knowledge is in narfs.model.library.

Dimension names used by the library and by sharding rules:
    hidden, heads, kv_heads, head_dim, ffn, vocab
    MLA:  q_lora, kv_lora, rope_dim, v_head_dim
    MoE:  experts, expert_ffn, top_k, shared_experts
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpec:
    count: int
    attention: str = "gqa"     # "gqa" or "mla"
    ffn: str = "dense"         # "dense" or "moe"


@dataclass(frozen=True)
class Architecture:
    name: str
    dims: tuple                # tuple of (name, int) pairs, so the object is hashable
    layers: tuple              # tuple of LayerSpec, in model order
    dtype_bytes: int = 2
    moe_imbalance: float = 1.15

    def __post_init__(self) -> None:
        names = [n for n, _ in self.dims]
        if len(names) != len(set(names)):
            raise ValueError("duplicate dimension name")
        for spec in self.layers:
            if spec.attention not in ("gqa", "mla") or spec.ffn not in ("dense", "moe"):
                raise ValueError(f"unknown layer type {spec}")

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
        return sum(s.count for s in self.layers)

    def layer_list(self) -> list[LayerSpec]:
        """One LayerSpec (with count 1) for every layer, in order."""
        out = []
        for s in self.layers:
            out += [LayerSpec(1, s.attention, s.ffn)] * s.count
        return out


def llama(name: str, hidden: int, layers: int, heads: int, kv_heads: int, ffn: int,
          vocab: int, head_dim: int | None = None) -> Architecture:
    return Architecture(
        name=name,
        dims=(("hidden", hidden), ("heads", heads), ("kv_heads", kv_heads),
              ("head_dim", head_dim or hidden // heads), ("ffn", ffn), ("vocab", vocab)),
        layers=(LayerSpec(layers, "gqa", "dense"),),
    )


LLAMA3_8B = llama("llama3-8b", 4096, 32, 32, 8, 14336, 128256)
LLAMA3_70B = llama("llama3-70b", 8192, 80, 64, 8, 28672, 128256)

DEEPSEEK_V3 = Architecture(
    name="deepseek-v3",
    dims=(("hidden", 7168), ("heads", 128), ("head_dim", 128), ("v_head_dim", 128),
          ("q_lora", 1536), ("kv_lora", 512), ("rope_dim", 64), ("ffn", 18432),
          ("experts", 256), ("expert_ffn", 2048), ("top_k", 8), ("shared_experts", 1),
          ("vocab", 129280)),
    layers=(LayerSpec(3, "mla", "dense"), LayerSpec(58, "mla", "moe")),
)
