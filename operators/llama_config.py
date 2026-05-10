from __future__ import annotations
from dataclasses import dataclass

from model import ArchitectureConfig


@dataclass(frozen=True)
class LlamaConfig(ArchitectureConfig):
    """Llama-style decoder architecture (GQA + SwiGLU FFN + RoPE)."""

    h: int = 4096
    n_layers: int = 32
    n_q: int = 32           # query heads
    n_kv: int = 8           # KV heads (GQA)
    head_size: int = 128
    m: int = 14336          # FFN intermediate
    rope_dim: int = 128
    vocab: int = 128_000
    flash_block_size: int = 64
    dtype_bytes: int = 2
