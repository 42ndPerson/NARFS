from __future__ import annotations
from dataclasses import dataclass

from model import ArchitectureConfig


@dataclass(frozen=True)
class V3Config(ArchitectureConfig):
    """DeepSeek-V3 architectural parameters."""

    h: int = 7168
    n_h: int = 128
    head_size: int = 128
    d_q_compress: int = 1536          # Q latent dim
    d_kv_compress: int = 512          # KV latent dim
    d_rope: int = 64                  # decoupled RoPE per head
    m_dense: int = 18432              # dense FFN intermediate
    m_expert: int = 2048              # per-expert FFN intermediate
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    top_k: int = 8
    vocab: int = 129280
    n_layers: int = 61
    n_dense_layers: int = 3
    moe_imbalance_factor: float = 1.15
    flash_block_size: int = 64
    dtype_bytes: int = 2
