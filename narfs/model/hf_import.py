"""Import adapter: a Hugging Face `config.json` dict -> Architecture.

Supported families: Llama-style dense models (Llama, Mistral, Qwen2),
Mixtral-style MoE, and DeepSeek-V3-style MLA + MoE. The adapter is thin on
purpose: it only renames fields.
"""
from __future__ import annotations

import json
from pathlib import Path

from narfs.model.architecture import Architecture, LayerSpec


def from_hf_config(cfg: dict | str | Path, name: str | None = None) -> Architecture:
    if not isinstance(cfg, dict):
        cfg = json.loads(Path(cfg).read_text())
    name = name or cfg.get("_name_or_path") or cfg.get("model_type", "imported")
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    dims = {"hidden": hidden, "heads": heads, "vocab": vocab, "ffn": cfg.get("intermediate_size", 4 * hidden)}

    mla = "kv_lora_rank" in cfg and cfg.get("kv_lora_rank")
    if mla:
        dims.update(head_dim=cfg["qk_nope_head_dim"], rope_dim=cfg["qk_rope_head_dim"],
                    v_head_dim=cfg.get("v_head_dim", cfg["qk_nope_head_dim"]),
                    q_lora=cfg.get("q_lora_rank") or hidden, kv_lora=cfg["kv_lora_rank"])
        attention = "mla"
    else:
        dims.update(kv_heads=cfg.get("num_key_value_heads", heads),
                    head_dim=cfg.get("head_dim") or hidden // heads)
        attention = "gqa"

    experts = cfg.get("n_routed_experts") or cfg.get("num_local_experts") or cfg.get("num_experts")
    if experts:
        dims.update(experts=experts, top_k=cfg["num_experts_per_tok"],
                    expert_ffn=cfg.get("moe_intermediate_size") or cfg["intermediate_size"],
                    shared_experts=cfg.get("n_shared_experts") or 0)
        dense = cfg.get("first_k_dense_replace", 0)
        specs = []
        if dense:
            specs.append(LayerSpec(dense, attention, "dense"))
        specs.append(LayerSpec(layers - dense, attention, "moe"))
    else:
        specs = [LayerSpec(layers, attention, "dense")]
    return Architecture(name=str(name), dims=tuple(dims.items()), layers=tuple(specs))
