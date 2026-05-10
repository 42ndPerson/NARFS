"""Inter-stage resharding strategies.

A ReshardingStrategy converts a (source_mapping, dest_mapping) pair into a
list of NetworkOps that physically move parameter bytes from the source
parallelism layout to the destination layout.

NaiveP2PStrategy is the v1 implementation: it walks Llama parameter
tensors, computes per-tensor (src_host, dst_host) byte movements based on
the natural Megatron sharding scheme, aggregates by host pair, and emits
one P2P NetworkOp per pair. The interface is intentionally generic — a
future HybridFlowMicroDPStrategy can return arbitrary NetworkOp subclasses
(e.g., AllGather within micro-DP groups) without changing transition.py.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from network_op import LogicalTransfer, NetworkOp, P2P
from operators.llama_config import LlamaConfig
from shard import Shard

if TYPE_CHECKING:
    from model_mapping import ModelMapping


@dataclass(frozen=True)
class ParameterTensor:
    """A single weight tensor with the info needed to compute per-rank slices.

    shard_axis:
      0  → row-parallel (split rows across TP)
      1  → column-parallel (split cols across TP)
      None → fully replicated across TP
    pp_anchor:
      "first" → lives on PP rank 0 (embedding)
      "last"  → lives on PP rank pp_size - 1 (LM head)
      "layer" → derived from layer_id × pp_size // n_layers
    """

    name: str
    shape: tuple[int, int]
    dtype_bytes: int
    layer_id: Optional[int] = None
    shard_axis: Optional[int] = None
    pp_anchor: str = "layer"


def _llama_parameters(arch: LlamaConfig) -> list[ParameterTensor]:
    """Llama parameter list under the Megatron-style natural sharding scheme.

    QKV column-parallel, attention output row-parallel, FFN gate/up
    column-parallel, FFN down row-parallel, embedding+LM head vocab-parallel.
    Norm scales (RMSNorm) are negligible and intentionally omitted; total
    parameter accounting differs by O(n_layers · h) bytes.
    """
    out: list[ParameterTensor] = []
    out.append(
        ParameterTensor(
            name="embed",
            shape=(arch.vocab, arch.h),
            dtype_bytes=arch.dtype_bytes,
            shard_axis=0,
            pp_anchor="first",
        )
    )
    n_qkv = arch.n_q + 2 * arch.n_kv
    for layer_id in range(arch.n_layers):
        out.append(
            ParameterTensor(
                name=f"qkv_l{layer_id}",
                shape=(n_qkv * arch.head_size, arch.h),
                dtype_bytes=arch.dtype_bytes,
                layer_id=layer_id,
                shard_axis=0,
                pp_anchor="layer",
            )
        )
        out.append(
            ParameterTensor(
                name=f"o_l{layer_id}",
                shape=(arch.h, arch.n_q * arch.head_size),
                dtype_bytes=arch.dtype_bytes,
                layer_id=layer_id,
                shard_axis=1,
                pp_anchor="layer",
            )
        )
        out.append(
            ParameterTensor(
                name=f"gate_l{layer_id}",
                shape=(arch.m, arch.h),
                dtype_bytes=arch.dtype_bytes,
                layer_id=layer_id,
                shard_axis=0,
                pp_anchor="layer",
            )
        )
        out.append(
            ParameterTensor(
                name=f"up_l{layer_id}",
                shape=(arch.m, arch.h),
                dtype_bytes=arch.dtype_bytes,
                layer_id=layer_id,
                shard_axis=0,
                pp_anchor="layer",
            )
        )
        out.append(
            ParameterTensor(
                name=f"down_l{layer_id}",
                shape=(arch.h, arch.m),
                dtype_bytes=arch.dtype_bytes,
                layer_id=layer_id,
                shard_axis=1,
                pp_anchor="layer",
            )
        )
    out.append(
        ParameterTensor(
            name="lm_head",
            shape=(arch.vocab, arch.h),
            dtype_bytes=arch.dtype_bytes,
            shard_axis=0,
            pp_anchor="last",
        )
    )
    return out


def _pp_rank_for(param: ParameterTensor, pp_size: int, n_layers: int) -> int:
    if param.pp_anchor == "first":
        return 0
    if param.pp_anchor == "last":
        return pp_size - 1
    if param.pp_anchor == "layer":
        if param.layer_id is None:
            raise ValueError(f"layer-anchored param {param.name} missing layer_id")
        return (param.layer_id * pp_size) // n_layers
    raise ValueError(f"Unknown pp_anchor: {param.pp_anchor!r}")


def _shards_at(mapping: "ModelMapping", dp: int, pp: int) -> dict[int, Shard]:
    """Returns {tp_rank: Shard} for the given (dp, pp) coordinate."""
    return {
        s.tp: s
        for s in mapping.shards_to_host_ids
        if s.dp == dp and s.pp == pp
    }


class ReshardingStrategy(ABC):
    @abstractmethod
    def compute_network_ops(
        self,
        source: "ModelMapping",
        dest: "ModelMapping",
    ) -> list[NetworkOp]:
        ...


class NaiveP2PStrategy(ReshardingStrategy):
    """For each parameter tensor, compute (src_host, dst_host) byte movements,
    aggregate per pair across all parameters, emit one P2P per non-zero pair.

    Assumes the natural Megatron sharding scheme (QKV column-parallel, O
    row-parallel, gate/up column-parallel, down row-parallel, embed+LM head
    vocab-parallel). Treats DP as canonical (dp=0 source → dp=0 dest); a
    real implementation would also broadcast to other DP replicas, but the
    extra fan-out is the same parameter set replicated and orthogonal to
    the resharding cost we model here.

    Currently supports only LlamaConfig models. V3 (with MoE expert
    sharding) requires a separate enumeration and is left for a later phase.
    """

    def compute_network_ops(
        self,
        source: "ModelMapping",
        dest: "ModelMapping",
    ) -> list[NetworkOp]:
        arch = source.model.architecture
        if not isinstance(arch, LlamaConfig):
            raise NotImplementedError(
                "NaiveP2PStrategy only supports LlamaConfig models in v1; "
                "V3 / MoE resharding will need its own enumeration."
            )

        params = _llama_parameters(arch)
        n_layers = arch.n_layers

        agg: dict[tuple[int, int], int] = defaultdict(int)
        for p in params:
            for pair, bytes_moved in self._movements_for_param(
                p, source, dest, n_layers
            ).items():
                agg[pair] += bytes_moved

        ops: list[NetworkOp] = []
        for (src_host, dst_host), bytes_total in agg.items():
            if bytes_total <= 0:
                continue
            transfer = LogicalTransfer(
                src_host_id=src_host,
                dst_host_id=dst_host,
                data_GB=bytes_total / 1e9,
            )
            # P2P.DEFAULT_PHASE is BOUNDARY, which is what we want for
            # one-shot resharding traffic.
            ops.append(P2P(logical_transfers=[transfer]))
        return ops

    def _movements_for_param(
        self,
        param: ParameterTensor,
        source: "ModelMapping",
        dest: "ModelMapping",
        n_layers: int,
    ) -> dict[tuple[int, int], int]:
        src_pp = _pp_rank_for(param, source.parallelism.pp, n_layers)
        dst_pp = _pp_rank_for(param, dest.parallelism.pp, n_layers)
        src_shards = _shards_at(source, dp=0, pp=src_pp)
        dst_shards = _shards_at(dest, dp=0, pp=dst_pp)
        if not src_shards or not dst_shards:
            return {}

        rows, cols = param.shape
        movements: dict[tuple[int, int], int] = defaultdict(int)

        if param.shard_axis is None:
            # Replicated: one canonical source replica → one canonical dest
            # replica. Other dest TP ranks would receive via intra-TP broadcast
            # (not modeled here).
            src_host = source.shards_to_host_ids[src_shards[0]]
            dst_host = dest.shards_to_host_ids[dst_shards[0]]
            movements[(src_host, dst_host)] = rows * cols * param.dtype_bytes
            return movements

        sharded_dim = param.shape[param.shard_axis]
        other_dim = param.shape[1 - param.shard_axis]
        src_tp = source.parallelism.tp
        dst_tp = dest.parallelism.tp

        if sharded_dim % src_tp != 0:
            raise ValueError(
                f"Param {param.name}: shard dim {sharded_dim} not divisible by "
                f"source tp {src_tp}"
            )
        if sharded_dim % dst_tp != 0:
            raise ValueError(
                f"Param {param.name}: shard dim {sharded_dim} not divisible by "
                f"dest tp {dst_tp}"
            )
        src_chunk = sharded_dim // src_tp
        dst_chunk = sharded_dim // dst_tp

        for src_t in range(src_tp):
            src_lo, src_hi = src_t * src_chunk, (src_t + 1) * src_chunk
            src_host = source.shards_to_host_ids[src_shards[src_t]]
            for dst_t in range(dst_tp):
                dst_lo, dst_hi = dst_t * dst_chunk, (dst_t + 1) * dst_chunk
                overlap = max(0, min(src_hi, dst_hi) - max(src_lo, dst_lo))
                if overlap == 0:
                    continue
                dst_host = dest.shards_to_host_ids[dst_shards[dst_t]]
                movements[(src_host, dst_host)] += overlap * other_dim * param.dtype_bytes
        return movements


def llama_total_parameter_bytes(arch: LlamaConfig) -> int:
    """Sum of parameter bytes for a Llama model under the natural sharding scheme.

    Exposed so tests can compare resharding bytes against the architectural
    target. Mirrors `_llama_parameters`.
    """
    return sum(p.shape[0] * p.shape[1] * p.dtype_bytes for p in _llama_parameters(arch))
