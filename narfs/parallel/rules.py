"""Level 2: sharding rules per dimension name.

A rule maps a dimension name to a mesh axis. A dimension that no rule names
is replicated. Divisibility is checked generically for every sharded
dimension; no code here names a concrete architecture.
"""
from __future__ import annotations

from dataclasses import dataclass

from narfs.model.architecture import Architecture
from narfs.parallel.mesh import Degrees


@dataclass(frozen=True)
class ShardingRules:
    name: str
    dim_axis: tuple                          # (dim name, axis) pairs
    replicate_if_smaller: tuple = ("kv_heads",)

    def axis_of(self, dim: str) -> str | None:
        for d, a in self.dim_axis:
            if d == dim:
                return a
        return None


MEGATRON = ShardingRules(
    "megatron",
    (("heads", "tp"), ("kv_heads", "tp"), ("ffn", "tp"), ("vocab", "tp"), ("experts", "ep")),
)


def local_dims(arch: Architecture, degrees: Degrees, rules: ShardingRules) -> dict:
    """Per-rank dimensions. Raises ValueError if a sharded dimension does not
    divide. A dimension in `replicate_if_smaller` that is smaller than its
    axis degree is replicated (for example GQA KV heads when tp > kv_heads)."""
    out = {}
    for name, size in arch.dims:
        axis = rules.axis_of(name)
        deg = degrees.of(axis) if axis else 1
        if deg == 1:
            out[name] = size
        elif size % deg == 0:
            out[name] = size // deg
        elif name in rules.replicate_if_smaller and deg % size == 0:
            out[name] = 1
        else:
            raise ValueError(f"{arch.name}: {name} = {size} is not divisible by {axis} = {deg}")
    if arch.n_layers < degrees.pp:
        raise ValueError(f"{arch.name}: {arch.n_layers} layers cannot fill pp = {degrees.pp}")
    return out


def stage_layers(arch: Architecture, pp: int) -> list[list]:
    """Contiguous split of the layer list into pp stages, as even as possible."""
    layers = arch.layer_list()
    base, extra = divmod(len(layers), pp)
    out, i = [], 0
    for s in range(pp):
        n = base + (1 if s < extra else 0)
        out.append(layers[i:i + n])
        i += n
    return out
