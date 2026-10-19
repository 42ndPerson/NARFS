"""L2: layout transfers, the moves of a tensor from one layout to another.

A stage boundary and a weight reshard are the same operation. The pairs come
from intersecting the two layouts, and a framework policy picks which replica
sends: the source replica with dp index (destination dp) mod (source dp), and
for replicated data the source tp rank (destination tp) mod (source tp), so
reads spread over the source ranks.

For a stage boundary between uniform stages, the intersection is "the same
(dp, ep, cp, tp) coordinate in the next stage".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from narfs.parallel.mesh import Placement, stage_split


@dataclass(frozen=True)
class ParamLayout:
    """A model's parameter bytes by sharding class, as one reshard moves them.

    layers     per layer, full-model bytes: (sharded over tp, replicated,
               sharded over ep)
    embed      input embedding bytes (first stage, vocab-sharded over tp)
    head       output head bytes (last stage, vocab-sharded over tp)
    expert_tp  the destination also shards the routed experts over tp (an
               inference engine without expert parallelism does)
    """
    layers: tuple
    embed: float = 0.0
    head: float = 0.0
    expert_tp: bool = False

    @property
    def total(self) -> float:
        return sum(sum(x) for x in self.layers) + self.embed + self.head


def _overlap(i: int, n: int, j: int, m: int) -> float:
    """Length of [i/n, (i+1)/n) and [j/m, (j+1)/m) intersected."""
    return max(0.0, min((i + 1) / n, (j + 1) / m) - max(i / n, j / m))


def stage_boundary_pairs(pl: Placement, s: int, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Sources and destinations of an activation move from stage s to stage t
    (t = s + 1, or 0 for an interleaved wrap-around)."""
    return pl.dev_of[:, s].ravel(), pl.dev_of[:, t].ravel()


def reshard_pairs(layout: ParamLayout, src: Placement, dst: Placement):
    """(src devices, dst devices, bytes): every destination rank reads its
    slice of every parameter from the source rank that holds it."""
    Ds, Dd = src.degrees, dst.degrees
    n = len(layout.layers)
    s_stages = [set(r) for r in stage_split(n, Ds.pp)]
    d_stages = [set(r) for r in stage_split(n, Dd.pp)]
    srcs, dsts, sizes = [], [], []
    dp = np.arange(Dd.dp)[:, None, None]                        # dst coordinates (dp, ep, cp), broadcast
    ep = np.arange(Dd.ep)[None, :, None]
    for sd in range(Dd.pp):
        for ss in range(Ds.pp):
            common = d_stages[sd] & s_stages[ss]
            p_tp = sum(layout.layers[i][0] for i in common)
            p_rep = sum(layout.layers[i][1] for i in common)
            p_ep = sum(layout.layers[i][2] for i in common)
            if sd == 0 and ss == 0:
                p_tp += layout.embed
            if sd == Dd.pp - 1 and ss == Ds.pp - 1:
                p_tp += layout.head
            if p_tp + p_rep + p_ep == 0:
                continue
            for td in range(Dd.tp):
                d_dev = dst.dev_of[:, sd, :, :, td]                                   # [dp, ep, cp]
                if p_tp:
                    for ts in range(Ds.tp):
                        f = _overlap(td, Dd.tp, ts, Ds.tp)
                        if f > 0:
                            s_dev = src.dev_of[dp % Ds.dp, ss, ep % Ds.ep, 0, ts]
                            _push(srcs, dsts, sizes, s_dev, d_dev, p_tp * f)
                if p_rep:
                    s_dev = src.dev_of[dp % Ds.dp, ss, ep % Ds.ep, 0, td % Ds.tp]
                    _push(srcs, dsts, sizes, s_dev, d_dev, p_rep)
            if p_ep:
                tp = np.arange(Dd.tp)[None, None, :]
                for ed in range(Dd.ep):
                    d_dev = dst.dev_of[:, sd, ed, :, :]                               # [dp, cp, tp]
                    for es in range(Ds.ep):
                        f = _overlap(ed, Dd.ep, es, Ds.ep)
                        if f > 0:
                            s_dev = src.dev_of[np.arange(Dd.dp)[:, None, None] % Ds.dp, ss, es, 0, tp % Ds.tp]
                            _push(srcs, dsts, sizes, s_dev, d_dev, p_ep * f / (Dd.tp if layout.expert_tp else 1))
    if not srcs:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0)
    return np.concatenate(srcs), np.concatenate(dsts), np.concatenate(sizes)


def _push(srcs, dsts, sizes, s_dev, d_dev, size):
    s_dev, d_dev = np.broadcast_arrays(s_dev, d_dev)
    srcs.append(s_dev.ravel())
    dsts.append(d_dev.ravel())
    sizes.append(np.full(d_dev.size, float(size)))
