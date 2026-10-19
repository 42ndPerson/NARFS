"""L2: handles and the binder.

A lowering builds its program with handles: opaque, hashable keys that name
"the devices of stage 2 of call actor_train" or "the TP groups of call
generate". The binder maps each handle to arrays for one candidate. Builders
never see placements and placements never see components, so there is no
builder-by-placement compatibility matrix.

Contract: a lowering declares the handles it uses (the tree's leaves and
scopes hold them), and `bind` must supply every one; a mismatch is an error
at bind time that names the call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from narfs.blocks import Collective, OnDevices, Transfer, Work, walk
from narfs.parallel.layouts import ParamLayout, reshard_pairs, stage_boundary_pairs
from narfs.parallel.mesh import AXES, Placement


@dataclass(frozen=True)
class Devices:
    """The devices of a call, or of one of its PP stages."""
    call: str
    stage: int | None = None


@dataclass(frozen=True)
class Group:
    """The communication groups of a call along `axes`, inside one PP stage."""
    call: str
    axes: tuple
    stage: int | None = None

    def __post_init__(self) -> None:
        axes = (self.axes,) if isinstance(self.axes, str) else tuple(self.axes)
        if not axes or any(a not in AXES for a in axes):
            raise ValueError(f"group axes must be a non-empty subset of {AXES}, got {self.axes!r}")
        object.__setattr__(self, "axes", axes)


@dataclass(frozen=True)
class Boundary:
    """Activation sends of a call from PP stage `src` to stage `dst`, one per
    (dp, ep, cp, tp) coordinate."""
    call: str
    src: int
    dst: int


@dataclass(frozen=True)
class Reshard:
    """Moving a model's parameters from one call's layout to another's."""
    src: str
    dst: str
    layout: ParamLayout


@dataclass(frozen=True)
class Hosts:
    """Host threads of a call: one per GPU (`per="rank"`, a training
    process per GPU) or one per replica (`per="replica"`, an engine process
    keyed by its lowest GPU)."""
    call: str
    per: str = "rank"

    def __post_init__(self) -> None:
        if self.per not in ("rank", "replica"):
            raise ValueError("Hosts.per must be 'rank' or 'replica'")


@dataclass(frozen=True)
class Pool:
    """A named CPU pool of the problem."""
    name: str


def handles(tree) -> set:
    """Every handle a tree uses."""
    out = set()
    for n in walk(tree):
        if isinstance(n, Collective):
            out.add(n.comm)
        elif isinstance(n, Transfer):
            out.add(n.pairs)
        elif isinstance(n, Work):
            out.add(n.resource)
        elif isinstance(n, OnDevices):
            out.add(n.devset)
    return out


class Binder:
    """Binds handles to arrays for one candidate: {call: Placement}. Bound
    arrays are cached per handle, so a handle used by many leaves (every TP
    all-reduce of a call) costs one lookup and keeps one array identity."""

    def __init__(self, placements: dict, host_ids=None):
        self.placements = dict(placements)
        self.host_ids = host_ids                      # GPU ids -> host resource ids (the topology's)
        self._bound: dict = {}

    def __getitem__(self, h):
        v = self._bound.get(h)
        if v is None:
            v = self._bound[h] = self._bind(h)
        return v

    def _placement(self, call: str) -> Placement:
        try:
            return self.placements[call]
        except KeyError:
            raise KeyError(f"call {call!r} has no placement") from None

    def _stage(self, pl: Placement, call: str, stage):
        if stage is not None and not 0 <= stage < pl.degrees.pp:
            raise KeyError(f"call {call!r} has no PP stage {stage} (pp = {pl.degrees.pp})")

    def _bind(self, h):
        match h:
            case Devices():
                pl = self._placement(h.call)
                self._stage(pl, h.call, h.stage)
                return pl.devices if h.stage is None else pl.stage_devices(h.stage)
            case Group():
                pl = self._placement(h.call)
                self._stage(pl, h.call, h.stage)
                return pl.groups(h.axes, h.stage)
            case Boundary():
                pl = self._placement(h.call)
                self._stage(pl, h.call, h.src)
                self._stage(pl, h.call, h.dst)
                src, dst = stage_boundary_pairs(pl, h.src, h.dst)
                return src, dst, np.ones(src.size)
            case Reshard():
                return reshard_pairs(h.layout, self._placement(h.src), self._placement(h.dst))
            case Hosts():
                pl = self._placement(h.call)
                devs = pl.devices if h.per == "rank" else pl.groups(("pp", "cp", "tp")).min(axis=1)
                return devs if self.host_ids is None else self.host_ids(devs)
            case Pool():
                return h.name
        raise TypeError(f"not a handle: {h!r}")

    def check(self, tree) -> None:
        """Bind every handle of `tree` now, so a mismatch fails before evaluation."""
        for h in handles(tree):
            self[h]
