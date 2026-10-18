"""Level 3: mesh axes on devices.

A Placement is an integer array: mesh coordinate -> device ID. The axes are
always stored in the canonical order AXES; `order` only decides which axis
varies fastest when devices are assigned from a list.

Data: EP ranks hold different data, as in Megatron-Core and in DP + EP
inference engines. So the data replicas are dp x ep, and the replicas of a
non-expert parameter are the (dp, ep) groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AXES = ("dp", "pp", "ep", "cp", "tp")


@dataclass(frozen=True)
class Degrees:
    dp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    tp: int = 1

    def __post_init__(self) -> None:
        for a in AXES:
            if getattr(self, a) < 1:
                raise ValueError(f"degree {a} must be >= 1")

    @property
    def size(self) -> int:
        out = 1
        for a in AXES:
            out *= getattr(self, a)
        return out

    def of(self, axis: str | tuple) -> int:
        """Degree of one axis, or the product for a tuple of axes."""
        if isinstance(axis, str):
            return getattr(self, axis)
        out = 1
        for a in axis:
            out *= getattr(self, a)
        return out

    @property
    def data(self) -> int:
        """Data replicas: dp x ep."""
        return self.dp * self.ep

    def shape(self) -> tuple:
        return tuple(getattr(self, a) for a in AXES)


@dataclass(frozen=True)
class Placement:
    degrees: Degrees
    dev_of: np.ndarray = field(compare=False, repr=False)   # shape = degrees.shape()

    def __post_init__(self) -> None:
        if self.dev_of.shape != self.degrees.shape():
            raise ValueError(f"dev_of shape {self.dev_of.shape} does not match degrees {self.degrees.shape()}")
        if len(np.unique(self.dev_of)) != self.dev_of.size:
            raise ValueError("a device appears twice in one placement")

    @property
    def devices(self) -> np.ndarray:
        return self.dev_of.ravel()

    def groups(self, axis: str | tuple, stage: int | None = None) -> np.ndarray:
        """[n_instances, degree] device IDs of every instance along `axis`
        (one axis, or a tuple of axes that form one group). With `stage`,
        only the instances inside that PP stage."""
        axes = (axis,) if isinstance(axis, str) else tuple(axis)
        idx = [AXES.index(a) for a in axes]
        arr = self.dev_of
        if stage is not None and "pp" not in axes:
            arr = arr[:, stage:stage + 1]
        moved = np.moveaxis(arr, idx, list(range(-len(idx), 0)))
        size = 1
        for i in idx:
            size *= arr.shape[i]
        return moved.reshape(-1, size)

    def stage_devices(self, stage: int) -> np.ndarray:
        return self.dev_of[:, stage].ravel()

    def pp_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Sources and destinations of every stage s -> s+1 activation send."""
        src = self.dev_of[:, :-1].ravel()
        dst = self.dev_of[:, 1:].ravel()
        return src, dst


def map_devices(devices, degrees: Degrees, order: tuple = AXES) -> Placement:
    """Assign `devices` (in the given list order) to mesh coordinates.

    `order` lists the axes from slowest to fastest varying. The default puts
    tp innermost, so consecutive devices (usually on one node) form TP groups.
    Any list of devices is allowed, which supports irregular layouts.
    """
    devices = np.asarray(devices, dtype=np.int64)
    if devices.size != degrees.size:
        raise ValueError(f"placement needs {degrees.size} devices, got {devices.size}")
    if sorted(order) != sorted(AXES):
        raise ValueError(f"order must be a permutation of {AXES}")
    shaped = devices.reshape([degrees.of(a) for a in order])
    perm = [order.index(a) for a in AXES]
    return Placement(degrees, np.ascontiguousarray(shaped.transpose(perm)))
