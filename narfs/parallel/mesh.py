"""L2: mesh degrees, device mapping and the stage split.

A Placement is an integer array: mesh coordinate -> device id, stored in the
canonical axis order AXES. `order` only decides which axis varies fastest when
devices are assigned from a list, and any device list is allowed, so irregular
layouts work.

Data replicas are dp x ep: EP ranks hold different data, as in Megatron-Core
and in DP + EP inference engines.
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
        return self.of(AXES)

    def of(self, axes) -> int:
        """Degree of one axis, or the product for a tuple of axes."""
        if isinstance(axes, str):
            return getattr(self, axes)
        out = 1
        for a in axes:
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
    dev_of: np.ndarray = field(compare=False, repr=False)       # shape = degrees.shape()

    def __post_init__(self) -> None:
        if self.dev_of.shape != self.degrees.shape():
            raise ValueError(f"device array shape {self.dev_of.shape} does not match degrees {self.degrees.shape()}")
        if len(np.unique(self.dev_of)) != self.dev_of.size:
            raise ValueError("a device appears twice in one placement")

    @property
    def devices(self) -> np.ndarray:
        return self.dev_of.ravel()

    def stage_devices(self, stage: int) -> np.ndarray:
        return self.dev_of[:, stage].ravel()

    def groups(self, axes, stage: int | None = None) -> np.ndarray:
        """[instances, members]: the devices of every instance along `axes` (one
        axis, or a tuple that forms one group). With `stage`, only the instances
        inside that PP stage."""
        axes = (axes,) if isinstance(axes, str) else tuple(axes)
        idx = [AXES.index(a) for a in axes]
        arr = self.dev_of
        if stage is not None and "pp" not in axes:
            arr = arr[:, stage:stage + 1]
        moved = np.moveaxis(arr, idx, list(range(-len(idx), 0)))
        size = 1
        for i in idx:
            size *= arr.shape[i]
        return np.ascontiguousarray(moved.reshape(-1, size))


def map_devices(devices, degrees: Degrees, order: tuple = AXES) -> Placement:
    """Assign `devices`, in list order, to mesh coordinates. `order` lists the
    axes from slowest to fastest varying; the default puts tp innermost, so
    consecutive devices (usually one node) form the TP groups."""
    devices = np.asarray(devices, dtype=np.int64)
    if devices.size != degrees.size:
        raise ValueError(f"degrees {degrees.shape()} need {degrees.size} devices, got {devices.size}")
    if sorted(order) != sorted(AXES):
        raise ValueError(f"order must be a permutation of {AXES}")
    shaped = devices.reshape([degrees.of(a) for a in order])
    perm = [order.index(a) for a in AXES]
    return Placement(degrees, np.ascontiguousarray(shaped.transpose(perm)))


def stage_split(n_layers: int, pp: int) -> list[range]:
    """Contiguous layer ranges of the pp stages, as even as possible (earlier
    stages take the extra layers). A framework policy, owned here."""
    if n_layers < pp:
        raise ValueError(f"{n_layers} layers cannot fill pp = {pp} stages")
    base, extra = divmod(n_layers, pp)
    out, start = [], 0
    for s in range(pp):
        n = base + (1 if s < extra else 0)
        out.append(range(start, start + n))
        start += n
    return out
