"""The candidate encoding: what an external optimizer passes to the simulator.

The optimizer lives outside and calls the simulator; this module holds no
search code. A candidate gives, for each call of the flow, a device list,
degrees, an axis order and a micro-batch size. It is the only input that
changes between evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass

from narfs.parallel.mesh import AXES, Degrees


@dataclass(frozen=True)
class CallPlan:
    devices: tuple                   # device ids, in the order the mapping uses
    degrees: Degrees
    order: tuple = AXES              # mesh axes from slowest to fastest varying
    micro_batch: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(int(d) for d in self.devices))


@dataclass(frozen=True)
class Candidate:
    plans: tuple                     # (call name, CallPlan) pairs

    @classmethod
    def of(cls, plans: dict) -> "Candidate":
        return cls(tuple(plans.items()))

    def plan(self, name: str) -> CallPlan | None:
        for n, p in self.plans:
            if n == name:
                return p
        return None

    def with_plan(self, name: str, plan: CallPlan) -> "Candidate":
        others = tuple((n, p) for n, p in self.plans if n != name)
        return Candidate(others + ((name, plan),))


def colocated(groups: list, degrees: dict, micro_batch: dict | None = None, start: int = 0) -> Candidate:
    """Consecutive device ranges for groups of calls: `groups` is a list of
    (device count, [call names]); the calls of one group share its devices."""
    micro_batch = micro_batch or {}
    plans, at = {}, start
    for count, names in groups:
        devs = tuple(range(at, at + count))
        for name in names:
            plans[name] = CallPlan(devs, degrees[name], micro_batch=micro_batch.get(name, 1))
        at += count
    return Candidate.of(plans)
