"""L3: variation helpers.

Variation is computed from whole distributions with order statistics, so
results are deterministic and the core stays a mean-value algebra.

    critical_instance   the expected slowest of k lockstep instances whose work
                        varies (tokens per DP rank, rows per EP rank)
"""
from __future__ import annotations

from statistics import NormalDist

_STD = NormalDist()


def max_normal(k: float) -> float:
    """E[max of k independent standard normals], Blom's approximation
    (within 1% for k >= 8, 4% at k = 2)."""
    if k <= 1:
        return 0.0
    return _STD.inv_cdf((k - 0.375) / (k + 0.25))


def critical_instance(mean: float, std: float, k: float) -> float:
    """Expected work of the slowest of k lockstep instances, each with work of
    mean `mean` and standard deviation `std` (normal approximation of a sum of
    many independent items)."""
    return mean + std * max_normal(k)
