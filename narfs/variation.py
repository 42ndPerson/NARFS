"""L3: variation helpers.

Variation is computed from whole distributions with order statistics, so
results are deterministic and the core stays a mean-value algebra.

    Dist                a length distribution on 1..cap: the measured body, a
                        fitted tail above the 90th percentile, and the cap atom
    varying_repeat      N instances each repeat a body n_i ~ dist times, at most
                        C at once per group (FIFO), R groups end at one barrier:
                        a Seq of Repeat segments, each with its active count
    critical_instance   the expected slowest of k lockstep instances whose work
                        varies (tokens per DP rank, rows per EP rank)

The tail of a length distribution sets the expected longest of N requests.
Above a threshold u the tail is a generalized Pareto distribution
(Pickands-Balkema-de Haan) with scale beta and shape xi, fitted by censored
maximum likelihood to the observed lengths (Grimshaw's profile in xi/beta).
Below about 1,000 observations the shape is fixed at 0 (an exponential tail,
one fitted value), whose expected maximum has a closed form:

    E[longest of N] = u + beta [Ein(N p_u) - Ein(N p_C)],  p_C = p_u exp(-(C - u)/beta)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np

_STD = NormalDist()
EULER_GAMMA = 0.5772156649015329


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


# ------------------------------------------------------------ tail fits ---

def fit_exponential(excess: np.ndarray, n_capped: int, cap_excess: float) -> float:
    """Censored MLE of an exponential tail's scale: capped values count as exposure."""
    return (float(excess.sum()) + n_capped * cap_excess) / max(1, excess.size)


def fit_gpd(excess: np.ndarray, n_capped: int, cap_excess: float) -> tuple[float, float]:
    """Censored MLE (beta, xi) of a generalized Pareto tail, by Grimshaw's
    profile likelihood over theta = xi / beta."""
    y, nu = excess, max(1, excess.size)
    ymax = max(float(y.max()) if y.size else 0.0, cap_excess)
    thetas = np.concatenate([-np.logspace(-6, math.log10(0.999 / ymax), 40)[::-1], [0.0],
                             np.logspace(-7, 0, 160) / max(ymax, 1.0) * 10])
    best = None
    for th in thetas:
        if abs(th) < 1e-15:
            beta, xi = fit_exponential(y, n_capped, cap_excess), 0.0
            ll = -nu * math.log(beta) - (float(y.sum()) + n_capped * cap_excess) / beta
        else:
            if 1 + th * ymax <= 0:
                continue
            s = float(np.log1p(th * y).sum()) + n_capped * math.log1p(th * cap_excess)
            xi = s / nu
            beta = xi / th
            if beta <= 0 or xi == 0:
                continue
            ll = (-nu * math.log(beta) - (1 + 1 / xi) * float(np.log1p(th * y).sum())
                  - (n_capped / xi) * math.log1p(th * cap_excess))
        if best is None or ll > best[0]:
            best = (ll, beta, xi)
    return best[1], best[2]


def _gpd_sf(y: np.ndarray, beta: float, xi: float) -> np.ndarray:
    y = np.maximum(y, 0.0)
    if abs(xi) < 1e-12:
        return np.exp(-y / beta)
    return np.clip(1 + xi * y / beta, 0.0, None) ** (-1.0 / xi)


def _ein(z: float) -> float:
    """Ein(z) = integral from 0 to z of (1 - e^-t)/t dt."""
    if z <= 0:
        return 0.0
    if z < 1:
        s, term, k = 0.0, 1.0, 1
        while True:
            term *= -z / k
            s += term / k
            if abs(term / k) < 1e-16:
                break
            k += 1
        return -s
    # E1 by a continued fraction (modified Lentz), then Ein = E1 + ln z + gamma
    b, c, d = z + 1.0, 1e300, 1.0 / (z + 1.0)
    h = d
    for i in range(1, 300):
        a = -i * i
        b += 2.0
        d = 1.0 / (a * d + b)
        c = b + a / c
        h *= c * d
        if abs(c * d - 1.0) < 1e-15:
            break
    return h * math.exp(-z) + math.log(z) + EULER_GAMMA


def exponential_tail_max(u: float, beta: float, p_u: float, cap: float, n: float) -> float:
    """Closed form of E[min(longest of n, cap)] for an exponential tail above u."""
    p_c = p_u * math.exp(-(cap - u) / beta)
    return u + beta * (_ein(n * p_u) - _ein(n * p_c))


# ---------------------------------------------------------------- Dist ---

@dataclass(frozen=True, eq=False)
class Dist:
    """Lengths on the integers 1..cap: pmf[l - 1] = P(L = l)."""
    pmf: np.ndarray = field(repr=False)
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = np.asarray(self.pmf, dtype=float)
        if p.ndim != 1 or p.size == 0 or (p < -1e-12).any():
            raise ValueError("a Dist needs a non-negative 1-D probability array")
        total = p.sum()
        if not 0.999 < total < 1.001:
            raise ValueError(f"probabilities sum to {total}, not 1")
        p = np.clip(p, 0.0, None) / total
        p.setflags(write=False)
        object.__setattr__(self, "pmf", p)
        cdf = np.cumsum(p)
        cdf.setflags(write=False)
        object.__setattr__(self, "_cdf", cdf)
        lengths = np.arange(1, p.size + 1, dtype=float)
        object.__setattr__(self, "mean", float((lengths * p).sum()))
        object.__setattr__(self, "second", float((lengths ** 2 * p).sum()))

    @property
    def cap(self) -> int:
        return self.pmf.size

    @property
    def std(self) -> float:
        return math.sqrt(max(self.second - self.mean ** 2, 0.0))

    def cdf(self, x) -> np.ndarray:
        """P(L <= x) for real x."""
        x = np.asarray(x, dtype=float)
        idx = np.floor(x).astype(np.int64) - 1
        return np.where(x >= 1, self._cdf[np.clip(idx, 0, self.cap - 1)], 0.0)

    def sf(self, x) -> np.ndarray:
        return 1.0 - self.cdf(x)

    def quantile(self, q: float) -> int:
        return int(np.searchsorted(self._cdf, q) + 1)

    def expected_max(self, n: float) -> float:
        """E[longest of n independent lengths]."""
        F = np.concatenate([[0.0], self._cdf[:-1]])            # P(L <= l) for l = 0 .. cap - 1
        return float((1.0 - F ** n).sum())

    # ------------------------------------------------------- construction ---
    @classmethod
    def histogram(cls, values, probs, cap: int | None = None) -> "Dist":
        v = np.maximum(1, np.rint(np.asarray(values, dtype=float))).astype(np.int64)
        cap = int(cap or v.max())
        v = np.minimum(v, cap)
        pmf = np.bincount(v - 1, weights=np.asarray(probs, dtype=float), minlength=cap)
        return cls(pmf / pmf.sum(), {"kind": "histogram"})

    @classmethod
    def from_samples(cls, samples, cap: int | None = None, tail: str = "auto", q: float = 0.9,
                     min_exceed: int = 10) -> "Dist":
        """The measured body, a fitted tail above the q quantile ("gpd", "exp",
        "none", or "auto": GPD from 1,000 observations, exponential below),
        and the cap atom (responses cut at max_tokens)."""
        x = np.maximum(1, np.rint(np.asarray(samples, dtype=float))).astype(np.int64)
        cap = int(cap or x.max())
        x = np.minimum(x, cap)
        n = x.size
        if tail == "auto":
            tail = "gpd" if n >= 1000 else "exp"
        u = int(np.quantile(x, q))
        above = x[x > u]
        info = {"kind": "measured", "n_obs": n, "cap": cap, "p_cap": float(np.mean(x >= cap))}
        if tail == "none" or above.size < min_exceed or u >= cap - 1:
            return cls(np.bincount(x - 1, minlength=cap) / n, {**info, "tail": "none"})
        capped = above >= cap
        excess, cap_excess = (above[~capped] - u).astype(float), float(cap - u)
        if tail == "exp":
            beta, xi = fit_exponential(excess, int(capped.sum()), cap_excess), 0.0
        elif tail == "gpd":
            beta, xi = fit_gpd(excess, int(capped.sum()), cap_excess)
        else:
            raise ValueError(f"unknown tail {tail!r}")
        p_u = above.size / n
        pmf = np.bincount(x[x <= u] - 1, minlength=cap).astype(float) / n
        lengths = np.arange(u + 1, cap + 1, dtype=float)
        sf_prev = _gpd_sf(lengths - 1 - u, beta, xi)
        sf_here = _gpd_sf(lengths - u, beta, xi)
        tail_pmf = p_u * (sf_prev - sf_here)
        tail_pmf[-1] = p_u * sf_prev[-1]                       # everything past the cap is cut at the cap
        pmf[u:] = tail_pmf
        return cls(pmf, {**info, "tail": tail, "u": u, "beta": float(beta), "xi": float(xi), "p_u": p_u})

    @classmethod
    def lognormal(cls, mu: float, sigma: float, cap: int) -> "Dist":
        """A lognormal length distribution cut at `cap` (for examples and tests)."""
        edges = np.arange(0, cap + 1, dtype=float)
        z = (np.log(np.maximum(edges, 1e-300)) - mu) / sigma
        F = np.array([_STD.cdf(v) for v in z])
        F[0] = 0.0
        pmf = np.diff(F)
        pmf[-1] += 1.0 - F[-1]
        return cls(pmf, {"kind": "lognormal", "mu": mu, "sigma": sigma, "cap": cap})

    @classmethod
    def load(cls, path) -> "Dist":
        """{"values": [...], "probs": [...], "cap": C} or {"samples": [...], "cap": C}."""
        d = json.loads(Path(path).read_text())
        if "samples" in d:
            return cls.from_samples(d["samples"], d.get("cap"), d.get("tail", "auto"))
        return cls.histogram(d["values"], d["probs"], d.get("cap"))


# ------------------------------------------------------- varying repeat ---

@dataclass(frozen=True)
class Segment:
    steps: float          # repetitions of the body in this segment
    active: float         # mean running instances per group (at least 1)
    age: float            # mean repetitions already done by a running instance
    admissions: float     # instances started per repetition (after the first cohort)


@dataclass(frozen=True)
class Trajectory:
    segments: tuple
    end: float            # expected repetitions until the barrier
    first: int            # instances that start at once in each group
    per_group: int        # instances per group
    info: dict


_TRAJECTORIES: dict = {}


def trajectory(dist: Dist, instances: int, slots: int, groups: int, segments: int = 16,
               grid: int = 1024) -> Trajectory:
    """The fluid FIFO schedule in step space. Instance j of a group starts when
    a slot frees: s_j = max(0, j - C + 1) E[L] / C. The barrier ends at the
    order statistic of (start + length) over every instance of every group:
    P(end <= x) = prod over j of F(x - s_j), to the power R."""
    key = (id(dist), instances, slots, groups, segments, grid)
    hit = _TRAJECTORIES.get(key)
    if hit is not None and hit[0] is dist:
        return hit[1]
    n = math.ceil(instances / groups)
    C = max(1, min(slots, n))
    mean = dist.mean
    starts = np.maximum(0, np.arange(n) - C + 1) * mean / C
    uniq, counts = np.unique(starts, return_counts=True)
    x = np.linspace(0.0, float(starts[-1]) + dist.cap, grid)
    log_p = np.zeros(grid)
    act = np.zeros(grid)
    age = np.zeros(grid)
    for lo in range(0, uniq.size, 256):
        s, w = uniq[lo:lo + 256, None], counts[lo:lo + 256, None]
        y = x[None, :] - s
        F = dist.cdf(y)
        log_p += (w * np.log(np.maximum(F, 1e-300))).sum(axis=0)
        alive = np.where(y >= 0, 1.0 - F, 0.0) * w
        act += alive.sum(axis=0)
        age += (alive * np.maximum(y, 0.0)).sum(axis=0)
    p_end = np.exp(groups * log_p)
    end = float(np.trapezoid(1.0 - p_end, x))
    edges = np.linspace(0.0, end, segments + 1)
    sample = np.linspace(0.0, end, segments * 32 + 1)
    a_s = np.interp(sample, x, act)
    g_s = np.interp(sample, x, age)
    out = []
    late = starts[C:]
    for k in range(segments):
        sel = (sample >= edges[k]) & (sample <= edges[k + 1])
        a = float(a_s[sel].mean())
        ag = float(g_s[sel].sum() / max(a_s[sel].sum(), 1e-12))
        span = edges[k + 1] - edges[k]
        adm = float(((late >= edges[k]) & (late < edges[k + 1])).sum()) / span if span > 0 else 0.0
        out.append(Segment(span, min(max(a, 1.0), C), ag, adm))   # fluid starts can overshoot the slots early on
    info = {"end": end, "slots": C, "per_group": n, "longest_of_all": dist.expected_max(n * groups)}
    t = Trajectory(tuple(out), end, C, n, info)
    if len(_TRAJECTORIES) > 512:
        _TRAJECTORIES.clear()
    _TRAJECTORIES[key] = (dist, t)
    return t


def varying_repeat(dist: Dist, instances: int, slots: int, groups: int, body, segments: int = 16,
                   name: str = "varying repeat"):
    """A Seq of Repeat segments; `body(segment)` builds one repetition at the
    segment's active count, age and admissions."""
    from narfs.blocks import Repeat, Seq
    t = trajectory(dist, instances, slots, groups, segments)
    return Seq(tuple(Repeat(s.steps, body(s), name=f"segment {i + 1}: {s.active:.0f} active")
                     for i, s in enumerate(t.segments)), name=name)
