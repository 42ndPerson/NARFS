"""Network contention via a composition tree.

The contention model expresses *which collectives co-occur on the fabric*
through a `NetworkComposition` tree composed of two primitives:

  - `Sequential` — children run one-at-a-time. Per-link load is the
    length-weighted average of the children's loads (compute-only
    stretches modeled via `Empty` weigh into the denominator).
  - `Parallel`   — children run concurrently. Per-link load sums.

A `NetworkLeaf` wraps one `NetworkOp` and reports its real per-link
transfer count. `ContentionModel` walks the composition to derive, for
each `NetworkOp`, its *background* load on every link (the sum of
`Parallel`-ancestor siblings' loads). The op's effective bandwidth on a
link is then

    link_bw / max(1, own_real_count + background)

with the op's own transfers contributing their full real count (an
AllToAll's 992 transfers genuinely self-contend) and other concurrent
units showing up only via the averaged background. Sequential-ancestor
siblings contribute 0 — they don't co-occur in time.

The output is a `dict[LogicalTransfer, float]` of per-transfer wall
times — the same interface `steady_transfer_times` exposed before — so
`CombinedOp.wall_time` consumes it unchanged.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from network_op import LogicalTransfer, NetworkOp
    from topo import HostTopo, Link


# ---- composition primitives -------------------------------------------------


class NetworkComposition(ABC):
    """Abstract base for composition tree nodes."""

    @abstractmethod
    def effective_length(self) -> int:
        """Number of timeline positions this composition covers (denominator
        weight in `Sequential` averaging)."""
        ...

    @abstractmethod
    def per_link_load(self, link: "Link", topo: "HostTopo") -> float:
        """Expected concurrent transfer count this composition puts on `link`."""
        ...

    @abstractmethod
    def all_ops(self) -> Iterator["NetworkOp"]:
        """All NetworkOps reachable inside this composition."""
        ...


@dataclass(frozen=True)
class NetworkLeaf(NetworkComposition):
    """One NetworkOp as a leaf. Reports its real transfer count on each
    link by walking its endpoint_pairs through the topology's paths."""
    op: "NetworkOp"

    def effective_length(self) -> int:
        return 1

    def per_link_load(self, link: "Link", topo: "HostTopo") -> float:
        count = 0
        for src, dst in self.op.endpoint_pairs:
            path = topo.device_connections.get((src, dst))
            if path is None:
                continue
            if link in path.path:
                count += 1
        return float(count)

    def all_ops(self) -> Iterator["NetworkOp"]:
        yield self.op


@dataclass(frozen=True)
class Empty(NetworkComposition):
    """A stretch of `length` compute-only timeline positions. Carries no
    network load but contributes to `Sequential`'s averaging denominator
    so compute steps correctly dilute network occupancy."""
    length: int = 1

    def effective_length(self) -> int:
        return self.length

    def per_link_load(self, link: "Link", topo: "HostTopo") -> float:
        return 0.0

    def all_ops(self) -> Iterator["NetworkOp"]:
        return iter(())


@dataclass(frozen=True)
class Sequential(NetworkComposition):
    """Children run one-at-a-time on the timeline. Per-link load is the
    length-weighted average of children's loads."""
    children: list[NetworkComposition] = field(default_factory=list)

    def effective_length(self) -> int:
        return sum(c.effective_length() for c in self.children)

    def per_link_load(self, link: "Link", topo: "HostTopo") -> float:
        total = 0.0
        total_len = 0
        for child in self.children:
            cl = child.effective_length()
            total += child.per_link_load(link, topo) * cl
            total_len += cl
        if total_len == 0:
            return 0.0
        return total / total_len

    def all_ops(self) -> Iterator["NetworkOp"]:
        for c in self.children:
            yield from c.all_ops()


@dataclass(frozen=True)
class Parallel(NetworkComposition):
    """Children run concurrently. Per-link load sums."""
    children: list[NetworkComposition] = field(default_factory=list)

    def effective_length(self) -> int:
        return max((c.effective_length() for c in self.children), default=0)

    def per_link_load(self, link: "Link", topo: "HostTopo") -> float:
        return sum(c.per_link_load(link, topo) for c in self.children)

    def all_ops(self) -> Iterator["NetworkOp"]:
        for c in self.children:
            yield from c.all_ops()


# ---- ContentionModel --------------------------------------------------------


def _collect_relevant_links(
    composition: NetworkComposition, topo: "HostTopo",
) -> set["Link"]:
    """Every Link touched by any path used by any op in the composition."""
    links: set["Link"] = set()
    for op in composition.all_ops():
        for src, dst in op.endpoint_pairs:
            path = topo.device_connections.get((src, dst))
            if path is None:
                continue
            for link in path.path:
                links.add(link)
    return links


def _op_real_count_on_link(
    op: "NetworkOp", link: "Link", topo: "HostTopo",
) -> int:
    """An op's true transfer count on `link` (used as own-load in the
    apply-back, not as averaged background)."""
    count = 0
    for src, dst in op.endpoint_pairs:
        path = topo.device_connections.get((src, dst))
        if path is None:
            continue
        if link in path.path:
            count += 1
    return count


class ContentionModel:
    """Steady-state contention from a `NetworkComposition` tree.

    Walks the tree once to compute each op's *background* load per link
    (sum of `Parallel`-ancestor siblings' loads; `Sequential`-ancestor
    siblings contribute 0). For an op at a link the effective bandwidth
    is `link.bandwidth / max(1, own_real_count + background)`. Each of
    the op's logical transfers then takes `bytes / min effective_bw over
    its path's links` plus the path's latency.

    The returned `dict[LogicalTransfer, float]` matches the legacy
    `steady_transfer_times` shape, so `CombinedOp.wall_time` consumes
    it unchanged.
    """

    def __init__(self, topo: "HostTopo"):
        self.topo = topo

    def steady_snapshot(
        self,
        composition: NetworkComposition,
        op_transfers: dict["NetworkOp", list["LogicalTransfer"]],
    ) -> dict["LogicalTransfer", float]:
        """Per-transfer wall times under the composition's contention."""
        if not op_transfers:
            return {}
        links = _collect_relevant_links(composition, self.topo)
        backgrounds = self._compute_backgrounds(composition, links)

        transfer_times: dict["LogicalTransfer", float] = {}
        for op, transfers in op_transfers.items():
            if not transfers:
                continue
            bg_for_op = backgrounds.get(op, {})
            # Effective bandwidth per relevant link for this op.
            eff_bw_per_link: dict["Link", float] = {}
            for link in links:
                own = _op_real_count_on_link(op, link, self.topo)
                bg = bg_for_op.get(link, 0.0)
                load = max(1.0, own + bg)
                eff_bw_per_link[link] = link.bandwidth / load
            # Per-transfer time = bytes / min(eff_bw over its path's links) + path latency.
            for t in transfers:
                path = self.topo.device_connections.get((t.src_device, t.dst_device))
                if path is None or not path.path:
                    transfer_times[t] = 0.0
                    continue
                min_bw = min(eff_bw_per_link.get(link, link.bandwidth) for link in path.path)
                latency_s = path.latency / 1000.0
                transfer_times[t] = (t.data_GB * 8) / min_bw + latency_s
        return transfer_times

    def _compute_backgrounds(
        self,
        composition: NetworkComposition,
        links: set["Link"],
    ) -> dict["NetworkOp", dict["Link", float]]:
        """For each NetworkOp in the tree, the background load it
        experiences from Parallel-ancestor siblings, per link."""
        result: dict["NetworkOp", dict["Link", float]] = {}
        empty: dict["Link", float] = {link: 0.0 for link in links}
        self._walk(composition, empty, links, result)
        return result

    def _walk(
        self,
        node: NetworkComposition,
        background: dict["Link", float],
        links: set["Link"],
        result: dict["NetworkOp", dict["Link", float]],
    ) -> None:
        if isinstance(node, NetworkLeaf):
            result[node.op] = dict(background)
        elif isinstance(node, Sequential):
            # Sequential siblings don't co-occur in time → no background addition.
            for child in node.children:
                self._walk(child, background, links, result)
        elif isinstance(node, Parallel):
            # Each child sees the OTHER children's loads as background.
            child_loads = [
                {link: child.per_link_load(link, self.topo) for link in links}
                for child in node.children
            ]
            for i, child in enumerate(node.children):
                child_bg = {
                    link: background[link]
                    + sum(child_loads[j][link] for j in range(len(node.children)) if j != i)
                    for link in links
                }
                self._walk(child, child_bg, links, result)
        # Empty: contributes nothing, has no leaves.
