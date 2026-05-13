"""Validation runner: compares simulator predictions to published references.

Run as a standalone script:
    .venv/bin/python validation/run_validation.py

Prints a prediction-vs-reference table to stdout. Writes findings.md alongside
this script if any scenario is more than 30% off the reference (regardless of
whether it passes its tolerance). Exits non-zero if any scenario exceeds its
tolerance.

Does NOT auto-tune efficiency factors. Investigation lives in findings.md.
"""
from __future__ import annotations

# Put the project root on sys.path so the bare absolute imports below
# resolve when this file is run as a script.
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from model_mapping import ModelMapping, SimulationResult
from network_requirement import ParallelismConfig
from operator_base import Operator
from simulator import simulate_rlhf_iteration
from stages import RLHFTimeline
from strategies import MegatronStrategy
from model_mapping_group import ModelMappingGroup
from topo import Device, HostTopo, Link, Path as TopoPath
from validation.scenarios import SCENARIOS, Scenario


def _dev(host: int) -> Device:
    """Single-device-per-host convention used by the validation harness."""
    return Device(host_id=host, gpu_id=0)


def _path_through_link(src: Device, dst: Device, shared: Link) -> TopoPath:
    """Path from src to dst going through `shared`, with 0-latency
    huge-bandwidth attach links so the shared link drives contention.
    """
    huge = 1e9
    attach_src = Link(node_a_id=src, node_b_id=shared.node_a_id, bandwidth=huge, latency=0.0)
    attach_dst = Link(node_a_id=shared.node_b_id, node_b_id=dst, bandwidth=huge, latency=0.0)
    return TopoPath([attach_src, shared, attach_dst])


# ---- topology + mapping helpers ---------------------------------------------


def build_topology(num_hosts: int, spine_bandwidth_gbps: float) -> HostTopo:
    """Single shared spine link between every device pair.

    Pessimistic baseline: ALL device pairs share one Link instance, so every
    inter-device transfer in the cluster contends on the same logical pipe.
    This bounds the worst case for cross-device collective contention but
    over-prices it on real NVLink/NVSwitch fabrics.
    """
    if num_hosts <= 1:
        return HostTopo(device_connections={})
    spine = Link(node_a_id=-1, node_b_id=-2, bandwidth=spine_bandwidth_gbps, latency=0.0)
    connections = {
        (_dev(i), _dev(j)): _path_through_link(_dev(i), _dev(j), spine)
        for i in range(num_hosts)
        for j in range(num_hosts)
        if i != j
    }
    return HostTopo(device_connections=connections)


def build_nvswitch_topology(
    num_hosts: int, link_bandwidth_gbps: float
) -> HostTopo:
    """Full-bisection NVSwitch model: every ordered device pair gets its own
    dedicated `Link` at NVLink-class bandwidth.

    `link_loads` in `topo.HostTopo` is keyed by Link *object identity*
    (frozen-dataclass hash), so distinct Link instances don't contend with
    each other. This matches a real single-node NVSwitch fabric where each
    GPU pair has a private full-bandwidth lane: an 8-rank ring AllReduce
    running at NVLink-class speeds doesn't bottleneck on a shared spine.
    """
    if num_hosts <= 1:
        return HostTopo(device_connections={})
    connections: dict[tuple[Device, Device], TopoPath] = {}
    for i in range(num_hosts):
        for j in range(num_hosts):
            if i == j:
                continue
            link = Link(
                node_a_id=i, node_b_id=j,
                bandwidth=link_bandwidth_gbps, latency=0.0,
            )
            connections[(_dev(i), _dev(j))] = _path_through_link(_dev(i), _dev(j), link)
    return HostTopo(device_connections=connections)


def build_multinode_topology(
    num_hosts: int,
    hosts_per_node: int,
    intra_node_bandwidth_gbps: float,
    inter_node_bandwidth_gbps: float,
) -> HostTopo:
    """Multi-node fat-tree-ish: NVSwitch within a node, IB between nodes.

    Within-node pairs each get a private full-bandwidth Link (NVSwitch).
    Cross-node pairs (node_a ≠ node_b) share one Link per directed
    (node_a, node_b) pair — every GPU on node_a talking to every GPU on
    node_b contends on the same inter-node IB link. This is conservative
    versus a true rail-aligned fat tree but materially better than the
    single-spine pessimistic baseline.
    """
    if num_hosts <= 1:
        return HostTopo(device_connections={})
    if hosts_per_node <= 0:
        raise ValueError("hosts_per_node must be > 0")

    def _node_of(h: int) -> int:
        return h // hosts_per_node

    # One inter-node Link per directed (node_a, node_b) pair.
    inter_node_links: dict[tuple[int, int], Link] = {}

    connections: dict[tuple[Device, Device], TopoPath] = {}
    for i in range(num_hosts):
        for j in range(num_hosts):
            if i == j:
                continue
            ni, nj = _node_of(i), _node_of(j)
            if ni == nj:
                # Intra-node: private NVSwitch lane per pair.
                link = Link(
                    node_a_id=i, node_b_id=j,
                    bandwidth=intra_node_bandwidth_gbps, latency=0.0,
                )
            else:
                # Inter-node: shared IB link per (node, node) pair.
                key = (ni, nj)
                link = inter_node_links.get(key)
                if link is None:
                    link = Link(
                        node_a_id=-(ni * 1000 + nj + 1),
                        node_b_id=-(nj * 1000 + ni + 1),
                        bandwidth=inter_node_bandwidth_gbps, latency=0.0,
                    )
                    inter_node_links[key] = link
            connections[(_dev(i), _dev(j))] = _path_through_link(_dev(i), _dev(j), link)
    return HostTopo(device_connections=connections)


def shard_grid(
    parallelism: ParallelismConfig,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Return (num_hosts, [(dp, pp, tp, ep), ...]).

    Phase 2 made EP an independent axis from TP, so this enumerates the
    full 4D coordinate set. One host per (dp, pp, tp, ep) cell.
    """
    coords = [
        (d, p, t, e)
        for d in range(parallelism.dp)
        for p in range(parallelism.pp)
        for t in range(parallelism.tp)
        for e in range(parallelism.ep)
    ]
    n = parallelism.dp * parallelism.pp * parallelism.tp * parallelism.ep
    return n, coords


def build_model_mapping(scenario: Scenario) -> tuple[ModelMapping, HostTopo]:
    model = scenario.model_class(
        name=scenario.name,
        role="actor",
        architecture=scenario.architecture,
    )
    num_hosts, _ = shard_grid(scenario.parallelism)
    mm = MegatronStrategy(scenario.parallelism).build_mm(
        model=model,
        workload=scenario.workload,
        devices=[_dev(i) for i in range(num_hosts)],
    )
    topo = _topology_for_scenario(scenario, num_hosts)
    return mm, topo


def _topology_for_scenario(scenario: Scenario, num_hosts: int) -> HostTopo:
    """Dispatch on `scenario.topology_kind`. Defaults to shared-spine."""
    kind = scenario.topology_kind
    if kind == "shared_spine":
        return build_topology(num_hosts, scenario.spine_bandwidth_gbps)
    if kind == "nvswitch":
        return build_nvswitch_topology(num_hosts, scenario.spine_bandwidth_gbps)
    if kind == "multinode":
        return build_multinode_topology(
            num_hosts,
            hosts_per_node=scenario.hosts_per_node,
            intra_node_bandwidth_gbps=scenario.spine_bandwidth_gbps,
            inter_node_bandwidth_gbps=scenario.inter_node_bandwidth_gbps,
        )
    raise ValueError(f"{scenario.name}: unknown topology_kind {kind!r}")


# ---- per-scenario evaluation ------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    description: str
    predicted_ms: float
    reference_ms: float
    error_pct: float
    tolerance_pct: float
    passes_tolerance: bool
    needs_findings: bool
    op_breakdown: list[tuple[str, float]] = field(default_factory=list)
    notes: str = ""


def op_type_breakdown(
    mm: ModelMapping, sim_result: SimulationResult
) -> list[tuple[str, float]]:
    """Aggregate per-op wall-time by Operator subclass name; return sorted fractions."""
    by_type: dict[str, float] = defaultdict(float)
    # GENERATION concatenates prefill + decode breakdown.
    ops: list[Operator] = list(mm.operator_pattern) + list(mm.decode_operator_pattern)
    breakdown = sim_result.per_op_breakdown
    if len(breakdown) != len(ops):
        # Defensive: shouldn't happen with current simulator.
        return []
    for op, t in zip(ops, breakdown):
        by_type[type(op).__name__] += t
    total = sum(by_type.values())
    if total <= 0:
        return []
    return sorted(
        [(name, t / total) for name, t in by_type.items()],
        key=lambda x: -x[1],
    )


def run_scenario(scenario: Scenario) -> ScenarioResult:
    mm, topo = build_model_mapping(scenario)
    group = ModelMappingGroup(model_mappings=[mm])
    timeline = RLHFTimeline(stages=[[group]], boundaries=[])

    iter_result = simulate_rlhf_iteration(timeline, scenario.workload_ctx, scenario.hardware, topo)
    if not iter_result.per_mm_results:
        raise RuntimeError(f"{scenario.name}: simulator returned no per-MM results")
    mm_result = iter_result.per_mm_results[0]

    if scenario.metric == "decode_step":
        response_len = scenario.workload_ctx.response_len
        if response_len <= 0 or mm_result.decode_time is None:
            raise ValueError(
                f"{scenario.name}: decode_step metric needs response_len > 0 and "
                f"a GENERATION workload (got response_len={response_len}, "
                f"decode_time={mm_result.decode_time})"
            )
        predicted_seconds = mm_result.decode_time / response_len
    elif scenario.metric == "prefill":
        if mm_result.prefill_time is None:
            raise ValueError(f"{scenario.name}: prefill metric needs a GENERATION workload")
        predicted_seconds = mm_result.prefill_time
    elif scenario.metric == "total":
        predicted_seconds = iter_result.total_time
    else:
        raise ValueError(f"{scenario.name}: unknown metric {scenario.metric!r}")

    predicted_ms = predicted_seconds * 1000.0
    error_pct = (predicted_ms - scenario.reference_latency_ms) / scenario.reference_latency_ms * 100.0
    passes = abs(error_pct) <= scenario.tolerance_percent
    needs_findings = abs(error_pct) > 30.0

    return ScenarioResult(
        name=scenario.name,
        description=scenario.description,
        predicted_ms=predicted_ms,
        reference_ms=scenario.reference_latency_ms,
        error_pct=error_pct,
        tolerance_pct=scenario.tolerance_percent,
        passes_tolerance=passes,
        needs_findings=needs_findings,
        op_breakdown=op_type_breakdown(mm, mm_result),
    )


# ---- output -----------------------------------------------------------------


def print_table(results: list[ScenarioResult]) -> None:
    header = (
        f"{'scenario':<46} {'predicted':>12} {'reference':>12} "
        f"{'error %':>10} {'tol %':>8} {'pass':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        flag = "PASS" if r.passes_tolerance else "FAIL"
        print(
            f"{r.name:<46} {r.predicted_ms:>10.1f}ms {r.reference_ms:>10.1f}ms "
            f"{r.error_pct:>9.1f}% {r.tolerance_pct:>7.1f}% {flag:>6}"
        )


def _findings_hypothesis(r: ScenarioResult) -> str:
    """Best-effort hypothesis for why a scenario is more than 30% off.

    These are heuristics — the user reviews findings.md and decides.
    """
    err = r.error_pct
    name = r.name
    if err < -30:
        # Predicted too low.
        if "decode" in name:
            return (
                "Pure roofline + GEMV efficiency tier (0.3 compute / 0.65 memory for M<16 "
                "GEMMs) is now the prediction. Total memory traffic per decode step ≈ model "
                "weights once + KV cache reads. The remaining gap to measured wall time is "
                "kernel-launch overhead, sampling, and KV-cache management — the simulator's "
                "scope stops at the roofline. To close further: (a) profile actual GEMV "
                "memory_efficiency on the target A100 — typical measurements are 0.5-0.6 "
                "rather than the 0.65 default; (b) bake in a per-decode-step constant overhead "
                "after profiling a real run. Do NOT auto-tune from validation alone."
            )
        if "prefill" in name:
            return (
                "Pure compute prediction is below measured wall time. Likely contributions: "
                "(a) compute_efficiency=0.7 may be high for small-batch prefill where "
                "tensor-core occupancy is limited (effective ~0.5-0.6 at batch=1); "
                "(b) measured times include framework overhead (allocator, dispatch, sampling) "
                "the simulator does not model; (c) reference may include one-off setup costs."
            )
        return "Predicted below reference; investigate compute_efficiency and overhead."
    if err > 30:
        # Predicted too high.
        if "training" in name:
            return (
                "Above reference. Layer-aware PP partitioning is now in place "
                "(the previous flat-partition bug is fixed); remaining gap likely comes from: "
                "(a) BackwardOp's 3× compute / 3× memory scaling with full activation "
                "recomputation is conservative — FlashAttention backward + selective "
                "checkpointing typically achieve 2.0-2.5× rather than 3×; "
                "(b) compute_efficiency=0.7 may be high for backward kernels where "
                "memory-traffic patterns differ from forward; (c) the optimizer step time "
                "(per-layer AdamOptimizerOps) gets multiplied through the (num_microbatches + "
                "pp - 1) pipeline formula even though optimizer is one-shot per iteration — "
                "minor effect since optimizer time is small relative to backward, but a "
                "principled fix would isolate optimizer from the 1F1B multiplier."
            )
        if "deepseek_v3" in name:
            return (
                "Above reference. PrefillAttention dominates ~80% of predicted wall time. "
                "memory_bytes = 3·n_h·head_size·t²/B·dtype gives ~14 ms per attention layer "
                "for V3 (n_h=128, head_size=192, t=4096, B=64), or ~880 ms across 61 layers "
                "alone. Two refinements would close the gap: (a) memory_efficiency=0.8 is "
                "conservative for H100 HBM3 — measured 0.85-0.9 — bumping it to 0.9 trims "
                "~10%; (b) FlashAttention-3 has lower effective memory traffic than the "
                "3·l²/B·d_h tile model assumes (recomputes Q/K/V tiles instead of materializing). "
                "Re-deriving the FA3 memory factor on H100 would shave the rest."
            )
        if "tp" in (r.parallelism_hint or ""):
            return (
                "Above reference. If data_GB_estimates is non-empty, steady-op contention is "
                "computed against ALL steady transfers as if concurrent — for a single submesh "
                "with serialized layers this over-counts. With empty estimates (the validation "
                "default), this branch shouldn't fire; investigate operator-level cost."
            )
        return (
            "Above reference. Either compute_efficiency too low for this workload, or the "
            "operator pattern includes work the reference doesn't model."
        )
    return "Within 30% — no findings entry needed."


def write_findings_md(results: list[ScenarioResult], out_path: Path) -> None:
    relevant = [r for r in results if r.needs_findings]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = []
    lines.append(f"# Simulator validation findings — {timestamp}")
    lines.append("")
    lines.append(
        "Generated automatically by `run_validation.py` whenever any scenario is "
        "more than 30% off its published reference. Default tolerance is 50%; "
        "training and DeepSeek-V3 are tagged as sanity bounds (100%)."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| scenario | predicted | reference | error % | tolerance % | pass |")
    lines.append("|---|---:|---:|---:|---:|:---:|")
    for r in results:
        flag = "✓" if r.passes_tolerance else "✗"
        lines.append(
            f"| {r.name} | {r.predicted_ms:.1f} ms | {r.reference_ms:.1f} ms | "
            f"{r.error_pct:+.1f}% | {r.tolerance_pct:.0f}% | {flag} |"
        )
    lines.append("")

    lines.append("## Per-scenario investigation")
    lines.append("")
    for r in relevant:
        lines.append(f"### {r.name}")
        lines.append("")
        lines.append(f"_{r.description}_")
        lines.append("")
        lines.append(
            f"- **Predicted:** {r.predicted_ms:.1f} ms  "
            f"**Reference:** {r.reference_ms:.1f} ms  "
            f"**Error:** {r.error_pct:+.1f}%  "
            f"**Tolerance:** ±{r.tolerance_pct:.0f}% "
            f"({'within' if r.passes_tolerance else 'exceeds'} tolerance)"
        )
        lines.append("")
        lines.append("**Dominant operators by wall-time fraction:**")
        lines.append("")
        for name, frac in r.op_breakdown[:6]:
            lines.append(f"- `{name}` — {frac * 100:.1f}%")
        lines.append("")
        lines.append("**Hypothesis:**")
        lines.append("")
        lines.append(_findings_hypothesis(r))
        lines.append("")

    lines.append("## Suggested measurements to fit constants")
    lines.append("")
    lines.append(
        "These would let us replace the literature-typical defaults "
        "(0.7 / 0.8 / 0.4) with profiled values for the specific hardware target. "
        "**Do not auto-tune.**"
    )
    lines.append("")
    lines.append(
        "- Profile a representative QKV GEMM at shape (M=2048, N=12288, K=4096) on the "
        "target A100 to fit `compute_efficiency` for compute-bound prefill."
    )
    lines.append(
        "- Profile a memory-bound GEMM at shape (M=1, N=4096, K=4096) on A100 to fit "
        "`memory_efficiency` and `small_gemm_efficiency` for decode."
    )
    lines.append(
        "- Measure end-to-end Llama-2-7B decode on a single A100 with kernel-launch "
        "overhead profiled (e.g., via NSight): if the gap to roofline is ~constant per "
        "token, bake it in as a per-step overhead constant rather than touching efficiencies."
    )
    lines.append(
        "- For multi-GPU TP scenarios, instrument NVLink utilization during a single "
        "AllReduce — current simulator assumes one shared spine; real NVLink switch fabric "
        "behaves differently under contention."
    )
    lines.append(
        "- For DeepSeek-V3, measure per-token AllToAll wall time on the target topology "
        "to fit a per-EP contention factor (single-spine model is approximate)."
    )
    lines.append("")
    lines.append("## What this run did NOT do")
    lines.append("")
    lines.append("- No efficiency factors were modified.")
    lines.append("- No reference numbers were adjusted.")
    lines.append("- No new operators or formulas were added.")
    lines.append("")
    lines.append(
        "Review this report and decide whether to (a) profile to fit constants, "
        "(b) extend the operator/contention model, or (c) accept current accuracy "
        "and tighten the tolerances on stable scenarios."
    )
    lines.append("")

    out_path.write_text("\n".join(lines))


# ---- main -------------------------------------------------------------------


def main() -> int:
    results = []
    for scenario in SCENARIOS:
        try:
            r = run_scenario(scenario)
        except Exception as e:
            print(f"ERROR running {scenario.name}: {type(e).__name__}: {e}", file=sys.stderr)
            raise
        # Stash some hint for the hypothesis text.
        r.parallelism_hint = (
            f"tp={scenario.parallelism.tp},pp={scenario.parallelism.pp},"
            f"dp={scenario.parallelism.dp},ep={scenario.parallelism.ep}"
        )
        results.append(r)

    print_table(results)

    out_path = Path(__file__).resolve().parent / "findings.md"
    if any(r.needs_findings for r in results):
        write_findings_md(results, out_path)
        print(f"\nfindings.md written to {out_path}")
    elif out_path.exists():
        # Stale file from a previous run that had findings; leave it.
        print(f"\nNo new findings; leaving prior {out_path} in place.")
    else:
        print("\nAll scenarios within 30% of reference. No findings.md generated.")

    failures = [r for r in results if not r.passes_tolerance]
    if failures:
        print(
            f"\n{len(failures)} scenario(s) exceeded their tolerance:",
            file=sys.stderr,
        )
        for r in failures:
            print(
                f"  - {r.name}: error {r.error_pct:+.1f}% (tolerance ±{r.tolerance_pct:.0f}%)",
                file=sys.stderr,
            )
        return 1
    print("\nAll scenarios within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
