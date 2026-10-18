# RLHF latency simulator

Analytical latency simulator for RLHF scheduling, modeled after the
DistServe latency model (Appendix A of the DistServe paper) and adapted for
GQA/MLA, SwiGLU FFN, MoE with shared experts, TP/PP/DP/EP parallelism, 1F1B
pipeline scheduling, and inter-stage parameter resharding.

The intent is **fast analytical predictions** for RLHF orchestration
research — not cycle-accurate simulation. At the default efficiency
factors and the validation scenarios shipped with this repo, predictions
land within ±25% on compute-bound prefill workloads and within ~2× on
training and decode workloads (where overheads the roofline doesn't model
dominate). See [validation/findings.md](validation/findings.md) and the
limitations section below for the per-scenario story.

> **New:** a rebuild of the simulator for placement search is in
> [narfs/](narfs/README.md). It sits beside this code and does not change it.

## Architecture

The codebase is organized into five layers, bottom-up:

```
topology/   →  operators/   →  schemes/   →  mapping/   →  simulation/
(fabric)      (kernels)        (sharding)     (per-MM)     (timeline)
```

Plus a small set of root foundations consumed by every layer
([model.py](model.py), [workload_context.py](workload_context.py),
[parallelism.py](parallelism.py), [network_requirement.py](network_requirement.py),
[network_op.py](network_op.py)).

### 1. Topology layer — [topology/](topology/)

[topology/topo.py](topology/topo.py) describes a cluster as a graph of
`Device(host_id, gpu_id)` endpoints connected by `Path`s of `Link`s
(each link has bandwidth Gbps and latency ms). `Path`'s constructor can
attach **phantom 0-latency self-loop links** at the endpoints to model
per-GPU NVLink-aggregate / NIC egress caps — concurrent transfers
leaving the same GPU share those phantoms and contend.

`HostTopo` exposes two snapshot styles consumed by upper layers:

- `get_simultaneous_logical_transfer_times` — shared-bandwidth
  contention. Each transfer's effective bandwidth = link bandwidth /
  number of transfers traversing that link. Used by the legacy steady
  snapshot and as the building block of the composition-tree model
  (see [§ContentionModel](#contention-model)).
- `get_simultaneous_system_flow_time` — flow-time contention. Each
  link's drain time = total bytes on that link / bandwidth. Used at
  stage boundaries (parameter resharding P2Ps) where the queue
  genuinely drains once.

[topology/hardware.py](topology/hardware.py) carries `HardwareSpec`
(see [§Calibration](#calibration-story)).

### 2. Operator layer — [operators/](operators/)

[operators/base.py](operators/base.py) — `Operator` is a per-kernel
cost function (`compute_flops(ctx)`, `memory_bytes(ctx)`,
`is_small_dim(ctx)`). The base class derives `compute_time`,
`memory_time`, and `kernel_time = max(compute_time, memory_time)` —
the standard roofline. Operators are **parallelism-agnostic**: they
report full-model FLOPs / bytes, and the scheme layer bakes per-rank
scaling into a closure attached to each `CombinedOp`.

Concrete operators:

- [gemm.py](operators/gemm.py) — `Gemm`. Reports an efficiency tier
  per call: large GEMMs use `compute_efficiency`, M < 16 uses
  `gemv_efficiency` (decode), any dim < 1024 (non-GEMV) uses
  `small_gemm_efficiency`.
- [attention.py](operators/attention.py) — `PrefillAttention`
  (FlashAttention prefill) and `DecodeAttention` (GQA / MLA decode
  with configurable `kv_bytes_per_token`).
- [elementwise.py](operators/elementwise.py) — `RMSNorm`,
  `SwiGLUActivation`, `RoPE`.
- [embed.py](operators/embed.py) — `TokenEmbedding`.
- [optimizer.py](operators/optimizer.py) — `AdamOptimizerOp`.
- [backward.py](operators/backward.py) — `BackwardOp` wraps a forward
  op and reports 3× / 2× the forward cost (full activation
  recomputation vs. selective checkpointing; toggled by `recompute`
  on the strategy, default `False` → 2× for the realistic
  selective-checkpointing case).
- [mla.py](operators/mla.py), [moe.py](operators/moe.py),
  [llama_builder.py](operators/llama_builder.py),
  [v3_block.py](operators/v3_block.py) — whole-model graph builders.
  `DispatchMarker` and `CombineMarker` (in `moe.py`) are zero-cost
  placeholders the scheme recognizes as attachment points for MoE
  dispatch/combine all-to-alls.
- [pattern.py](operators/pattern.py) — pattern primitives:
  `LabeledOp(role, op)`, `LayerRepeat(count, members)`,
  `OperatorWithReqs`, `OpRole`. Builders emit a `list[PatternEntry]`
  in this vocabulary, parallelism-agnostic.

### 3. Scheme layer — [schemes/](schemes/)

A `ParallelismScheme` owns *how* a model shards onto hardware. The
split is two-stage:

- `scheme.build_plan(model, workload)` → `Plan`. Plan is
  **device-agnostic**: it carries the decorated forward (and decode,
  for GENERATION) patterns plus the abstract `Shard`s the scheme
  requires. Decorations bake per-rank kernel-time scaling and emit
  `NetworkRequirement`s on operators that trigger collectives.
- `assemble(plan, placement, devices, topology)` → `ModelMapping`.
  Placement is a function `(plan, devices, topology) → dict[Shard,
  Device]` that binds shards to physical GPUs. Different placements
  can be tried against the same plan to optimize cross-stage
  resharding cost (HybridFlow's 3D HybridEngine pattern).

For convenience, `scheme.build_mm(model, workload, devices,
recompute=False)` does both steps with a default placement.

[schemes/strategies.py](schemes/strategies.py) — the
`ParallelismScheme` ABC, `Plan`, `MegatronScheme` (the canonical
column/row TP + EP + per-layer DP-grad AR), and the `ReshardingScheme`
ABC.

[schemes/llama_resharding.py](schemes/llama_resharding.py) —
`MegatronToMegatronP2P`, the Llama-specific resharding scheme used by
`StageTransition`. Walks Llama parameter tensors under Megatron
sharding (QKV column-parallel, O row-parallel, gate/up column-parallel,
down row-parallel, embed/LM-head vocab-parallel), computes
per-(src_device, dst_device) byte movement via slice-overlap
arithmetic, and emits one `P2P` per non-zero pair.

### 4. Per-model layer — [mapping/](mapping/)

[mapping/model_mapping.py](mapping/model_mapping.py) —
`ModelMapping(model, parallelism, workload, shards_to_devices,
forward_pattern, decode_pattern, recompute, ...)` holds one model
placed on one parallelism configuration for one `Workload`
(GENERATION / PREPARATION / TRAINING / DORMANT). One ModelMapping per
(model, parallelism, workload) — they don't try to serve multiple
workloads.

`__post_init__` runs the following in order:

1. Validates the shard layout against `parallelism` along TP, EP, PP,
   DP. Each axis with degree > 1 must have that many ranks with the
   *other three* axes held fixed; under-provisioned layouts raise
   `ValueError` rather than silently shrinking a collective.
2. Expands the scheme-decorated pattern into a `forward_seq:
   list[CombinedOp]`. For GENERATION, also builds `decode_seq` from
   the decode pattern. For TRAINING, derives backward via
   `derive_backward_pattern(forward_pattern, recompute)` and appends
   a per-layer `AdamOptimizerOp` sized from the strategy's
   `per_rank_parameter_bytes`.
3. Resolves each unique `NetworkRequirement` into a concrete
   `NetworkOp` (deduplicated by `(kind, relation, eta, group,
   payload_id)`).
4. Layer-aware, cost-balanced PP partition — co-locates each layer's
   forward / backward / optimizer ops on the same stage (backward
   needs the activations and weights stored locally after its
   corresponding forward).
5. When `parallelism.pp > 1`, installs per-microbatch PP activation
   `P2P` ops (forward; plus backward in TRAINING) with phase override
   `STEADY` — these stream per microbatch alongside compute, not as
   one-shot boundary handoffs.
6. Builds the steady and fill/drain **network composition trees** (see
   [§ContentionModel](#contention-model)) that the upper layer feeds
   to `ContentionModel`.

`simulate(ctx, hw, op_transfers, transfer_times_steady,
transfer_times_boundary, transfer_times_fill_drain=None)` walks
`forward_seq` (and `decode_seq` in GENERATION), combines each op's
`kernel_time` with its network requirements via
`NetworkRequirement.combine_times`, and applies a 1F1B pipeline
formula when `pp > 1`. Returns a `SimulationResult` with `total_time`,
`prefill_time` / `decode_time` / `training_time` (when relevant),
`pp_stage_times`, and `per_op_breakdown`.

The PP bubble formula:

- **Single-snapshot** (no `transfer_times_fill_drain` passed, legacy
  behavior): `(nmb + pp − 1) × (max_stage + p2p_per_mb) +
  optimizer_total`. `optimizer_total = max(stage_optimizer_times)`
  is separated from the per-microbatch loop because the optimizer
  runs once per iteration in parallel across PP stages, not once per
  microbatch slot.
- **Weighted two-snapshot** (used by `simulate_stage`):
  `steady_slots × (max_stage_steady + p2p_steady) +
  fill_drain_slots × (max_stage_fd + p2p_fd) + optimizer_total`,
  with `steady_slots = max(0, nmb − pp + 1)` and
  `fill_drain_slots = 2(pp − 1)`. During fill and drain, only some
  pipeline stages have work, so the steady-state composition
  (everything concurrent) overestimates contention there.

For GENERATION, `decode_time = (decode_step.total_time +
hw.decode_overhead_per_token_ms / 1000) × response_len`. The overhead
captures per-token framework + kernel-launch costs (~300 kernels ×
~10-30 μs/launch plus paged-attention block management on vLLM-class
systems) that the pure roofline doesn't see.

#### Network composition

[mapping/contention.py](mapping/contention.py) expresses *which
collectives co-occur on the fabric* through a `NetworkComposition`
tree:

- `NetworkLeaf(op)` — one `NetworkOp`. Reports its real per-link
  transfer count from `endpoint_pairs` walked through the topology.
- `Empty(length)` — `length` compute-only timeline positions. Carries
  no load but contributes to `Sequential` averaging.
- `Sequential(children)` — children run one-at-a-time. Per-link load
  is the length-weighted average of children's loads.
- `Parallel(children)` — children run concurrently. Per-link load
  sums.

Each `ModelMapping` builds two compositions for its forward seq:

- **steady** — one pipeline slot's worth of activity. At pp=1 it's a
  `Sequential` over `forward_seq`. At pp>1 it's a `Parallel` over
  per-stage `Sequential`s plus the PP-P2P leaves (every stage
  processes some microbatch concurrently in steady state).
- **fill_drain** — for pp>1, the per-stage Sequentials are wrapped in
  an outer `Sequential` (only one stage active at a time). For pp=1,
  identical to steady.

`ModelMappingGroup` wraps these in a per-group `Sequential` (members
within a group run one after another); `simulate_stage` wraps the
groups in a per-stage `Parallel`. `ContentionModel.steady_snapshot`
then walks the assembled tree once to derive each op's *background*
load on every link (sum of `Parallel`-ancestor siblings' loads —
`Sequential`-ancestor siblings contribute zero), and projects per-link
effective bandwidths `link.bw / max(1, own_real_count + background)`
back to per-transfer wall times.

#### Workload-dependent collective sizing

`NetworkRequirement.data_GB_fn: Callable[[WorkloadContext], float]`
is the sizing closure attached to a requirement at build time.
Builders close over architecture parameters (h, vocab, top_k, dtype)
and produce a function whose only argument is the current ctx — so
transfer sizes automatically track changes in `microbatch_size`,
`prompt_len`, etc. between simulate calls.

`resolve_op_transfers(ctx, model_mappings)` (in
[mapping/model_mapping_group.py](mapping/model_mapping_group.py))
evaluates every registered `NetworkRequirement` against `ctx` and
materializes a fresh `dict[NetworkOp, list[LogicalTransfer]]`. Each
op's `logical_transfers(total_GB)` splits the total according to its
collective shape (ring → total/N per hop, AllToAll → total/N per
pair, P2P / Broadcast → full payload). The result feeds both the
contention snapshot and `CombinedOp.wall_time` — coherently sourced
from the same evaluation.

Two requirements with identical `(kind, relation, eta, group,
payload_id)` dedupe to one NetworkOp — `data_GB_fn` is excluded from
equality and hash. `payload_id` is the discriminator that prevents
collapsing collectives that fire repeatedly with *different* sizes:
the per-layer DP-gradient AllReduces in training carry distinct
`payload_id`s for the V3 dense and MoE templates so their parameter
counts don't silently merge to whichever inserted first.

### 5. Simulation layer — [simulation/](simulation/)

An RLHF iteration is a `RLHFTimeline` of `stages:
list[list[ModelMappingGroup]]` with `boundaries: list[StageBoundary]`
between them ([simulation/stages.py](simulation/stages.py)). A
`ModelMappingGroup` runs one or more `ModelMapping`s sequentially in
its own slice of the cluster; multiple groups within a stage run
concurrently and share the cluster fabric.

[simulation/simulator.py](simulation/simulator.py):

- `simulate_stage(stage, ctx, hw, topo)` — calls `resolve_op_transfers`,
  builds the stage-level steady and fill/drain compositions
  (`Parallel` over per-group `Sequential` over per-MM compositions),
  takes both snapshots plus a boundary snapshot, then per-group sums
  its sequential `ModelMapping`s' times against those snapshots and
  returns the slowest group.
- `simulate_rlhf_iteration(timeline, ctx, hw, topo)` sums stage times
  and `StageBoundary.simulate(topo, hw)` times. `IterationResult`
  exposes `per_stage_times`, `per_boundary_times`, and
  `per_mm_results` (flattened) for introspection.

`ModelMappingGroup.simulate_isolated(ctx, hw, topo)` computes a single
group's time using only **its own** composition. Less accurate than
`simulate_stage` (cross-group contention is silently ignored); intended
for early exploration where stage-wide context isn't available.

[simulation/transition.py](simulation/transition.py) —
`StageTransition(source, dest, scheme=MegatronToMegatronP2P())` is one
resharding hop between two `ModelMapping`s of the same model.
`StageBoundary.simulate` takes a single joint flow-time snapshot
across every transition's transfers (so transitions in the same
boundary contend with each other on the fabric) and returns the
slowest op. `__post_init__` validates that every emitted op is
`NetworkPhase.BOUNDARY`.

## Calibration story

The simulator's accuracy class is set by a small number of efficiency
factors on `HardwareSpec`. These are deliberately **not** auto-tuned by
the validation harness — every change is intentional.

| factor | default | what it controls |
|---|---:|---|
| `compute_efficiency` | 0.7 | achieved fraction of peak FLOPs for large GEMMs |
| `memory_efficiency` | 0.8 | achieved fraction of peak HBM bandwidth |
| `small_gemm_efficiency` | 0.4 | replaces compute_efficiency when min(M,N,K) < 1024 |
| `gemv_efficiency` | 0.3 | replaces compute_efficiency when M < 16 (decode GEMVs) |
| `gemv_memory_efficiency` | 0.65 | replaces memory_efficiency when M < 16 |
| `decode_overhead_per_token_ms` | 0.0 | per-token framework / kernel-launch / paged-attention overhead added once per generated token |

The GEMV tier triggers when M < 16 and takes precedence over
`is_small_dim`. It exists because batch=1 decode produces M=1 GEMMs
with K, N >> 1024 — which miss `is_small_dim`'s threshold but have
fundamentally GEMV-style efficiency characteristics. Only `Gemm`
consults the GEMV tier; attention, RMSNorm, RoPE, etc. use the base
factors directly.

`decode_overhead_per_token_ms` is the only non-efficiency knob. Set
per scenario to a profiled value (the validation suite uses 10 ms on
A100, 5 ms on H100 as vLLM-class representative numbers); leave at 0
to get pure-roofline decode predictions.

To improve accuracy on a specific deployment, profile representative
operator shapes and replace the defaults.
[validation/findings.md](validation/findings.md) lists the per-scenario
errors the current defaults produce.

## Resharding extensibility

`ReshardingScheme` (in [schemes/strategies.py](schemes/strategies.py))
is the pluggable interface:

```python
class ReshardingScheme(ABC):
    @abstractmethod
    def compute_network_ops(
        self, source: ModelMapping, dest: ModelMapping,
    ) -> list[NetworkOp]: ...
```

v1 ships with `MegatronToMegatronP2P` (Llama-specific, in
[schemes/llama_resharding.py](schemes/llama_resharding.py)). V3 / MoE
resharding is left to a future scheme.

A `HybridFlowMicroDPScheme` could inject AllGathers within micro-DP
groups instead of pure P2Ps; the framework consumes whatever
`NetworkOp` subclasses the scheme returns, uniformly. `StageTransition`
validates that every emitted op is `BOUNDARY`-phase.

## Known limitations

a. **PP activation P2P is sized off one representative hop.** The
   per-microbatch P2P time threaded into the bubble formula is computed
   from the single (pp=0 → pp=1) transfer between the reference
   `(dp=0, tp=0, ep=0)` coordinate. Heterogeneous boundaries (uneven
   layer partitions, asymmetric link bandwidths between adjacent
   stages) are not modeled — the contention snapshot does amortize
   topology sharing across coordinates, but the *additive* term in
   the bubble formula is the same for every microbatch.

b. **Fill/drain composition models stages as fully sequentialized.**
   The fill/drain snapshot wraps per-stage Sequentials in an outer
   Sequential (one stage at a time). Real 1F1B fill/drain has up to
   `pp − 1` stages active at any moment with overlap; the chosen bound
   is conservative on the *low* side for those phases.

c. **Backward FLOPs derived as 3× / 2× forward.** Per-operator
   backward shapes are not modeled exactly — `recompute=True` →
   3× (full activation checkpointing), default `recompute=False` →
   2× (selective checkpointing). This is a rule-of-thumb upper bound
   on the per-op backward cost.

d. **Decode attention uses average context length** (`prompt_len +
   response_len/2`). Per-step variation is not modeled, so total
   decode time is `decode_step_avg × response_len` plus the
   per-token overhead.

e. **Pipeline schedule is 1F1B only.** Interleaved 1F1B, ZB-V (zero
   bubble), Chimera, and other schedules are not supported.

f. **Group-local simulation mode**
   (`ModelMappingGroup.simulate_isolated`) ignores cross-group
   contention by design — accuracy is reduced. Prefer `simulate_stage`
   whenever stage-wide information is available; the two modes can
   disagree.

g. **GEMV efficiency is a rough default.** `gemv_efficiency=0.3` and
   `gemv_memory_efficiency=0.65` are literature-typical numbers; real
   kernels vary by 1.5-2× depending on M, K, N, and the actual GEMV
   library used. For production accuracy, profile a representative
   GEMV kernel (M=1, your decode K/N) on the target hardware and
   update `HardwareSpec`.

h. **Decode predictions need a per-step overhead constant.** With
   `decode_overhead_per_token_ms = 0` the simulator returns the
   analytical floor and underpredicts measured wall time by ~2× at
   batch=1. The right knob is the per-stack measured overhead (the
   validation suite uses 10 ms on A100, 5 ms on H100 as representative
   vLLM-class numbers); set it per scenario from real measurement, not
   validation auto-tuning.

i. **DP-gradient AllReduce is keyed by layer template, not layer
   instance.** Within a `LayerRepeat` block (e.g., V3's 58 MoE
   layers), all repetitions share one resolved `AllReduceRing`. The
   simulator counts that op's time once per attaching operator in the
   wall-time sum, which is correct *if* every layer's parameter shape
   and DP-group topology match — true today by construction (uniform
   layers per template). `payload_id` keeps distinct templates (V3
   dense vs MoE) from collapsing, but per-instance heterogeneity
   within a template would still merge.

j. **LM-head AllGather is unconditional under `tp > 1`.** The
   simulator emits a `tp`-way AllGather of `T × vocab × dtype` after
   the LM head regardless of phase. Some training stacks compute
   cross-entropy with TP-aware kernels and skip the gather; for those
   configurations the prediction is conservative.

k. **MoE routed experts are EP-sharded only, not TP-sharded.** TP
   sharding under V3 affects MLA heads, dense FFN intermediate, and
   the shared expert. The per-expert routed FFN keeps full `m_expert`.
   Real stacks that further TP-shard routed experts (rare; most use
   EP-only for routed) would see a different intra-expert TP-AR cost
   than the simulator predicts.

l. **MoE imbalance is a fixed scalar** (`moe_imbalance_factor`
   default 1.15). Real imbalance is workload-dependent and can vary
   per layer.

## How to extend

### Add a new model architecture

1. Subclass `ArchitectureConfig` with the model's hyperparameters
   (e.g., `MyModelConfig` mirroring `LlamaConfig` / `V3Config`).
2. Subclass `Model` and implement `forward_sequence(phase: str) ->
   list[PatternEntry]`. See [operators/llama_builder.py](operators/llama_builder.py)
   for the canonical example. Use `LayerRepeat(count, members)` to
   express repeated layers and emit `LabeledOp(role, op)` with
   appropriate `OpRole` tags — the existing `MegatronScheme`
   decoration table covers the standard set
   (QKV_PROJ, ATTN_OUT, FFN_REDUCE, LM_HEAD, ...).
3. For collectives whose payload depends on workload
   (activation-sized AR, EP-AllToAll, LM-head AllGather), attach a
   `data_GB_fn` closure to the `NetworkRequirement` that closes over
   architecture parameters and reads the runtime ctx —
   `_activation_size_fn` / `_lm_head_gather_size_fn` in
   `llama_builder.py` and `v3_block.py` are the canonical patterns.
   For collectives with payload independent of workload (DP-gradient
   AR sized to parameter bytes), the closure is a constant; pair it
   with a `payload_id` discriminator if multiple layer templates emit
   the same kind/group/relation but differ in size.
4. Validate divisibility early (`raise ValueError`) so
   misconfigurations fail at build time rather than producing
   incorrect collective sizing.
5. Use the existing operator vocabulary (`Gemm`, `RMSNorm`, etc.)
   where possible; only add new operators when the architecture
   introduces a genuinely new kernel.
6. Run via `MegatronScheme(parallelism).build_mm(model=..., workload=...,
   devices=..., recompute=...)`.

### Add a new operator

1. Subclass `Operator` (from [operators/base.py](operators/base.py)).
   Implement `compute_flops(ctx)` and `memory_bytes(ctx)`. Override
   `is_small_dim(ctx)` if the operator should consult the
   small-GEMM efficiency tier; override
   `compute_efficiency_factor` / `memory_efficiency_factor` if the
   shape-dependent tier choice is more involved (as `Gemm` does for
   GEMV).
2. If the operator carries learnable weights, override
   `parameter_bytes()` (and `activated_parameter_bytes()` if
   sparsely activated, e.g., MoE experts).
3. If the operator triggers a collective at this position in the
   graph, the **builder** is responsible for emitting the
   `NetworkRequirement` alongside the operator (the operator itself
   stays parallelism-agnostic).

### Add a new resharding scheme

1. Subclass `ReshardingScheme` (in
   [schemes/strategies.py](schemes/strategies.py)). Implement
   `compute_network_ops(source, dest) -> list[NetworkOp]`. Return
   any mix of `P2P`, `AllReduceRing`, `AllGatherRing`, etc. —
   `StageTransition` consumes them uniformly as long as every op's
   `phase` is `NetworkPhase.BOUNDARY`.
2. Pass an instance to `StageTransition(source, dest, scheme=...)`.

## Layout

```
NARFS/
├── README.md                          ← you are here
├── model.py                           ← Model, ArchitectureConfig
├── workload_context.py                ← WorkloadContext, Workload, TokenCount
├── parallelism.py                     ← ParallelismConfig, ParallelismGroup, ParallelismRole, Shard
├── network_requirement.py             ← NetworkRequirement, CollectiveKind, Relation
├── network_op.py                      ← NetworkOp + concrete collectives + NetworkPhase
├── topology/
│   ├── topo.py                        ← HostTopo, Path, Link, Device
│   └── hardware.py                    ← HardwareSpec
├── operators/
│   ├── base.py                        ← Operator ABC
│   ├── pattern.py                     ← LabeledOp, LayerRepeat, OpRole, OperatorWithReqs
│   ├── gemm.py, attention.py, elementwise.py, embed.py,
│   │   optimizer.py, backward.py, mla.py, moe.py
│   ├── llama_config.py, llama_builder.py
│   └── v3_config.py, v3_block.py
├── schemes/
│   ├── strategies.py                  ← ParallelismScheme, Plan, MegatronScheme, ReshardingScheme, assemble
│   └── llama_resharding.py            ← MegatronToMegatronP2P
├── mapping/
│   ├── model_mapping.py               ← ModelMapping + CombinedOp + simulate
│   ├── model_mapping_group.py         ← ModelMappingGroup + resolve_op_transfers + steady/boundary transfer times
│   ├── compute_model.py               ← per-MM seq walker + aggregator
│   └── contention.py                  ← NetworkComposition tree + ContentionModel
├── simulation/
│   ├── simulator.py                   ← simulate_stage, simulate_rlhf_iteration
│   ├── stages.py                      ← RLHFTimeline, StageBoundary
│   └── transition.py                  ← StageTransition
├── tests/                             ← pytest suite
└── validation/                        ← prediction-vs-reference harness
```
