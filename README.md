# RLHF latency simulator

Analytical latency simulator for RLHF scheduling, modeled after the
DistServe latency model (Appendix A of the DistServe paper) and adapted for
GQA/MLA, SwiGLU FFN, MoE with shared experts, TP/PP/DP/EP parallelism, 1F1B
pipeline scheduling, and inter-stage parameter resharding.

The intent is **fast analytical predictions** for RLHF orchestration
research — not cycle-accurate simulation. The simulator produces
predictions within roughly ±30% of measured systems for compute-bound
prefill workloads at the default efficiency factors. Decode and overhead-
heavy workloads sit further off the analytical floor; see
`validation/findings.md` and the limitations section below.

## Architecture

The codebase has four layers, bottom-up:

### 1. Contention layer — `topo.py`, `network_op.py`

`HostTopo` describes a cluster as a graph of hosts connected by `Path`s of
`Link`s with bandwidth and latency. It exposes two contention models:

- `get_simultaneous_logical_transfer_times` — shared-bandwidth contention.
  Used for steady-state ops that repeat per layer (AllReduce after attn_out
  / ffn_down, AllToAll for MoE dispatch/combine). Each transfer's effective
  bandwidth = link bandwidth / number of transfers on that link.
- `get_simultaneous_system_flow_time` — flow-time contention. Used for
  one-shot ops at stage boundaries (parameter resharding P2Ps). Each
  link's completion time = total bytes on that link / bandwidth.

`NetworkOp` (and its subclasses `AllReduceRing`, `AllGatherRing`,
`ReduceScatterRing`, `AllToAll`, `Broadcast`, `P2P`) decomposes a logical
collective into `LogicalTransfer`s and computes per-op wall time given a
transfer-times dict. Each instance carries a `phase: NetworkPhase`
(`STEADY` / `BOUNDARY`) that selects which contention snapshot it consults.

### 2. Operator layer — `operator.py`, `hardware.py`, `operators/*.py`

`Operator` (ABC) is a per-kernel cost function:
`compute_flops(ctx)`, `memory_bytes(ctx)`, `is_compute_bound(ctx, hw)`,
`is_small_dim(ctx)`. The base class derives `compute_time`, `memory_time`,
and `kernel_time = max(compute_time, memory_time)` (the standard roofline).

`HardwareSpec` carries the calibration knobs:

- `peak_compute_flops`, `peak_memory_bandwidth`, `ridge_flops_per_byte` —
  the device's analytical roofline.
- `compute_efficiency` (default 0.7) — achieved fraction of peak for large
  GEMMs.
- `memory_efficiency` (default 0.8) — achieved fraction of peak HBM
  bandwidth.
- `small_gemm_efficiency` (default 0.4) — used when any GEMM dim < 1024.

These are the calibration points (originally DistServe's C₁..C₅, exposed
as named factors). Defaults are literature-typical; replace with profiled
values for production accuracy.

Concrete operators in `operators/`:

- `gemm.py` — `Gemm(n, k, token_count | derive_M, dtype_bytes, n_replicas, replicas_active)`.
- `attention.py` — `PrefillAttention` (FlashAttention prefill) and
  `DecodeAttention` (GQA / MLA decode with configurable
  `kv_bytes_per_token`).
- `elementwise.py` — `RMSNorm`, `SwiGLUActivation`, `RoPE`.
- `embed.py` — `TokenEmbedding` (zero-flop output write).
- `optimizer.py` — `AdamOptimizerOp` for the training optimizer step.
- `backward.py` — `BackwardOp` wraps a forward op and reports backward
  cost (3× compute / 3× memory with full activation recomputation;
  2× / 1.5× without).
- `mla.py`, `moe.py`, `composite.py`, `v3_block.py`, `llama_builder.py` —
  whole-model graph builders.
- `markers.py` (in `moe.py`) — `DispatchMarker`, `CombineMarker` are
  zero-cost placeholders that ModelMapping recognizes as the attachment
  point for MoE dispatch/combine all-to-alls.

Operators stay parallelism-agnostic. They operate on whatever shapes the
builder constructed them with.

### 3. Per-model layer — `model_mapping.py`

`ModelMapping(model, mesh, workload, parallelism, shards_to_host_ids, ...)`
holds one `Model` placed on one parallelism configuration for one
`Workload` (GENERATION / PREPARATION / TRAINING / DORMANT). One
ModelMapping per (model, parallelism, workload) — they don't try to serve
multiple workloads.

`__post_init__` does several things in order:

1. Validates the shard layout against `parallelism` along TP, EP, PP, DP
   axes. Each axis with degree > 1 must have that many ranks in the shard
   set with the *other three* axes held fixed; layouts that under-provision
   raise `ValueError` rather than silently shrinking a collective.
2. Calls `model.build_pattern(architecture, phase, parallelism)` to get
   the operator graph as a tagged pattern (operators paired with their
   network requirements). For GENERATION, builds two patterns
   (prefill + decode); for TRAINING, also derives backward and appends an
   AdamOptimizerOp. In TRAINING, layer builders also emit a per-layer
   DP-gradient AllReduce when `parallelism.dp > 1` (see
   "Workload-dependent collective sizing" below).
3. Expands the pattern into parallel lists `operator_pattern: list[Operator]`
   and `per_op_network_requirements: list[list[NetworkRequirement]]`.
4. Resolves each unique `NetworkRequirement` into a concrete `NetworkOp`
   (deduplicated by `(kind, relation, eta, group, payload_id)`; one
   `AllReduceRing` shared across every TP-AR usage).
5. Layer-aware partition across PP stages, co-locating each layer's
   forward / backward / optimizer ops on the same stage so backward and
   optimizer share the partition with their forward.
6. When `parallelism.pp > 1`, installs a per-microbatch PP activation
   `P2P` (forward; plus a backward leg in TRAINING) with phase override
   `STEADY` — these stream per microbatch alongside compute, not as
   one-shot boundary handoffs.

`simulate(ctx, hw, transfer_times_steady, transfer_times_boundary)` walks
the operator pattern, combines each op's `kernel_time` with its network
requirements via `NetworkAssociation.combine_times`, and applies a
1F1B-shaped pipeline formula `(num_microbatches + pp - 1) × (max_stage +
p2p_per_mb)` when pp > 1, where `p2p_per_mb` is the per-microbatch PP
activation cost (forward + backward when training). Returns a
`SimulationResult` with `total_time`, `prefill_time` / `decode_time` /
`training_time` (depending on workload), `pp_stage_times`, and
`per_op_breakdown`.

#### Workload-dependent collective sizing

`NetworkRequirement.data_GB_fn: Callable[[WorkloadContext], float]` is an
optional sizing closure attached to a requirement at build time. Builders
close over architecture parameters (h, vocab, top_k, dtype) and produce a
function whose only argument is the current ctx — so transfer sizes
automatically track changes in `microbatch_size`, `prompt_len`, etc.
between simulate calls.

The simulator runs `ModelMapping.refresh_collective_sizes(ctx)` at the
top of every `simulate_stage` / `simulate_isolated` /
`simulate_rlhf_iteration` invocation; for each registered NetworkOp the
closure is evaluated and `NetworkOp.refresh_data_size(total_GB)` rebuilds
the `LogicalTransfer` list with the per-transfer split appropriate to
that collective (ring → total/N per hop, AllToAll → total/N per pair,
P2P / Broadcast → full payload).

Two requirements with identical `(kind, relation, eta, group,
payload_id)` dedupe to one NetworkOp — `data_GB_fn` is excluded from
equality and hash. `payload_id` is the discriminator that prevents
collapsing collectives that fire repeatedly with *different* sizes:
the per-layer DP-gradient AllReduces in training carry distinct
`payload_id`s for the V3 dense and MoE templates so their parameter
counts don't silently merge to whichever inserted first.

### 4. Stage orchestration — `stages.py`, `submesh_mapping.py`, `transition.py`, `simulator.py`

An RLHF iteration is a `RLHFTimeline` of `stages: list[list[SubmeshMapping]]`
with `boundaries: list[StageBoundary]` between them. Each
`SubmeshMapping` runs one or more `ModelMapping`s sequentially in its own
submesh; multiple submeshes within a stage run concurrently and share the
cluster fabric.

`simulate_stage(stage, ctx, hw, topo)` first calls
`refresh_collective_sizes(ctx)` on every ModelMapping in the stage so
each NetworkOp's transfer sizes reflect the current workload, then takes
a single steady-state contention snapshot and a single boundary snapshot
across **every** NetworkOp in **every** ModelMapping, per-submesh sums
its sequential ModelMappings' times against those snapshots, and returns
the slowest submesh.

`StageBoundary.simulate(topo, hw)` takes a single joint flow-time snapshot
across every transition's transfers (so transitions in the same boundary
contend with each other on the fabric) and returns the slowest op.

`simulate_rlhf_iteration(timeline, ctx, hw, topo)` sums stage times +
boundary times. `IterationResult` exposes `per_stage_times`,
`per_boundary_times`, and `per_mm_results` (flattened) for introspection.

`SubmeshMapping.simulate_isolated(ctx, hw, topo)` computes per-submesh
time using only **its own** NetworkOps' contention. Less accurate than
`simulate_stage` (cross-submesh contention is silently ignored); intended
for early-stage exploration where stage-wide context isn't available.
**The two modes can disagree; prefer `simulate_stage` whenever possible.**

## Calibration story

The simulator's accuracy class is set by a small number of efficiency
factors on `HardwareSpec`. These are deliberately **not** auto-tuned by
the validation harness — every change is intentional.

Default values (literature-typical):

| factor | default | what it controls |
|---|---:|---|
| `compute_efficiency` | 0.7 | achieved fraction of peak FLOPs for large GEMMs |
| `memory_efficiency` | 0.8 | achieved fraction of peak HBM bandwidth |
| `small_gemm_efficiency` | 0.4 | replaces compute_efficiency when min(M,N,K) < 1024 |
| `gemv_efficiency` | 0.3 | replaces compute_efficiency when M < 16 (decode GEMVs) |
| `gemv_memory_efficiency` | 0.65 | replaces memory_efficiency when M < 16 |

The GEMV tier (`Gemm.is_gemv(ctx)` triggers when M < 16, taking precedence
over `is_small_dim`) exists because batch=1 decode produces M=1 GEMMs with
K, N >> 1024 — which miss `is_small_dim`'s threshold but have fundamentally
GEMV-style efficiency characteristics. Only `Gemm` consults the GEMV tier;
attention, RMSNorm, RoPE, etc. use the base `compute_efficiency` /
`memory_efficiency` directly.

To improve accuracy on a specific deployment, profile representative
operator shapes and replace the defaults. `validation/findings.md` lists
the specific measurements that would pin each constant.

## Resharding extensibility

`ReshardingStrategy` (in `resharding.py`) is the pluggable interface:

```python
class ReshardingStrategy(ABC):
    @abstractmethod
    def compute_network_ops(self, source: ModelMapping, dest: ModelMapping) -> list[NetworkOp]: ...
```

v1 ships with `NaiveP2PStrategy`: walks Llama parameter tensors under the
natural Megatron sharding scheme (QKV column-parallel, O row-parallel,
gate/up column-parallel, down row-parallel, embed/LM-head vocab-parallel),
computes per-(src_host, dst_host) byte movement via slice-overlap
arithmetic, aggregates per pair across all parameters, and emits one
`P2P` per non-zero pair. V3/MoE resharding is left to a future strategy.

A `HybridFlowMicroDPStrategy` could inject AllGathers within micro-DP
groups instead of pure P2Ps; the framework consumes whatever NetworkOp
subclasses the strategy returns, uniformly. `StageTransition` validates
that every emitted op is BOUNDARY-phase.

## Known limitations

a. **Host-level granularity hides intra-node fabric topology** Intra-node
   transfers do not face contention.

b. **Bidirectional links require explicit directed-pair construction**
   in the topology, otherwise full-duplex contention is double-counted
   on a single Link.

c. **MoE imbalance is a fixed scalar** (`moe_imbalance_factor` default
   1.15). Real imbalance is workload-dependent and can vary per layer.

d. **Backward FLOPs derived as 3× forward** (with full activation
   recomputation) or 2× / 1.5× without. Per-operator backward shapes
   are not modeled exactly; this is an upper bound on true backward
   cost.

e. **Decode attention uses average context length** (`prompt_len +
   response_len/2`). Per-step variation is not modeled, so total decode
   time is `decode_step_avg × response_len`.

f. **Pipeline schedule is 1F1B only.** Interleaved 1F1B, ZB-V (zero
   bubble), Chimera, and other schedules are not supported.

g. **Submesh-local simulation mode** (`SubmeshMapping.simulate_isolated`)
   ignores cross-submesh contention by design — accuracy is reduced.

h. **Submesh-local mode and stage-level simulation can disagree.** When
   stage-wide information is available, prefer `simulate_stage`.

i. **GEMV efficiency is a rough default.** `gemv_efficiency=0.3` and
   `gemv_memory_efficiency=0.65` are literature-typical numbers; real
   kernels vary by 1.5-2× depending on M, K, N, and the actual GEMV
   library used. For production accuracy, profile a representative
   GEMV kernel (M=1, your decode K/N) on the target hardware and update
   `HardwareSpec`. The brief's per-scenario fitting is a sounder route
   than tweaking the global default.

j. **Decode predictions are still systematically optimistic.** Even with
   the GEMV efficiency tier, the simulator predicts the analytical floor
   for decode and does not model kernel-launch overhead, sampling,
   allocator/dispatch costs, or KV-cache management. At batch=1 these
   dominate measured wall time (~2× the simulator floor on Llama-2-7B
   decode). Treat decode predictions as **lower bounds** rather than
   point estimates; the gap can be closed by adding a per-decode-step
   constant overhead measured on the target stack, but baking that in
   should be done from real measurement, not validation auto-tuning.

k. **PP activation P2P is sized off one representative hop.** The
   per-microbatch P2P time threaded into the bubble formula is computed
   from the single (pp=0 → pp=1) transfer between the reference (dp=0,
   tp=0, ep=0) coordinate. Heterogeneous boundaries (uneven layer
   partitions, asymmetric link bandwidths between adjacent stages) are
   not modeled — the contention snapshot does amortize topology sharing
   across coordinates, but the *additive* term in the bubble formula is
   the same for every microbatch.

l. **DP-gradient AllReduce is keyed by layer template, not layer
   instance.** Within a `TaggedRepeat` block (e.g., V3's 58 MoE layers),
   all repetitions share one resolved `AllReduceRing`. The simulator
   counts that op's time once per attaching operator in the wall-time
   sum, which is correct *if* every layer's parameter shape and DP-group
   topology match — true today by construction (uniform layers per
   template). Phase 4 introduces `payload_id` precisely so distinct
   templates (V3 dense vs MoE) don't accidentally collapse, but
   per-instance heterogeneity within a template would still merge.

m. **LM-head AllGather is unconditional under `tp > 1`.** The simulator
   emits a `tp`-way AllGather of `T × vocab × dtype` after the LM head
   regardless of phase. Some training stacks compute cross-entropy with
   TP-aware kernels and skip the gather; for those configurations the
   prediction is conservative.

n. **MoE routed experts are EP-sharded only, not TP-sharded.** TP
   sharding under V3 affects MLA heads, dense FFN intermediate, and the
   shared expert. The per-expert routed FFN keeps full `m_expert`. Real
   stacks that further TP-shard routed experts (rare; most use EP-only
   for routed) would see a different intra-expert TP-AR cost than the
   simulator predicts.

## How to extend

### Add a new model architecture

1. Subclass `ArchitectureConfig` with the model's hyperparameters
   (e.g., `MyModelConfig` mirroring `LlamaConfig` / `V3Config`).
2. Write a `build_pattern(arch, phase, parallelism)` function that
   returns `list[PatternEntry]` — see `operators/llama_builder.py` for
   the canonical example. Use `TaggedRepeat(count, members)` to express
   repeated layers and emit `NetworkRequirement`s on operators that
   trigger collectives.
3. For collectives whose payload depends on workload (activation-sized
   AR, EP-AllToAll, LM-head AllGather), attach a `data_GB_fn` closure to
   the `NetworkRequirement` that closes over architecture parameters and
   reads the runtime ctx — `_activation_size_fn` /
   `_lm_head_gather_size_fn` in `llama_builder.py` and `v3_block.py` are
   the canonical patterns. For collectives with payload independent of
   workload (DP-gradient AR sized to parameter bytes), the closure is a
   constant; pair it with a `payload_id` discriminator if multiple
   layer templates emit the same kind/group/relation but differ in
   size.
4. TP-shard the operator shapes the architecture intends to split: head
   counts, FFN intermediate dim, vocab-sharded LM head with an emitted
   `ALL_GATHER` requirement on TP. Validate divisibility (`raise
   ValueError` early) so misconfigurations fail at build time rather
   than producing incorrect collective sizing.
5. Use the existing operator vocabulary (`Gemm`, `RMSNorm`, etc.) where
   possible; only add new operators when the architecture introduces a
   genuinely new kernel.
6. Construct a `Model(name, role, architecture=MyModelConfig(...),
   build_pattern=build_my_pattern)` and use it like any other model.

### Add a new operator

1. Subclass `Operator`. Implement `compute_flops(ctx)`,
   `memory_bytes(ctx)`, `is_compute_bound(ctx, hw)`. Override
   `is_small_dim(ctx)` if the operator should use `small_gemm_efficiency`.
2. If the operator carries learnable weights, override
   `parameter_bytes()` (and `activated_parameter_bytes()` if sparsely
   activated, e.g., MoE experts).
3. If the operator triggers a collective at this position in the graph,
   the **builder** is responsible for emitting the `NetworkRequirement`
   alongside the operator (the operator itself stays
   parallelism-agnostic).

### Add a new resharding strategy

1. Subclass `ReshardingStrategy`. Implement `compute_network_ops(source,
   dest) -> list[NetworkOp]`. Return any mix of `P2P`, `AllReduceRing`,
   `AllGatherRing`, etc. — `StageTransition` consumes them uniformly as
   long as every op's `phase` is `NetworkPhase.BOUNDARY`.
2. Pass an instance to `StageTransition(source, dest, strategy=...)`.

## Layout

```
verl/trainer/ppo/simu/
├── README.md                  ← you are here
├── topo.py                    ← HostTopo, Path, Link
├── network_op.py              ← NetworkOp + concrete collectives
├── network_requirement.py     ← NetworkRequirement, ParallelismConfig
├── relation.py                ← NetworkAssociation, NetworkPhase
├── hardware.py                ← HardwareSpec
├── operator.py                ← Operator base class
├── workload_context.py        ← WorkloadContext, Workload, TokenCount
├── operators/                 ← concrete operators + builders
├── model.py                   ← Model, ArchitectureConfig
├── shard.py                   ← Shard
├── mesh.py                    ← Mesh
├── model_mapping.py           ← ModelMapping + simulate
├── submesh_mapping.py         ← SubmeshMapping + simulate_isolated
├── stages.py                  ← StageBoundary, RLHFTimeline
├── transition.py              ← StageTransition
├── resharding.py              ← ReshardingStrategy + NaiveP2PStrategy
├── simulator.py               ← simulate_stage, simulate_rlhf_iteration
├── tests/                     ← pytest suite (Phases 1-5)
└── validation/                ← prediction-vs-reference harness (Phase 5)
```
