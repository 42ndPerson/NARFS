# Simulator validation findings — 2026-05-13T03:36:59Z

Generated automatically by `run_validation.py` whenever any scenario is more than 30% off its published reference. Default tolerance is 50%; training and DeepSeek-V3 are tagged as sanity bounds (100%).

## Summary

| scenario | predicted | reference | error % | tolerance % | pass |
|---|---:|---:|---:|---:|:---:|
| llama2_7b_prefill_a100 | 57.6 ms | 50.0 ms | +15.2% | 50% | ✓ |
| llama2_7b_decode_a100 | 10.3 ms | 25.0 ms | -58.6% | 50% | ✗ |
| llama3_70b_prefill_8xa100_tp8 | 455.3 ms | 250.0 ms | +82.1% | 50% | ✗ |
| llama3_8b_training_8xa100_tp2_pp2_dp2 | 8637.8 ms | 4000.0 ms | +115.9% | 100% | ✗ |
| deepseek_v3_prefill_32xh100_ep32 | 4849.8 ms | 500.0 ms | +870.0% | 100% | ✗ |
| llama3_70b_prefill_8xa100_tp8_nvswitch | 258.2 ms | 250.0 ms | +3.3% | 50% | ✓ |
| llama3_8b_training_8xa100_tp2_pp2_dp2_nvswitch | 7614.0 ms | 4000.0 ms | +90.4% | 100% | ✓ |
| deepseek_v3_prefill_32xh100_ep32_multinode | 1640.8 ms | 500.0 ms | +228.2% | 100% | ✗ |

## Per-scenario investigation

### llama2_7b_decode_a100

_Llama-2-7B decode, batch=1, single A100-80GB. Reference ~25ms/token (vLLM)._

- **Predicted:** 10.3 ms  **Reference:** 25.0 ms  **Error:** -58.6%  **Tolerance:** ±50% (exceeds tolerance)

**Dominant operators by wall-time fraction:**

- `Gemm` — 94.8%
- `PrefillAttention` — 3.0%
- `SwiGLUActivation` — 1.0%
- `RMSNorm` — 0.5%
- `RoPE` — 0.5%
- `DecodeAttention` — 0.2%

**Hypothesis:**

Pure roofline + GEMV efficiency tier (0.3 compute / 0.65 memory for M<16 GEMMs) is now the prediction. Total memory traffic per decode step ≈ model weights once + KV cache reads. The remaining gap to measured wall time is kernel-launch overhead, sampling, and KV-cache management — the simulator's scope stops at the roofline. To close further: (a) profile actual GEMV memory_efficiency on the target A100 — typical measurements are 0.5-0.6 rather than the 0.65 default; (b) bake in a per-decode-step constant overhead after profiling a real run. Do NOT auto-tune from validation alone.

### llama3_70b_prefill_8xa100_tp8

_Llama-3-70B prefill, batch=1, prompt=2048, 8xA100-80GB TP=8. Reference ~250ms._

- **Predicted:** 455.3 ms  **Reference:** 250.0 ms  **Error:** +82.1%  **Tolerance:** ±50% (exceeds tolerance)

**Dominant operators by wall-time fraction:**

- `Gemm` — 93.5%
- `PrefillAttention` — 4.4%
- `RMSNorm` — 1.5%
- `SwiGLUActivation` — 0.5%
- `RoPE` — 0.1%
- `TokenEmbedding` — 0.0%

**Hypothesis:**

Above reference. If data_GB_estimates is non-empty, steady-op contention is computed against ALL steady transfers as if concurrent — for a single submesh with serialized layers this over-counts. With empty estimates (the validation default), this branch shouldn't fire; investigate operator-level cost.

### llama3_8b_training_8xa100_tp2_pp2_dp2

_Llama-3-8B training step, batch=64, microbatch=4, prompt=2048, 8xA100-80GB TP=2 PP=2 DP=2. Reference ~3-5s (Megatron-class). Sanity bound._

- **Predicted:** 8637.8 ms  **Reference:** 4000.0 ms  **Error:** +115.9%  **Tolerance:** ±100% (exceeds tolerance)

**Dominant operators by wall-time fraction:**

- `BackwardOp` — 68.7%
- `Gemm` — 24.6%
- `PrefillAttention` — 3.6%
- `AdamOptimizerOp` — 2.4%
- `SwiGLUActivation` — 0.4%
- `RMSNorm` — 0.3%

**Hypothesis:**

Above reference. Layer-aware PP partitioning is now in place (the previous flat-partition bug is fixed); remaining gap likely comes from: (a) BackwardOp's 3× compute / 3× memory scaling with full activation recomputation is conservative — FlashAttention backward + selective checkpointing typically achieve 2.0-2.5× rather than 3×; (b) compute_efficiency=0.7 may be high for backward kernels where memory-traffic patterns differ from forward; (c) the optimizer step time (per-layer AdamOptimizerOps) gets multiplied through the (num_microbatches + pp - 1) pipeline formula even though optimizer is one-shot per iteration — minor effect since optimizer time is small relative to backward, but a principled fix would isolate optimizer from the 1F1B multiplier.

### deepseek_v3_prefill_32xh100_ep32

_DeepSeek-V3 prefill, batch=1, prompt=4096, 32xH100-80GB EP=32 TP=1 PP=1. Reference ~500ms (looser bound)._

- **Predicted:** 4849.8 ms  **Reference:** 500.0 ms  **Error:** +870.0%  **Tolerance:** ±100% (exceeds tolerance)

**Dominant operators by wall-time fraction:**

- `DispatchMarker` — 38.7%
- `CombineMarker` — 38.7%
- `PrefillAttention` — 18.1%
- `Gemm` — 4.2%
- `RMSNorm` — 0.1%
- `RoPE` — 0.1%

**Hypothesis:**

Above reference. PrefillAttention dominates ~80% of predicted wall time. memory_bytes = 3·n_h·head_size·t²/B·dtype gives ~14 ms per attention layer for V3 (n_h=128, head_size=192, t=4096, B=64), or ~880 ms across 61 layers alone. Two refinements would close the gap: (a) memory_efficiency=0.8 is conservative for H100 HBM3 — measured 0.85-0.9 — bumping it to 0.9 trims ~10%; (b) FlashAttention-3 has lower effective memory traffic than the 3·l²/B·d_h tile model assumes (recomputes Q/K/V tiles instead of materializing). Re-deriving the FA3 memory factor on H100 would shave the rest.

### llama3_8b_training_8xa100_tp2_pp2_dp2_nvswitch

_Llama-3-8B training step on 8xA100 with NVSwitch (per-pair private lanes). batch=64, microbatch=4, prompt=2048, TP=2 PP=2 DP=2. Reference ~3-5s sanity bound._

- **Predicted:** 7614.0 ms  **Reference:** 4000.0 ms  **Error:** +90.4%  **Tolerance:** ±100% (within tolerance)

**Dominant operators by wall-time fraction:**

- `BackwardOp` — 71.2%
- `Gemm` — 21.3%
- `PrefillAttention` — 4.0%
- `AdamOptimizerOp` — 2.7%
- `SwiGLUActivation` — 0.4%
- `RMSNorm` — 0.3%

**Hypothesis:**

Above reference. Layer-aware PP partitioning is now in place (the previous flat-partition bug is fixed); remaining gap likely comes from: (a) BackwardOp's 3× compute / 3× memory scaling with full activation recomputation is conservative — FlashAttention backward + selective checkpointing typically achieve 2.0-2.5× rather than 3×; (b) compute_efficiency=0.7 may be high for backward kernels where memory-traffic patterns differ from forward; (c) the optimizer step time (per-layer AdamOptimizerOps) gets multiplied through the (num_microbatches + pp - 1) pipeline formula even though optimizer is one-shot per iteration — minor effect since optimizer time is small relative to backward, but a principled fix would isolate optimizer from the 1F1B multiplier.

### deepseek_v3_prefill_32xh100_ep32_multinode

_DeepSeek-V3 prefill, batch=1, prompt=4096, 32xH100-80GB EP=32 across 4 nodes × 8 GPUs each: NVSwitch within node, IB between. Reference ~500ms (looser bound)._

- **Predicted:** 1640.8 ms  **Reference:** 500.0 ms  **Error:** +228.2%  **Tolerance:** ±100% (exceeds tolerance)

**Dominant operators by wall-time fraction:**

- `PrefillAttention` — 53.6%
- `DispatchMarker` — 16.6%
- `CombineMarker` — 16.6%
- `Gemm` — 12.6%
- `RMSNorm` — 0.3%
- `RoPE` — 0.2%

**Hypothesis:**

Above reference. PrefillAttention dominates ~80% of predicted wall time. memory_bytes = 3·n_h·head_size·t²/B·dtype gives ~14 ms per attention layer for V3 (n_h=128, head_size=192, t=4096, B=64), or ~880 ms across 61 layers alone. Two refinements would close the gap: (a) memory_efficiency=0.8 is conservative for H100 HBM3 — measured 0.85-0.9 — bumping it to 0.9 trims ~10%; (b) FlashAttention-3 has lower effective memory traffic than the 3·l²/B·d_h tile model assumes (recomputes Q/K/V tiles instead of materializing). Re-deriving the FA3 memory factor on H100 would shave the rest.

## Suggested measurements to fit constants

These would let us replace the literature-typical defaults (0.7 / 0.8 / 0.4) with profiled values for the specific hardware target. **Do not auto-tune.**

- Profile a representative QKV GEMM at shape (M=2048, N=12288, K=4096) on the target A100 to fit `compute_efficiency` for compute-bound prefill.
- Profile a memory-bound GEMM at shape (M=1, N=4096, K=4096) on A100 to fit `memory_efficiency` and `small_gemm_efficiency` for decode.
- Measure end-to-end Llama-2-7B decode on a single A100 with kernel-launch overhead profiled (e.g., via NSight): if the gap to roofline is ~constant per token, bake it in as a per-step overhead constant rather than touching efficiencies.
- For multi-GPU TP scenarios, instrument NVLink utilization during a single AllReduce — current simulator assumes one shared spine; real NVLink switch fabric behaves differently under contention.
- For DeepSeek-V3, measure per-token AllToAll wall time on the target topology to fit a per-EP contention factor (single-spine model is approximate).

## What this run did NOT do

- No efficiency factors were modified.
- No reference numbers were adjusted.
- No new operators or formulas were added.

Review this report and decide whether to (a) profile to fit constants, (b) extend the operator/contention model, or (c) accept current accuracy and tighten the tolerances on stable scenarios.
