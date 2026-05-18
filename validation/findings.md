# Simulator validation findings — 2026-05-18T10:36:34Z

Generated automatically by `run_validation.py` whenever any scenario is more than 30% off its published reference. Default tolerance is 50%; training and DeepSeek-V3 are tagged as sanity bounds (100%).

## Summary

| scenario | predicted | reference | error % | tolerance % | pass |
|---|---:|---:|---:|---:|:---:|
| llama2_7b_prefill_a100 | 57.6 ms | 50.0 ms | +15.2% | 50% | ✓ |
| llama2_7b_decode_a100 | 20.3 ms | 25.0 ms | -18.6% | 50% | ✓ |
| llama3_70b_prefill_8xa100_tp8 | 258.2 ms | 250.0 ms | +3.3% | 50% | ✓ |
| llama3_8b_training_8xa100_tp2_pp2_dp2 | 7619.9 ms | 4000.0 ms | +90.5% | 100% | ✓ |
| deepseek_v3_prefill_32xh100_ep8_tp4 | 614.9 ms | 500.0 ms | +23.0% | 100% | ✓ |

## Per-scenario investigation

### llama3_8b_training_8xa100_tp2_pp2_dp2

_Llama-3-8B training step, batch=64, microbatch=4, prompt=2048, 8xA100-80GB TP=2 PP=2 DP=2 on an NVSwitch fabric. Reference ~3-5s (Megatron-class). Sanity bound._

- **Predicted:** 7619.9 ms  **Reference:** 4000.0 ms  **Error:** +90.5%  **Tolerance:** ±100% (within tolerance)

**Dominant operators by wall-time fraction:**

- `BackwardOp` — 59.6%
- `Gemm` — 33.0%
- `PrefillAttention` — 3.9%
- `AdamOptimizerOp` — 2.6%
- `SwiGLUActivation` — 0.4%
- `RMSNorm` — 0.3%

**Hypothesis:**

Above reference; BackwardOp dominates the breakdown. Likely contributions: (a) BackwardOp's 3× compute / 3× memory scaling with full activation recomputation is conservative — FlashAttention backward + selective checkpointing typically achieve 2.0-2.5× rather than 3×; (b) compute_efficiency=0.7 may be high for backward kernels where memory-traffic patterns differ from forward; (c) per-layer AdamOptimizerOps get multiplied through the (num_microbatches + pp - 1) pipeline formula even though optimizer is one-shot per iteration — minor effect, but a principled fix would isolate the optimizer from the 1F1B multiplier.

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
