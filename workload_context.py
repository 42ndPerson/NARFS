from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Workload(Enum):
    GENERATION = "generation"
    PREPARATION = "preparation"
    TRAINING = "training"
    DORMANT = "dormant"


class TokenCount(Enum):
    """Resolves to the number of tokens an operator processes in a single invocation.

    PREFILL: every prompt token in one microbatch (`microbatch_size × prompt_len`).
        Used for PREPARATION and TRAINING forward (each invocation walks one
        microbatch's sequences in one shot). The pipeline scheduler loops this
        invocation `num_microbatches` times.
    DECODE: one new token per concurrent request (`microbatch_size`), the
        per-step cost during generation.
    """

    PREFILL = "prefill"
    DECODE = "decode"

    def resolve(self, ctx: "WorkloadContext") -> int:
        if self is TokenCount.PREFILL:
            return ctx.microbatch_size * ctx.prompt_len
        if self is TokenCount.DECODE:
            return ctx.microbatch_size
        raise ValueError(f"Unknown TokenCount: {self}")


@dataclass(frozen=True)
class WorkloadContext:
    """Workload shape: per-invocation operator sizing × pipeline loop count.

    - **`microbatch_size`** is the per-invocation batch dimension that
      operators size off (`microbatch_size × prompt_len` tokens per call).
    - **`num_microbatches`** is the pipeline-loop multiplier — how many
      times the same graph is invoked per training step on each DP rank.
    - **`batch_size`** is the global iteration batch on this DP rank,
      tied to the others by the invariant
      `batch_size = microbatch_size × num_microbatches`.

    No operator's `compute_flops` / `memory_bytes` should ever read
    `batch_size`. Operators size off `microbatch_size` (via
    `TokenCount.resolve` or a `derive_M` callable). The simulator's
    pipeline formula reads `num_microbatches`.
    """

    workload_type: Workload
    batch_size: int
    microbatch_size: int
    prompt_len: int          # input sequence length per request
    response_len: int        # number of tokens to generate (0 for non-generation)
    num_microbatches: int    # pipeline-loop multiplier; >= 1

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.microbatch_size <= 0:
            raise ValueError(f"microbatch_size must be positive, got {self.microbatch_size}")
        if self.num_microbatches <= 0:
            raise ValueError(f"num_microbatches must be positive, got {self.num_microbatches}")
        product = self.microbatch_size * self.num_microbatches
        if product != self.batch_size:
            raise ValueError(
                f"num_microbatches × microbatch_size must equal batch_size; "
                f"got {self.num_microbatches} × {self.microbatch_size} = "
                f"{product} ≠ {self.batch_size}."
            )
