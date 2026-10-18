"""The component library: operator costs, backward rules and parameter counts
for each component type, on LOCAL (per-rank) shapes.

This is the only place with operator knowledge. Architectures only name
component types (gqa / mla attention, dense / moe FFN) and dimensions.

`ShapeCtx` carries everything one expansion needs: the local dimensions, the
mesh degrees, the framework profile, and the token counts.
"""
from __future__ import annotations

from dataclasses import dataclass

from narfs.blocks import Collective, FlowShop2, Op, Scale
from narfs.model.architecture import Architecture, LayerSpec
from narfs.parallel.mesh import Degrees
from narfs.profiles import FrameworkProfile


@dataclass(frozen=True)
class ShapeCtx:
    arch: Architecture
    local: dict
    degrees: Degrees
    profile: FrameworkProfile
    tokens: float            # tokens per rank in one pass (after the cp split)
    seq: float               # attention context length
    dt: int = 2

    def d(self, name: str, default: int | None = None) -> int:
        if name in self.local:
            return self.local[name]
        if default is not None:
            return default
        raise KeyError(name)


# ------------------------------------------------------------ primitives ---

def gemm(m: float, n: float, k: float, dt: int, name: str) -> Op:
    return Op("gemm", 2.0 * m * n * k, dt * (m * k + k * n + m * n), (m, n, k), name=name)


def elementwise(flops: float, bytes_: float, name: str) -> Op:
    return Op("elementwise", flops, bytes_, name=name)


def attention_core(c: ShapeCtx, heads: int, d_qk: int, d_v: int, kv_heads: int, name: str) -> Op:
    """Causal flash attention over context `seq`: QK^T and PV, half masked."""
    t, s = c.tokens, c.seq
    flops = t * s * heads * (d_qk + d_v)
    kv_reads = max(1.0, s / 128.0)               # K and V streamed once per query block
    bytes_ = c.dt * t * (heads * (d_qk + d_v) + kv_heads * (d_qk + d_v) * kv_reads)
    return Op("attention", flops, bytes_, name=name)


def tp_degree(c: ShapeCtx) -> int:
    return c.degrees.tp


def _tp_after(c: ShapeCtx, op: Op, size: float) -> list:
    """An op whose output must be reduced over TP: blocking or overlapped,
    as the framework profile says."""
    if tp_degree(c) == 1:
        return [op]
    kind = "reducescatter" if c.profile.sequence_parallel else "allreduce"
    if c.profile.tp_comm == "overlapped":
        k = c.profile.tp_chunks
        chunk_op = Scale(1.0 / k, op, name=op.name + " chunk")
        chunk_comm = Collective(kind, "tp", size / k, name=f"tp {kind} chunk")
        return [FlowShop2((chunk_op,) * k, (chunk_comm,) * k, name=op.name + " + tp comm")]
    return [op, Collective(kind, "tp", size, name=f"tp {kind}")]


def _tp_before(c: ShapeCtx, size: float) -> list:
    """With sequence parallelism, the input of a TP region is all-gathered."""
    if tp_degree(c) > 1 and c.profile.sequence_parallel:
        return [Collective("allgather", "tp", size, name="tp allgather")]
    return []


def _norm(c: ShapeCtx, name: str) -> Op:
    h = c.d("hidden")
    t = c.tokens / (tp_degree(c) if c.profile.sequence_parallel else 1)
    return elementwise(5.0 * t * h, 2.0 * t * h * c.dt, name)


# ------------------------------------------------------------ components ---

def attention_forward(c: ShapeCtx, kind: str) -> list:
    t, h, dt = c.tokens, c.d("hidden"), c.dt
    act = t * h * dt
    out = [_norm(c, "attn norm")] + _tp_before(c, act)
    if kind == "gqa":
        hq, hkv, dh = c.d("heads"), c.d("kv_heads"), c.d("head_dim")
        out += [gemm(t, (hq + 2 * hkv) * dh, h, dt, "qkv"),
                elementwise(3.0 * t * (hq + hkv) * dh, 2.0 * t * (hq + hkv) * dh * dt, "rope")]
        if c.degrees.cp > 1:
            out.append(Collective("allgather", "cp", 2 * t * hkv * dh * dt, name="cp kv allgather"))
        out.append(attention_core(c, hq, dh, dh, hkv, "attention"))
        out += _tp_after(c, gemm(t, h, hq * dh, dt, "attn out"), act)
    else:  # mla
        hq, dh, dr, dv = c.d("heads"), c.d("head_dim"), c.d("rope_dim"), c.d("v_head_dim")
        ql, kl = c.d("q_lora"), c.d("kv_lora")
        out += [gemm(t, ql, h, dt, "q down"),
                gemm(t, hq * (dh + dr), ql, dt, "q up"),
                gemm(t, kl + dr, h, dt, "kv down"),
                gemm(t, hq * (dh + dv), kl, dt, "kv up"),
                elementwise(3.0 * t * hq * dr, 2.0 * t * hq * dr * dt, "rope")]
        if c.degrees.cp > 1:
            out.append(Collective("allgather", "cp", t * (kl + dr) * dt, name="cp latent allgather"))
        out.append(attention_core(c, hq, dh + dr, dv, hq, "attention"))
        out += _tp_after(c, gemm(t, h, hq * dv, dt, "attn out"), act)
    return out


def dense_ffn_forward(c: ShapeCtx, ffn: int | None = None, prefix: str = "ffn") -> list:
    t, h, dt = c.tokens, c.d("hidden"), c.dt
    f = c.d("ffn") if ffn is None else ffn
    act = t * h * dt
    return (_tp_before(c, act)
            + [gemm(t, 2 * f, h, dt, f"{prefix} gate+up"),
               elementwise(3.0 * t * f, 3.0 * t * f * dt, f"{prefix} act")]
            + _tp_after(c, gemm(t, h, f, dt, f"{prefix} down"), act))


def moe_ffn_forward(c: ShapeCtx) -> list:
    """Router, dispatch, routed experts (EP-sharded), combine, shared expert
    (TP-sharded). Routed experts are not TP-sharded."""
    t, h, dt = c.tokens, c.d("hidden"), c.dt
    e_all, top_k, ef = c.arch.dim("experts"), c.d("top_k"), c.d("expert_ffn")
    e_local = c.d("experts")
    ep = c.degrees.ep
    imb = c.arch.moe_imbalance if ep > 1 else 1.0
    rows = t * top_k * imb                         # token-expert rows on this rank
    per_expert = rows / e_local
    routed = [
        gemm(t, e_all, h, dt, "router"),
        Op("gemm", 2.0 * rows * 2 * ef * h, dt * (rows * h + e_local * 2 * ef * h + rows * 2 * ef),
           (per_expert, 2 * ef, h), name="experts gate+up"),
        elementwise(3.0 * rows * ef, 3.0 * rows * ef * dt, "experts act"),
        Op("gemm", 2.0 * rows * ef * h, dt * (rows * ef + e_local * ef * h + rows * h),
           (per_expert, h, ef), name="experts down"),
    ]
    if ep > 1:
        size = t * top_k * h * dt
        routed.insert(1, Collective("alltoall", "ep", size, name="ep dispatch"))
        routed.append(Collective("alltoall", "ep", size, name="ep combine"))
    shared = c.arch.dim("shared_experts", 0)
    out = [_norm(c, "ffn norm")] + routed
    if shared:
        out += dense_ffn_forward(c, ef * shared // tp_degree(c), prefix="shared expert")
    return out


def layer_forward(c: ShapeCtx, spec: LayerSpec) -> list:
    ffn = moe_ffn_forward(c) if spec.ffn == "moe" else [_norm(c, "ffn norm")] + dense_ffn_forward(c)
    return attention_forward(c, spec.attention) + ffn


def embed_forward(c: ShapeCtx) -> list:
    t, h = c.tokens, c.d("hidden")
    return [Op("memory", 0.0, 2.0 * t * h * c.dt, name="embedding")]


def head_forward(c: ShapeCtx, with_loss: bool) -> list:
    t, h, v = c.tokens, c.d("hidden"), c.d("vocab")
    out = [_norm(c, "final norm"), gemm(t, v, h, c.dt, "lm head")]
    if with_loss:
        out.append(elementwise(5.0 * t * v, 2.0 * t * v * c.dt, "loss"))
    if tp_degree(c) > 1 and not c.profile.vocab_parallel_loss:
        out.append(Collective("allgather", "tp", t * v * tp_degree(c) * c.dt, name="logits allgather"))
    return out


# ---------------------------------------------------------- backward rules ---

_BWD_FLOPS = {"gemm": 2.0, "attention": 2.5, "elementwise": 2.0, "memory": 1.0}
_BWD_BYTES = {"gemm": 2.0, "attention": 2.0, "elementwise": 2.0, "memory": 1.0}
_BWD_COLLECTIVE = {"allreduce": "allreduce", "allgather": "reducescatter",
                   "reducescatter": "allgather", "alltoall": "alltoall"}


def backward(block):
    """Backward of one forward block, by operator type."""
    match block:
        case Op():
            return Op(block.kind, _BWD_FLOPS[block.kind] * block.flops, _BWD_BYTES[block.kind] * block.bytes,
                      block.shape, kernels=2 if block.kind == "gemm" else 1, count=block.count,
                      name=block.name + " bwd")
        case Collective():
            return Collective(_BWD_COLLECTIVE[block.kind], block.axis, block.size, block.count, block.name + " bwd")
        case Scale():
            return Scale(block.factor, backward(block.body), block.name + " bwd")
        case FlowShop2():
            memo: dict = {}

            def bw(x):
                if id(x) not in memo:
                    memo[id(x)] = backward(x)
                return memo[id(x)]
            return FlowShop2(tuple(bw(x) for x in block.a), tuple(bw(x) for x in block.b),
                             block.count_a, block.buffer, block.name + " bwd")
    raise TypeError(f"no backward rule for {type(block).__name__}")


def layer_backward(c: ShapeCtx, fwd: list) -> list:
    """Backward of a layer, in reverse order, with the profile's recompute."""
    bwd = [backward(b) for b in reversed(fwd)]
    if c.profile.recompute == "full":
        return list(fwd) + bwd
    if c.profile.recompute == "selective":
        return [b for b in fwd if isinstance(b, Op) and b.kind == "attention"] + bwd
    return bwd


# --------------------------------------------------------- parameter counts ---

def layer_params(arch: Architecture, spec: LayerSpec, local: dict, degrees: Degrees, full: bool = False) -> float:
    """Parameters of one layer: full model (full=True) or per rank."""
    L = (lambda n, default=None: arch.dim(n, default)) if full else (lambda n, default=None: local.get(n, default))
    h = arch.dim("hidden")
    p = 2.0 * h                                                    # two norms
    if spec.attention == "gqa":
        hq, hkv, dh = L("heads"), L("kv_heads"), arch.dim("head_dim")
        p += h * (hq + 2 * hkv) * dh + hq * dh * h
    else:
        hq, dh, dr, dv = L("heads"), arch.dim("head_dim"), arch.dim("rope_dim"), arch.dim("v_head_dim")
        ql, kl = arch.dim("q_lora"), arch.dim("kv_lora")
        p += h * ql + ql * hq * (dh + dr) + h * (kl + dr) + kl * hq * (dh + dv) + hq * dv * h
    if spec.ffn == "dense":
        p += 3.0 * h * L("ffn")
    else:
        ef = arch.dim("expert_ffn")
        shared = arch.dim("shared_experts", 0)
        tp = 1 if full else degrees.tp
        p += h * arch.dim("experts") + L("experts") * 3.0 * h * ef + 3.0 * h * ef * shared / tp
    return p


def layer_expert_params(arch: Architecture, spec: LayerSpec, local: dict) -> float:
    """Per-rank routed-expert parameters of one layer (EP-sharded; their
    replicas are the dp groups, not the (dp, ep) groups)."""
    if spec.ffn != "moe":
        return 0.0
    return local["experts"] * 3.0 * arch.dim("hidden") * arch.dim("expert_ffn")


def layer_active_params(arch: Architecture, spec: LayerSpec, local: dict, degrees: Degrees) -> float:
    """Per-rank parameters used by one token (MoE: top_k experts only)."""
    if spec.ffn == "dense":
        return layer_params(arch, spec, local, degrees)
    full = layer_params(arch, spec, local, degrees)
    ef, h = arch.dim("expert_ffn"), arch.dim("hidden")
    return full - local["experts"] * 3.0 * h * ef + arch.dim("top_k") * 3.0 * h * ef


def embed_params(arch: Architecture, local: dict) -> float:
    return float(local["vocab"] * arch.dim("hidden"))


def model_params(arch: Architecture) -> float:
    full = {n: v for n, v in arch.dims}
    deg = Degrees()
    return sum(layer_params(arch, s, full, deg, full=True) * s.count for s in arch.layers) \
        + 2.0 * arch.dim("vocab") * arch.dim("hidden")
