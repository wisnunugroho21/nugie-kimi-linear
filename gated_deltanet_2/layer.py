"""
Gated DeltaNet-2 token-mixer layer in Flax NNX, ANNOTATED against the paper
(arXiv:2605.22791): Section 3.5 (block design) and Appendix C.1 (layer
parameterization), with supporting equations 11, 12, 85, 86 and the numerical
notes in Appendix D.

Block design (Fig. 1 right; Sec. 3.5 "Gated DeltaNet-2 token mixer"):
  q,k = L2norm(SiLU(ShortConv(Linear(x))))      # key-side paths + L2 norm (Sec. 3.5, App. D.2)
  v   =        SiLU(ShortConv(Linear(x)))        # value path (Sec. 3.5; Fig. 1 caption)
  g   = -exp(a) ⊙ softplus(Linear_f(x) + delta)  # log-decay, fp32 (Eq. 12 / 86, App. D.1)
  b   = sigmoid(Linear_b(x))                     # erase gate (Eq. 11 / 85); x2 if neg-eigenvalue
  w   = sigmoid(Linear_w(x))                     # write gate (Eq. 11 / 85)
  O   = chunkwise_gated_delta_rule_2(q,k,v,g,b,w, state)   # Gated Delta Rule-2 (Eq. 10)
  out = Linear_o( RMSNorm(O) * SiLU(gate) )      # gated RMSNorm + out proj (Sec. 3.5, App. D.5)

Grouped value heads (Sec. 3.5 last sentence / App. C.1): with num_v_heads = G*num_heads,
the key-side tensors q, k, the log-decay g, and b are repeated across the G value-head
groups; v and w already live on the value-head axis.

Scope: this is the recurrent TOKEN MIXER only (Fig. 1 right). The recurrent model
(Sec. 3.5 "Model families") stacks [this + MLP]; the hybrid model inserts
Sliding-Window Attention after it, repeating the cell [GDN-2, MLP, SWA, MLP]
(Fig. 1 left). Those wrappers are not implemented here.

Two honest deviations from the paper, flagged inline below:
  (1) A_log is stored per (head, key-channel); App. C.1 stores 'a' per key HEAD and
      broadcasts it over d_k. This implementation is a strict generalization (tie the
      d_k columns to recover the paper).
  (2) All Linear kernels use the paper's Xavier-uniform init, gain 2^{-2.5}, with zero
      biases (App. D.5). The one exception: the decay bias δ starts negative (−4), not
      the paper's value, to keep early decay mild for fp32 stability (App. D.1).
"""

from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from gated_deltanet_2.core import (
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)

F32 = jnp.float32

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² =
# 2^{-5}), replacing Flax NNX's default Linear kernel init. Biases stay at zero (the
# NNX default) — except the decay bias δ, set negative in __init__ for fp32 safety.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


# --------------------------------------------------------------------------- #
#  Inference cache for streaming (incremental) decode.
#
#  Linear attention's headline property: the entire history collapses into a
#  FIXED-SIZE recurrent state S [B,Hv,dk,dv] — it does NOT grow with sequence
#  length (contrast a softmax KV-cache). To decode token-by-token we just carry S
#  across calls.  The short causal conv ALSO has a (kernel_size)-wide receptive
#  field, so we must additionally cache its last (kernel_size-1) inputs — otherwise
#  the first streamed tokens would see wrong, zero-padded context.  That is the
#  WHOLE state of a GDN-2 layer; both pieces are fixed-size.
# --------------------------------------------------------------------------- #
class GDN2Cache(NamedTuple):
    recurrent_state: jax.Array  # [B, Hv, dk, dv]  the gated-delta-rule memory S
    q_conv: jax.Array  # [B, conv_size-1, H*dk]   last inputs to the q short-conv
    k_conv: jax.Array  # [B, conv_size-1, H*dk]   last inputs to the k short-conv
    v_conv: jax.Array  # [B, conv_size-1, Hv*dv]  last inputs to the v short-conv


class RMSNorm(nnx.Module):
    """Plain RMSNorm used for the pre-norms around mixer / channel-mixer."""

    def __init__(self, dim: int, *, eps: float = 1e-5, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        xf = x.astype(F32)
        mean = jnp.mean(xf * xf, axis=-1, keepdims=True)
        rms = jax.lax.rsqrt(mean + self.eps)

        return (xf * rms).astype(x.dtype) * self.weight.value


class LowRankLinear(nnx.Module):
    """y = W_up(W_down x). The W↑(W↓·) factorization Kimi Linear uses for its
    output gate and decay, kept at rank = head dim for parameter parity."""

    def __init__(
        self,
        in_features: int,
        rank: int,
        out_features: int,
        *,
        use_bias: bool = False,
        rngs: nnx.Rngs,
    ):
        self.down = nnx.Linear(
            in_features, rank, use_bias=False, kernel_init=_XAVIER, rngs=rngs
        )
        self.up = nnx.Linear(
            rank, out_features, use_bias=use_bias, kernel_init=_XAVIER, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.down(x)
        return self.up(x)


class GatedRMSNorm(nnx.Module):
    """Head-wise RMSNorm of the recurrent output, gated by a LOW-RANK SIGMOID gate.

    Implements Kimi Linear Eq. 10's output stage:

        Sigmoid(W↑g W↓g x) ⊙ RMSNorm(O)

    Two corrections vs the GDN-2 paper block used in your gdn2_layer:
      (1) sigmoid, not SiLU/swish  — Kimi Linear's ablation found the swish output
          gate (GDN's choice) performs substantially worse than sigmoid, and they
          adopt sigmoid across all experiments including their GDN hybrid baseline.
      (2) low-rank gate (W↑ W↓), rank = head dim — Kimi Linear factorizes the gate
          "to ensure a fair parameter comparison" against the full-attention baseline.

    The gate is produced INSIDE the norm from the block input x, so the call site in
    your layer collapses to `O = self.o_norm(O_heads, x)` (no separate gate_proj).
    """

    def __init__(
        self,
        head_dim: int,
        d_model: int,
        inner_dim: int,
        gate_rank: int,
        *,
        eps: float = 1e-5,
        rngs: nnx.Rngs,
    ):
        self.norm = RMSNorm(head_dim, eps=eps, rngs=rngs)
        self.gate = LowRankLinear(
            d_model, gate_rank, inner_dim, use_bias=False, rngs=rngs
        )

    def __call__(self, O_heads: jax.Array, x: jax.Array) -> jax.Array:
        """O_heads: [B, L, Hv, dv]   x: [B, L, d_model]  ->  [B, L, Hv*dv]."""
        B, L, Hv, dv = O_heads.shape

        o = O_heads.astype(F32)  # [B,L,Hv,dv] -> fp32 for RMSNorm
        o = self.norm(o)  # head-wise RMSNorm

        g = self.gate(x).astype(F32)  # low-rank gate
        g = jax.nn.sigmoid(g)  # low-rank SIGMOID gate
        g = g.reshape(B, L, Hv, dv)

        return (o * g).reshape(B, L, Hv * dv)


class ShortConv(nnx.Module):
    """Causal depthwise 1-D convolution — the 'Conv' boxes in Fig. 1 (Sec. 3.5).

    The paper says only "short causal convolution"; the kernel width (default 4)
    is an implementation choice, as in the Mamba/GatedDeltaNet lineage.

    nnx.Conv is channels-last ([B, L, C]) and owns the kernel+bias, so the manual
    NCW transposes and the raw conv call disappear. Padding is fixed at construction,
    so we run the conv in 'VALID' mode and keep the causal left-context / streaming
    state ourselves (state = the trailing kernel_size-1 inputs).
    """

    def __init__(self, channels: int, kernel_size: int = 4, *, rngs: nnx.Rngs):
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=(kernel_size,),
            feature_group_count=channels,  # depthwise: one filter per channel
            padding="VALID",  # left context supplied manually below
            use_bias=True,
            rngs=rngs,
        )

    def _apply(
        self, x: jax.Array, conv_state: jax.Array | None
    ) -> tuple[jax.Array, jax.Array]:
        """Shared conv core. `conv_state` is the previous (kernel_size-1) inputs used
        as left context, or None on the full/training path (pad with zeros == the
        causal left-pad). Returns (y: [B, L, C], new_state: [B, kernel_size-1, C])."""
        B, L, C = x.shape
        kc = self.kernel_size - 1

        left = jnp.zeros((B, kc, C), x.dtype) if conv_state is None else conv_state
        xc = jnp.concatenate([left, x], axis=1)  # [B, kc+L, C]
        new_state = xc[:, xc.shape[1] - kc :, :]  # last kc inputs -> next context

        y = self.conv(xc)  # VALID: (kc+L)-(kc+1)+1 = L
        return y, new_state  # [B, L, C]

    def __call__(
        self, x: jax.Array
    ) -> jax.Array:  # full-sequence (training) path; left context = zeros
        y, _ = self._apply(x, conv_state=None)
        return y

    def step(
        self, x: jax.Array, conv_state: jax.Array
    ) -> tuple[jax.Array, jax.Array]:  # streaming path; carry the left context in/out
        return self._apply(x, conv_state)


class GatedDeltaNet2(nnx.Module):
    """Gated DeltaNet-2 recurrent token mixer (Fig. 1 right; Sec. 3.5 / App. C.1)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,  # H key heads; App. E.1 uses H=16 at 1.3B
        head_k_dim: int = 128,  # d_k; App. E.1 uses 128
        head_v_dim: int = 128,  # d_v; App. E.1 uses 128
        num_v_heads: int | None = None,  # H_v for GQA; defaults to H (App. C.1)
        chunk_size: int = 64,  # C; App. C.2 fixes C = 64
        conv_size: int = 4,
        expanded_erase: bool = False,  # erase gate in [0,2] (neg-eigenvalue variant; Sec. 3.1, App. C.1)
        compute_dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        # Matmul dtype for the q/k/v/b/w/o projection Linears (bf16 on H200). The
        # chunkwise/recurrent core (core.py) upcasts to fp32 regardless, and the
        # log-decay branch (f_proj) is kept fp32 below — both for numerical safety.
        self.compute_dtype = compute_dtype
        self.d_model = d_model
        self.H = num_heads
        self.Hv = num_v_heads or num_heads

        assert self.Hv % self.H == 0, "num_v_heads must be a multiple of num_heads"

        self.group = self.Hv // self.H  # G, value-head group size (App. C.1)
        self.dk = head_k_dim
        self.dv = head_v_dim
        self.chunk_size = chunk_size
        self.conv_size = conv_size  # kernel width; sizes the streaming conv cache
        self.expanded_erase = expanded_erase

        # App. C.1 projection shapes: erase/key side -> H·d_k, write/value side -> H_v·d_v.
        k_proj_dim = self.H * self.dk  # q, k, b live on the key-head axis
        v_proj_dim = self.Hv * self.dv  # v, w live on the value-head axis

        # Linear projections feeding the SiLU/conv paths (Sec. 3.5; Fig. 1 'Linear' boxes).
        self.q_proj = nnx.Linear(
            d_model,
            k_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            d_model,
            k_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            d_model,
            v_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.b_proj = nnx.Linear(
            d_model,
            k_proj_dim,
            use_bias=True,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )  # Proj_b, Eq. 85: b = σ(Proj_b x)
        self.w_proj = nnx.Linear(
            d_model,
            v_proj_dim,
            use_bias=True,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )  # Proj_w, Eq. 85: w = σ(Proj_w x)
        self.f_proj = LowRankLinear(
            d_model, self.dk, k_proj_dim, use_bias=True, rngs=rngs
        )  # Proj_f, Eq. 86 (log-decay)

        # Short causal convs on q, k, v (App. C.1: "short-convolutional projections for q, k, v").
        self.q_conv = ShortConv(k_proj_dim, conv_size, rngs=rngs)
        self.k_conv = ShortConv(k_proj_dim, conv_size, rngs=rngs)
        self.v_conv = ShortConv(v_proj_dim, conv_size, rngs=rngs)

        # Log-decay parameters (Eq. 12 / 86; App. C.1).
        #   DEVIATION (1): paper stores 'a' per key HEAD (shape [H]) broadcast over d_k.
        #   Here A_log is [H, d_k] (per head AND per channel) — a strict generalization.
        self.A_log = nnx.Param(
            jnp.zeros((self.H, self.dk))
        )  # 'a' in -exp(a)·softplus(·)

        # App. C.1: bias δ is stored per key channel -> shape [H·d_k]. Eq. 86 adds it pre-softplus.
        #   Init negative (not the paper's value) so per-token decay starts mild (α≈1),
        #   keeping cumulative decay / γ^{-1} in a safe fp32 range (cf. App. D.1).
        self.dt_bias = nnx.Param(jnp.full((self.H * self.dk,), -4.0))  # δ

        # Output gate + gated RMSNorm + output projection (Sec. 3.5 / App. D.5).
        self.o_norm = GatedRMSNorm(
            head_dim=self.dv,
            d_model=d_model,
            inner_dim=self.Hv * self.dv,
            gate_rank=self.dv,
            rngs=rngs,
        )

        self.o_proj = nnx.Linear(
            v_proj_dim,
            d_model,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )  # back to d_model

        # App. D.5: every Linear kernel above uses Xavier-uniform init, gain 2^{-2.5}
        # (_XAVIER); biases are zero except the decay bias δ (-4, for fp32 safety).

    def _split_k(self, x: jax.Array, B: int, L: int) -> jax.Array:
        # Head reshaping for key-side tensors (App. C.1: "followed by head reshaping").
        return x.reshape(B, L, self.H, self.dk).swapaxes(1, 2)  # [B,H,L,dk]

    def _split_v(self, x: jax.Array, B: int, L: int) -> jax.Array:
        # Head reshaping for value-side tensors.
        return x.reshape(B, L, self.Hv, self.dv).swapaxes(1, 2)  # [B,Hv,L,dv]

    def _project(
        self, x: jax.Array, conv_states: tuple[jax.Array, jax.Array, jax.Array] | None
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        tuple[jax.Array, jax.Array, jax.Array] | None,
    ]:
        """Shared front-end used by BOTH the training and streaming paths:
        Linear -> ShortConv -> SiLU -> head split -> L2 norm, plus the log-decay g
        and the channel-wise gates b, w.  `conv_states` is None on the full/training
        path, or a (q, k, v) tuple of conv caches when streaming.  Returns
        (q, k, v, g, b, w) on the value-head (Hv) axis and the updated conv states
        (or None)."""
        B, L, _ = x.shape

        # q,k,v paths: Linear -> ShortConv -> SiLU (Sec. 3.5; Fig. 1 caption).
        if conv_states is None:  # full/training: conv pads with zeros (causal)
            q = self.q_conv(self.q_proj(x))
            k = self.k_conv(self.k_proj(x))
            v = self.v_conv(self.v_proj(x))
            new_conv = None
        else:  # streaming: conv uses the cached left context and returns a new one
            qcs, kcs, vcs = conv_states
            q, qcs = self.q_conv.step(self.q_proj(x), qcs)
            k, kcs = self.k_conv.step(self.k_proj(x), kcs)
            v, vcs = self.v_conv.step(self.v_proj(x), vcs)
            new_conv = (qcs, kcs, vcs)

        q, k, v = jax.nn.silu(q), jax.nn.silu(k), jax.nn.silu(v)

        q = self._split_k(q, B, L)
        k = self._split_k(k, B, L)
        v = self._split_v(v, B, L)

        # L2-normalize q, k per head (Sec. 3.5 "L2 normalization applied to q_t and k_t"; App. D.2).
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

        # Log-decay branch, computed in fp32 outside the kernel (Eq. 12 / 86; App. C.1 / D.1).
        #   g_t = -exp(a) ⊙ softplus(Proj_f(x_t) + δ),  then α_t = exp(g_t) inside the core.
        f_p = self.f_proj(x).astype(jnp.float32)  # [B,L,H*dk]  Proj_f(x) in Eq. 86
        d_t = self.dt_bias.value.astype(jnp.float32)  # [H*dk]  decay bias δ, Eq. 86
        a_l = self.A_log.value.astype(
            jnp.float32
        )  # [H, d_k]  log-decay matrix A, Eq. 86

        f = f_p + d_t  # Proj_f(x)+δ
        f = self._split_k(f, B, L)
        a = jnp.exp(a_l)[None, :, None, :]  # exp(a); [1,H,1,dk]
        g = -a * jax.nn.softplus(f)  # [B,H,L,dk] ≤ 0  (Eq. 86)

        # Channel-wise gates (Eq. 11 / 85).
        b = jax.nn.sigmoid(self.b_proj(x))  # b = σ(Proj_b x) ∈ [0,1]^{d_k}
        b = self._split_k(b, B, L)

        if self.expanded_erase:
            b = 2.0 * b  # neg-eigenvalue variant: scale ONLY b to [0,2] (Sec. 3.1)

        w = jax.nn.sigmoid(self.w_proj(x))  # w = σ(Proj_w x) ∈ [0,1]^{d_v}
        w = self._split_v(w, B, L)

        # GQA: repeat key-side tensors across value-head groups (Sec. 3.5 / App. C.1).
        #   q, k, g, b are repeated; v, w already live on the value-head axis.
        if self.group > 1:

            def rep(t: jax.Array) -> jax.Array:
                return jnp.repeat(t, self.group, axis=1)

            q, k, g, b = rep(q), rep(k), rep(g), rep(b)

        return q, k, v, g, b, w, new_conv

    def _output(self, o: jax.Array, x: jax.Array) -> jax.Array:
        """Gated RMSNorm + output projection (Sec. 3.5 / App. D.5). o: [B,Hv,L,dv]."""
        o = o.swapaxes(1, 2)  # [B,Hv,L,dv] -> [B,L,Hv,dv]
        o = self.o_norm(o, x).astype(
            x.dtype
        )  # low-rank sigmoid gate computed inside, from x

        return self.o_proj(o)  # project back to d_model

    def __call__(
        self, x: jax.Array, initial_state: jax.Array | None = None
    ) -> jax.Array:
        """Full-sequence (training) forward via the CHUNKWISE parallel core.
        x: [B, L, d_model] -> out: [B, L, d_model].

        The core also produces the end-of-sequence recurrent state, but training only
        needs the outputs, so it is discarded here (the streaming `step` below is what
        threads the state across calls). `initial_state` lets a caller warm-start from a
        prior state; it defaults to zeros."""
        B, _, _ = x.shape
        q, k, v, g, b, w, _ = self._project(x, conv_states=None)

        if initial_state is None:
            initial_state = jnp.zeros((B, self.Hv, self.dk, self.dv), jnp.float32)

        # Gated Delta Rule-2 chunkwise core (Eq. 10); forms cumsum γ internally (Eq. 30).
        o, _ = chunkwise_gated_delta_rule_2(
            q, k, v, g, b, w, initial_state, chunk_size=self.chunk_size
        )

        return self._output(o, x)

    # ----------------------------------------------------------------------- #
    #  Streaming / inference.  Same math, but via the RECURRENT core, which works
    #  for ANY length (no chunk-size divisibility constraint) and naturally threads
    #  the fixed-size state in -> out.  One method serves both phases of decoding:
    #     prefill: out, cache = layer.step(prompt, layer.init_cache(B, ...))
    #     decode : out, cache = layer.step(one_token, cache)   # repeat
    # ----------------------------------------------------------------------- #
    def init_cache(
        self, batch_size: int, max_len: int | None = None, dtype=jnp.float32
    ) -> GDN2Cache:
        """Empty streaming cache. `max_len` is accepted for a uniform interface with
        the MLA cache but UNUSED here — the GDN-2 state is fixed-size, independent of
        sequence length (the point of linear attention)."""
        kc = self.conv_size - 1
        return GDN2Cache(
            recurrent_state=jnp.zeros(
                (batch_size, self.Hv, self.dk, self.dv), jnp.float32
            ),
            q_conv=jnp.zeros((batch_size, kc, self.H * self.dk), dtype),
            k_conv=jnp.zeros((batch_size, kc, self.H * self.dk), dtype),
            v_conv=jnp.zeros((batch_size, kc, self.Hv * self.dv), dtype),
        )

    def step(self, x: jax.Array, cache: GDN2Cache) -> tuple[jax.Array, GDN2Cache]:
        """Streaming forward. x: [B, L, d_model] (L>=1). Returns (out, new_cache)."""
        q, k, v, g, b, w, new_conv = self._project(
            x, conv_states=(cache.q_conv, cache.k_conv, cache.v_conv)
        )
        # We passed real conv_states, so _project always returns updated ones here
        # (it only returns None on the full-sequence/training path) — assert narrows
        # the tuple|None type for the checker and documents the invariant.
        assert new_conv is not None
        qcs, kcs, vcs = new_conv

        # Recurrent core: token-by-token, threading S_in -> S_out (Eq. 9 / 29).
        o, new_state = recurrent_gated_delta_rule_2(
            q, k, v, g, b, w, cache.recurrent_state
        )

        return self._output(o, x), GDN2Cache(new_state, qcs, kcs, vcs)
