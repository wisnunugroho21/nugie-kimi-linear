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
  (2) Init uses Flax NNX defaults; App. D.5 specifies Xavier-uniform with gain 2^{-2.5}
      and zero biases (except the decay bias, set negative here for fp32 safety).
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from gated_deltanet_2.core import chunkwise_gated_delta_rule_2


class ShortConv(nnx.Module):
    """Causal depthwise 1-D convolution — the 'Conv' boxes in Fig. 1 (Sec. 3.5).

    The paper says only "short causal convolution"; the kernel width (default 4)
    is an implementation choice, as in the Mamba/GatedDeltaNet lineage.
    """

    def __init__(self, channels: int, kernel_size: int = 4, *, rngs: nnx.Rngs):
        self.channels = channels
        self.kernel_size = kernel_size
        key = rngs.params()
        w = jax.random.normal(key, (channels, 1, kernel_size)) * (kernel_size**-0.5)
        self.weight = nnx.Param(w)
        self.bias = nnx.Param(jnp.zeros((channels,)))

    def __call__(self, x):  # x: [B, L, C]
        xt = jnp.transpose(x, (0, 2, 1))  # [B, C, L]
        xt = jnp.pad(
            xt, ((0, 0), (0, 0), (self.kernel_size - 1, 0))
        )  # causal pad (left-only)
        y = jax.lax.conv_general_dilated(
            xt,
            self.weight.value,
            window_strides=(1,),
            padding="VALID",
            feature_group_count=self.channels,  # depthwise: one filter per channel
            dimension_numbers=("NCW", "OIW", "NCW"),
        )
        y = y + self.bias.value[None, :, None]
        return jnp.transpose(y, (0, 2, 1))  # [B, L, C]


class GatedRMSNorm(nnx.Module):
    """RMSNorm then SiLU output gate — the 'Norm' + gate path in Fig. 1.

    Sec. 3.5 / Fig. 1 caption: "recurrent output is RMS-normalized, multiplied by a
    separate SiLU output gate, and passed through the output projection." App. D.5
    lists the RMSNorm+SiLU output gate among the family's training-recipe choices.
    """

    def __init__(self, dim: int, *, eps: float = 1e-5, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x, gate):  # both [..., dim]
        x = x.astype(jnp.float32)  # normalize in fp32
        rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        x = x * rms * self.weight.value  # RMSNorm
        return x * jax.nn.sigmoid(
            gate.astype(jnp.float32)
        )  # multiply by Sigmoid output gate


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
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.H = num_heads
        self.Hv = num_v_heads or num_heads
        assert self.Hv % self.H == 0, "num_v_heads must be a multiple of num_heads"
        self.group = self.Hv // self.H  # G, value-head group size (App. C.1)
        self.dk = head_k_dim
        self.dv = head_v_dim
        self.chunk_size = chunk_size
        self.expanded_erase = expanded_erase

        # App. C.1 projection shapes: erase/key side -> H·d_k, write/value side -> H_v·d_v.
        k_proj_dim = self.H * self.dk  # q, k, b live on the key-head axis
        v_proj_dim = self.Hv * self.dv  # v, w live on the value-head axis

        # Linear projections feeding the SiLU/conv paths (Sec. 3.5; Fig. 1 'Linear' boxes).
        self.q_proj = nnx.Linear(d_model, k_proj_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, k_proj_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, v_proj_dim, use_bias=False, rngs=rngs)
        self.b_proj = nnx.Linear(
            d_model, k_proj_dim, use_bias=True, rngs=rngs
        )  # Proj_b, Eq. 85: b = σ(Proj_b x)
        self.w_proj = nnx.Linear(
            d_model, v_proj_dim, use_bias=True, rngs=rngs
        )  # Proj_w, Eq. 85: w = σ(Proj_w x)
        self.f_proj = nnx.Linear(
            d_model, k_proj_dim, use_bias=True, rngs=rngs
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
        self.gate_proj = nnx.Linear(
            d_model, v_proj_dim, use_bias=False, rngs=rngs
        )  # SiLU output gate source
        self.o_norm = GatedRMSNorm(self.dv, rngs=rngs)
        self.o_proj = nnx.Linear(
            v_proj_dim, d_model, use_bias=False, rngs=rngs
        )  # back to d_model
        # NOTE (deviation 2): App. D.5 specifies Xavier-uniform init, gain 2^{-2.5}, zero biases;
        # this module uses Flax NNX default initializers instead.

    def _split_k(self, x, B, L):
        # Head reshaping for key-side tensors (App. C.1: "followed by head reshaping").
        return x.reshape(B, L, self.H, self.dk).transpose(0, 2, 1, 3)  # [B,H,L,dk]

    def _split_v(self, x, B, L):
        # Head reshaping for value-side tensors.
        return x.reshape(B, L, self.Hv, self.dv).transpose(0, 2, 1, 3)  # [B,Hv,L,dv]

    def __call__(self, x, initial_state=None):
        """x: [B, L, d_model]. Returns (out: [B, L, d_model], final_state: [B,Hv,dk,dv])."""
        B, L, _ = x.shape

        # q,k,v paths: Linear -> ShortConv -> SiLU (Sec. 3.5; Fig. 1 caption).
        q = jax.nn.silu(self.q_conv(self.q_proj(x)))
        k = jax.nn.silu(self.k_conv(self.k_proj(x)))
        v = jax.nn.silu(self.v_conv(self.v_proj(x)))

        q = self._split_k(q, B, L)
        k = self._split_k(k, B, L)
        v = self._split_v(v, B, L)

        # L2-normalize q, k per head (Sec. 3.5 "L2 normalization applied to q_t and k_t"; App. D.2).
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

        # Log-decay branch, computed in fp32 outside the kernel (Eq. 12 / 86; App. C.1 / D.1).
        #   g_t = -exp(a) ⊙ softplus(Proj_f(x_t) + δ),  then α_t = exp(g_t) inside the core.
        f = self.f_proj(x).astype(jnp.float32) + self.dt_bias.value.astype(
            jnp.float32
        )  # Proj_f(x)+δ
        f = self._split_k(f, B, L)
        a = jnp.exp(self.A_log.value.astype(jnp.float32))[
            None, :, None, :
        ]  # exp(a); [1,H,1,dk]
        g = -a * jax.nn.softplus(f)  # [B,H,L,dk] ≤ 0  (Eq. 86)

        # Channel-wise gates (Eq. 11 / 85).
        b = jax.nn.sigmoid(self.b_proj(x))  # b = σ(Proj_b x) ∈ [0,1]^{d_k}
        b = self._split_k(b, B, L)
        if self.expanded_erase:
            b = (
                2.0 * b
            )  # neg-eigenvalue variant: scale ONLY b to [0,2] (Sec. 3.1 / App. C.1)
        w = jax.nn.sigmoid(
            self.w_proj(x)
        )  # w = σ(Proj_w x) ∈ [0,1]^{d_v}  (write gate stays in [0,1])
        w = self._split_v(w, B, L)

        # GQA: repeat key-side tensors across value-head groups (Sec. 3.5 / App. C.1).
        #   q, k, g, b are repeated; v, w already live on the value-head axis.
        if self.group > 1:
            rep = lambda t: jnp.repeat(t, self.group, axis=1)
            q, k, g, b = rep(q), rep(k), rep(g), rep(b)

        if initial_state is None:
            initial_state = jnp.zeros((B, self.Hv, self.dk, self.dv), jnp.float32)

        # Gated Delta Rule-2 chunkwise core (Eq. 10); the kernel forms cumsum γ internally (Eq. 30).
        O, final_state = chunkwise_gated_delta_rule_2(
            q, k, v, g, b, w, initial_state, chunk_size=self.chunk_size
        )

        # Gated RMSNorm + output projection (Sec. 3.5 / App. D.5).
        O = O.transpose(0, 2, 1, 3)  # [B,Hv,L,dv] -> [B,L,Hv,dv]
        gate = self.gate_proj(x).reshape(
            B, L, self.Hv, self.dv
        )  # SiLU output gate (applied in o_norm)
        O = self.o_norm(O, gate).reshape(
            B, L, self.Hv * self.dv
        )  # RMSNorm(O) · SiLU(gate)
        out = self.o_proj(O.astype(x.dtype))  # project back to d_model
        return out, final_state
