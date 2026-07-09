"""
Dispatched grouped-GEMM MoE channel mixer for Kimi Linear (JAX / Flax NNX).

Replaces the dense O(E) reference MoE in kimi_linear_gdn2.py with the production
pattern: permute tokens so each expert's assignments are contiguous (dispatch),
run one matmul per expert as a single grouped GEMM (`jax.lax.ragged_dot`), then
un-permute and weighted-sum (combine). No token dropping, no capacity padding.

Pipeline per forward:
    1. Route:    sigmoid affinities (+ aux-loss-free bias on the SELECTION only)
                 -> keep only the token's top expert GROUPS (group-limited routing)
                 -> top-k experts among them -> normalize the k gate weights.
    2. Dispatch: build (token, expert) assignments, sort by expert id, gather the
                 hidden states into expert-contiguous order; group_sizes = per-expert
                 counts.
    3. Grouped GEMM: ragged_dot(x_sorted, W_in, group_sizes) -> SwiGLU ->
                 ragged_dot(a, W_out, group_sizes).   (gate+up fused into W_in)
    4. Combine:  scale rows by gate weight, scatter-add back to tokens (sums the
                 top-k contributions), add the always-on shared expert.

Routing follows the DeepSeek-V3 / Moonlight / Kimi lineage: sigmoid scoring,
normalized top-k weights, a shared expert, aux-loss-free load balancing via a
per-expert selection bias updated outside the gradient (see `update_router_bias`),
and group-limited ("node-limited") routing: the E experts are split into n_groups
groups and each token draws its top-k only from its topk_groups best-scoring
groups. At scale the groups map to devices, so this bounds all-to-all dispatch
traffic per token; it changes WHICH experts are candidates, never the dispatch /
grouped-GEMM / combine machinery. Set n_groups=1 to disable.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

F32 = jnp.float32

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² =
# 2^{-5}), replacing Flax NNX's default Linear kernel init. Biases stay at zero (the
# NNX default). The stacked expert weights below keep their own explicit fan-in init.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


class GroupedGemmMoE(nnx.Module):
    """Token-dispatched grouped-GEMM MoE with a shared expert.

    Args:
        d_model:   model width.
        d_ff:      per-expert hidden width (SwiGLU inner dim).
        n_routed:  number of routed experts E.
        n_shared:  number of shared experts (always-on), folded into one SwiGLU.
        top_k:     experts activated per token.
        n_groups:  expert groups for group-limited routing (1 = no restriction).
        topk_groups: groups a token may draw its top-k experts from.
        norm_topk: renormalize the top-k gate weights to sum to 1.
        routed_scale: multiply normalized gate weights (DeepSeek's routed_scaling_factor).
        bias_balancing: enable the aux-loss-free selection bias.
        aux_alpha: coefficient for the optional sequence-level load-balancing aux loss.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_routed: int = 256,
        n_shared: int = 1,
        top_k: int = 8,
        *,
        n_groups: int = 1,
        topk_groups: int = 1,
        norm_topk: bool = True,
        routed_scale: float = 1.0,
        bias_balancing: bool = True,
        aux_alpha: float = 1e-3,
        compute_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        assert n_routed % n_groups == 0, "n_routed must be divisible by n_groups"
        assert 1 <= topk_groups <= n_groups, "need 1 <= topk_groups <= n_groups"
        assert top_k <= topk_groups * (n_routed // n_groups), (
            "top_k experts must fit inside the topk_groups selected groups"
        )
        self.d_model = d_model
        self.d_ff = d_ff
        self.E = n_routed
        self.top_k = top_k
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.norm_topk = norm_topk
        self.routed_scale = routed_scale
        self.bias_balancing = bias_balancing
        self.aux_alpha = aux_alpha
        # Matmul dtype for the expert grouped GEMMs + shared expert (bf16 on H200).
        # Weights are stored fp32; the router, combine (scatter-add), and aux loss
        # stay fp32 below for stable routing/load-balancing.
        self.compute_dtype = compute_dtype

        self.router = nnx.Linear(
            d_model, n_routed, use_bias=False, kernel_init=_XAVIER, rngs=rngs
        )
        self.router_bias = nnx.Variable(jnp.zeros((n_routed,), F32))

        # Stacked routed-expert weights. Gate and up are fused into W_in so the
        # forward needs only TWO grouped GEMMs (W_in, W_out) instead of three.
        kin, kout = jax.random.split(rngs.params(), 2)
        self.w_in = nnx.Param(
            jax.random.normal(kin, (n_routed, d_model, 2 * d_ff), F32) * (d_model**-0.5)
        )
        self.w_out = nnx.Param(
            jax.random.normal(kout, (n_routed, d_ff, d_model), F32) * (d_ff**-0.5)
        )

        # Shared expert(s) as a single wider SwiGLU (always applied to every token).
        sg, su, sd = jax.random.split(rngs.params(), 3)
        ish = d_ff * n_shared
        self.ws_gate = nnx.Param(
            jax.random.normal(sg, (d_model, ish), F32) * (d_model**-0.5)
        )
        self.ws_up = nnx.Param(
            jax.random.normal(su, (d_model, ish), F32) * (d_model**-0.5)
        )
        self.ws_down = nnx.Param(
            jax.random.normal(sd, (ish, d_model), F32) * (ish**-0.5)
        )

    # ----------------------------------------------------------------------- #
    def _route(self, x_flat: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Route tokens to experts. Returns (top_idx [T,k], gate [T,k], logits [T,E]).

        DeepSeek-V3-style routing: the aux-loss-free selection bias is applied only
        to the SELECTION (which experts win top-k), never to the gate weights, and
        under stop_gradient — the bias is a non-learned nudge updated outside the
        gradient (see `update_router_bias`). Group-limited routing then restricts
        the candidates to the token's topk_groups best expert groups before the
        final top-k. Gate weights come from the TRUE sigmoid affinities of the
        selected experts, so the router still gets exact gradients. The raw logits
        are returned so the caller can reuse them for the aux loss without a second
        router matmul."""
        logits = self.router(x_flat).astype(F32)
        scores = jax.nn.sigmoid(logits)  # affinities [T,E]

        # Selection: true scores + per-expert bias (bias only shifts WHO wins top-k).
        sel = scores + self.router_bias if self.bias_balancing else scores
        sel = jax.lax.stop_gradient(sel) if self.bias_balancing else sel

        # Group-limited routing (DeepSeek-V3 "node-limited"): score each of the
        # n_groups expert groups by the sum of its top-2 selection scores (V3's
        # group metric), keep the token's topk_groups best groups, and mask every
        # other group's experts to -inf so the expert top-k below cannot pick them.
        # The bias participates via `sel`, so load balancing steers group choice too.
        if self.n_groups > 1:
            T = sel.shape[0]
            gsize = self.E // self.n_groups  # experts per group
            sel_g = sel.reshape(T, self.n_groups, gsize)
            top2, _ = jax.lax.top_k(sel_g, min(2, gsize))
            group_score = top2.sum(-1)  # [T, n_groups]
            _, gidx = jax.lax.top_k(group_score, self.topk_groups)  # [T, topk_groups]
            keep = (
                jnp.zeros((T, self.n_groups), bool)
                .at[jnp.arange(T)[:, None], gidx]
                .set(True)
            )
            sel = jnp.where(jnp.repeat(keep, gsize, axis=-1), sel, -jnp.inf)

        _, top_idx = jax.lax.top_k(sel, self.top_k)  # selection [T,k]

        # Gate weights from the true (un-biased) scores of the selected experts.
        gate = jnp.take_along_axis(scores, top_idx, axis=-1)
        if self.norm_topk:
            gate = gate / (gate.sum(-1, keepdims=True) + 1e-9)
        gate = gate * self.routed_scale

        return top_idx, gate, logits

    def _shared(self, x_flat: jax.Array) -> jax.Array:
        """Shared expert(s) as a single wider SwiGLU (always applied to every token).
        Runs the matmuls in compute_dtype (bf16 on H200); the caller upcasts the
        result to fp32 for the residual combine."""
        cd = self.compute_dtype
        xf = x_flat.astype(cd)
        a = jax.nn.silu(xf @ self.ws_gate.astype(cd)) * (xf @ self.ws_up.astype(cd))
        return a @ self.ws_down.astype(cd)

    # ----------------------------------------------------------------------- #
    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        B, L, d = x.shape
        T = B * L
        k = self.top_k
        xf = x.reshape(T, d)
        cdtype = self.compute_dtype  # force bf16 GEMMs even though the residual is fp32

        top_idx, gate, router_logits = self._route(xf)

        # ---- dispatch: flatten assignments and sort by expert id ----
        flat_e = top_idx.reshape(T * k).astype(jnp.int32)  # expert per assignment
        flat_tok = jnp.repeat(jnp.arange(T, dtype=jnp.int32), k)  # token per assignment
        flat_w = gate.reshape(T * k).astype(F32)

        order = jnp.argsort(flat_e)  # group same-expert rows
        sort_tok = flat_tok[order]
        sort_w = flat_w[order]
        group_sizes = jnp.bincount(flat_e, length=self.E)  # [E], sums to T*k

        x_sorted = xf[sort_tok].astype(cdtype)  # [M, d], M = T*k

        # ---- grouped GEMM: one matmul per expert over its contiguous rows ----
        h = jax.lax.ragged_dot(x_sorted, self.w_in.astype(cdtype), group_sizes)
        g_, u_ = jnp.split(h, 2, axis=-1)  # [M, d_ff] each
        a = jax.nn.silu(g_) * u_
        y_sorted = jax.lax.ragged_dot(
            a, self.w_out.astype(cdtype), group_sizes
        )  # [M,d]

        # ---- combine: weight, un-permute, sum top-k per token ----
        y_sorted = y_sorted.astype(F32) * sort_w[:, None]
        routed = (
            jnp.zeros((T, d), F32).at[sort_tok].add(y_sorted)
        )  # scatter-add over slots

        out = routed + self._shared(xf).astype(F32)
        out = out.reshape(B, L, d).astype(cdtype)

        # ---- diagnostics for the training loop ----
        load = group_sizes.astype(F32) / (T * k)  # fraction per expert

        # ---- aux loss ----
        # Switch/DeepSeek-style aux loss: E * <f_e, P_e>, where f_e is the realized
        # per-expert load fraction (non-differentiable; acts as a constant) and P_e
        # the mean softmax routing probability (this is where the gradient flows).
        # Reuses the logits already computed by _route (no second router matmul).
        probs = jax.nn.softmax(router_logits, axis=-1).mean(0)  # [E]
        aux_loss = self.aux_alpha * self.E * jnp.sum(load * probs)
        aux = {"load": load, "aux_loss": aux_loss, "group_sizes": group_sizes}
        return out, aux

    # ----------------------------------------------------------------------- #
    def dense_forward(self, x: jax.Array) -> jax.Array:
        """Reference path computing every expert densely (for tests only).
        Uses the SAME weights as __call__, so any mismatch is a dispatch/GEMM bug."""
        B, L, d = x.shape
        T = B * L
        xf = x.reshape(T, d)
        top_idx, gate, _ = self._route(xf)

        full = (
            jnp.zeros((T, self.E), F32).at[jnp.arange(T)[:, None], top_idx].add(gate)
        )  # [T,E] sparse weights
        h = jnp.einsum("td,edf->tef", xf, self.w_in)  # [T,E,2*d_ff]
        g_, u_ = jnp.split(h, 2, axis=-1)
        a = jax.nn.silu(g_) * u_
        ye = jnp.einsum("tef,efd->ted", a, self.w_out)  # [T,E,d]
        routed = jnp.einsum("te,ted->td", full, ye)
        out = routed + self._shared(xf)
        return out.reshape(B, L, d)


# --------------------------------------------------------------------------- #
def update_router_bias(
    bias: jax.Array, group_sizes: jax.Array, lr: float = 1e-3
) -> jax.Array:
    """Aux-loss-free load balancing (DeepSeek-V3 style), called in the training loop
    AFTER each step, outside the gradient:

        moe.router_bias = update_router_bias(
            moe.router_bias, aux['group_sizes'], lr)

    Nudges the selection bias up for under-loaded experts and down for over-loaded
    ones by a fixed step, driving per-expert load toward uniform without an aux loss.
    """
    load = group_sizes.astype(F32) / jnp.sum(group_sizes).astype(F32)
    target = 1.0 / bias.shape[0]
    return bias + lr * jnp.sign(target - load)


# Note on group-limited routing: implemented inside `_route` (n_groups / topk_groups),
# following Kimi K2 / DeepSeek-V3. At real scale each group of experts lives on one
# device, so limiting a token to topk_groups groups bounds its all-to-all dispatch
# fan-out; in this single-device implementation it is faithful-but-cosmetic — the
# restriction shapes routing exactly as at scale, while the dispatch / grouped-GEMM /
# combine machinery below is untouched (it never assumes anything about groups).
