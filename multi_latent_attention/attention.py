from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² =
# 2^{-5}), replacing Flax NNX's default Linear kernel init. Biases stay at zero.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


class MLACache(NamedTuple):
    """Streaming KV cache for the MLA layer. Thanks to MLA we cache only the small
    COMPRESSED latent `l_kv` (one latent serves as BOTH K and V — see below), in a
    preallocated [B, max_len, Hkv*Dh] buffer written at position `pos`. Unlike GDN-2's
    fixed-size state, this GROWS with context: these full-attention layers are exactly
    the ones that pay the long-context KV-cache cost in the hybrid (3:1 keeps them few)."""

    l_kv: jax.Array  # [B, max_len, num_kv_heads*head_dim]  preallocated latent buffer
    pos: jax.Array  # scalar int32: number of filled positions so far


class GroupedQueryLatentAttention(nnx.Module):
    """Grouped-Query attention over a low-rank KV *latent*, in MLA "absorbed" form.

    This is NoPE (no rotary embeddings) Multi-head Latent Attention written in its
    matrix-absorbed form, fused with GQA-style KV-head sharing. Each of the three
    projections folds together two of the usual MLA matrices:

        w_q_uk : W_Q  . W_UK   -> queries are produced *directly* in the
                                  compressed K space, so they can dot against the
                                  latent without an explicit key up-projection.
        w_dkv  : W_DKV          -> down-projects x to the shared KV latent (c_kv).
        w_uv_o : W_UV . W_O     -> up-projects the value latent and applies the
                                  output projection in a single matmul.

    Key consequence: because there is no RoPE, W_UK and W_UV can be absorbed away
    *exactly*, and in the compressed latent space the keys and the values are the
    same tensor. That is why a single `l_kv` plays the role of BOTH K and V below.

    Note: `head_dim` here is the per-head latent (rank) dimension, not a
    conventional attention head width.
    """

    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_length: int,
        rngs: nnx.Rngs,
    ):
        # GQA constraint: every KV (latent) head must serve a whole number of
        # query heads, so that `repeat` below tiles the latent evenly.
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # How many query heads share each KV/latent head (the GQA group size).
        self.group_size = num_q_heads // num_kv_heads

        d_q = num_q_heads * head_dim  # total width of the query projection
        d_kv = num_kv_heads * head_dim  # total width of the (shared) KV latent

        # W_Q . W_UK absorbed: x -> queries already living in the latent K space.
        self.w_q_uk = nnx.Linear(
            embed_dim, d_q, use_bias=False, kernel_init=_XAVIER, rngs=rngs
        )

        # W_DKV: x -> low-rank KV latent c_kv (one latent per KV head).
        self.w_dkv = nnx.Linear(
            embed_dim, d_kv, use_bias=False, kernel_init=_XAVIER, rngs=rngs
        )

        # W_UV . W_O absorbed: value-latent -> up-projected, output-projected.
        self.w_uv_o = nnx.Linear(
            d_q, embed_dim, use_bias=False, kernel_init=_XAVIER, rngs=rngs
        )

        # Lower-triangular causal mask (True = keep), built once at the
        # construction seq_length and sliced at call time, so it also covers any
        # shorter sequence. The diagonal is included, guaranteeing at least one
        # unmasked key per row (so the -inf masking below cannot NaN).
        #
        # Wrapped in nnx.Variable so NNX treats it as a proper *state leaf* (data)
        # rather than a static attribute. It is a plain Variable, not an nnx.Param,
        # so optimizers that filter on Param leave it untouched -- correct for a
        # constant. It is still carried in the module state (moved/checkpointed
        # with the model). Indexing it (below) returns the underlying array.
        self.causal_mask = nnx.Variable(
            jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, T, embed_dim)
        batch_size, seq_length, _ = x.shape

        # --- Queries (already in the compressed K space via the absorbed W_UK) ---
        q_latent = self.w_q_uk(x)  # (B, T, num_q_heads * head_dim)

        # Split the flat projection into per-head latent vectors.
        q_reshaped = q_latent.reshape(
            batch_size, seq_length, self.num_q_heads, self.head_dim
        )  # (B, T, Hq, Dh)

        # Move the head axis next to batch for batched matmuls: (B, Hq, T, Dh)
        q_heads = q_reshaped.swapaxes(1, 2)

        # --- Shared KV latent (serves as both keys and values) ---
        l_kv = self.w_dkv(x)  # (B, T, num_kv_heads * head_dim)

        l_kv_reshaped = l_kv.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        )  # (B, T, Hkv, Dh)

        l_kv_heads = l_kv_reshaped.swapaxes(1, 2)  # (B, Hkv, T, Dh)

        # GQA tiling: repeat each latent head `group_size` times so it lines up
        # with the query heads. `repeat` interleaves, so KV head i feeds query
        # heads [i*group_size : (i+1)*group_size]. Result: (B, Hq, T, Dh).
        # (This materializes the full Hq KV stack; broadcasting would save memory
        # but materializing keeps the einsums simple.)
        l_kv_repeated = l_kv_heads.repeat(self.group_size, axis=1)

        # --- Attention scores: Q . K^T, contracting the latent feature dim `d` ---
        # 'd' is shared (contracted); 'k' indexes key/latent positions (kept).
        qk_t = jnp.einsum("bhqd, bhkd -> bhqk", q_heads, l_kv_repeated)  # (B, Hq, T, T)

        # Scale by sqrt of the latent per-head dim.
        scaled_logits = qk_t / jnp.sqrt(self.head_dim)

        # Apply causal mask: future positions -> -inf so they vanish under softmax.
        # Indexing the nnx.Variable yields the raw bool array. Safe to use -inf
        # here because the diagonal is always kept (no fully-masked rows).
        scaled_logits = jnp.where(
            self.causal_mask[None, None, :seq_length, :seq_length],
            scaled_logits,
            -jnp.inf,
        )

        # Softmax over the key axis -> per-query attention distribution.
        a = jax.nn.softmax(scaled_logits, axis=-1)  # (B, Hq, T, T)

        # --- Weighted sum of value-latents ---
        # 'k' is shared between the weights and the value positions, so it is the
        # contracted axis (the actual attention sum); 'd' is the kept feature dim.
        # Because keys and values are the same latent, l_kv_repeated reappears here.
        weighted_heads = jnp.einsum(
            "bhqk, bhkd -> bhqd", a, l_kv_repeated
        )  # (B, Hq, T, Dh)

        # Move head axis back and flatten heads: (B, T, Hq, Dh) -> (B, T, Hq*Dh)
        weighted_reshaped = weighted_heads.swapaxes(1, 2)
        weighted_latents = weighted_reshaped.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )

        # Absorbed W_UV . W_O: up-project the value latent and output-project.
        output = self.w_uv_o(weighted_latents)  # (B, T, embed_dim)

        return output

    # ----------------------------------------------------------------------- #
    #  Streaming / inference.  Same softmax attention, but the KV latents of past
    #  positions are read from a preallocated cache instead of recomputed, and the
    #  new positions are written into it.  Use it for prefill (L = prompt length)
    #  and per-token decode (L = 1) alike.
    # ----------------------------------------------------------------------- #
    def init_cache(self, batch_size: int, max_len: int, dtype=jnp.float32) -> MLACache:
        """Initialize the streaming KV cache for a given batch size and max length.
        The cache is a preallocated buffer of shape [B, max_len, Hkv*Dh] and a position counter. 
        The buffer is filled with zeros initially."""
        d_kv = self.num_kv_heads * self.head_dim
        return MLACache(
            l_kv=jnp.zeros((batch_size, max_len, d_kv), dtype),
            pos=jnp.array(0, jnp.int32),
        )

    def step(self, x: jax.Array, cache: MLACache) -> tuple[jax.Array, MLACache]:
        """Process a new chunk of input x, updating the cache and returning the output.
        x: [B, L, embed_dim]  cache: MLACache with l_kv: [B, max_len, Hkv*Dh], pos: scalar int32"""
        B, L, _ = x.shape
        max_len = cache.l_kv.shape[1]
        new_pos = cache.pos + L

        # Queries for the new positions (already in the compressed K space via W_UK).
        q_heads = (
            self.w_q_uk(x).reshape(B, L, self.num_q_heads, self.head_dim).swapaxes(1, 2)
        )  # (B, Hq, L, Dh)

        # New latents -> write them into the cache buffer at the current position.
        l_new = self.w_dkv(x)  # (B, L, Hkv*Dh)
        l_kv = jax.lax.dynamic_update_slice(
            cache.l_kv, l_new.astype(cache.l_kv.dtype), (0, cache.pos, 0)
        )

        # --- Shared KV latent (serves as both keys and values) ---
        l_kv_heads = l_kv.reshape(
            B, max_len, self.num_kv_heads, self.head_dim
        ).swapaxes(1, 2)  # (B, Hkv, max_len, Dh)
        l_kv_rep = l_kv_heads.repeat(self.group_size, axis=1)  # (B, Hq, max_len, Dh)

        # Scores: the L new queries attend over all max_len cached slots.
        logits = jnp.einsum("bhqd, bhkd -> bhqk", q_heads, l_kv_rep) / jnp.sqrt(
            self.head_dim
        )

        # Causal mask offset by the cache position: query i sits at absolute position
        # pos+i and may attend to slot j iff j <= pos+i.  This also masks the not-yet-
        # filled slots (j >= pos+L > pos+i), so no separate validity mask is needed.
        q_pos = cache.pos + jnp.arange(L)  # (L,)
        k_pos = jnp.arange(max_len)  # (max_len,)
        mask = k_pos[None, :] <= q_pos[:, None]  # (L, max_len)
        logits = jnp.where(mask[None, None], logits, -jnp.inf)

        # Softmax over the key axis -> per-query attention distribution.
        a = jax.nn.softmax(logits, axis=-1)

        # Weighted sum of value-latents: the same latent serves as both K and V.
        weighted = jnp.einsum("bhqk, bhkd -> bhqd", a, l_kv_rep)  # (B, Hq, L, Dh)
        weighted = weighted.swapaxes(1, 2).reshape(
            B, L, self.num_q_heads * self.head_dim
        )

        output = self.w_uv_o(weighted)  # (B, L, embed_dim)
        return output, MLACache(l_kv, new_pos)
