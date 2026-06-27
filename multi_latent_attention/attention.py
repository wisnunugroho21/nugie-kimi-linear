import jax
import jax.numpy as jnp
from flax import nnx


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
        dropout_rate: float,
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
        self.w_q_uk = nnx.Linear(embed_dim, d_q, use_bias=False, rngs=rngs)
        # W_DKV: x -> low-rank KV latent c_kv (one latent per KV head).
        self.w_dkv = nnx.Linear(embed_dim, d_kv, use_bias=False, rngs=rngs)
        # W_UV . W_O absorbed: value-latent -> up-projected, output-projected.
        self.w_uv_o = nnx.Linear(d_q, embed_dim, use_bias=False, rngs=rngs)

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
