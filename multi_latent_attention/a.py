import jax
import jax.numpy as jnp
from flax import nnx


class GroupedQueryLatentAttentionV2(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_rate: float,
        seq_length: int,
        rngs: nnx.Rngs,
        rope_base: float = 10000.0,
    ):
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE.")

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_q_heads // num_kv_heads

        # Projections
        d_q = num_q_heads * head_dim
        d_kv = num_kv_heads * head_dim

        self.w_q_uk = nnx.Linear(d_model, d_q, use_bias=False, rngs=rngs)
        self.w_dkv = nnx.Linear(d_model, d_kv, use_bias=False, rngs=rngs)
        self.w_uv_o = nnx.Linear(d_q, d_model, use_bias=False, rngs=rngs)

        self.causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # --- RoPE PRECOMPUTATION ---
        half_dim = head_dim // 2
        inv_freq = 1.0 / (rope_base ** (jnp.arange(0, half_dim) / half_dim))
        positions = jnp.arange(seq_length)

        # Outer product to get angles: (seq_length, half_dim)
        angles = jnp.outer(positions, inv_freq)

        # Duplicate angles for both halves of the embedding: (seq_length, head_dim)
        angles = jnp.concatenate([angles, angles], axis=-1)

        # Add batch and head dimensions for easy broadcasting: (1, 1, seq_length, head_dim)
        self.freqs_cos = jnp.cos(angles)[None, None, :, :]
        self.freqs_sin = jnp.sin(angles)[None, None, :, :]

    def apply_rope(self, x: jax.Array) -> jax.Array:
        # Splits the tensor in half along the head_dim
        x1, x2 = jnp.split(x, 2, axis=-1)
        # Rotate half the dimensions: [-x2, x1]
        rotated_half_x = jnp.concatenate([-x2, x1], axis=-1)
        # Apply Euler's formula representation
        return (x * self.freqs_cos) + (rotated_half_x * self.freqs_sin)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        batch_size, seq_length, _ = x.shape

        # 1. QUERIES (N Heads)
        q_latent = self.w_q_uk(x)
        q_reshaped = q_latent.reshape(
            batch_size, seq_length, self.num_q_heads, self.head_dim
        )
        q_heads = q_reshaped.swapaxes(1, 2)

        # 2. KV LATENTS (M Heads)
        l_kv = self.w_dkv(x)
        l_kv_reshaped = l_kv.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        )
        l_kv_heads = l_kv_reshaped.swapaxes(1, 2)

        # 3. APPLY RoPE
        # Apply rotation to Queries
        q_heads = self.apply_rope(q_heads)

        # Apply rotation to KV Latents to act as Keys.
        # Values (v_heads) remain unrotated standard practice.
        k_heads = self.apply_rope(l_kv_heads)
        v_heads = l_kv_heads

        # 4. EXPLICITLY REPEAT KV HEADS
        # Repeat the M heads along the num_heads axis (axis=1) by the group_size
        k_repeated = k_heads.repeat(self.group_size, axis=1)
        v_repeated = v_heads.repeat(self.group_size, axis=1)

        # 5. ATTENTION SCORES
        # Q: (batch, num_q_heads, seq_len, head_dim)
        # K transposed: (batch, num_q_heads, head_dim, seq_len)
        qk_t = q_heads @ k_repeated.swapaxes(-1, -2)

        # 6. MASK
        scaled_logits = qk_t / jnp.sqrt(self.head_dim)
        scaled_logits = jnp.where(
            self.causal_mask[None, None, :seq_length, :seq_length],
            scaled_logits,
            -jnp.inf,
        )

        # 7. SOFTMAX
        a = jax.nn.softmax(scaled_logits, axis=-1)
        dropped_a = self.dropout(a, deterministic=deterministic)

        # 8. WEIGHTED LATENTS
        # A: (batch, num_q_heads, seq_len, seq_len)
        # V: (batch, num_q_heads, seq_len, head_dim) (Unrotated)
        weighted_heads = dropped_a @ v_repeated

        # 9. RECOMBINE HEADS
        weighted_reshaped = weighted_heads.swapaxes(1, 2)
        weighted_latents = weighted_reshaped.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )

        # 10. OUTPUT PROJECTION
        output = self.w_uv_o(weighted_latents)

        return output
