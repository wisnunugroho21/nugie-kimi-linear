# Kimi Linear (Gated DeltaNet-2 variant) — minimal JAX / Flax NNX

A small, heavily-commented reference implementation of the **Kimi Linear**
architecture ("Kimi Linear: An Expressive, Efficient Attention Architecture"),
with one deliberate substitution: the linear-attention token mixer is **Gated
DeltaNet-2** ("Decoupling Erase and Write in Linear Attention", arXiv:2605.22791)
instead of Kimi's own KDA. Built with **JAX**, **Flax NNX**, **Optax**, and **Orbax**.

The goal is readability: every module is annotated against the equations in the
papers, and the fast/parallel paths are checked against simple reference paths.

## The architecture in one paragraph

Kimi Linear is a decoder-only transformer whose **token mixer alternates** between
a cheap O(L) linear-attention layer and an ordinary softmax full-attention layer,
at a fixed **3:1 ratio** (three linear layers per full-attention layer). Linear
layers carry positional information through their recurrence, so the full-attention
layers use **NoPE** (no positional encoding). Every layer's channel mixer (FFN) is a
DeepSeek-V3 / Moonlight-style **MoE**. Here the linear layer is **Gated DeltaNet-2**,
a gated-delta-rule attention whose novelty is a *decoupled* erase gate `b` and write
gate `w` (the original GDN/KDA tie them with a single `beta`).

```
            ┌─────────────────────────── repeat n_layers ──────────────────────────┐
 tokens ─► Embed ─►│ RMSNorm ─► [GDN-2 | MLA] ─►(+) ─► RMSNorm ─► MoE ─►(+) │─► RMSNorm ─► LM head
                   └──────────── token mixer ────────┘     └─ channel mixer ─┘
   token mixer = GDN-2 (linear) on 3 of every 4 layers, MLA (full, NoPE) on the 4th
```

## Files

| File | What it is |
|------|------------|
| `kimi_linear_gdn2.py` | **Top-level model**: config, the pre-norm decoder block, the 3:1 hybrid stacking, embedding + LM head. |
| `gated_deltanet_2/core.py` | GDN-2 **chunkwise-parallel** core + a token-by-token **recurrent reference**, each line mapped to the paper's equations. |
| `gated_deltanet_2/layer.py` | GDN-2 **token mixer** (projections, short convs, L2-norm q/k, decoupled gates, gated RMSNorm output). |
| `multi_latent_attention/attention.py` | **MLA** full-attention in the absorbed, NoPE, GQA form (shared KV latent acts as both K and V). |
| `multi_latent_attention/moe.py` | Token-dispatched **grouped-GEMM MoE** with a shared expert + aux-loss-free load balancing. |
| `train.py` | **Training + checkpoint demo**: Optax (AdamW + clip + warmup-cosine) and Orbax (save/restore). |
| `sanity_check.py` | Correctness checks (parallel paths == reference paths). |

## Run

```bash
python sanity_check.py   # verify the building blocks
python train.py          # train the toy model + round-trip an Orbax checkpoint
```

`train.py` learns a "continue the counting sequence" task and generalizes to
held-out starts (held-out accuracy → 1.0), then reloads its Orbax checkpoint and
checks the logits are bit-identical.

## Faithfulness notes

* **Kept as in the paper:** 3:1 linear/full hybrid, NoPE MLA, MoE FFN, pre-norm
  RMSNorm blocks, GDN-2's chunkwise WY recurrence, channel-wise gating, L2-normed
  q/k, sigmoid output gate.
* **Deliberate simplifications (flagged inline):** tiny default dims; the GDN-2
  layer stores the decay `a` per (head, channel) rather than per head; default NNX
  initializers instead of the paper's Xavier/`2^-2.5` scheme; no incremental-decode
  state threading; group-limited MoE routing omitted.
* **Numerical caveat:** the GDN-2 chunkwise core runs in fp32 and forms `exp(-G)`
  with `G` the cumulative log-decay. Very strong decay within a chunk can overflow
  (`exp(-G)=inf`, `exp(G)=0` → `0*inf=NaN`). Keep `gdn_chunk_size` modest and the
  learning rate moderate, as `train.py` does.
