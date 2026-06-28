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

## Two forward modes: training and streaming inference

Every token mixer supports a parallel full-sequence path *and* a stateful streaming
path, so the model can score a whole sequence at once or decode token-by-token while
**reusing state across steps**:

| | Full sequence — `model(ids)` | Streaming — `model.step` / `model.generate` |
|---|---|---|
| GDN-2 (linear) | chunkwise-parallel core | recurrent core, carrying a **fixed-size** state `S` + short-conv cache |
| MLA (full attn) | full causal-attention matrix | cached KV **latent** (grows with context) |

```python
cont = model.generate(prompt_ids, max_new_tokens=32)   # greedy decode, reuses per-layer state
```

The headline property: a GDN-2 layer's entire history collapses into a fixed-size
matrix `S` (`[B, Hv, dk, dv]`) — so decoding is **O(1) per token, independent of
context length**. Only the 1-in-4 MLA layers keep a growing cache. `sanity_check.py`
verifies that streaming token-by-token reproduces the full-sequence forward exactly
(identical argmax predictions).

## Files

| File | What it is |
|------|------------|
| `kimi_linear_gdn2.py` | **Top-level model**: config, the pre-norm decoder block, the 3:1 hybrid stacking, embedding + LM head. |
| `gated_deltanet_2/core.py` | GDN-2 **chunkwise-parallel** core + a token-by-token **recurrent reference**, each line mapped to the paper's equations. |
| `gated_deltanet_2/layer.py` | GDN-2 **token mixer** (projections, short convs, L2-norm q/k, decoupled gates, gated RMSNorm output). |
| `multi_latent_attention/attention.py` | **MLA** full-attention in the absorbed, NoPE, GQA form (shared KV latent acts as both K and V). |
| `multi_latent_attention/moe.py` | Token-dispatched **grouped-GEMM MoE** with a shared expert + aux-loss-free load balancing. |
| `train.py` | **Training + checkpoint demo**: Optax (AdamW + clip + warmup-cosine) and Orbax (save/restore). |
| `train_chat_gutenberg.py` | **Real-data training + chat**: Project Gutenberg via HuggingFace `datasets`/`transformers`, batched with `grain`; trains with per-step eval, then an interactive chat REPL. |
| `sanity_check.py` | Correctness checks (parallel paths == reference paths). |

## Run

```bash
python sanity_check.py          # verify the building blocks
python train.py                 # toy task + round-trip an Orbax checkpoint

pip install grain datasets transformers   # extra deps for the next one
python train_chat_gutenberg.py  # train on Project Gutenberg, then chat
```

`train.py` learns a "continue the counting sequence" task and generalizes to
held-out starts (held-out accuracy → 1.0), then reloads its Orbax checkpoint and
checks the logits are bit-identical.

`train_chat_gutenberg.py` streams Project Gutenberg through a Grain + HuggingFace
pipeline (GPT-2 tokenizer), trains the model while printing **train vs. held-out loss
every step**, then drops into an infinite **chat** loop that continues your prompt via
the streaming `generate`. It's a tiny model on CPU, so loss drops but the prose stays
rough; press Ctrl-C during training to jump straight to chat (`/exit` to quit).

## Faithfulness notes

* **Kept as in the paper:** 3:1 linear/full hybrid, NoPE MLA, MoE FFN, pre-norm
  RMSNorm blocks, GDN-2's chunkwise WY recurrence, channel-wise gating, L2-normed
  q/k, sigmoid output gate, Xavier-uniform init with gain `2^{-2.5}` and zero biases
  (App. D.5).
* **Deliberate simplifications (flagged inline):** tiny default dims; the GDN-2
  layer stores the decay `a` per (head, channel) rather than per head; the decay bias
  δ starts at −4 (not the paper's value) for fp32 safety; group-limited MoE routing
  omitted; streaming prefill uses the recurrent core (chunkwise prefill would be faster
  for long prompts).
* **Numerical caveat:** the GDN-2 chunkwise core runs in fp32 and forms `exp(-G)`
  with `G` the cumulative log-decay. Very strong decay within a chunk can overflow
  (`exp(-G)=inf`, `exp(G)=0` → `0*inf=NaN`). Keep `gdn_chunk_size` modest and the
  learning rate moderate, as `train.py` does.
