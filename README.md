# GQA: One Change to the Transformer That Made Inference Practical

**Paper:** Ainslie et al. (2023) — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
**Baseline:** Phuong & Hutter (2022) — *Formal Algorithms for Transformers*

---

## The Narrative in One Sentence

The original transformer has H key-value heads per layer. GQA changes that to G and that single number is responsible for making modern LLM deployment economically viable.

---

## Overview

**Context.** Large language models generate text one token at a time. At every
step, the model loads its weights plus a growing cache of every previous
attention key and value from memory. As models scale to billions of parameters
and thousands of context tokens, this memory bandwidth cost, not computation, 
becomes the dominant bottleneck for inference speed.

**Problem.** The standard transformer (Phuong & Hutter, 2022) uses H separate
key and value heads in every attention layer. Every decoding step must load all
H of them. Multi-query attention (MQA) compressed these to a single shared head,
cutting memory cost dramatically — but at the cost of quality degradation,
training instability, and requiring a model to be trained from scratch.
Practitioners were forced to choose between a fast model or a good one.

**Approach.** This paper introduces grouped-query attention (GQA): instead of
H key-value heads or 1, use G — one per group of query heads. This interpolates
between MHA and MQA, tunable by a single hyperparameter. The paper also
introduces an uptraining recipe: any existing MHA checkpoint can be converted
to GQA using only 5% of original pretraining compute, rather than training
a new model.

**Resolution.** Uptrained GQA-8 (8 groups) matches MHA-XXL quality within 0.1
average points across summarization, translation, and QA benchmarks, while
running at speeds close to MQA — 6.3× faster than MHA-XXL. GQA has since
become a standard component of every major production LLM.

---

## 1. Start Here: The Baseline Transformer

Before GQA, the standard transformer (formalized by Phuong & Hutter) defines multi-head attention as:

```
Algorithm 5 (Phuong & Hutter): MHAttention(X, Z | W, Mask)
─────────────────────────────────────────────────────────────
Hyperparameters: H  (number of heads)

For h ∈ [H]:
    Y_h ← Attention(X, Z | W^h_qkv, Mask)     ← H separate K,V projections

Y   ← [Y_1 ; Y_2 ; ... ; Y_H]
return V̂ = W_o · Y + b_o
```

Each head h has its own `W^h_k` and `W^h_v`. During autoregressive decoding,
every one of those H key-value pairs must be **loaded from memory at every
single generation step.**

That is the bottleneck. Memory bandwidth.

---

## 2. Why Memory Bandwidth Is the Bottleneck

At each decoding step the model needs:

```
Load from memory every step:
  ┌─────────────────────────────────────────┐
  │  All model weights            (fixed)   │
  │  KV cache: H × seq_len × d_h (grows)   │
  └─────────────────────────────────────────┘
```

The KV cache grows with every new token. For a model with 64 heads
generating a 4096-token sequence across 80 layers:

```
MHA KV cache  =  2 × 64 × 4096 × 128 × 80 layers × 2 bytes
              ≈  107 GB
```

The GPU must reload this from memory at every step. That is the wall.

---

## 3. The One Change GQA Makes

GQA replaces the H-head K/V structure with G groups. Queries remain H heads.
Keys and values collapse to G heads — one per group.

```
┌──────────────────┬────────────────────┬──────────────────────────┐
│   Multi-Head     │   Multi-Query      │   Grouped-Query (GQA)    │
│   (baseline)     │   (prior work)     │   (this paper)           │
├──────────────────┼────────────────────┼──────────────────────────┤
│  Q: H heads      │  Q: H heads        │  Q: H heads              │
│  K: H heads      │  K: 1 head         │  K: G heads              │
│  V: H heads      │  V: 1 head         │  V: G heads              │
├──────────────────┼────────────────────┼──────────────────────────┤
│  Quality: ████   │  Quality: ██       │  Quality: ████           │
│  Speed:   ██     │  Speed:   ████     │  Speed:   ████           │
│  Stable:  yes    │  Stable:  no       │  Stable:  yes            │
└──────────────────┴────────────────────┴──────────────────────────┘
  G = H                G = 1                  1 < G < H
                                          (paper uses G = 8)
```

GQA-1 = MQA. GQA-H = MHA. 
---

## 4. The Diff: What Changed in the Algorithm

Below is the **only part of the transformer that GQA modifies**, shown as a
diff against Phuong & Hutter Algorithm 5.

```diff
  Algorithm 5 → GQA-MHAttention(X, Z | W, Mask)
  ────────────────────────────────────────────────────────────────
  Hyperparameters: H  (query heads, unchanged)
+ Hyperparameters: G  (key/value groups, NEW — was implicitly G=H)

  For h ∈ [H]:
-     Y_h ← Attention(X, Z | W^h_qkv, Mask)
+     g   ← ceil(h × G / H)                 ← assign head h to group g
+     Y_h ← Attention(X, Z | W^h_q, W^g_k, W^g_v, Mask)
+                              ──────  ──────────────────
+                              own Q   shared K,V for group g

  Y   ← [Y_1 ; Y_2 ; ... ; Y_H]
  return V̂ = W_o · Y + b_o
```

**Everything else in the transformer is identical.** The concatenation and
output projection are unchanged. All encoder layers, feed-forward layers,
embeddings, and training algorithms (Algs 11–14 in Phuong & Hutter) are
untouched. GQA is a surgical modification to one parameter set inside one
sub-algorithm.

<img width="719" height="364" alt="image" src="https://github.com/user-attachments/assets/952bcace-beca-4d2c-93a8-e06b909061ae" />

---

## 5. What Stayed the Same (Most of It)

| Component (Phuong & Hutter ref) | Changed by GQA? |
|---|---|
| Token embedding — Alg 1 | No |
| Positional embedding — Alg 2 | No |
| Attention score computation — Alg 3/4 | No |
| Query projections W^h_q | No |
| Output projection W_o | No |
| Layer norm — Alg 6 | No |
| Feed-forward MLP sublayer | No |
| Unembedding — Alg 7 | No |
| Training loop — Alg 13 (DTraining) | No |
| Inference loop — Alg 14 (DInference) | No |
| **Key projections W^h_k** | **Yes — G shared heads instead of H** |
| **Value projections W^h_v** | **Yes — G shared heads instead of H** |

Two parameter matrices. One new hyperparameter G. That is the entire change.

---

## 6. The KV Cache Impact

The reduction is direct and proportional to G:

```
KV cache  =  2 × G × seq_len × d_h × layers × bytes

  G = H = 64  (MHA):   ~107 GB
  G = 8  (GQA-8):       ~13 GB   ← 8× smaller
  G = 1  (MQA):         ~1.7 GB  ← 64× smaller, but quality and stability suffer
```

GQA-8 achieves an 8× cache reduction while keeping quality nearly
indistinguishable from full MHA. That is the core claim.

---

## 7. Uptraining: Getting There Without Retraining

The second contribution is practical: you do not need to train a new model.
Any existing MHA checkpoint can be converted in two steps.

### Step 1 — Checkpoint Conversion (mean pooling)

```
For each group g ∈ [G]:
    heads in group  =  { h : ceil(h × G / H) = g }

    W^g_k  ←  mean( W^h_k  for h in group g )
    W^g_v  ←  mean( W^h_v  for h in group g )

All other parameters unchanged.
```

The paper tests three conversion strategies:

```
  Method          Avg performance    Why
  ──────────────  ───────────────    ─────────────────────────────────────
  Mean pool       55.4  (best)       Preserves information from all heads
  First head      55.1               Discards all but one head's knowledge
  Random init     54.6  (worst)      Discards all pretrained knowledge
```

### Step 2 — Short Additional Pretraining

The converted checkpoint runs through DTraining (Phuong & Hutter Alg 13)
for α = 0.05 of original training steps — identical algorithm, just fewer steps:

```
α = 0.05  →  ~600 TPUv3 chip-days  ≈  5% of original T5-XXL training cost

Performance vs uptraining proportion α:

  α = 0.00  GQA works reasonably (MQA does not — needs uptraining to be useful)
  α = 0.05  Both GQA and MQA recover most quality  ← recommended
  α = 0.10  Diminishing returns
```

The fact that GQA works at α=0 while MQA does not is itself informative:
GQA's intermediate structure preserves more of the pretrained model's
representational quality. This also explains GQA's superior training stability.

---

## 8. Results

Evaluated on T5-XXL across summarization, translation (WMT), and QA (TriviaQA):

```
Model           Inference time    Avg score
──────────────  ──────────────    ─────────
MHA-Large            0.37s          46.0
MHA-XXL              1.51s          47.2   ← gold standard
MQA-XXL              0.24s          46.6
GQA-8-XXL            0.28s          47.1   ← 6.3× faster than MHA-XXL, -0.1 pts

CNN/DM  arXiv  PubMed  MediaSum  MultiNews  WMT    TriviaQA
 43.5   45.4   47.7     36.3      47.2      28.4     81.6    ← GQA-8-XXL
 43.8   45.6   47.5     36.4      46.9      28.4     81.9    ← MHA-XXL
```

GQA-8 is 0.04 seconds slower than MQA but 0.5 points better on average.

### Groups vs. inference time

```
G =   1    4    8    16   32   64
      │    │    │    │    │    │
      ████ ████ ███▌ ███  ██▌  ─  ← relative speed (longer = faster)

1→8:   modest overhead
8→64:  increasing cost, diminishing quality benefit
```

---

## 9. Critical Analysis

**Evaluation metric weakness.** ROUGE measures n-gram overlap, not semantic
quality. The differences between GQA and MHA may not reflect actual quality
gaps as perceived by humans or stronger evaluation frameworks like MT-Bench.

**No from-scratch baseline.** Uptrained GQA-8 is never compared against
GQA-8 trained natively from scratch. The paper cannot claim uptraining
reaches the quality ceiling of GQA — only that it gets close to MHA-XXL.

**Encoder-decoder only.** All experiments use T5. The paper itself speculates
decoder-only architectures benefit more from GQA (since GQA applies to a
larger share of computation without a separate encoder), but does not test
this. The field has since validated this — every major decoder-only model
now uses GQA.

**MQA instability is unexplained.** Appendix A documents loss spikes and
fine-tuning divergence in MQA models. GQA avoids these. The root cause is
never investigated, which limits theoretical understanding.

**Resource framing.** 600 TPUv3 chip-days is presented as "cheap" (5% of
original training). This is accurate relative to Google-scale pretraining,
but inaccessible to most academic labs.

---

## 10. Impact

GQA was released May 2023. It is now a default component of modern LLMs:

```
Model             Date      GQA?
────────────────  ────────  ─────────────────────────
Llama 2 (70B)     Jul 2023  Yes
Mistral 7B        Sep 2023  Yes
Llama 3           Apr 2024  Yes (all sizes)
Gemma             Feb 2024  Yes
```

The key to rapid adoption: GQA is a *recipe*, not a new architecture.
Practitioners could apply it to existing checkpoints at 5% of training cost.
That practicality is what drove immediate and universal adoption.

---

## Resource Links

1. **GQA Paper** — https://arxiv.org/abs/2305.13245
2. **Formal Algorithms for Transformers** — https://arxiv.org/abs/2207.09238
3. **Original MQA Paper (Shazeer 2019)** — https://arxiv.org/abs/1911.02150
4. **Llama 2 (GQA in production)** — https://arxiv.org/abs/2307.09288
5. **Jay Alammar — The Illustrated Transformer** — https://jalammar.github.io/illustrated-transformer/

---

## Citation

```bibtex
@article{ainslie2023gqa,
  title   = {GQA: Training Generalized Multi-Query Transformer Models
             from Multi-Head Checkpoints},
  author  = {Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and
             Zemlyanskiy, Yury and Lebr{\'o}n, Federico and Sanghai, Sumit},
  journal = {arXiv:2305.13245},
  year    = {2023}
}

@techreport{phuong2022formal,
  title       = {Formal Algorithms for Transformers},
  author      = {Phuong, Mary and Hutter, Marcus},
  year        = {2022},
  institution = {DeepMind}
}
```
