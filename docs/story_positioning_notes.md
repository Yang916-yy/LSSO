# LSSO Story Positioning Notes

This note records the current paper story so future experiments and writing
stay focused. The goal is not to add more tasks broadly, but to identify pain
points of attention-approximation methods and show where LSSO answers a real
need.

## Core Story

Existing efficient-attention methods mainly reduce the cost of pairwise
routing. Linformer, Performer, Nystromformer, Longformer, BigBird, and related
families still organize the mixer around approximating or sparsifying the
attention matrix.

LSSO takes a different route. It does not approximate softmax routing. It
learns a low-rank global relation field `U U^T` and solves a positive-shifted
linear system for hidden states:

```text
(mu I + gamma U U^T) Y = C
LSSO(X) = Y W_o
```

The useful story is therefore:

```text
efficient attention compresses "who attends to whom";
LSSO learns a global relation field and solves for a representation;
rank becomes a controllable and diagnosable global-capacity knob.
```

## Pain Points to Target

### 1. Approximation structure is task-dependent

Approximate attention methods usually assume one useful structure: low-rank,
kernel features, landmarks, windows, or sparse patterns. Real attention
patterns vary by task, layer, and input. A fixed approximation family can save
compute but lose quality when the needed interaction pattern changes.

LSSO response:

```text
Do not approximate the attention matrix. Learn a low-rank relation field and
solve the state directly.
```

Potential evidence:

```text
retrieval length sweep
approximation-budget vs solve-rank Pareto plots
```

### 2. Cheaper routing does not guarantee transferable representations

For encoder retrieval, the objective is not to reproduce attention weights. The
encoder must compress a document or evidence chunk into a representation that
transfers across domains.

LSSO response:

```text
Use global solving as the inductive bias for bidirectional representation
learning.
```

Strong current evidence:

```text
MS MARCO -> BEIR transfer
MHA macro nDCG@10:        0.22945, mixer MACs 2.147G
Nystromformer:            0.20523, mixer MACs 1.270G
LSSO-r16:                 0.22973, mixer MACs 0.714G
LSSO-r32:                 0.22940, mixer MACs 0.910G
```

This is the clearest current separation from attention-approximation baselines.

Best follow-up:

```text
MS MARCO -> BEIR capacity/budget sweep
LSSO-r4/r8/r16/r32
Nystrom landmarks 16/32/64 if time permits
```

### 3. Approximation budget is not directly diagnostic

Landmarks, random features, sparse windows, or projection rank are often
approximation budgets. They do not directly tell us how much global semantic
capacity the model used, whether the global term participates in the output, or
which coordinates can be removed at inference.

LSSO response:

```text
Expose gamma/mu, correction ratio, effective rank, and rank pruning.
```

Current evidence:

```text
operator_diagnostics/
rank_pruning/
retrieval_ablation/
```

Best presentation:

```text
one rank-capacity figure:
rank -> retrieval metric
rank -> effective rank
rank -> correction ratio
keep_rank -> metric after pruning
```

### 4. Long sequence quality-efficiency tradeoff

Approximate attention methods reduce the cost of long-context mixing, but the
quality loss depends on the approximation budget and task. LSSO keeps global
interaction through a fixed rank solve, which gives a clear knob when `r << N`.

LSSO response:

```text
Use solve rank as the global-capacity knob while sequence length grows.
```

Potential evidence:

```text
doc_len 512/1024/2048 retrieval sweep
metric + throughput + mixer MACs vs length
```

## Task Mapping

### NLP tasks that match the story

High priority:

```text
long-document retrieval
RAG chunk retrieval
MS MARCO -> BEIR transfer
QASPER evidence retrieval
SciFact / NFCorpus / FIQA retrieval
```

Good secondary tasks:

```text
long-document classification
legal document classification
scientific document classification
multi-hop evidence retrieval
HotpotQA / FEVER-style paragraph retrieval
```

Avoid as first-claim tasks:

```text
decoder-only generation
copy-heavy QA
span extraction
code completion
causal language modeling
```

These rely more on precise token routing, causal masking, or copying behavior.

### CV tasks that match the story

High priority:

```text
high-token image classification
Food-101
ImageNet-100 / ImageNet-1K resolution sweep
iNaturalist / Places365 / fine-grained classification
```

Good secondary tasks:

```text
remote sensing classification
medical image classification
histopathology patch classification
multi-frame or video classification
```

Avoid as first-claim tasks:

```text
semantic segmentation
object detection
instance segmentation
super-resolution
diffusion generation
low-level texture restoration
```

These may need local detail routing, spatial alignment, or decoder-specific
structure. LSSO may still help in hybrid models, but they are not the clean
first story.

## Experiment Discipline

Do not add tasks just for coverage. A new experiment should answer one of:

```text
1. Does attention-approximation lose transferable encoder quality?
2. Does solve rank act as global representation capacity?
3. Is that capacity diagnosable or compressible?
4. Does the quality-efficiency tradeoff improve as sequence/token count grows?
```

If an experiment does not answer one of these, it is probably a distraction.
