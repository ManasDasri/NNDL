# 7 · The pretrained models

[← The CNN baseline](06-cnn-baseline.md) · [Index](README.md) · [Next: Training →](08-training.md)

**Files:** `pretrained.py`, `train_transformer.py`, [`docs/model-selection.md`](../docs/model-selection.md)

## Three models, two experiments

| Model | Pretrained on | Context | Parameters |
|---|---|---:|---:|
| `bert-base-uncased` | Wikipedia + books, ~3.3B words | 512 | 109,486,854 |
| `nlpaueb/legal-bert-base-uncased` | 12GB legislation, cases, contracts | 512 | 109,486,854 |
| `allenai/longformer-base-4096` | RoBERTa, continued on long documents | 4,096 | ~149M |

They are **not three attempts at the same thing**. They are arranged as two
controlled comparisons, each changing exactly one variable:

```
BERT ←─── pretraining corpus ───→ Legal-BERT      (Problem 1)
  │
  └────── context length ────────→ Longformer     (Problem 2)
```

Adding a fourth architecture would add a result. It would not add an answer.

## What a transformer does differently

The CNN looks for fixed patterns in small windows. A transformer uses
**attention**: every word looks at every other word and decides which ones are
relevant to interpreting it.

In *"the Supplier shall maintain such insurance throughout the Term"*, the word
"such" is meaningless alone. Attention lets it look back and bind to whatever
insurance was described earlier.

That is the capability a CNN lacks, and the reason transformers usually win on
language tasks. It is also why they cost O(n²) — every word attending to every
other word — which is the whole of [Problem 2](04-the-two-problems.md#problem-2--how-should-a-whole-contract-be-represented).

## Why BERT vs Legal-BERT is a fair fight

The comparison only means something if everything except pretraining is held
constant. We verified this rather than assuming it:

- same architecture (12-layer encoder)
- **identical parameter count: 109,486,854 each**, confirmed by loading both
- same 512-token window and same 300-word chunks
- same training loop, splits, loss weighting, threshold tuning

`test_bert_and_legal_bert_differ_only_in_pretraining` asserts the specs differ
only in checkpoint. The moment something else drifts, Problem 1 stops being
answerable.

## Longformer and the `[CLS]` detail

Longformer replaces all-pairs attention with a **sliding window**: each token
attends only to its neighbours. That is what makes 4,096 tokens affordable.

But it creates a problem. A classifier reads a single summary position —
`[CLS]`, the first token — and with pure local attention, `[CLS]` only ever sees
its own neighbourhood. It would summarise the first few words, not the window.

The fix is **global attention** on `[CLS]`: that one token attends to
everything, and everything attends to it.

```python
global_attention_mask = torch.zeros_like(input_ids)
global_attention_mask[:, 0] = 1     # [CLS] only
```

Without this, the long context is wasted on the one position that matters. A
test spies on the model call and asserts the mask marks `[CLS]` and nothing
else, and that BERT is never passed one.

## Chunk width is the only thing that varies

| Model | Window | Why |
|---|---:|---|
| BERT / Legal-BERT | 300 words | ~435 tokens, fits 512 with room for special tokens |
| Longformer | 2,600 words | ~3,770 tokens, fits 4,096 |

This is a deliberate exception. Everything else is identical across models, but
chunk width **must** vary because it is bounded by the context window — and that
bound is exactly the variable Problem 2 measures. Holding it constant would
leave Longformer's extra capacity unused and make the comparison meaningless.

A test (`test_chunk_widths_fit_their_context_windows`) checks each width still
fits after tokenization, because an overflow would truncate silently and undo
the experiment.

## One interface for four models

The training loop calls `model(**batch)` and gets back logits. Hugging Face
models return a structured object instead, so a thin wrapper unwraps it:

```python
return self.backbone(**inputs).logits
```

That is the whole adapter. It means **one training loop drives the CNN, BERT,
Legal-BERT and Longformer** — which is the difference between a comparison and
four unrelated runs.

## Models we considered and rejected

**RoBERTa, DeBERTa, ELECTRA** — stronger general-domain encoders, but adopting
one changes architecture *and* pretraining together. A win would not tell you
which caused it.

**A second legal model** — measures variance between legal models, which is not
one of our questions.

**A generative LLM prompted to extract clauses** — a different paradigm with no
fine-tuning on our splits, so its score would not be comparable. The assignment
also asks for trained classifiers.

**Hierarchical BERT over chunk embeddings** — the obvious next step *if* chunk
pooling turns out to be the bottleneck. Deferred rather than dismissed: worth
building only once evaluation shows that pooling, not window classification, is
what limits the score.

## Status

The registry, wrapper, training CLI and tests are complete and verified — both
BERT variants load and produce correct-shaped output.

**The fine-tuning runs have not been completed.** They need a GPU;
[chapter 13](13-run-it-yourself.md) has the Colab link. [Chapter 12](12-results.md)
records exactly what is measured and what is not.

---

[← The CNN baseline](06-cnn-baseline.md) · [Index](README.md) · [Next: Training →](08-training.md)
