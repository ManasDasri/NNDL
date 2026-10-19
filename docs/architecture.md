# The system, end to end

What exists, what each piece does, and which numbers it produced. This is the
engineer-facing reference; [the handbook](../handbook/README.md) tells the same
story from zero background, and [pooling.md](./pooling.md) and
[model-selection.md](./model-selection.md) go deeper on the two decisions that
needed their own argument.

## The pipeline

```text
CUAD_v1.json
  prepare.py      spans per category + per-document split assignment
  chunking.py     overlapping word windows; a window inherits a label when it
                  overlaps an annotated span by >= 50% of the shorter range
  training.py     one loop, every model: weighted BCE, AMP, early stopping
  evaluate.py     window metrics, then noisy-OR pooling to document metrics
  compare.py / export_dashboard.py / demo.py / export_hf.py
```

Four properties hold across every model, and they are what make the comparisons
mean anything:

| Property | Why |
|---|---|
| Windows measured in **words**, not tokens | Identical boundaries for all four models, so a score gap is a model gap and not a tokenizer gap |
| Splits assigned **per document**, persisted to `splits.json` | Overlapping windows would otherwise straddle train and test |
| Six independent sigmoids + `BCEWithLogitsLoss(pos_weight=…)` | Labels co-occur, so softmax is wrong; positives run 1.5–4.2% so unweighted loss predicts nothing |
| Thresholds tuned per label **on validation only** | Each label has its own positive rate; a single 0.5 cut is arbitrary |

## The models

### TextCNN — the floor

`textcnn.py`. Embedding (128d, learned from 358 contracts) → three parallel
`Conv1d` banks at kernel 3/4/5, 128 filters each → ReLU → max-over-time →
dropout 0.5 → linear to 6 logits. ~4M parameters.

A filter is a literal n-gram detector and max-over-time records *where* it
fired, which is why attribution returns an actual phrase from the contract
rather than a contested attention weight. That is also its job in the project:
a 110M-parameter pretrained encoder that cannot beat convolutions over
embeddings learned from 358 contracts is not earning its cost.

### The three encoders

`pretrained.py` wraps each checkpoint so the training loop calls
`model(**batch)` and gets logits back, exactly as it does for the CNN. One loop,
four models.

| Key | Checkpoint | Max tokens | Window | Overlap | Chunks/contract |
|---|---|---:|---:|---:|---:|
| `bert` | `bert-base-uncased` | 512 | 300 w | 100 w | 37.9 |
| `legal-bert` | `nlpaueb/legal-bert-base-uncased` | 512 | 300 w | 100 w | 37.9 |
| **`longformer`** | `allenai/longformer-base-4096` | 4096 | 2600 w | 400 w | **3.9** |

Window width is sized from a measured ~1.45 subword tokens per word in legal
English. It is the one deliberate difference between models, because it is
bounded by the context window — and that bound is what Problem 2 asks about.

Longformer is the default for `legal-risk-train-transformer`, and the one
correction worth stating plainly: a wider window produces **fewer, larger**
chunks (3.9 per contract against BERT's 37.9), not smaller ones. Every
chunk-count argument in this repo follows from that direction.

`[CLS]` is given global attention under Longformer. Its local window cannot
pool a sequence on its own, so without this the classification head reads a
position that only ever saw its own neighbourhood.

## Training

`training.py`, one loop for all four models. `TrainingConfig` defaults: 8
epochs, lr 1e-3, batch 32, grad-norm clip 1.0, patience 3, seed 42, `amp="auto"`.
The transformer CLI overrides these to 3 epochs, lr 2e-5, batch 8, grad
accumulation 2, weight decay 0.01, patience 2.

Three details that are easy to get wrong:

- **Mixed precision is CUDA-only.** `autocast` + `GradScaler` give 2–3× on a T4
  at no measurable cost; MPS and CPU autocast are unsupported or slower, so
  `auto` disables them.
- **`scaler.unscale_` runs before clipping.** Clipping scaled gradients makes
  the 1.0 threshold meaningless.
- **Tokenization happens once**, batched, at dataset construction — not per
  `__getitem__`.

Early stopping selects on validation macro F1 and the chosen thresholds are
written into `training.json`, which both `train_cnn` and `evaluate` read. They
did not always: an earlier version silently fell back to 0.5 and reported 0.539
against training's 0.591 for identical weights.

## Evaluation

Two levels, because the task has two levels. Windows are what the model
classifies; contracts are what the question is about.

Document scoring pools chunk probabilities with **noisy-OR** by default —
`P = 1 − Π(1 − pᵢ)` — so two chunks at 0.80 and 0.79 combine to 0.958 instead
of collapsing to 0.80. Max-pooling cannot express corroboration at all; the
full comparison of seven rules, the learned aggregator that lost to noisy-OR,
and the auto-selection that overfits are all in [pooling.md](./pooling.md).

Thresholds come from validation and are applied unchanged to test. Correcting a
leak here — document thresholds had been tuned on the split being scored —
moved document F1 from 0.840 to 0.773 while average precision stayed at 0.872,
which is what confirmed the diagnosis rather than a regression.

## Results

Test split: 76 contracts, 3,314 windows, never trained on.

| Model | Window macro F1 | Document macro F1 | Document macro AP |
|---|---:|---:|---:|
| TextCNN | 0.663 | 0.846 | 0.904 |
| BERT | — | — | — |
| Legal-BERT | — | — | — |
| Longformer | — | — | — |

Per-label CNN thresholds, tuned on validation: Cap on Liability 0.78 ·
Non-Compete 0.28 · License Grant 0.32 · Audit Rights 0.39 · Termination for
Convenience 0.22 · Insurance 0.83. The spread is the point — a single cut would
be wrong for five of the six.

**Run-to-run variance is about ±0.03 macro F1** from float non-determinism on
GPU. Nothing smaller than that is a finding.

### Status of the two problems

Both are **unanswered**, and the reason is compute, not code. BERT, Legal-BERT
and Longformer are implemented, tested offline, and unrun — no transformer has
been trained yet. Problem 1 (does legal pretraining help?) needs BERT and
Legal-BERT; Problem 2 (does longer context help?) needs BERT and Longformer.
Until those runs exist the only measured model is the baseline.

Queued behind the first GPU run: `legal-risk-compare`, a regenerated dashboard,
`legal-risk-export-hf` for the new checkpoints, handbook chapter 12, and a
re-run of the learned aggregator — 3.9 chunks per contract is a different
learning problem from 37.9.

## Interpretability and delivery

`attribution.py` maps a filter's receptive field back to the phrase that fired
it. It has already earned its place: asked why it flagged Cap on Liability, the
CNN returns the same `product liability insurance` phrases it uses for
Insurance, so it keys on the shared word "liability" rather than on capping
language.

`demo.py` is the Gradio app, `export_dashboard.py` generates the static results
page, `export_hf.py` pushes a checkpoint with `id2label` intact. The published
model is `KenX049/cuad-clause-risk-cnn`; the dashboard and demo are Hugging Face
Spaces, deliberately not GitHub Pages, so this project never appears on the
personal portfolio domain.

## Tests

90 checks across nine files, no framework — each runs as a script.

```text
test_data_pipeline.py    12   stdlib only
test_analysis.py          9   stdlib only
test_model_layer.py      14   torch
test_pretrained.py       11   torch, offline
test_pooling.py           7   torch
test_pooling_rules.py     9   the seven rules and their failure modes
test_learned_pooling.py  10   the aggregator and its padding contract
test_interpretability.py 11   torch
test_export_hf.py         7   no network
```

They carry the reasoning, so a decision cannot be quietly reversed: that a CUAD
question is not a category, that a clause must not label windows which do not
contain it, that mean pooling buries a true positive, that max cannot see
corroboration, that noisy-OR saturates on many weak chunks, that short-document
padding must read as absence and not as agreement, and that BERT and Legal-BERT
differ only in checkpoint.
