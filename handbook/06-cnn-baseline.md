# 6 · The CNN baseline

[← The data pipeline](05-data-pipeline.md) · [Index](README.md) · [Next: The pretrained models →](07-pretrained-models.md)

**Files:** `textcnn.py`, `vocab.py`, `train_cnn.py`

## What a convolution is, without the maths

A **convolution** slides a small window across the input and looks for one
specific pattern at every position.

In images this is intuitive: a filter that detects a vertical edge is dragged
across the picture, and wherever there is a vertical edge, it lights up. The
filter does not care *where* the edge is — same pattern, same response.

Text works the same way, with one axis instead of two. Slide a window of 3 words
along a sentence:

```
"the supplier shall maintain comprehensive general liability insurance"
 └── 1 ──┘
     └── 2 ──┘
         └── 3 ──┘
             └── 4 ──┘  ← "maintain comprehensive general" fires hard
```

A filter is just an n-gram detector that learned its own n-gram.

## Why this architecture fits the task

The match is genuinely good, not a default.

**A clause is announced by a phrase.** "In no event shall", "shall maintain
insurance", "the licensor hereby grants". These are 3–5 word fixed expressions,
which is exactly the span a kernel of width 3, 4 or 5 covers.

**Position does not matter.** An insurance clause is an insurance clause whether
it is at character 2,000 or 42,000. This is handled by **max-over-time
pooling**: after sliding a filter across the whole window, keep only its single
strongest response.

> Read that as: *"did this pattern appear anywhere in this window, and how
> strongly?"* — discarding where it appeared. That is precisely the question our
> task asks. A test (`test_cnn_pooling_is_position_invariant`) asserts the model
> gives the same answer for a phrase at the start and at the end.

**It is small and fast.** ~4 million parameters, trains in minutes on a CPU.
That makes it a practical first check that the data pipeline carries any signal
at all.

## The architecture

```
input: 400 word IDs
   │
   ├─ Embedding (30,968 words × 128 dims)      ← learned from scratch
   │
   ├─ Conv1d k=3 ─┐
   ├─ Conv1d k=4 ─┼─ 128 filters each, ReLU
   ├─ Conv1d k=5 ─┘
   │
   ├─ Max-over-time pooling  → 384 numbers (128 × 3)
   ├─ Dropout 0.5
   ├─ Linear → 6
   └─ Sigmoid → six independent probabilities
```

Three kernel widths run in **parallel**, not in sequence, so the model detects
3-word, 4-word and 5-word patterns simultaneously. This is Kim (2014), the
standard TextCNN.

**Dropout** randomly zeroes half the features during training. It stops the
model leaning on any single filter and is the main defence against memorising
358 contracts.

## The vocabulary

The CNN has no pretrained vocabulary, so we build one: every word appearing at
least twice in the training split. **30,968 words.**

> Built from the **training split only**. Building it over the whole corpus
> would let test-set vocabulary influence training — a subtle leak, and an easy
> one to commit by accident.

Unknown words at test time map to a shared `<unk>` token. Our tokenizer keeps
alphanumerics including section numbers, because "10.2" is meaningful in a
contract.

## Why have a baseline at all

This is the part people skip, and it is the most important.

Without a floor, a score is uninterpretable. Is 0.663 good? Compared to what?

The CNN answers that. It is **4 million parameters learned from 358 contracts**.
If BERT — 110 million parameters pretrained on 3.3 billion words — cannot
clearly beat it, then the pretraining is not earning its cost on this task, and
that is a finding worth reporting rather than hiding.

It also gave us the first proof the rebuilt data pipeline works. The old
pipeline produced no labels, so *no* architecture could have scored above zero.
The CNN reaching 0.663 is the label fix, measured.

## Results

Test split: 3,314 windows from 76 unseen contracts.

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Insurance | 0.877 | 0.814 | **0.844** | 70 |
| License Grant | 0.806 | 0.767 | **0.786** | 146 |
| Cap on Liability | 0.760 | 0.731 | **0.745** | 104 |
| Audit Rights | 0.651 | 0.793 | **0.715** | 87 |
| Termination for Convenience | 0.491 | 0.540 | **0.514** | 50 |
| Non-Compete | 0.400 | 0.351 | **0.374** | 57 |

**Macro F1 0.663** at window level, **0.840** at document level.

The pattern is hard to miss: **the strongest labels are the most frequent
ones.** Non-Compete is both the rarest (1.5% of windows) and the weakest.
Rarity and difficulty track each other almost exactly, which tells you the
bottleneck is data volume, not architecture.

## A caveat that applies to every number in this handbook

Two runs of the *identical* command, same seed, same settings, gave **0.633 and
0.663**.

The cause is floating-point non-determinism on the GPU backend. So:

> **A difference smaller than about ±0.03 macro F1 is not evidence of
> anything.**

This matters most for [Problem 1](04-the-two-problems.md). If Legal-BERT beats
BERT by 0.02, that is noise, and the honest response is to run several seeds and
report the spread.

---

[← The data pipeline](05-data-pipeline.md) · [Index](README.md) · [Next: The pretrained models →](07-pretrained-models.md)
