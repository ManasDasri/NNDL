# 5 · The data pipeline

[← The two hard problems](04-the-two-problems.md) · [Index](README.md) · [Next: The CNN baseline →](06-cnn-baseline.md)

How a 54,000-character contract becomes training examples a network can learn
from. This is Person A's deliverable, and it is where the project's most
instructive bug lived.

**Files:** `cuad.py` → `chunking.py` → `splits.py`, driven by `prepare.py`.

---

## The bug that made everything meaningless

Worth reading even if you skip everything else, because it is the kind of bug
that does not announce itself.

The first version of this project derived labels like this:

```python
label = normalize_label(qa["question"])
```

It looked up the question text in a table of the six label names. But a CUAD
question reads:

> Highlight the parts (if any) of this contract related to "Cap On Liability"
> that should be reviewed by a lawyer. Details: ...

That string never equals `"Cap on Liability"`. **The lookup matched nothing.
Every single training example was created with an empty label list.**

### Why nobody noticed

This is the instructive part. The pipeline did not crash. It produced 20,000
training examples, all labelled "no clauses here". The model trained happily,
the loss went *down* — because predicting "nothing" on an all-nothing dataset is
easy — and accuracy looked excellent.

A silent failure that produces plausible-looking output is far more dangerous
than a crash.

### The second, deeper bug

Even with the category read correctly, the design was broken. Recall from
[chapter 3](03-the-dataset.md) that a CUAD "paragraph" is the **whole
contract**. So a label derived at paragraph level is a fact about the entire
document. The old code then chunked the contract and copied that
document-level label onto **every chunk**:

```
Contract contains an Insurance clause (somewhere)
  → chunk 1  "Insurance" ✓   ← wrong, it's the title page
  → chunk 2  "Insurance" ✓   ← wrong, it's the payment terms
  → chunk 3  "Insurance" ✓   ← correct, by luck
  ...
  → chunk 60 "Insurance" ✓   ← wrong, it's the signature block
```

You would be teaching the model that signature blocks are insurance clauses.

### The root cause

Both bugs come from the same mistake: **throwing away `answer_start`.**

CUAD labels are *locations*. Discard the offsets and you are left with
"this contract mentions insurance somewhere", which cannot supervise a model
that works on windows. The fix was not a patch; it was rebuilding the data layer
around spans.

Two tests now pin this permanently: one asserts a whole question is not a
category, the other asserts a clause must not label windows that do not contain
it.

---

## Step 1 — Read the spans (`cuad.py`)

For each contract, for each of our six categories, collect every annotated
character range.

```python
{
  "doc_id": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
  "text":   "DISTRIBUTOR AGREEMENT\n\nTHIS AGREEMENT ...",   # all 54,290 chars
  "spans":  {
    "License Grant": ((2112, 2611), (3756, 4080)),
    "Insurance":     ((41736, 42217),),
    "Non-Compete":   (),
  }
}
```

Two details:

**The category comes from the `id` suffix**, split on `__`, not from the
question. That is both simpler and correct.

**Overlapping spans are merged.** Annotators sometimes marked the same clause
twice, or marked adjacent sentences separately. Merging keeps the overlap
arithmetic in step 2 honest.

Result: **510 documents, 3,123 spans.** (The old pipeline: zero.)

## Step 2 — Cut into windows (`chunking.py`)

A contract is too long, so slice it into overlapping windows.

```
contract:  [-------------------------------------------------------]
window 1:  [--------300 words--------]
window 2:            [--------300 words--------]
window 3:                      [--------300 words--------]
                      ^^^^^^^^^
                      100-word overlap
```

Default: **300-word windows, 100-word overlap.** 510 contracts → **20,048
windows**.

### Why overlap at all

Without it, a clause landing on a boundary is cut in half and may be
unrecognisable in both pieces. With a 100-word overlap, any clause shorter than
100 words appears *whole* in at least one window. A test verifies that
overlapping windows recover at least as many positives as non-overlapping ones.

### Why windows are measured in words, not tokens

This is a real decision and worth understanding.

The obvious approach is to chunk by *model tokens* — 512 for BERT. But every
model tokenizes differently, so BERT's chunks and Longformer's chunks would fall
in different places, and the models would be trained on different text.

If we then compared them, we would be measuring **tokenizer differences mixed
with model differences**, and could not separate them.

Words are tokenizer-agnostic. Chunking by words means all four models see
*identical* window boundaries, and each applies its own tokenizer to that same
text. A score difference is then a difference between models. That is what makes
[Problem 1](04-the-two-problems.md) answerable.

## The overlap rule

Now the question that makes or breaks supervision: **which windows get which
labels?**

A window takes a label when it overlaps an annotated span by at least **half of
whichever range is shorter**.

The "shorter range" part handles both directions cleanly:

| Situation | Overlap | Positive? |
|---|---|---|
| 100-char clause fully inside a 1,800-char window | 100/100 = 1.0 | yes |
| 1,800-char window fully inside a 2,000-char clause | 1,800/1,800 = 1.0 | yes |
| Window catches 50 chars of a clause at its edge | 50/1,800 = 0.03 | no |

That last row is the point: a few trailing characters of a clause should not
make a window positive.

**This is a judgement call, not a fact.** The 0.5 ratio is exposed as
`min_overlap_ratio` so it can be changed and the effect measured. We are
flagging it as a modelling decision rather than hiding it as a constant.

The result is genuine window-level supervision:

```
Non-Compete:  301 of 20,048 windows positive (1.50%)
```

Compare that to the old behaviour, which would have marked every window of all
119 Non-Compete contracts.

## Step 3 — Split by document (`splits.py`)

358 train / 76 validation / 76 test.

### Why per document and never per window

This is the leakage trap.

Our windows **overlap by 100 words**. Split randomly by window and window 5 goes
to train while window 6 — which shares 100 words with it — goes to test. You
would be testing on text the model already memorised, and your score would be
inflated nonsense.

Splitting by whole contract means every window of a contract lands on the same
side. No shared text across the boundary.

The assignment is **written to disk** (`data/processed/splits.json`) so every
model in the comparison is scored on exactly the same 76 test contracts. If
splits were recomputed per run, model differences and split differences would be
indistinguishable.

The generated report checks the splits are representative — no label is
concentrated in one side.

## Why chunking is not done in `prepare`

`prepare.py` writes documents and spans, **not chunks**.

BERT wants 300-word windows; Longformer wants 2,600. Persisting chunks would
mean maintaining a separate dataset file per model, and they would drift.
Chunking happens at load time, from one source of truth. The prepare step exists
only because parsing the 100MB CUAD JSON is slow.

## The whole pipeline

```
data/CUAD_v1.json
      │  legal-risk-prepare
      ▼
data/processed/documents.jsonl   ← full text + character spans per label
data/processed/splits.json       ← which contract is train/val/test
      │  at training time
      ▼
   chunk_documents(window, overlap)  ← model-specific width
      ▼
   20,048 windows, each with 0-6 labels
```

## Tests

`python tests/test_data_pipeline.py` — 12 checks, no pytest and no torch needed.
Notably: a whole question is not a category; a clause must not label windows that
do not contain it; overlap recovers boundary-straddling clauses; splits are
seeded and no document appears twice.

---

[← The two hard problems](04-the-two-problems.md) · [Index](README.md) · [Next: The CNN baseline →](06-cnn-baseline.md)
