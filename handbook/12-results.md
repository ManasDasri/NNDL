# 12 · Results

[← Deployment](11-deployment.md) · [Index](README.md) · [Next: Run it yourself →](13-run-it-yourself.md)

Status as of the last update. This chapter separates **what has been measured**
from **what has not**, because a handbook that blurs the two is worse than no
handbook.

## What is measured

Mean ± standard deviation over seeds, on the 76 held-out test contracts.

| Model | Window F1 | Window AP | Document F1 | Document AP | Runs |
|---|---:|---:|---:|---:|---|
| TextCNN | 0.658 ± 0.018 | 0.694 ± 0.013 | 0.834 ± 0.017 | 0.898 ± 0.017 | 3 seeds |
| BERT | 0.697 ± 0.014 | 0.750 ± 0.015 | 0.861 ± 0.017 | 0.917 ± 0.012 | 3 seeds |
| **Legal-BERT** | **0.744 ± 0.005** | **0.805 ± 0.004** | **0.883 ± 0.009** | **0.928 ± 0.005** | 3 seeds |
| Longformer | 0.743† | 0.770† | 0.831 | 0.886 | 2 runs |

Document scores use noisy-OR pooling, with thresholds tuned on validation.
†Longformer's window scores are on 2,600-word windows and are not comparable
with the 300-word rows; compare the document columns.

BERT and Legal-BERT are seeds 42, 1 and 2 from one Kaggle session (notebook 3);
Longformer is one Colab and one Kaggle run of notebook 4, both after the fixes
in [docs/weak-points.md](../docs/weak-points.md); the CNN row is the three-seed
baseline from the same document. Every run's full classification report is in
[docs/classification-reports.md](../docs/classification-reports.md), and the
executed notebooks, with their charts, are in [`results/`](../results).

### CNN, per class

Test split: 3,314 windows from 76 contracts never seen in training.

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Insurance | 0.877 | 0.814 | **0.844** | 70 |
| License Grant | 0.806 | 0.767 | **0.786** | 146 |
| Cap on Liability | 0.760 | 0.731 | **0.745** | 104 |
| Audit Rights | 0.651 | 0.793 | **0.715** | 87 |
| Termination for Convenience | 0.491 | 0.540 | **0.514** | 50 |
| Non-Compete | 0.400 | 0.351 | **0.374** | 57 |

## Problem 1 — answered: legal pretraining helps

BERT and Legal-BERT differ only in what they were pretrained on, so this is the
cleanest comparison in the project.

**Legal-BERT wins on every metric, and by more than the noise.** Window F1
0.744 ± 0.005 against 0.697 ± 0.014 (+0.047); window AP 0.805 ± 0.004 against
0.750 ± 0.015 (+0.055). Within the session, Legal-BERT's worst run (0.739) beats
BERT's best (0.713). It is also the steadier model: its spread across seeds is a
third of BERT's.

**At contract level the gain shrinks.** 0.883 ± 0.009 against 0.861 ± 0.017
(+0.022), and the runs overlap: BERT's best (0.878) beats Legal-BERT's worst
(0.875). Legal pretraining helps most at finding the passage; once windows are
pooled into a yes/no per contract, both models are usually right.

| Label (window F1, mean of 3 seeds) | CNN | BERT | Legal-BERT | Legal − BERT | Legal run beats BERT run |
|---|---:|---:|---:|---:|---:|
| Termination for Convenience | 0.471 | 0.586 | 0.693 | **+0.107** | 9 of 9 pairs |
| Cap on Liability | 0.756 | 0.740 | 0.817 | **+0.077** | 9 of 9 |
| License Grant | 0.753 | 0.741 | 0.797 | **+0.056** | 9 of 9 |
| Non-Compete | 0.364 | 0.547 | 0.580 | +0.033 | 6 of 9 |
| Audit Rights | 0.673 | 0.716 | 0.734 | +0.018 | 7 of 9 |
| Insurance | 0.819 | 0.854 | 0.847 | −0.007 | 4 of 9 |

(CNN column is the seed-42 CNN from the same session. "Pairs" compares each of
Legal-BERT's three runs with each of BERT's three.)

**The per-label prediction holds.** Legal pretraining was expected to help most
on labels carried by legal phrasing and least on Insurance, whose wording is
ordinary English. The three clear gains are Termination for Convenience, Cap on
Liability and License Grant, all won in every pairing; Insurance gained nothing.

> **Conclusion.** Pretraining on legal text helps: +0.047 window F1 and +0.055
> AP over the same architecture pretrained on general English, concentrated in
> the clauses written in legal phrasing, and absent on Insurance. At contract
> level the advantage is smaller (+0.022) and within run-to-run overlap.

### Why the first answer was "no advantage"

An earlier Colab session trained Legal-BERT once (0.718) and BERT twice (0.723,
0.729), and this chapter called it a tie. That Legal-BERT run is the lowest of
the four Legal-BERT runs that exist; the three Kaggle seeds scored 0.739–0.749.
With one seed per model, one low run decided the conclusion.

Pooling all nine runs from both sessions, Legal-BERT still leads on window F1
(0.738 mean over 4 runs against BERT's 0.709 over 5), by less than within one
session, and the runs overlap. **On window AP they do not overlap at all:**
every Legal-BERT run (0.769–0.808) ranks clauses better than every BERT run
(0.733–0.760). Per label, only Termination for Convenience gained in both
sessions (+0.064 in Colab, +0.107 on Kaggle); Cap on Liability gained on Kaggle
but not in Colab. The Kaggle session is the controlled comparison, since all six
runs share one environment, so it is the one reported above.

## Pretrained models beat the CNN — clearly

Legal-BERT is +0.086 window F1 and +0.049 document F1 above the CNN; BERT +0.039
and +0.027. The largest gain is on the rarest label: **Non-Compete, 0.364 →
0.547 (BERT) and 0.580 (Legal-BERT).**

## Problem 2 — answered: longer context did not help

BERT reads 300-word windows, Longformer 2,600-word ones. Only the contract-level
scores compare fairly: Longformer's test set is 326 windows against BERT's 3,314,
and a window nine times larger is far likelier to contain a clause, so its window
F1 measures a different task.

| Model | Window | Windows per contract | Document F1 | Document AP |
|---|---:|---:|---:|---:|
| TextCNN | 300 words | ~38 | 0.834 ± 0.017 | 0.898 ± 0.017 |
| BERT | 300 words | ~38 | 0.861 ± 0.017 | 0.917 ± 0.012 |
| **Legal-BERT** | 300 words | ~38 | **0.883 ± 0.009** | **0.928 ± 0.005** |
| Longformer | 2,600 words | ~4 | 0.831 (0.834, 0.828) | 0.886 (0.892, 0.880) |

Both Longformer runs score below every BERT run (lowest 0.844) and every
Legal-BERT run (lowest 0.875), and level with the CNN. Two runs agree to within
0.006, so the result is not one unlucky seed.

### Three runs, and what each one measured

1. **Three epochs, before the fixes: 0.759.** A 2,600-word window yields 1,380
   training windows against BERT's 13,829, so equal epochs meant 259 optimizer
   updates against about 2,600, and validation F1 was still rising 0.1 per
   epoch. That run measured undertraining, not context length.
2. **Ten epochs, before the fixes: 0.821 / 0.868.** Its tokenizer spent tokens on
   layout whitespace, which cut off the tail of 20% of its windows, and the last
   partial batch of each epoch was not trained.
3. **Ten epochs, after the fixes, run twice: 0.834 / 0.892 (Colab) and
   0.828 / 0.880 (Kaggle).** The fixes were worth about +0.01 F1 and +0.02 AP.

Both fixed runs stopped early at epoch 9 with their best epoch at 7, so
Longformer has now levelled off; the earlier caveat that it might still be
climbing no longer applies.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Val F1, Colab | 0.301 | 0.403 | 0.589 | 0.535 | 0.662 | 0.649 | **0.700** | 0.685 | 0.661 |
| Val F1, Kaggle | 0.102 | 0.366 | 0.448 | 0.549 | 0.563 | 0.649 | **0.692** | 0.680 | 0.682 |

### Where it won and lost, per label

Contract F1 per label (Colab run): License Grant 0.989, Audit Rights 0.943,
Insurance 0.935, Cap on Liability 0.933, Termination for Convenience 0.618,
Non-Compete 0.586. The two weak labels have high recall and low precision (0.54
and 0.46): Longformer flags many contracts that do not contain them. Both are
clauses that are typically one or two sentences, which a 2,600-word window
dilutes.

### What it cost

149M parameters against 110M, nine epochs against three, batch size 1 with
gradient checkpointing to fit a T4. More compute for a lower score.

> **Conclusion.** On this task, with 358 training contracts, a 2,600-word window
> did worse than a 300-word one. These clauses are locally signalled, so the
> extra context adds little, while larger windows leave ten times fewer training
> examples, each with a vaguer label. Chapter 4 said a null result here would be
> a finding about the task; this is that finding. A fairer test of long context
> would need a legal-pretrained long-context model and a larger GPU.

### Noisy-OR at four windows per contract

Pooling lifted Longformer by about 0.09 (0.749 → 0.834 and 0.736 → 0.828),
against 0.164 for BERT at ~38 windows. The prediction that noisy-OR would matter
*more* with fewer windows is **not confirmed**: with only ~4 windows there is
less to combine.

## What the CNN results already tell us

### Rarity predicts difficulty, almost exactly

Rank the labels by frequency and by F1 and you get nearly the same order:

| Label | Window rate | F1 |
|---|---:|---:|
| License Grant | 4.24% | 0.786 |
| Cap on Liability | 3.57% | 0.745 |
| Audit Rights | 2.95% | 0.715 |
| Insurance | 2.00% | 0.844 |
| Termination for Convenience | 1.51% | 0.514 |
| Non-Compete | 1.50% | 0.374 |

That suggested the bottleneck was **data volume**, not architecture, and
predicted that a bigger model would not help much on Non-Compete.

**The transformer runs refuted that.** BERT lifted Non-Compete from 0.317 to
0.560, the largest gain on any label. The bottleneck was not the 301 positive
windows themselves but how much a model had to learn from them: a pretrained
model arrives already knowing English, so it needs fewer examples to recognise
the clause. Rarity still predicts difficulty, since Non-Compete is BERT's weakest
label too, but the gap narrowed.

### Insurance is the informative exception

Insurance beats its frequency: 2.00% of windows, yet the best F1 at 0.844.

Why? Its vocabulary is unambiguous. "Shall maintain insurance", "additional
insured", "certificate of insurance" — these phrases appear almost nowhere else.
Compare Non-Compete, whose language ("shall not engage", "restricted
activities") overlaps heavily with ordinary contractual obligations.

**Signal clarity matters as much as frequency.** It also set up a prediction
for Problem 1: Legal-BERT should help *least* on Insurance. Over three seeds it
gained nothing there (−0.007) while gaining 0.06–0.11 on the labels written in
legal phrasing, so the prediction holds.

### The model confuses "liability" with "liability"

From [chapter 10](10-interpretability-and-demo.md): asked why it flagged Cap on
Liability, the model returns the same `product liability insurance` phrases it
uses for Insurance. It is keying on the shared word rather than on capping
language like "in no event shall".

This is a specific, fixable weakness, found by interpretability rather than
guesswork.

### Pooling to document level is worth 0.15–0.18

CNN 0.658 → 0.834, BERT 0.697 → 0.861, Legal-BERT 0.744 → 0.883 (three-seed
means). The
existential question — *does this contract contain the clause anywhere* — is
genuinely easier than localising every instance, and noisy-OR pooling exploits
that on transformer predictions as well as on the CNN's.

## The caveat that governs every number here

> **Two identical runs produced 0.633 and 0.663.**

Floating-point non-determinism on GPU. So a difference smaller than about
**±0.03 macro F1 is not evidence of anything.**

This was written down **before** the transformer results existed, with the
rule that a single-run difference of 0.02 would be reported as within variance
rather than as a finding, and settled with more seeds. That is what happened.
The first single runs put Legal-BERT 0.005 *below* BERT, which was reported as a
tie; three seeds each put it 0.047 above, with the runs not overlapping.

## What is left

1. **Download the run folders from Kaggle** (each notebook's Output tab). The numbers here were read from the printed output, so the Hub has the CNN only. Then regenerate the comparison with `legal-risk-compare` and push the models with `legal-risk-export-hf`.
2. Optional: a fair long-context test, with a legal-pretrained long-context model on a GPU large enough for real batches.

---

[← Deployment](11-deployment.md) · [Index](README.md) · [Next: Run it yourself →](13-run-it-yourself.md)
