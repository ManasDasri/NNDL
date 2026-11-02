# 12 · Results

[← Deployment](11-deployment.md) · [Index](README.md) · [Next: Run it yourself →](13-run-it-yourself.md)

Status as of the last update. This chapter separates **what has been measured**
from **what has not**, because a handbook that blurs the two is worse than no
handbook.

## What is measured

| Model | Window F1 | Window AP | Document F1 | Document AP | Status |
|---|---:|---:|---:|---:|---|
| TextCNN | 0.663 | — | 0.846 | 0.904 | trained, published |
| **BERT** | **0.723** | 0.736 | **0.879** | 0.933 | trained (notebook 3) |
| **Legal-BERT** | 0.718 | **0.769** | 0.875 | **0.936** | trained (notebook 3), one seed |
| Longformer | — | — | — | — | code complete, not run |

Document scores use noisy-OR pooling, with thresholds tuned on validation.
Notebook 3 also retrained the CNN, which scored 0.656 window F1, within noise of
the published 0.663.

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

## Problem 1 — answered: legal pretraining shows no measurable advantage

BERT and Legal-BERT differ only in what they were pretrained on, so this is the
cleanest comparison in the project.

| Label (window F1) | CNN | BERT | Legal-BERT | Legal − BERT |
|---|---:|---:|---:|---:|
| Termination for Convenience | 0.544 | 0.621 | 0.685 | **+0.064** |
| Non-Compete | 0.317 | 0.560 | 0.509 | −0.051 |
| Cap on Liability | 0.780 | 0.780 | 0.766 | −0.015 |
| Audit Rights | 0.696 | 0.738 | 0.725 | −0.013 |
| License Grant | 0.760 | 0.762 | 0.752 | −0.010 |
| Insurance | 0.841 | 0.876 | 0.871 | −0.004 |
| **Macro F1** | 0.656 | 0.723 | 0.718 | −0.005 |

(CNN column is the notebook 3 retrain, so all three come from the same session.)

**On F1, a tie.** Legal-BERT is 0.005 below BERT. BERT trained a second time
with a different seed scored 0.729, a gap of 0.006 from itself.

**On ranking, a slight edge for Legal-BERT.** Window AP 0.769 against BERT's
0.736 and 0.757 across its two seeds. At document level the two are tied, 0.936
against 0.933.

**The per-label prediction did not hold.** Legal pretraining was expected to help
most on labels carried by legal vocabulary and least on Insurance. Insurance did
gain nothing, but neither did Cap on Liability, the most jargon-heavy label. The
only clear gain is Termination for Convenience, whose AP rose from 0.573 to
0.711.

How much a single run can be trusted per label, from BERT's two seeds:

| Label | Seed 42 | Seed 1 | Swing |
|---|---:|---:|---:|
| Non-Compete | 0.560 | 0.468 | 0.092 |
| License Grant | 0.762 | 0.821 | 0.059 |
| Audit Rights | 0.738 | 0.772 | 0.034 |
| Insurance | 0.876 | 0.904 | 0.028 |
| Termination for Convenience | 0.621 | 0.631 | 0.010 |
| Cap on Liability | 0.780 | 0.780 | 0.000 |

Rare labels swing by up to 0.09 between runs of the *same* model, so Legal-BERT's
−0.051 on Non-Compete is noise. Its +0.064 on Termination for Convenience exceeds
that label's own swing, which makes it the one per-label result worth re-checking.

> **Conclusion.** With 358 training contracts, fine-tuning a general-English
> model closes the gap that legal pretraining would otherwise fill. Legal-BERT
> matches BERT on F1, ranks marginally better, and does not show the predicted
> vocabulary-specific gains.

**Still owed:** Legal-BERT has one seed and BERT two; the seed loop in notebook 3
was cut short. Two or three Legal-BERT seeds would settle whether the AP edge and
the Termination for Convenience gain are real.

## Pretrained models beat the CNN — clearly

BERT is +0.07 window F1 and +0.03 document F1 above the CNN, more than twice the
noise floor. The largest gain is on the rarest label: **Non-Compete, 0.317 →
0.560.**

## Problem 2 — not yet measured

Longformer needs notebook 4. Until it runs we do not know whether a 2,600-word
window beats a 300-word one, or whether noisy-OR gains more at ~4 windows per
contract than at ~38.

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
for Problem 1: Legal-BERT should help *least* on Insurance. It did gain nothing
there (−0.004), but it gained nothing almost everywhere else too, so the
prediction is not confirmed in any meaningful sense.

### The model confuses "liability" with "liability"

From [chapter 10](10-interpretability-and-demo.md): asked why it flagged Cap on
Liability, the model returns the same `product liability insurance` phrases it
uses for Insurance. It is keying on the shared word rather than on capping
language like "in no event shall".

This is a specific, fixable weakness, found by interpretability rather than
guesswork.

### Pooling to document level is worth 0.15–0.18

CNN 0.663 → 0.846, BERT 0.723 → 0.879, Legal-BERT 0.718 → 0.875. The
existential question — *does this contract contain the clause anywhere* — is
genuinely easier than localising every instance, and noisy-OR pooling exploits
that on transformer predictions as well as on the CNN's.

## The caveat that governs every number here

> **Two identical runs produced 0.633 and 0.663.**

Floating-point non-determinism on GPU. So a difference smaller than about
**±0.03 macro F1 is not evidence of anything.**

This was written down **before** the transformer results existed, with the
rule that a Legal-BERT win of 0.02 would be reported as within variance rather
than as a finding. The real gap came out at −0.005, so the rule was not needed
to deflate a claimed win; it is why the result above is called a tie rather than
a loss.

## What is left

1. **Save notebook 3's `outputs/` folder.** These numbers were read from the notebook's printed output; the run directories never left Colab, so the dashboard and the Hub models cannot be updated yet.
2. Run two or three more Legal-BERT seeds.
3. Run notebook 4 (Longformer) for Problem 2.
4. Then regenerate the comparison with `legal-risk-compare`, re-export the dashboard, and push the models with `legal-risk-export-hf`.

---

[← Deployment](11-deployment.md) · [Index](README.md) · [Next: Run it yourself →](13-run-it-yourself.md)
