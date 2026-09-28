# 12 · Results so far

[← Deployment](11-deployment.md) · [Index](README.md) · [Next: Run it yourself →](13-run-it-yourself.md)

Status as of the last update. This chapter separates **what has been measured**
from **what has not**, because a handbook that blurs the two is worse than no
handbook.

## What is measured

| Model | Window macro F1 | Document macro F1 | Status |
|---|---:|---:|---|
| **TextCNN** | **0.663** | **0.840** | trained, evaluated, published |
| BERT | — | — | code complete, not run |
| Legal-BERT | — | — | code complete, not run |
| Longformer | — | — | code complete, not run |

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

## What has not been measured

**The two experiments the project is built around have not been run.** BERT,
Legal-BERT and Longformer need a GPU; [chapter 13](13-run-it-yourself.md) has a
Colab link and the runs take about an hour.

Until then:

- **Problem 1 is unanswered.** We do not know whether legal pretraining helps.
- **Problem 2 is unanswered.** We do not know whether longer context helps.

Everything around them is finished: the registry, the shared training loop, the
evaluation, the comparison generator, and tests asserting the comparison is fair.
What is missing is the compute.

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

That suggests the bottleneck is **data volume**, not architecture — which in
turn predicts that a bigger model may not help much on Non-Compete, because the
problem is 301 positive windows rather than model capacity. That is a testable
prediction and the transformer runs will check it.

### Insurance is the informative exception

Insurance beats its frequency: 2.00% of windows, yet the best F1 at 0.844.

Why? Its vocabulary is unambiguous. "Shall maintain insurance", "additional
insured", "certificate of insurance" — these phrases appear almost nowhere else.
Compare Non-Compete, whose language ("shall not engage", "restricted
activities") overlaps heavily with ordinary contractual obligations.

**Signal clarity matters as much as frequency.** This also sets up a prediction
for Problem 1: Legal-BERT should help *least* on Insurance, because ordinary
English already carries that signal.

### The model confuses "liability" with "liability"

From [chapter 10](10-interpretability-and-demo.md): asked why it flagged Cap on
Liability, the model returns the same `product liability insurance` phrases it
uses for Insurance. It is keying on the shared word rather than on capping
language like "in no event shall".

This is a specific, fixable weakness, found by interpretability rather than
guesswork.

### Pooling to document level is worth 0.11

0.663 at window level, 0.773–0.840 at document level depending on thresholds.
The existential question — *does this contract contain the clause anywhere* — is
genuinely easier than localising every instance, and max-pooling exploits that.

## The caveat that governs every number here

> **Two identical runs produced 0.633 and 0.663.**

Floating-point non-determinism on GPU. So a difference smaller than about
**±0.03 macro F1 is not evidence of anything.**

This matters most for the experiment we have not run. If Legal-BERT beats BERT
by 0.02, the honest response is:

> *"Within run-to-run variance. We ran both across N seeds; the spread was X."*

not a claimed win. We are writing this down **before** there is a number,
deliberately — it is far easier to agree what counts as a finding when nobody is
attached to the outcome yet.

## What to do when the runs finish

1. Paste the comparison into `docs/results.md` — or better, regenerate it with `legal-risk-compare`, which builds it from the run directories so nothing is transcribed.
2. Re-export the dashboard; it picks up the new models automatically.
3. Push the models to the Hub with `legal-risk-export-hf`.
4. Write the Problem 1 and Problem 2 conclusions — **including the possibility that both are null results**, which chapter 4 argues are genuine findings.

---

[← Deployment](11-deployment.md) · [Index](README.md) · [Next: Run it yourself →](13-run-it-yourself.md)
