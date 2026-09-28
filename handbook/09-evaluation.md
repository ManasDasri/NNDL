# 9 · Evaluation

[← Training](08-training.md) · [Index](README.md) · [Next: Interpretability and the demo →](10-interpretability-and-demo.md)

**Files:** `metrics.py`, `pooling.py`, `evaluate.py`, `compare.py`

Measuring this model correctly is harder than training it. This chapter contains
two bugs we found in our own evaluation, both of which produced wrong numbers
without any error message.

## What we report, and what we refuse to

**Per class**: precision, recall, F1, average precision, support.
**Aggregate**: macro and micro F1, macro average precision.
**Never**: accuracy.

Accuracy is excluded on principle. An all-negative model scores above 95% on
every label here. Publishing it would be flattering and meaningless.

**Average precision** is worth knowing: it summarises performance across *all*
thresholds at once, so it is threshold-free. That makes it the stable number
when thresholds are in question — and it is how we confirmed the second bug
below.

## Thresholds

A sigmoid gives 0.73; you need a cutoff. We tune one **per label**, choosing
whichever value maximises F1.

Ours, from the trained CNN:

| Label | Threshold |
|---|---:|
| Termination for Convenience | 0.22 |
| Non-Compete | 0.28 |
| License Grant | 0.32 |
| Audit Rights | 0.39 |
| Cap on Liability | 0.78 |
| Insurance | 0.83 |

Nowhere near 0.5, and nowhere near each other. A single global cutoff would
throw away the rare labels entirely.

> **The rule that makes this legitimate: tune on validation, apply to test.**
> Tuning thresholds on the test set means fitting your decision boundary to the
> answers, and the resulting number is unreproducible.

This also has a user-interface consequence. In the demo, a label can be flagged
at 0.223 while another is *not* flagged at 0.341. That looks like a bug, so the
interface shows each label's threshold beside its score
([chapter 10](10-interpretability-and-demo.md)).

## Two levels of evaluation

The model classifies **windows**. A user asks about a **contract**. Both are
reported because they answer different questions:

| Level | Question | CNN score |
|---|---|---:|
| Window | Can it recognise a clause in the text in front of it? | 0.663 |
| Document | Does it find the clause somewhere in the contract? | 0.840 |

Document level is the product question. Window level is the harder, more honest
measure of the model. Quoting only one would misrepresent the system — flattering
in one direction, unfair in the other.

## Pooling: window answers → contract verdict

A contract yields ~30 windows and thus 30 sets of predictions. How do you get
one answer?

**Max-pooling**: the strongest window decides.

The task is **existential** — "does this contract contain a liability cap?" A
clause appearing in one window of a sixty-window contract means the contract
contains it. Averaging would bury that single positive under fifty-nine
negatives.

We tested this rather than assuming it, and a test constructs the exact case: a
true positive at 0.95 confidence in a 60-window document. Mean pooling drops it
below **0.03**.

Measured on real predictions:

| Pooling | Document macro F1 | Document macro AP |
|---|---:|---:|
| top-k (k=3) | 0.797 | 0.862 |
| max | 0.773 | **0.872** |
| mean | 0.624 | 0.721 |

**Mean is clearly worse**, by about 0.15, and that part is solid.

**Max versus top-k is not settled by these numbers.** Max ranks better; top-k
places its operating point better; across 76 documents the difference is inside
our ±0.03 noise floor. Max remains the default because ranking is the more
stable property and the existential argument holds — *not* because 0.797 beats
0.773. Saying so is more useful than picking the bigger number and calling it a
result.

Document-level thresholds are tuned **separately**, because a pooled maximum is
systematically higher than any single window probability.

---

## The threshold bug

`train_cnn` did not record its validation-tuned thresholds at the top level of
`metrics.json`. The evaluation script's lookup chain fell through to a default
of 0.5.

The result: evaluation reported **0.539** where training reported **0.591** —
*for identical weights.* No error, no warning. Just a quietly different
operating point, and a number that would have gone into the report.

**Fix:** both paths now read `training.json`, which the shared loop writes for
every model. One place thresholds live, and the silent fallback is gone.

**Lesson:** a default that silently substitutes for missing data is a bug
generator. Failing loudly would have taken ten seconds to diagnose.

## The leak we caught

This one was ours, and worse.

The first version of the evaluation tuned document-level thresholds on **the
split it was scoring**. For the test split that means fitting the decision
boundary to the test answers and then reporting the score it produces.

That is leakage. It inflates the number and nobody could reproduce it on new
contracts.

Correcting it moved document macro F1 from **0.840 to 0.773**. That 0.067 was
pure optimistic bias.

> The confirmation that the diagnosis was right: **average precision did not
> move at all** — 0.872 before and after. It is threshold-free, so honest and
> dishonest thresholds cannot change it. Exactly the behaviour you would predict
> if the problem was thresholds and nothing else.

The evaluation output now records which split thresholds were tuned on, so the
provenance travels with the number.

> Both bugs were found by **running the evaluation against a real checkpoint**
> rather than trusting it. Neither would have been caught by reading the code.

---

## Why the comparison is generated

`legal-risk-compare` builds the results table from the run directories
themselves. No number reaches the write-up by being copied by hand, so the
report cannot drift from what was measured. The same applies to the dataset
report and the dashboard.

## Tests

`tests/test_pooling.py` (7) and the metrics portion of `tests/test_model_layer.py`.
The notable ones encode reasoning, so a decision cannot be silently reversed:
mean pooling burying a true positive, tuned thresholds beating a fixed 0.5, and
metrics exposing an all-negative model that accuracy would rate above 95%.

---

[← Training](08-training.md) · [Index](README.md) · [Next: Interpretability and the demo →](10-interpretability-and-demo.md)
