# Combining chunk predictions

A contract is split into chunks, each chunk is classified, and the results must
be combined into one verdict. This document records which rules were tried,
what they scored, and why the default is what it is.

## The problem with max-pooling

Max keeps the single strongest chunk and discards the rest. A contract scoring
**0.80 and 0.79** in two chunks is indistinguishable from one scoring **0.80
and 0.02**:

| Rule | 0.80 + 0.79 | 0.80 alone | Separates? |
|---|---:|---:|---|
| max | 0.8000 | 0.8000 | **no** |
| mean | 0.0717 | 0.0460 | yes |
| top_k (k=3) | 0.5367 | 0.2800 | yes |
| power (r=4) | 0.4040 | 0.3418 | yes |
| logsumexp (T=0.1) | 0.5249 | 0.4611 | yes |
| **noisy_or** | **0.9761** | **0.8887** | yes |
| noisy_or_topk (k=5) | 0.9605 | 0.8155 | yes |

Two chunks independently suggesting a clause is stronger evidence than one.
Max cannot express that; it is an order statistic, so everything below the
maximum is thrown away.

## The three families

**Order statistics** — `max`, `top_k`. Look only at the strongest chunk(s).

**Smooth averages** — `power`, `logsumexp`. Use every chunk, weighted toward
the strong ones. Both have a knob that interpolates between the mean and the
max, so the aggressiveness is a tunable rather than a commitment.

**Evidence combiners** — `noisy_or`, `noisy_or_topk`. Treat each chunk as
independent evidence: `P(present) = 1 - Π(1 - pᵢ)`. This is the family that
matches the intuition above, because two moderate chunks can outrank one strong
one.

## Measured results

TextCNN, test split, 76 documents at ~44 chunks each. Every rule's constant and
all thresholds were calibrated on validation and applied unchanged to test.

| Rule | Params | Test macro F1 | Test macro AP |
|---|---|---:|---:|
| top_k | k=2 | **0.846** | 0.891 |
| **noisy_or** | — | **0.846** | **0.904** |
| noisy_or_topk | k=2 | 0.841 | 0.900 |
| max | — | 0.840 | 0.899 |
| logsumexp | T=0.03 | 0.836 | 0.873 |
| power | r=3 | 0.813 | 0.843 |
| mean | — | 0.511 | 0.817 |

### What this does and does not show

**Mean pooling is genuinely bad** (0.511), and that result is solid. Averaging
a real clause across forty chunks that legitimately contain nothing destroys
the signal.

**The gap between the serious rules is inside the noise floor.** Run-to-run
variance on identical settings is about ±0.03 macro F1; the spread from max
(0.840) to noisy_or (0.846) is 0.006. **F1 does not separate these rules on
this corpus**, and any write-up claiming otherwise is over-reading.

**Average precision is the more stable comparator** because it is
threshold-free, and it ranks noisy_or first at 0.904. noisy_or is also better
than max on both metrics at once, which is weak evidence but consistent
evidence.

## Why noisy_or is the default

Not because 0.846 beats 0.840 — that margin is noise. Three reasons that do
survive:

1. It is the only family whose mechanism matches the task. Chunks are separate
   passages; a clause appearing in two of them is two observations, not one.
2. It leads on average precision, the metric least sensitive to this corpus's
   small validation set.
3. It is never worse than max on either metric here.

## The failure mode, and the guard

Plain noisy_or saturates. Sixty chunks at 0.10 each contain nothing, but
`1 - 0.9⁶⁰ = 0.998` reads as near-certainty. This grows with chunk count, so it
matters most for wide-window models producing few chunks and least for narrow
ones producing many — the opposite of what intuition suggests.

`noisy_or_topk` exists for this: combine only the k strongest chunks, so
evidence still accumulates but background noise cannot. A test pins both
behaviours.

## Automatic rule selection does not work here

`--pooling auto` calibrates every rule on validation and picks the winner. It
was tried and it **overfits**, under either selection criterion:

| Selected on | Picks | Test macro F1 |
|---|---|---:|
| validation macro F1 | power (r=3) | 0.813 — the worst serious rule |
| validation macro AP | logsumexp (T=0.03) | 0.836 |
| *fixed default* | *noisy_or* | *0.846* |

With 76 validation documents and genuine differences smaller than the noise
floor, selection fits the split rather than the task. The flag remains for a
larger corpus, with this warning attached. **The default is a fixed, principled
choice, and that is deliberate.**

## A learned aggregator was tried, and lost

If choosing between fixed rules is hard, the obvious next step is to stop
choosing and learn the combination. `learned_pooling.py` does that: a
document's chunk probabilities for one label are sorted, the strongest K
converted to log-odds, and the score is a weighted sum fitted by logistic
regression.

The parameterisation was chosen so it *contains* the rules it competes with:

| Rule | Equivalent weights |
|---|---|
| max | `(1, 0, 0, …)` |
| top-k mean | `(1/k, …, 1/k, 0, …)` |
| noisy-OR | a positive, roughly equal weighting |

So it should not be able to lose except by overfitting. It lost anyway.

| Fitted on | K | L2 | Test macro F1 | Test macro AP |
|---|---:|---:|---:|---:|
| validation (76 docs) | 3 | 10.0 | 0.833 | 0.903 |
| validation | 5 | 1.0 | 0.843 | 0.899 |
| train (358 docs) | 3 | 1.0 | 0.833 | 0.899 |
| train | 5 | 1.0 | 0.834 | 0.899 |
| **noisy_or (fixed)** | — | — | **0.846** | **0.904** |

No configuration of the learned aggregator beats noisy_or on either metric.
The closest on F1 is 0.843 against 0.846; the closest on average precision is
0.903 against 0.904.

Two explanations were checked and one was eliminated.

**It is not miscalibration from fitting on seen data.** The obvious worry is
that chunk probabilities on the training split are overconfident, since the
model was trained there. Measured, they are not: mean per-chunk maximum
probability is 0.169 on train against 0.166 on validation. The distributions
are effectively the same, so fitting on train is legitimate — it simply does
not help.

**It is that there is nothing better to find.** The fitted weights come out as
`(+0.50, -0.08, +0.06)` — overwhelmingly the top chunk, with the remaining
terms near zero and alternating in sign, which is what noise looks like. Given
the freedom to weight five chunks however it liked, and 358 documents to learn
from, the fit converged on approximately max-pooling and could not beat a fixed
evidence combiner.

The module is kept rather than deleted, for two reasons. It is the first thing
a reviewer asks about after seeing a hand-chosen rule, and the answer should be
evidence rather than speculation. And the experiment is worth repeating once
Longformer is trained, where three or four large chunks make a different
learning problem than forty small ones.

It is not the default, and nothing depends on it.

## Effect of chunk count

Longformer's window is ~2,600 words against BERT's ~300, so it produces far
*fewer and larger* chunks — roughly 3–5 per contract instead of 44. Since every
rule here combines chunks, the count changes the arithmetic.

A simulation was attempted by grouping adjacent chunks to imitate a wider
window. **It was not conclusive**, for a reason worth recording: taking the max
within each group makes max-of-max identical to the global max, so the baseline
is invariant by construction and the comparison is unfair to it. The remaining
differences stayed inside the noise floor.

This question is genuinely open until Longformer is trained. The expectation is
that noisy_or's advantage grows as chunk count falls, because saturation is
driven by chunk count and corroboration still works with three chunks — but
that is a prediction, not a result.

## Reproducing

```bash
legal-risk-evaluate --run_dir outputs/cnn --split test --pooling noisy_or
legal-risk-evaluate --run_dir outputs/cnn --split test --pooling auto
```

Tests: `tests/test_pooling_rules.py` (9) and `tests/test_pooling.py` (7).
