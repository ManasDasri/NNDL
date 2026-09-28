# 8 · Training

[← The pretrained models](07-pretrained-models.md) · [Index](README.md) · [Next: Evaluation →](09-evaluation.md)

**Files:** `training.py`, `datasets.py`, `metrics.py`

## One loop for every model

`training.py` trains the CNN, BERT, Legal-BERT and Longformer. Not four loops —
one.

This is deliberate. Every model gets identical chunks, identical splits,
identical loss weighting, identical early stopping. A difference in score is
therefore a difference **between the models**, not between their training
recipes.

Four separate loops would drift — someone tunes one, changes a learning rate in
another — and the comparison would quietly stop being a comparison.

## The loop

```
for each epoch:
    for each batch of windows:
        predict          → six numbers per window
        compare to truth → loss
        backpropagate    → nudge every weight
    score on validation
    if best so far: remember these weights
    if no improvement for 2-3 epochs: stop
restore the best weights
```

Two safeguards:

**Early stopping.** Training past the point of improvement makes the model
memorise the training set. We stop after 2–3 epochs without progress and restore
the best checkpoint — not the last one.

**Gradient clipping.** Occasionally a batch produces an enormous gradient that
would wreck the weights. We cap the magnitude at 1.0.

## The imbalance problem

This is the part that decides whether the model works at all.

85% of windows carry no label. Per label, positives are 1.5%–4.2%. Presented to
a normal loss function, the maths is brutal:

> If a label appears in 1.5% of windows, a model that answers "no" to everything
> is right 98.5% of the time. The loss from being wrong on the rare positives is
> swamped by the reward for being right on the negatives. **The optimizer's best
> move is to predict nothing, forever.**

This is not hypothetical — it is where the model lands by default.

### The fix: weighted loss

We use `BCEWithLogitsLoss(pos_weight=...)`, which multiplies the penalty for
missing a positive by that label's negative-to-positive ratio:

| Label | pos_weight |
|---|---:|
| License Grant | 23× |
| Cap on Liability | 27× |
| Audit Rights | 33× |
| Insurance | 49× |
| Termination for Convenience | 65× |
| Non-Compete | 66× |

So missing a Non-Compete clause hurts 66 times more than a false alarm. That
re-balances the incentive and the model starts predicting.

Weights are computed from the **training split only**.

You can see this for yourself — `--no_pos_weight` trains without it, and the
model collapses toward predicting nothing. Notebook 2 runs that ablation
deliberately, because it is the clearest single demonstration of what the
imbalance does.

### BCE, and why not cross-entropy

**Binary cross-entropy** treats each of the six outputs as its own independent
yes/no question. Standard cross-entropy (with softmax) forces the six
probabilities to sum to 1, which would mean more Insurance necessarily implies
less Non-Compete.

That is wrong here. A contract can have both. BCE is the multi-label choice —
the practical consequence of [chapter 2](02-crash-course.md#multi-class-versus-multi-label).

## Hyperparameters, and why

| Setting | CNN | Transformers | Reason |
|---|---|---|---|
| Learning rate | 1e-3 | 2e-5 | Pretrained weights are already good; large steps destroy them. 50× smaller is standard for fine-tuning. |
| Epochs | 12 | 3 | Pretrained models converge in 2–3; a from-scratch CNN needs more. |
| Batch size | 32 | 8 (+accumulation) | Transformers at 512 tokens are memory-hungry. |
| Dropout | 0.5 | built in | Regularisation. |
| Optimizer | AdamW | AdamW | Standard. |

**Gradient accumulation** deserves a note: if a batch of 32 will not fit in
memory, run four batches of 8, sum the gradients, then update once. Same effect,
less memory. This is how Longformer trains at all.

## Reproducibility, and its limit

Every run seeds Python, NumPy and PyTorch, and the split assignment is written
to disk. Same seed, same data, same splits.

And yet **two identical runs produced 0.633 and 0.663**.

The cause is floating-point non-determinism in GPU kernels — operations complete
in non-deterministic order, and floating-point addition is not associative, so
tiny differences compound over thousands of steps.

You can force determinism, at a significant speed cost. We did not. Instead we
**measured the variance and report it**:

> ±0.03 macro F1 between identical runs. Any claimed difference smaller than
> that needs multiple seeds before it is a finding.

Reporting a known error bar is more useful than pretending to a precision we do
not have.

## What a run leaves behind

```
outputs/cnn/
  model.pt          trained weights
  vocab.json        vocabulary (CNN only)
  training.json     config, per-epoch history, tuned thresholds
  metrics.json      full test results
```

`training.json` is the single source of truth for thresholds. That matters —
[chapter 9](09-evaluation.md#the-threshold-bug) explains the bug that came from
having more than one place to look.

---

[← The pretrained models](07-pretrained-models.md) · [Index](README.md) · [Next: Evaluation →](09-evaluation.md)
