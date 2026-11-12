# Weak points, and what was done about each

An audit of the trained models and the pipeline around them: where they fail,
why, which fixes were tried, and which survived measurement. Every fix was tried
as an option first and kept as a default only if it beat run-to-run noise.

## What the classification reports show

From [classification-reports.md](./classification-reports.md), three patterns
hold across every model.

**The same two labels are weakest everywhere.** Non-Compete and Termination for
Convenience are the bottom two for all four models. They are also the two rarest
(1.5% of windows each), and both are usually one short sentence inside a longer
section.

| Window F1 | Non-Compete | Termination for Convenience | Best other label |
|---|---:|---:|---:|
| TextCNN | 0.317 | 0.544 | 0.841 |
| BERT | 0.560 | 0.621 | 0.876 |
| Legal-BERT | 0.509 | 0.685 | 0.871 |
| Longformer | 0.567 | 0.333 | 0.901 |

**False alarms outnumber misses.** Recall is above precision for every
transformer: BERT's micro average is 0.691 precision against 0.802 recall. Audit
Rights is the clearest case, 0.639 against 0.874. The tuned thresholds settle
low because the weighted loss pushes the models towards flagging.

**Longformer is weakest exactly where the clause is shortest.** Termination for
Convenience at 0.333 F1, precision 0.263. A one-sentence clause is a small part
of a 2,600-word window, and the window-level label says only that the clause is
somewhere in it.

## Two bugs, fixed

### Longformer was silently losing a fifth of its windows' text

CUAD contracts are OCR'd and keep their page layout as runs of spaces. BERT's
tokenizer discards whitespace. Longformer's, inherited from RoBERTa, encodes it:
**16.6% of Longformer's tokens were whitespace.** That pushed 19.8% of its
windows past the 4,096-token limit, where the tokenizer cut the tail off without
warning. The model never read that text, and 4.7% of positive labels pointed at
a clause lying mostly in the cut-off part.

**Fix:** collapse whitespace before tokenizing (`collapse_whitespace` in
`datasets.py`).

| Longformer, training split | Before | After |
|---|---:|---:|
| Windows over the token limit | 19.8% | 1.5% |
| Tokens cut off | 487,550 | 19,090 |

For BERT and Legal-BERT the token sequences are identical before and after, so
the Problem 1 results are unaffected. **The Longformer result in the handbook
predates this fix.** Notebook 4 reruns it; Problem 2 is provisional until then.

### The last partial batch of every epoch never trained

With gradient accumulation, the optimizer steps once every N batches. When an
epoch's batch count was not a multiple of N, the leftover batches' gradients were
computed and then zeroed without a step. For Longformer (accumulation 16, 1,380
windows) that was the last 4 windows of every epoch. BERT and the CNN use no
accumulation and were unaffected.

**Fix:** step on the epoch's last batch too. `test_a_partial_accumulation_group_still_trains`
pins it.

## Training changes, measured and rejected

Five configurations of the CNN, three seeds each, same splits. Means ± standard
deviation across seeds.

| Configuration | Window F1 | Window AP | Contract F1 | Contract AP |
|---|---:|---:|---:|---:|
| **Baseline** | 0.658 ± 0.018 | 0.694 ± 0.013 | 0.834 ± 0.017 | 0.898 ± 0.017 |
| Pick the best epoch on validation AP, not F1 | 0.652 ± 0.009 | 0.691 ± 0.008 | 0.835 ± 0.011 | 0.888 ± 0.018 |
| + learning-rate warmup and decay | 0.657 ± 0.004 | 0.701 ± 0.007 | 0.827 ± 0.014 | 0.897 ± 0.005 |
| AP selection + square-root class weights | 0.652 ± 0.009 | 0.695 ± 0.006 | 0.838 ± 0.018 | 0.896 ± 0.008 |
| All three | 0.655 ± 0.022 | 0.706 ± 0.017 | 0.834 ± 0.014 | 0.901 ± 0.010 |

**None of these beats the baseline by more than its own seed-to-seed spread.**
Each targeted a real symptom: F1-at-0.5 checkpoint selection bounces between
epochs, a constant learning rate is unusual for fine-tuning, and full class
weighting drives the over-flagging above. None of them moved the score, so none
was kept: an option measured to do nothing is one more thing to get wrong.

The square-root class weights are worth one more sentence, because they target
the precision problem directly. They did not fix it. The thresholds are retuned
on validation either way, and retuning absorbs most of what the loss weighting
changes.

A fourth idea, training on more overlapping windows from the same contracts to
give the rare labels more examples, was started and not finished, so it is not
reported here.

## Smaller weak points, left as they are

- **3.7% of BERT's windows exceed 512 tokens**, mostly tables and numeric
  schedules. Only 0.2% of positive labels sit in the cut-off text, against 4.7%
  for Longformer before its fix, so it was not worth changing BERT's windows and
  disturbing the Problem 1 comparison.
- **Thresholds are tuned on 76 validation contracts.** That is the main source of
  the ±0.03 run-to-run noise, and the only cure is more labelled data.
- **Legal-BERT and Longformer have one seed each.** More seeds would tighten every
  comparison in this project more than any modelling change above.

## Classification reports for every run

`legal-risk-evaluate` now writes `classification_report_<split>.txt` for every
model it scores, window level and contract level, and prints it. Notebooks 3 and
4 evaluate every run they train, including each extra seed, and end with a cell
that prints all the reports. The reports for the runs made before this change are
rebuilt in [classification-reports.md](./classification-reports.md).

## What the notebooks now run

- **Notebook 3:** CNN, BERT and Legal-BERT, then BERT and Legal-BERT at seeds 1
  and 2 with the same settings as the main runs. That gives Legal-BERT three seeds
  instead of one, which is what its small AP edge needs.
- **Notebook 4:** Longformer again, with the whitespace and accumulation fixes.
