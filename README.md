# Clause Risk Review

Multi-label classification of commercial contract clauses into six risk-relevant
categories, on [CUAD v1](https://www.atticusprojectai.org/cuad) — 510 contracts
annotated by lawyers at The Atticus Project.

> **New to the project? Read [the handbook](./handbook/README.md).** Fifteen
> chapters covering the whole thing in order, assuming no machine learning
> background: what the problem is, why each architecture was chosen, what every
> number means, and which decisions we got wrong the first time. For the
> engineer-facing version of the same ground, see
> [docs/architecture.md](./docs/architecture.md).

**Labels:** Cap on Liability · Non-Compete · License Grant · Audit Rights ·
Termination for Convenience · Insurance

**Results page:** [algorithmicbit.tech/NNDL](https://www.algorithmicbit.tech/NNDL/) ·
**Try the demo:** [algorithmicbit.tech/NNDL/demo](https://www.algorithmicbit.tech/NNDL/demo/)
(runs in your browser)

```bash
pip install -e ".[analysis]"
legal-risk-prepare                     # CUAD spans + document-level splits
legal-risk-analyze                     # dataset report and figures
legal-risk-train-cnn --epochs 12       # the CNN baseline
legal-risk-train-transformer           # defaults to Legal-BERT, the best model; needs a GPU
legal-risk-evaluate --run_dir outputs/cnn --split test
```

Everything runs from the command line, so a notebook never holds logic. The
[notebooks](./notebooks) are Colab runners that install and call these commands.

## Results

Held-out test split: 76 contracts, 3,314 windows, never seen in training.
Mean ± standard deviation over seeds.

| Model | Window F1 | Window AP | Document F1 | Document AP | Runs |
|---|---:|---:|---:|---:|---|
| TextCNN | 0.658 ± 0.018 | 0.694 ± 0.013 | 0.834 ± 0.017 | 0.898 ± 0.017 | 3 seeds, published |
| BERT | 0.697 ± 0.014 | 0.750 ± 0.015 | 0.861 ± 0.017 | 0.917 ± 0.012 | 3 seeds |
| **Legal-BERT** | **0.744 ± 0.005** | **0.805 ± 0.004** | **0.883 ± 0.009** | **0.928 ± 0.005** | 3 seeds |
| Longformer | 0.743† | 0.770† | 0.831 | 0.886 | 2 runs |

Every run's classification report is in
[docs/classification-reports.md](./docs/classification-reports.md); the executed
notebooks, with confusion matrices and metric charts, are in [`results/`](./results).

Document scores use noisy-OR pooling over windows, the default since it beat max
on both metrics — see [docs/pooling.md](./docs/pooling.md).

Per class, CNN, window level:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Insurance | 0.877 | 0.814 | 0.844 | 70 |
| License Grant | 0.806 | 0.767 | 0.786 | 146 |
| Cap on Liability | 0.760 | 0.731 | 0.745 | 104 |
| Audit Rights | 0.651 | 0.793 | 0.715 | 87 |
| Termination for Convenience | 0.491 | 0.540 | 0.514 | 50 |
| Non-Compete | 0.400 | 0.351 | 0.374 | 57 |

Rarity and difficulty track each other: the weakest label is also the rarest.

**Run-to-run variance is about ±0.03 macro F1** on identical settings, from float
nondeterminism on GPU. A difference smaller than that is not evidence of
anything, and should be run across several seeds before it is claimed. Problem 1
shows why: one run each said "no advantage", three seeds each said otherwise.

## The two problems

### Problem 1 — how should legal text be represented?

BERT against Legal-BERT. They share an architecture, a parameter count (109,486,854
each), a context window and a chunk width, differing only in pretraining corpus, so
any gap between them measures domain adaptation and nothing else.

The hypothesis is specific: the gain should concentrate in labels carried by legal
vocabulary and be smallest for labels carried by ordinary English. Read the
per-class columns, not the macro score. See [docs/model-selection.md](./docs/model-selection.md).

**Result: legal pretraining helps.** Over three seeds each, Legal-BERT leads
BERT by +0.047 window F1 (0.744 ± 0.005 against 0.697 ± 0.014) and +0.055 window
AP, and its worst run beats BERT's best. The per-label prediction holds: the
gains are largest on Termination for Convenience (+0.107), Cap on Liability
(+0.077) and License Grant (+0.056), and zero on Insurance (−0.007), the label
written in ordinary English. At contract level the lead shrinks to +0.022 and
the runs overlap. An earlier single run of each had called it a tie; one low
Legal-BERT run decided that. Full analysis in
[handbook chapter 12](./handbook/12-results.md).

### Problem 2 — how should whole contracts be represented?

Contracts run to a median of 5,039 words and a maximum of 47,733.

**97% exceed BERT's 512-token window. 70% exceed Longformer's 4,096 as well.**

So long context does not remove the need for chunking — both models train on
chunks. What differs is how much context one window holds: ~300 words against
~2,600. The experiment measures whether a clause is easier to recognise with more
contract around it, which is narrower and more defensible than "long-context models
handle long documents".

A null result is a real finding. Most of these clauses are locally signalled:
*"shall maintain insurance"* is recognisable from its own sentence.

**Result: longer context did not help.** At contract level Longformer scores
0.831 F1 / 0.886 AP (two runs, 0.834 and 0.828), below every BERT and Legal-BERT
run and level with the CNN. Both runs stopped early at epoch 9 with their best
at epoch 7, so it had levelled off. 2,600-word windows leave 1,380 training
windows against BERT's 13,829, each with a vaguer label, and these clauses are
recognisable from their own sentence. Full analysis in
[handbook chapter 12](./handbook/12-results.md).

†Longformer's window scores are on 2,600-word windows, not comparable with the
300-word rows above them.

## Dataset challenges

Generated in full by `legal-risk-analyze` into
[docs/dataset-report.md](./docs/dataset-report.md).

| Challenge | Evidence | Consequence |
|---|---|---|
| Documents far exceed context windows | 97% over 512 tokens; longest 47,733 words | Chunking is mandatory for every model |
| Severe negative dominance | 85% of windows carry no label | Weighted loss, tuned thresholds, no accuracy |
| Sparse positives per label | as few as 301 positive windows | High variance; report per-class support |
| Labels co-occur | 275 contracts carry Cap on Liability, 182 of those also License Grant | Six sigmoids, not a softmax |
| Annotation is span-level, supervision is window-level | 3,123 spans mapped by overlap | The overlap rule is a modelling choice |
| Legal English is domain-specific and OCR-noisy | irregular whitespace and page furniture throughout | Motivates the Legal-BERT comparison |

The positive rate per label runs 1.5%–4.2% against every window a classifier sees.
A balanced loss needs weights of 22×–68×.

## How it works

```text
CUAD_v1.json
  └─ prepare    spans extracted by category, splits assigned per document
       └─ chunking   overlapping word windows; a window takes a label when it
       │             meaningfully overlaps an annotated span
       └─ training   one shared loop for every model
            └─ evaluate   window-level and document-level, thresholds from validation
                 └─ compare / dashboard / demo
```

Four decisions worth knowing before reading the code:

**Windows are counted in words, not tokens.** A word window is tokenizer-agnostic,
so all four models train on identical boundaries and a score difference is a
difference between models rather than between tokenizers.

**Splits are per document and persisted.** Splitting per window would put
overlapping text on both sides. The assignment is written to disk so every model is
scored on an identical test set.

**Thresholds are tuned per label on validation, never on test.** They range 0.22
to 0.83, because each label has its own positive rate and a single 0.5 cut is
arbitrary.

**A contract's windows are combined as independent evidence, not by taking the
strongest.** Max-pooling cannot tell 0.80-and-0.79 apart from 0.80-alone, though the
first has two observations; noisy-OR combines them to 0.958. Plain averaging is far
worse than either, at 0.511 document macro F1 against noisy-OR's 0.846, because a
real clause gets diluted across forty windows that legitimately contain nothing.
Seven rules and a learned aggregator were measured: [docs/pooling.md](./docs/pooling.md).

## Layout

```text
src/legal_risk_classifier/
  cuad.py chunking.py splits.py      data layer — standard library only
  analysis.py figures.py analyze.py  dataset report
  textcnn.py vocab.py                CNN baseline
  pretrained.py                      BERT / Legal-BERT / Longformer
  training.py datasets.py metrics.py one training loop for every model
  prepare.py                         CUAD spans and persisted splits
  train_cnn.py train_transformer.py  the two entry points
  pooling.py learned_pooling.py      seven rules plus the one that lost
  evaluate.py compare.py plots.py    window and document scoring, and their plots
  attribution.py runtime.py demo.py  interpretability and the browser demo
  export_dashboard.py export_hf.py   results page and Hugging Face publishing
notebooks/                           Colab runners
docs/                                generated reports and the model justification
tests/                               95 checks, no pytest required
```

## Tests

```bash
python  tests/test_data_pipeline.py      # 12 — standard library only
python  tests/test_analysis.py           #  9 — standard library only
python  tests/test_model_layer.py        # 17 — needs torch
python  tests/test_pretrained.py         # 12 — needs torch, runs offline
python  tests/test_pooling.py            #  7 — needs torch
python  tests/test_pooling_rules.py      #  9 — the seven rules, no torch
python  tests/test_learned_pooling.py    # 10 — the aggregator's contract
python  tests/test_interpretability.py   # 11 — needs torch
python  tests/test_export_hf.py          #  8 — no network
```

No framework. Each file runs as a script and prints what passed.

Tests carry the reasoning behind decisions, so a decision cannot be quietly
reversed: that a whole CUAD question is not a category, that a clause must not
label windows which do not contain it, that mean pooling buries a true positive, that max cannot see corroboration,
that BERT and Legal-BERT differ only in checkpoint, and that metrics expose an
all-negative model which accuracy would rate above 95%.

## Deployment

The results page (`site/index.html`) and the demo (`site/demo/`) are published by
GitHub Pages at [algorithmicbit.tech/NNDL](https://www.algorithmicbit.tech/NNDL/).
The demo runs the CNN in the visitor's browser, so there is no server to keep
running. To publish a new CNN run to it:

```bash
legal-risk-evaluate --run_dir outputs/cnn --split test
legal-risk-export-web --run_dir outputs/cnn --out_dir site/demo
```

A generated dashboard (`legal-risk-export-dashboard`) can also be pushed to a
static Hugging Face Space. See [deploy/](./deploy/README.md).

## Demo

```bash
pip install -e ".[demo]"
legal-risk-demo --run_dir outputs/cnn          # --share for a temporary public link
```

Paste a contract; get the six labels scored against their own thresholds, the
flagged passages with confidence and character offsets, and the phrases that
triggered each flag.

Attribution works because a convolutional filter is a literal n-gram detector:
max-over-time pooling records which position fired it hardest, so the explanation
is an actual phrase from the contract. It already earns its place — asked why it
flagged Cap on Liability, the model returns the same `product liability insurance`
phrases it uses for Insurance, meaning it keys on the shared word "liability"
rather than on capping language.

License: [MIT](./LICENSE)
