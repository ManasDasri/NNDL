# 13 · Run it yourself

[← Results so far](12-results.md) · [Index](README.md) · [Next: Decisions →](14-decisions.md)

Copy-paste, from nothing to a trained model.

## Fastest path: Colab, no setup

Notebooks install everything and call the same command line the repository
exposes, so nothing in them can drift from what the code actually does.

| Notebook | What it does | Time |
|---|---|---|
| [1 · Dataset exploration](https://colab.research.google.com/github/ManasDasri/NNDL/blob/main/notebooks/1_dataset_exploration.ipynb) | Measures the corpus, writes the report and figures | 5 min, CPU |
| [2 · CNN baseline](https://colab.research.google.com/github/ManasDasri/NNDL/blob/main/notebooks/2_cnn_baseline.ipynb) | Trains the CNN, plus the unweighted-loss ablation | 15 min |
| [3 · Problem 1](https://colab.research.google.com/github/ManasDasri/NNDL/blob/main/notebooks/3_problem1_legal_bert.ipynb) | **BERT vs Legal-BERT** | ~1 hour, GPU |
| [4 · Problem 2](https://colab.research.google.com/github/ManasDasri/NNDL/blob/main/notebooks/4_problem2_longformer.ipynb) | **BERT vs Longformer** | several hours, GPU |
| [5 · Demo](https://colab.research.google.com/github/ManasDasri/NNDL/blob/main/notebooks/5_demo_and_interpretability.ipynb) | Browser demo and attribution | 10 min |

> **Set the runtime first: Runtime → Change runtime type → T4 GPU.** Notebook 3
> prints the GPU name in its second cell and tells you if it is missing. Do not
> run past that on CPU.

**Notebook 3 is the one that matters** — it answers Problem 1, which is
currently unanswered.

## Local setup

```bash
git clone https://github.com/ManasDasri/NNDL.git
cd NNDL
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"
```

Python 3.10–3.13. **Not 3.14** — PyTorch has no wheels for it yet.

## The pipeline, step by step

```bash
# 1. CUAD JSON -> documents with spans + split assignment   (~2 min)
legal-risk-prepare

# 2. Measure the corpus -> docs/dataset-report.md + figures
legal-risk-analyze

# 3. Train the CNN baseline                                 (~10 min CPU)
legal-risk-train-cnn --epochs 12

# 4. Score it at window and document level
legal-risk-evaluate --run_dir outputs/cnn --split test --pooling max

# 5. Try it on text
legal-risk-demo --run_dir outputs/cnn
```

## The transformers

```bash
legal-risk-train-transformer --model bert       --epochs 3
legal-risk-train-transformer --model legal-bert --epochs 3

legal-risk-train-transformer --model longformer --epochs 3 \
    --batch_size 1 --grad_accumulation 16 --gradient_checkpointing
```

**A GPU is effectively required.** On an Apple M-series laptop a single
BERT-base run takes about 4.6 hours. On a free Colab T4 it is 30–50 minutes.

Longformer needs `--batch_size 1 --gradient_checkpointing` to fit in memory at
4,096 tokens.

## Comparing and publishing

```bash
legal-risk-compare outputs/cnn outputs/bert outputs/legal_bert   # -> docs/results.md

hf auth login
legal-risk-export-hf --run_dir outputs/legal_bert \
    --repo_id <user>/cuad-clause-risk-legal-bert --push

python -m legal_risk_classifier.export_dashboard \
    --run_dir outputs/cnn --repo_id <user>/clause-risk-review --push
```

Note `hf auth login`, not `huggingface-cli login` — the latter is deprecated and
exits without doing anything.

## Tests

```bash
python  tests/test_data_pipeline.py      # 12 · standard library only
python  tests/test_analysis.py           #  9 · standard library only
python  tests/test_model_layer.py        # 14 · needs torch
python  tests/test_pretrained.py         # 11 · needs torch, offline
python  tests/test_pooling.py            #  7 · needs torch
python  tests/test_interpretability.py   # 11 · needs torch
python  tests/test_export_hf.py          #  7 · needs torch, offline
```

**71 checks, no framework.** Each file runs as a script and prints what passed.
The first two need nothing installed at all.

## Useful flags

| Flag | Effect |
|---|---|
| `--limit_documents 25` | Smoke-test on a subset. Use this before any long run. |
| `--no_pos_weight` | Disable class balancing, to watch the model collapse (notebook 2) |
| `--pooling mean` | Compare pooling strategies |
| `--window` / `--overlap` | Change chunk geometry |
| `--seed` | Change the seed — useful for measuring variance |

## If something breaks

**`python: command not found`** — use `python3`, or activate the venv.

**No module named torch** — you are on Python 3.14. Make a 3.12 venv.

**Training is extremely slow** — you are on CPU. Check `torch.cuda.is_available()`.

**CUDA out of memory** — lower `--batch_size`, raise `--grad_accumulation`, add
`--gradient_checkpointing`.

**Scores differ from this handbook by ~0.03** — expected. See
[chapter 12](12-results.md#the-caveat-that-governs-every-number-here).

---

[← Results so far](12-results.md) · [Index](README.md) · [Next: Decisions →](14-decisions.md)
