# 11 · Deployment

[← Interpretability and the demo](10-interpretability-and-demo.md) · [Index](README.md) · [Next: Results so far →](12-results.md)

**Files:** `export_hf.py`, `export_dashboard.py`, `deploy/`

Three things get published, each to a different place because each has different
needs.

| What | Where | Live |
|---|---|---|
| Results dashboard | Hugging Face static Space | [clause-risk-review](https://kenx049-clause-risk-review.static.hf.space) |
| Trained model | Hugging Face Hub | [cuad-clause-risk-cnn](https://huggingface.co/KenX049/cuad-clause-risk-cnn) |
| Live demo | Hugging Face Space (Gradio) | prepared, not yet launched |

## Publishing a model is not uploading a file

The naive approach — upload `model.pt` — produces something nobody can use.

A training run saves what the **training loop** needs. For the transformer runs
that is a wrapper state dict whose keys are prefixed `backbone.`:

```
backbone.bert.embeddings.word_embeddings.weight
backbone.classifier.weight
```

`from_pretrained` cannot load that. You would have published a file that fails
on first use.

`legal-risk-export-hf` does three things a file upload would not:

**Strips the prefix and saves properly.** Verified against real Legal-BERT
weights: the exported model reproduces the training wrapper's outputs with a
maximum absolute difference of **0.0**.

**Names the outputs.** Without `id2label`, a loaded model reports `LABEL_0`
through `LABEL_5` and the caller has to know our ordering by hand — which is
exactly how label orderings get silently transposed:

```python
model.config.id2label
# {0: 'Cap on Liability', 1: 'Non-Compete', 2: 'License Grant', ...}
```

**Publishes the thresholds.** They run 0.22–0.83. Anyone applying a default 0.5
loses recall on the rare labels with no way to know why, so they ship as a table
in the model card.

## The model card

Carries the run's **real** metrics rather than placeholders, the chunking and
imbalance that shaped training, the ±0.03 variance caveat so small differences
between checkpoints are not over-read, and a plain statement that this is a
research artifact and **not legal advice** — it locates clauses a lawyer should
read, it does not replace reading them.

The CNN's card is handled honestly too: it is not a transformers architecture,
so the card points at `CNNRuntime` rather than promising a `from_pretrained`
that would fail.

## Why not GitHub Pages

We tried it first. It is the wrong tool, for a structural reason worth
recording.

A custom domain attached to a **user-level** GitHub Pages site makes *every*
project repository serve as a subpath of that domain. Enabling Pages on this
repository published a course project under a personal portfolio at
`algorithmicbit.tech/NNDL/`, and there is **no per-repository way** to keep
Pages while opting out of the domain.

Any GitHub Pages hosting under that account lands on the portfolio. So the
answer was not "configure it differently" but "use something else".

A static Hugging Face Space has its own URL, no relationship to any other site,
costs nothing, and uses the **same account as the model and demo** — one
credential for the whole deployment story instead of two.

> If Pages was ever enabled on a fork of this repository, turn it off under
> Settings → Pages → Source: None. The REST API refuses to deactivate a Pages
> site, so it cannot be scripted.

## The dashboard is generated, not written

```bash
python -m legal_risk_classifier.export_dashboard \
    --run_dir outputs/cnn --repo_id <user>/clause-risk-review --push
```

Data is exported from the current runs and inlined into the page at build time —
one self-contained file with no request that can fail. Re-run after any training
run and the published page picks up the new numbers. **No figure on the page was
typed by hand.**

## Training on managed GPUs

`deploy/jobs/train_on_hub.py` runs the fine-tuning on Hugging Face Jobs:

```bash
hf jobs uv run deploy/jobs/train_on_hub.py --flavor t4-small \
    --secrets HF_TOKEN --env MODEL=legal-bert --env REPO_ID=user/name
```

The job **clones this repository** rather than vendoring the pipeline, so it
trains exactly the code the results were measured on, and pushes the finished
model straight to the Hub.

It refuses to run when no GPU is visible rather than falling back to CPU, where
a fine-tune would burn money for days and produce nothing.

Jobs requires a **pre-paid credit balance**. At `t4-small` ($0.40/hr) both
transformers would cost well under a dollar, but with an empty balance the
submission fails immediately and costs nothing. Colab remains the free
alternative ([chapter 13](13-run-it-yourself.md)).

---

[← Interpretability and the demo](10-interpretability-and-demo.md) · [Index](README.md) · [Next: Results so far →](12-results.md)
