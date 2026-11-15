# 11 · Deployment

[← Interpretability and the demo](10-interpretability-and-demo.md) · [Index](README.md) · [Next: Results so far →](12-results.md)

**Files:** `export_hf.py`, `export_web.py`, `site/`, `deploy/`

| What | Where | Live |
|---|---|---|
| Results page | GitHub Pages, from `site/` | [algorithmicbit.tech/NNDL](https://www.algorithmicbit.tech/NNDL/) |
| Live demo | In the browser, from `site/demo/` | [algorithmicbit.tech/NNDL/demo](https://www.algorithmicbit.tech/NNDL/demo/) |
| Results page, mirror | Hugging Face static Space | [clause-risk-review](https://kenx049-clause-risk-review.static.hf.space) |
| Trained model | Hugging Face Hub | [cuad-clause-risk-cnn](https://huggingface.co/KenX049/cuad-clause-risk-cnn) |

The demo was planned as a Gradio Space, but free Hugging Face Spaces now run
only static sites; Gradio and Docker Spaces need a PRO plan. So the CNN runs in
the visitor's browser instead (see [chapter 10](10-interpretability-and-demo.md#the-demo)).

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

## GitHub Pages, and why it was first turned off

A custom domain attached to a **user-level** GitHub Pages site makes every
project repository's Pages site a subpath of that domain. Enabling Pages here
from the repository root published the whole README under the personal domain
at `algorithmicbit.tech/NNDL/`, which was not wanted, so it was switched off and
the results page went to a static Hugging Face Space.

It was later turned back on deliberately, once the project was wanted on that
domain, with one difference: `.github/workflows/pages.yml` publishes only the
`site/` folder, so the results page and demo are public and nothing else is.
To take it down: Settings → Pages → Source: None.

---

[← Interpretability and the demo](10-interpretability-and-demo.md) · [Index](README.md) · [Next: Results so far →](12-results.md)
