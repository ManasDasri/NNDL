# Deploying

Everything public is static: the results page and the browser demo live in
`site/`, and the model is on the Hugging Face Hub. There is no server.

## Results page and demo — GitHub Pages

`site/index.html` is a single hand-written HTML file, no build step, covering
what the project does and the results for all four models. `site/demo/` is the
browser demo.

`.github/workflows/pages.yml` publishes the `site/` folder on every push to main
that touches it. Because the account's user-level Pages site carries a custom
domain, it is served at https://www.algorithmicbit.tech/NNDL/. Only `site/` is
published, so the README and docs are not exposed. To take it down: Settings ->
Pages -> Source: None.

The same folder is mirrored to a static Hugging Face Space:

```bash
hf upload <your-username>/clause-risk-review site . --repo-type space
```

## The demo runs in the browser

The TextCNN runs in the visitor's browser, so there is no server to pay for,
keep awake or restart. `site/demo/model.js` is a line-for-line port of the
chunking, the model, noisy-OR pooling and the phrase attribution; on test
contracts it matches Python to within 1e-6 and gives identical verdicts and
phrases. To publish a new CNN run:

```bash
legal-risk-evaluate --run_dir outputs/cnn --split test   # contract thresholds
legal-risk-export-web --run_dir outputs/cnn --out_dir site/demo
```

Then commit `site/demo/`; the Pages workflow deploys it, and push the same run
to the Hub with `legal-risk-export-hf` so notebook 5's Gradio demo, which loads
the Hub checkpoint, keeps matching. The weights are stored as float16 (8.3 MB),
which moves scores by under 0.001 and changes no test-set verdict.

Only the CNN can run this way. A fine-tuned BERT or Legal-BERT is about 440 MB.

A Gradio Space was the original plan, but since November 2026 Hugging Face hosts
Gradio and Docker Spaces on its free CPU only for PRO subscribers; static Spaces
stay free. The Gradio app (`legal-risk-demo`) still runs anywhere Python does,
which is what notebook 5 does with `--share`.

## Model — Hugging Face Hub

A training run saves what the training loop needs, which is not what a stranger
needs: the transformer checkpoints are wrapper state dicts with `backbone.`
prefixed keys, so uploading one unchanged hands people a file they cannot load.

`legal-risk-export-hf` converts a run into the standard layout, names the six
outputs in the config so the model is self-describing, and writes a model card
carrying that run's real numbers, its per-label thresholds and its limitations.

```bash
pip install -e ".[hub]"
hf auth login                              # a token with write access

legal-risk-export-hf --run_dir outputs/legal_bert \
    --repo_id <your-username>/cuad-clause-risk-legal-bert --push
```

Leave `--push` off to inspect `outputs/<run>/hf` first.

A transformer exported this way loads with plain `transformers`, no repository
required, and reproduces the training wrapper exactly:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained(name)
model.config.id2label      # {0: 'Cap on Liability', 1: 'Non-Compete', ...}
```

The TextCNN is not a transformers architecture, so its card says so and points
at `CNNRuntime` rather than promising a `from_pretrained` that would fail.
