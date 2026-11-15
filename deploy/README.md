# Deploying

Two surfaces, deployed separately because they have different needs: the
dashboard is a static file, the demo needs a Python process holding a model.

## Dashboard — Hugging Face Space (static)

```bash
pip install -e ".[hub,analysis]"
hf auth login

python -m legal_risk_classifier.export_dashboard \
    --run_dir outputs/cnn --repo_id <your-username>/clause-risk-review --push
```

Rerun it after any training run; the page is generated, so the published
dashboard picks up the new numbers with nothing edited by hand.

### The plain results page

`site/index.html` is a single hand-written HTML file, no scripts and no build
step, covering what the project does and the results for all four models. It is
what the Space serves while the transformer run directories are unavailable to
the generated dashboard. Publish it with:

```bash
hf upload <your-username>/clause-risk-review site/index.html index.html --repo-type space
```

### GitHub Pages, for the plain results page only

`.github/workflows/pages.yml` publishes the `site/` folder on every push to main
that touches it. Because the account's user-level Pages site carries a custom
domain, the page is served under that domain at `/NNDL/`.

This reverses an earlier decision. Pages was first enabled from the repository
root, which published the README under the personal portfolio domain, and it was
switched off for that reason. It is back on deliberately, with two differences:
the owner now wants the project on that domain, and the workflow publishes only
`site/`, so nothing but the results page is exposed.

To take it down: Settings -> Pages -> Source: None.

## Demo — in the browser, on the results site

Live at https://www.algorithmicbit.tech/NNDL/demo/, linked from the results page.

The TextCNN runs in the visitor's browser, so there is no server to pay for,
keep awake or restart. `site/demo/model.js` is a line-for-line port of the
chunking, the model, noisy-OR pooling and the phrase attribution; on test
contracts it matches Python to within 1e-6 and gives identical verdicts and
phrases. To publish a new CNN run:

```bash
legal-risk-evaluate --run_dir outputs/cnn --split test   # contract thresholds
legal-risk-export-web --run_dir outputs/cnn --out_dir site/demo
```

Then commit `site/demo/`; the Pages workflow deploys it. The weights are stored
as float16 (8.3 MB), which moves scores by under 0.001 and changes no test-set
verdict.

Only the CNN can run this way. A fine-tuned BERT or Legal-BERT is about 440 MB.

### Why not a Hugging Face Space

`deploy/space/` holds a Gradio app for a Space, and was the original plan. As of
November 2026, Hugging Face hosts Gradio and Docker Spaces on its free CPU only
for PRO subscribers; static Spaces stay free. The app still runs anywhere
Python does, for example `legal-risk-demo --run_dir outputs/cnn --share` from a
notebook, which is what notebook 5 does.

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

## Why not one deployment for both

A static page costs nothing to host and never sleeps. A Space holding a model
sleeps after inactivity and takes time to wake. Keeping the results on Pages
means the thing most people open is always instant, and the demo is there for
anyone who wants to try their own contract.
