# 10 · Interpretability and the demo

[← Evaluation](09-evaluation.md) · [Index](README.md) · [Next: Deployment →](11-deployment.md)

**Files:** `attribution.py`, `runtime.py`, `demo.py`

## Why "why" matters here

A lawyer will not act on "the model says 0.87". They need to see the text that
triggered it. An unexplained flag is a flag nobody can act on — and for a triage
tool, that defeats the purpose.

## Why the CNN is the better explainer

This is counter-intuitive, because transformers are the stronger models.

**Attention weights are contested as explanations.** There is a real research
literature arguing attention does not reliably indicate what a model used. A
heatmap looks convincing and may mean little.

**A convolution's receptive field is not a matter of interpretation.** A filter
of width 4 looked at exactly 4 words. Max-pooling recorded which position fired
hardest. That position maps to a literal span of the contract. There is nothing
to dispute.

So our explanation is **an actual phrase from the document**, not a diffuse
weight over tokens.

## How attribution works

A filter's contribution to a label is its pooled activation × the classifier
weight connecting it to that label — exactly its additive term in the final
score. Rank the filters, map each back to the words it fired on, deduplicate.

Asked why it flagged **Insurance** on a real distribution agreement:

```
1.34   "liability insurance. company"
1.24   "product liability insurance"
1.19   "thirty 30 days advance notice"
1.07   "product liability insurance through a"
1.03   "reasonably acceptable products liability insurance"
```

That is readable evidence. A lawyer can check it in seconds.

## What it immediately exposed

The feature paid for itself on its first run.

Asked why it flagged **Cap on Liability**, the model returned *the same*
`product liability insurance` phrases it uses for Insurance.

It is keying on the shared word **"liability"** rather than on capping language
like *"in no event shall"* or *"shall not exceed"*.

That is a concrete, actionable lead on the per-class results — a candidate
explanation for why Cap on Liability sits at 0.745 despite being one of the most
frequent labels — rather than a guess. This is what interpretability is *for*:
not decoration, but debugging.

## The shared runtime

`runtime.py` loads a trained run and scores raw text. The demo uses it rather
than reimplementing inference.

That is deliberate. A demo that chunked or pooled differently from the
evaluation would show numbers that do not match the reported ones — worse than
having no demo. The runtime reads the pooling rule (noisy-OR) and the
contract-level thresholds from the run's `evaluation_*.json`, the same ones the
reported results used, and tests check both, so the two cannot drift apart
silently.

It once did drift: the runtime pooled with max and judged contracts against the
window-level thresholds, after evaluation had moved to noisy-OR with its own
contract-level thresholds. The demo's verdicts were therefore not the reported
model's verdicts. That is now fixed, and it is the reason for the tests.

## The demo

**Live at [algorithmicbit.tech/NNDL/demo](https://www.algorithmicbit.tech/NNDL/demo/).**
Paste a contract, get the six labels scored, the flagged windows quoted with
their scores, and the phrases that triggered the selected label.

It runs in the visitor's browser. `legal-risk-export-web` writes the CNN as
float16 weights, and `site/demo/model.js` ports the chunking, model, pooling and
attribution to plain JavaScript; on test contracts it matches Python to within
1e-6, with identical verdicts and phrases. No server means nothing to pay for or
keep awake, and nothing pasted leaves the browser. Only the CNN is small enough
for this; Legal-BERT is about 50 times larger.

`legal-risk-demo` is the same interface in Python (Gradio). It runs anywhere
Python does, including a notebook cell with `--share` for a temporary public
link, which is what notebook 5 does.

### One interface decision worth understanding

The score panel shows **each label's own contract-level threshold** beside its
score. With the published CNN's thresholds (scores illustrative):

```
Cap on Liability            ·  thr 0.82     0.700
Non-Compete                 ·  thr 0.75     0.600
Termination for Convenience ✓  thr 0.42     0.450
Insurance                   ✓  thr 0.09     0.150
```

Look at rows 2 and 4. Insurance is **flagged at 0.150** while Non-Compete is
**not flagged at 0.600**. Without the threshold visible, that reads as an
obvious bug — and someone would "fix" it by forcing a single cutoff, undoing the
tuning from [chapter 9](09-evaluation.md#thresholds).

Showing the threshold makes the design legible. Interface honesty is part of
model honesty.

## The dashboard

A static page reporting the dataset findings, model performance, and — the part
worth having — a **contract explorer**: ten held-out contracts, chosen to cover
all six labels and **weighted toward the ones the model gets wrong**, showing
CUAD's expert annotation and the model's flagged window side by side.

A page showing only successes is a page nobody believes. Of the ten embedded
contracts, one is fully correct and the rest carry one to three errors.

Every figure is generated from the run data, so regenerating after a new
training run updates the page with nothing edited by hand.

## Still to come

**Attention heatmaps for Legal-BERT** are the planned comparison against filter
attribution — genuinely interesting because it puts a contested explanation
method next to an uncontested one on the same task. Blocked on the fine-tuning
runs.

---

[← Evaluation](09-evaluation.md) · [Index](README.md) · [Next: Deployment →](11-deployment.md)
