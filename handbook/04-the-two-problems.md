# 4 · The two hard problems

[← The dataset](03-the-dataset.md) · [Index](README.md) · [Next: The data pipeline →](05-data-pipeline.md)

Our teacher asked us to identify and justify the two genuinely hard technical
problems in this project, rather than just wiring together a pipeline. These are
they. Everything in chapters 5 to 9 is machinery for answering them.

---

## Problem 1 — How should legal text be represented numerically?

### Why it is a real problem

A neural network sees numbers, not words, and those numbers come from embeddings
([chapter 2](02-crash-course.md#turning-text-into-numbers)). Embeddings are
learned from whatever text the model was pretrained on.

General-purpose models learn from Wikipedia and books. Legal English is a
different dialect, and the differences are exactly where our clauses live:

| Phrase | Everyday meaning | Legal meaning |
|---|---|---|
| "consideration" | thoughtfulness | the payment that makes a contract binding |
| "without prejudice" | impartially | without waiving any rights |
| "in no event shall" | — | the standard opening of a liability cap |
| "party" | social gathering | a signatory |

A model that has barely seen "indemnify" or "liquidated damages" has weak,
imprecise vectors for them. If our six clause types are signalled by exactly this
vocabulary, that weakness costs us accuracy.

### How we test it

Compare **BERT** against **Legal-BERT**.

This pairing is chosen so the comparison means something. The two models share:

- the same architecture (12-layer transformer encoder)
- the same parameter count — confirmed identical at **109,486,854** when fitted with our 6-label head
- the same context window (512 tokens)
- the same chunk width, splits, loss, and threshold tuning

They differ in exactly one respect: **what they read during pretraining**. BERT
read Wikipedia and books. Legal-BERT read 12GB of legislation, court cases and
contracts.

So any difference in score is attributable to the pretraining corpus and nothing
else. A test in the repository (`test_bert_and_legal_bert_differ_only_in_pretraining`)
asserts this, because the moment something else drifts, the experiment stops
answering the question.

> **Why not compare against RoBERTa or DeBERTa?** They are stronger models, but
> they differ in architecture *and* pretraining at once. A win would not tell
> you which caused it. That would improve a leaderboard and ruin an experiment.

### What we expect, and what would surprise us

The hypothesis is specific, not "legal is better". Legal-BERT's advantage should
**concentrate** in labels carried by legal vocabulary, and be **smallest** for
labels carried by ordinary English — "shall maintain insurance" needs no
specialist knowledge.

So read the per-class columns, not the headline average. A *uniform* gain across
all six labels would actually be evidence that something other than domain
knowledge is responsible.

---

## Problem 2 — How should a whole contract be represented?

### Why it is a real problem

Transformers compare every token with every other token. That is what makes them
good, and it costs **O(n²)** — double the text, quadruple the work. This is why
BERT stops at 512 tokens.

Our contracts:

```
median   5,039 words   ≈  7,300 tokens
longest 47,733 words   ≈ 69,200 tokens
BERT limit                   512 tokens
```

The longest contract is **135 times** BERT's capacity. Truncating to the first
512 tokens would keep the title page and throw away the contract. Worse, our
clauses tend to sit *late* — insurance and liability terms are usually near the
end — so truncation discards exactly the text we need.

### The two techniques

**Chunking.** Split the contract into overlapping windows, classify each one,
then combine the answers. Simple and it always works.

**Long-context architecture.** Use a model built to read further. **Longformer**
replaces all-pairs attention with a sliding window plus a few "global" tokens,
which drops the cost from O(n²) to O(n) and makes 4,096 tokens affordable.

### The finding that reframed this problem

Our brief described chunking as a fallback "for contracts still exceeding 4096
tokens". We measured it:

> **97% of contracts exceed BERT's 512-token window. 70% exceed Longformer's
> 4,096-token window as well.**

The leftovers are the *majority*. Long context does not remove the need for
chunking — **both models train on chunks**. That changes what the experiment can
honestly claim.

### What we are actually measuring

Not "can Longformer handle long documents" — neither model handles a whole
contract. The real variable is **how much context one window holds**:

| Model | Window |
|---|---:|
| BERT / Legal-BERT | ~300 words |
| Longformer | ~2,600 words |

So the question is: **is a clause easier to recognise with more surrounding
contract around it?** That is narrower, and it is the claim the data supports.

### A null result is a real result

There is a genuine chance Longformer does not win, and the project should be
able to say so.

Most of these clauses are **locally signalled**. "The Supplier shall maintain
comprehensive general liability insurance" is recognisable from that sentence
alone. You do not need the preceding 2,000 words.

If Longformer does not beat BERT, the finding is: *clause detection in contracts
is a local problem, and expensive long-context machinery is not what it needs.*
That is worth knowing, and it comes with a cost argument attached — Longformer
is 35% larger and far slower to train.

We are stating this **before** running the experiment, deliberately. It is much
easier to agree what a null result means before there is a number someone has
grown attached to.

---

## How the two problems interact

They are deliberately separable:

- **Problem 1** holds context length constant (both 512) and varies pretraining.
- **Problem 2** holds pretraining constant (both general-domain) and varies context length.

Each comparison changes one thing. That is the whole reason the model registry
has exactly three entries ([chapter 7](07-pretrained-models.md)).

---

[← The dataset](03-the-dataset.md) · [Index](README.md) · [Next: The data pipeline →](05-data-pipeline.md)
