# 1 · What this project is

[← Index](README.md) · [Next: Crash course →](02-crash-course.md)

## The situation

A commercial contract is a long legal document — a supply agreement, a
distribution deal, a licensing arrangement. A big company signs thousands of
them. Buried inside each one are a handful of clauses that carry real financial
risk:

- **Does this contract cap how much we can be sued for?**
- **Does it stop us competing in a market?**
- **Does it let the other side walk away whenever they like?**

A lawyer finds these by reading the whole contract. That is slow, expensive, and
does not scale to a shelf of 500 agreements.

## What we built

A system that reads a contract and flags six clause types:

| Clause type | Why it matters commercially |
|---|---|
| **Cap on Liability** | Puts a ceiling on what you can be sued for. Its absence is a large hidden risk. |
| **Non-Compete** | Restricts which markets or customers you can pursue. |
| **License Grant** | Defines what intellectual property rights you gave away or received. |
| **Audit Rights** | Lets the other party inspect your books and records. |
| **Termination for Convenience** | Lets a party exit with notice and no reason. Kills revenue predictability. |
| **Insurance** | Obliges you to carry and maintain cover. An ongoing cost and compliance duty. |

The output is not a legal opinion. It is a **triage aid**: it points a lawyer at
the paragraphs worth reading first. That distinction matters, and it is written
into the model card that ships with the published model.

## What kind of machine learning problem this is

Three properties define it, and each one drives a design decision later.

**It is classification, not generation.** We are not writing text. We are
answering fixed questions about text.

**It is multi-label, not multi-class.** A contract can contain a liability cap
*and* a license grant *and* an insurance obligation, all at once. In our data,
only 79 of 510 contracts carry exactly one of the six labels; 338 carry two or
more, and 18 carry all six.

> This is why the model has **six independent yes/no outputs** rather than one
> "pick the best category" output. A model forced to pick one would be wrong by
> construction on the majority of contracts. See
> [chapter 2](02-crash-course.md#multi-class-versus-multi-label).

**It is supervised.** We have 510 contracts where lawyers have already marked
where each clause appears. The model learns by copying those judgements. The
annotations come from CUAD ([chapter 3](03-the-dataset.md)).

## The complication that shapes everything

Contracts are long. The models that are good at understanding language are
short-sighted.

```
median contract      5,039 words
longest contract    47,733 words
what BERT can read     ~380 words at a time
```

**493 of 510 contracts (97%) are longer than BERT can read in one go.**

You cannot simply feed a contract to the model. Everything in chapters 4 and 5 —
splitting documents into windows, deciding which window carries which label,
recombining the answers — exists because of those three numbers.

## What a "good" result looks like

The model currently gets **0.663** on a measure called macro F1 at the window
level, and **0.840** when its window-level answers are combined into a verdict
about the whole contract. [Chapter 9](09-evaluation.md) explains what those
numbers mean and why we report both.

For intuition now: 1.0 is perfect, and roughly 0.0 is what you would get by
predicting nothing. The strongest clause type (Insurance, 0.844) is recognised
reliably. The weakest (Non-Compete, 0.374) is missed more often than it is
caught, and [chapter 12](12-results.md) explains why.

## Who did what

The project was split four ways, and each person's work maps to specific files:

| Person | Owns | Where it lives |
|---|---|---|
| A — Data | Cleaning, chunking, splitting | `cuad.py`, `chunking.py`, `splits.py` |
| B — Embeddings | BERT vs Legal-BERT (Problem 1) | `pretrained.py`, `docs/model-selection.md` |
| C — Long documents | Longformer, pooling (Problem 2) | `pretrained.py`, `pooling.py` |
| D — Evaluation | Metrics, interpretability, demo | `metrics.py`, `attribution.py`, `demo.py` |

---

[← Index](README.md) · [Next: Crash course →](02-crash-course.md)
