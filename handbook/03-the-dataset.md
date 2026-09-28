# 3 · The dataset

[← Crash course](02-crash-course.md) · [Index](README.md) · [Next: The two hard problems →](04-the-two-problems.md)

## What CUAD is

The **Contract Understanding Atticus Dataset** (CUAD v1) is 510 real commercial
contracts, filed publicly with the US Securities and Exchange Commission, in
which lawyers at The Atticus Project marked every occurrence of 41 clause
categories. It took them roughly 9,000 hours.

It is free, it is the standard benchmark for this task, and it is the reason
this project is possible at all — hand-labelling contracts ourselves was never
on the table.

The whole corpus lives in `data/` in this repository.

## What one contract actually looks like

CUAD is stored in a format borrowed from question-answering datasets, which is
initially confusing. One contract looks like this:

```json
{
  "title": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
  "paragraphs": [{
    "context": "DISTRIBUTOR AGREEMENT\n\nTHIS AGREEMENT made ...",
    "qas": [
      {
        "id": "LIMEENERGYCO_..._AGREEMENT__Insurance",
        "question": "Highlight the parts (if any) of this contract related to \"Insurance\" ...",
        "answers": [{"text": "Company will carry a reasonable amount of product liability insurance ...",
                     "answer_start": 41736}]
      }
    ]
  }]
}
```

Three things to notice, because each one caused a design decision:

**`context` is the entire contract.** Not a paragraph. The field is called
"paragraphs" but each contract has exactly one, holding all 54,290 characters.
Anything that treats a "paragraph" as a small unit of text is wrong.

**The category is in the `id`, not the question.** The `id` ends with
`__Insurance`. The `question` is a long natural-language prompt that merely
*mentions* the category inside quote marks. Reading the category off the
question text is the mistake that broke the first version of this project
([chapter 5](05-data-pipeline.md#the-bug-that-made-everything-meaningless)).

**`answer_start` is a character offset.** This is the single most important field
in the dataset. It says the Insurance clause starts at character 41,736 of this
contract. **Labels are locations, not document properties.** Keeping those
offsets is what makes it possible to know *which part* of a contract carries
which clause.

## The numbers that matter

All produced by `legal-risk-analyze`, which regenerates
[`docs/dataset-report.md`](../docs/dataset-report.md) from the data.

### Length

| Statistic | Words |
|---|---:|
| Shortest contract | 109 |
| Median | 5,039 |
| Mean | 7,861 |
| 95th percentile | 25,298 |
| Longest | 47,733 |

**493 of 510 (97%) exceed BERT's ~380-word window. 358 (70%) exceed
Longformer's ~2,800-word window.** Chapter 4 is entirely about this.

### Our six labels

We kept 6 of CUAD's 41 categories — commercially material, and frequent enough
to learn from. At document level:

| Label | Contracts containing it | Annotated spans |
|---|---:|---:|
| Cap on Liability | 275 | 662 |
| License Grant | 255 | 772 |
| Audit Rights | 214 | 639 |
| Termination for Convenience | 183 | 245 |
| Insurance | 166 | 554 |
| Non-Compete | 119 | 251 |

### Labels overlap heavily

How many of the six labels a single contract carries:

| Labels in one contract | Contracts |
|---:|---:|
| 0 | 93 |
| 1 | 79 |
| 2 | 109 |
| 3 | 86 |
| 4 | 76 |
| 5 | 49 |
| 6 | 18 |

Only 79 contracts carry exactly one. Of the 275 with a liability cap, 182 also
have a license grant. **This is the evidence for multi-label**
([chapter 2](02-crash-course.md#multi-class-versus-multi-label)) — not a
preference, a property of the data.

## The imbalance, and a correction to our own brief

Here is where the honest number differs from the flattering one.

Our project brief said the classes were *"reasonably balanced (12–20% each)"*
and concluded this *"simplifies the class-imbalance handling"*. That is true
only if you count clause occurrences **against each other**.

A classifier does not see clauses against each other. It sees every window of
every contract and must decide, for each, whether a clause is present. Against
that denominator:

| Label | Positive windows | Rate | Loss weight needed |
|---|---:|---:|---:|
| License Grant | 851 | 4.24% | 23× |
| Cap on Liability | 716 | 3.57% | 27× |
| Audit Rights | 592 | 2.95% | 33× |
| Insurance | 400 | 2.00% | 49× |
| Termination for Convenience | 302 | 1.51% | 65× |
| Non-Compete | 301 | 1.50% | 66× |

**85% of the 20,048 windows carry no label at all.**

So the imbalance is severe, not mild, and three consequences follow that shape
the rest of the project:

1. Accuracy is meaningless — an all-negative model scores above 95%.
2. An unweighted loss collapses to predicting nothing ([chapter 8](08-training.md#the-imbalance-problem)).
3. A 0.5 threshold is arbitrary ([chapter 9](09-evaluation.md#thresholds)).

This is the most valuable thing the dataset analysis found, and it is why
chapter 8 looks the way it does.

## Other challenges in the data

**The text is OCR-noisy.** These are scanned filings. Expect irregular
whitespace, page numbers mid-sentence, and broken line wrapping:

```
Company will carry a reasonable                   amount of product  liability
insurance  through a  reasonably                   acceptable  products
```

Our tokenizer splits on whitespace runs, so this is survivable, but it is one
argument for a model pretrained on legal documents rather than clean Wikipedia.

**Annotations are spans, supervision is windows.** Lawyers marked character
ranges. Our model classifies windows of text. Converting between the two
requires a rule, and that rule is a modelling choice we had to make and defend
([chapter 5](05-data-pipeline.md#the-overlap-rule)).

**One labelling question is still open.** CUAD has a separate
`No-Solicit Of Employees` category. An earlier version of our code folded it
into Non-Compete, which is why our brief says 257 Non-Compete clauses while we
count 119 contracts. We chose *not* to merge them, because they are legally
distinct. It is a defensible decision either way, but it should be a decision
rather than an accident.

---

[← Crash course](02-crash-course.md) · [Index](README.md) · [Next: The two hard problems →](04-the-two-problems.md)
