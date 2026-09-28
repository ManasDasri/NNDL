# 2 · Crash course

[← What this project is](01-what-this-project-is.md) · [Index](README.md) · [Next: The dataset →](03-the-dataset.md)

Everything you need to follow the rest of the handbook. Skip any section you
already know.

## A neural network, honestly

A neural network is a function with adjustable knobs. You feed it an input, it
produces an output, and you compare that output to the right answer. The
difference is the **loss**. The network then nudges every knob slightly in
whichever direction makes the loss smaller. Repeat a few hundred thousand times
and the knobs settle somewhere useful.

- The knobs are called **parameters** or **weights**. Our CNN has about 4 million. BERT has 110 million.
- One pass over all the training data is an **epoch**. We train for 3–12.
- The nudging algorithm is an **optimizer**. We use one called AdamW.
- How big each nudge is, is the **learning rate**. Too big and it never settles; too small and it takes forever.

That is genuinely the whole idea. Everything else is detail about what shape the
function has and what the input looks like.

## Turning text into numbers

A network does arithmetic, so text must become numbers. Two steps.

**Tokenization** splits text into pieces and gives each piece an ID.

```
"shall maintain insurance"  →  ["shall", "maintain", "insurance"]  →  [4618, 5441, 5427]
```

A **token** is usually a word, but not always. Models like BERT use *subword*
tokens, so a rare word gets split into familiar fragments:

```
"indemnification"  →  ["indemn", "##ification"]
```

This matters for us in a very concrete way: legal English is full of rare words,
so it inflates into **roughly 1.45 tokens per word**. That ratio is why a
512-token limit means only about 380 words of contract.

**Embedding** turns each ID into a list of numbers (a *vector*) that captures
meaning. Words used in similar ways end up with similar vectors. "insurance" and
"indemnity" land near each other; "insurance" and "Tuesday" do not.

The crucial point for [chapter 7](07-pretrained-models.md): embeddings are
*learned*, so what they capture depends on what the model read during training.
A model that read Wikipedia learns everyday meanings. A model that read
contracts learns that "consideration" is money, not thoughtfulness.

## Pretraining and fine-tuning

Training a language model from scratch needs enormous text and compute. Nobody
does it for a course project. Instead:

1. **Pretraining** — someone else trains a model on billions of words, learning general language structure. Expensive, done once, published free.
2. **Fine-tuning** — you take that model and train it a bit more on *your* small labelled dataset for *your* specific task. Cheap, minutes to hours.

We fine-tune. "BERT" in this project means "the published BERT weights, then
trained further on our 358 contracts".

## Multi-class versus multi-label

This distinction decides the shape of our model's output.

**Multi-class** — exactly one answer is correct. *Is this animal a cat, dog, or
horse?* Implemented with a **softmax**, which forces the probabilities to sum to
1: more cat necessarily means less dog.

**Multi-label** — any number of answers can be correct at once. *Which of these
six clause types appear in this contract?* Implemented with six independent
**sigmoids**, each producing a probability between 0 and 1 that does not affect
the others.

We are multi-label. A contract with both a liability cap and a license grant is
normal, not a contradiction. Using softmax here would make the model
structurally unable to be right.

## The confusion matrix vocabulary

For one label on one piece of text there are four outcomes:

|  | Model says YES | Model says NO |
|---|---|---|
| **Truly present** | True Positive (TP) | False Negative (FN) — *a miss* |
| **Truly absent** | False Positive (FP) — *a false alarm* | True Negative (TN) |

From these come the only three metrics you need:

**Precision** = TP / (TP + FP) — *when it raises a flag, how often is it right?*
Low precision means it cries wolf.

**Recall** = TP / (TP + FN) — *of the clauses really there, how many did it
find?* Low recall means it misses things. For a legal triage tool, a miss is
usually worse than a false alarm: a lawyer can dismiss a wrong flag in seconds,
but cannot review a clause they were never shown.

**F1** = the harmonic mean of precision and recall. One number, between 0 and 1,
that is only high when *both* are high. It is the headline metric in this
project.

**Macro F1** means: compute F1 separately for each of the six labels, then
average. This weights every clause type equally, including the rare ones — which
is what we want, because a rare clause is not an unimportant clause.

## Why accuracy is banned in this project

**Accuracy** = fraction of predictions that were correct. It sounds like the
obvious metric. It is actively misleading here.

85% of our text windows contain none of the six clauses. A model that simply
answers "no" to everything, always, scores **above 95% accuracy on every label**
while being completely useless.

F1 catches this instantly — a model that never says yes has zero recall, so zero
F1. This is why you will not find accuracy anywhere in our results.

## Thresholds

A sigmoid outputs a probability like 0.73. To turn that into yes/no you need a
cutoff — the **threshold**. The obvious choice is 0.5.

The obvious choice is wrong when classes are imbalanced. Our rare labels appear
in 1.5% of windows, and the model is appropriately hesitant about them, so their
probabilities cluster low. A 0.5 cutoff throws away correct-but-cautious
predictions.

We therefore **tune a separate threshold per label**, choosing whichever value
maximises F1. Our thresholds range from **0.22 to 0.83** — wildly different from
each other and from 0.5. [Chapter 9](09-evaluation.md#thresholds) covers the one
rule that makes this legitimate rather than cheating.

## Train, validation, test

Three separate piles of data:

- **Train** (358 contracts) — the model learns from these.
- **Validation** (76) — used to make decisions *about* the model: when to stop training, what thresholds to use.
- **Test** (76) — touched only once, at the very end, to report a number.

The rule: **never make a decision using the test set.** If you tune anything on
test, your reported score is optimistic and nobody can reproduce it on new data.
We broke this rule once by accident; [chapter 9](09-evaluation.md#the-leak-we-caught)
describes it and what it cost.

---

[← What this project is](01-what-this-project-is.md) · [Index](README.md) · [Next: The dataset →](03-the-dataset.md)
