# The NNDL Handbook

Everything about this project, in order, assuming you know nothing about it.

If you have never trained a neural network, start at chapter 1 and read straight
through. Chapter 2 teaches the vocabulary the rest of the handbook uses, so you
do not need any machine learning background before starting.

If you already know the machine learning and just want this project's decisions,
skip to [chapter 14](14-decisions.md), which lists every choice and the reason
behind it on one page.

## The short version

We built a system that reads a commercial contract and flags six kinds of
clause that a lawyer would want to look at. It is a **neural network
classification project**: the input is contract text, the output is six
yes/no answers.

The interesting part is not the classifier. It is that contracts are enormous —
a median of 5,039 words, up to 47,733 — while the models everyone uses can only
read about 380 words at a time. Most of this project is about that mismatch.

## Read in this order

| # | Chapter | What it answers |
|---|---|---|
| 1 | [What this project is](01-what-this-project-is.md) | What problem are we solving, and for whom? |
| 2 | [Crash course](02-crash-course.md) | What is a neural network, a token, an embedding, an F1 score? |
| 3 | [The dataset](03-the-dataset.md) | What is CUAD and what is actually inside it? |
| 4 | [The two hard problems](04-the-two-problems.md) | The two questions our teacher asked us to answer. |
| 5 | [The data pipeline](05-data-pipeline.md) | How raw contracts become training examples. |
| 6 | [The CNN baseline](06-cnn-baseline.md) | What a convolutional network is and why it works on text. |
| 7 | [The pretrained models](07-pretrained-models.md) | BERT, Legal-BERT, Longformer, and why exactly these three. |
| 8 | [Training](08-training.md) | How the model actually learns, and the imbalance problem. |
| 9 | [Evaluation](09-evaluation.md) | How we measure success, and why accuracy is a lie here. |
| 10 | [Interpretability and the demo](10-interpretability-and-demo.md) | How we see *why* the model said something. |
| 11 | [Deployment](11-deployment.md) | Where everything is published and how. |
| 12 | [Results so far](12-results.md) | What we have measured, and what is still pending. |
| 13 | [Run it yourself](13-run-it-yourself.md) | Copy-paste commands, from nothing to trained model. |
| 14 | [Decisions](14-decisions.md) | Every design choice and its justification, on one page. |
| 15 | [Glossary](15-glossary.md) | Every term, defined in one line. |

## A promise about honesty

This handbook reports what we actually measured, including the parts that are
unfinished and the mistakes we made along the way. Three of the most useful
sections are about bugs: the labelling bug that made the first version of this
project silently meaningless ([chapter 5](05-data-pipeline.md)), the evaluation
bug that inflated our own headline number ([chapter 9](09-evaluation.md)), and
the reason a small difference between two models may mean nothing at all
([chapter 12](12-results.md)).

If a number here disagrees with a number elsewhere in the repository, the
generated reports in `docs/` win — those are produced directly from the data on
every run, while this handbook is written by hand.
