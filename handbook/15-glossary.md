# 15 · Glossary

[← Decisions](14-decisions.md) · [Index](README.md)

Every term this handbook uses, in one line each.

## Machine learning

**Accuracy** — fraction of predictions correct. Useless here: an all-negative model scores >95%. Never reported in this project.

**Attention** — mechanism letting every token look at every other token to decide which are relevant. What makes transformers strong, and why they cost O(n²).

**Average precision (AP)** — performance summarised across all thresholds at once. Threshold-free, so it is the stable number when thresholds are in doubt.

**Backpropagation** — computing how much each weight contributed to the error, so the optimizer knows which way to nudge it.

**BCE (binary cross-entropy)** — loss treating each output as an independent yes/no question. The multi-label choice.

**Batch** — group of examples processed together before updating weights.

**Convolution** — sliding window that looks for one pattern at every position. On text, a learned n-gram detector.

**Dropout** — randomly zeroing some features during training so the model cannot lean on any single one.

**Early stopping** — halting when validation stops improving, and restoring the best checkpoint rather than the last.

**Embedding** — a vector of numbers representing a token's meaning. Learned, so it reflects whatever text the model was pretrained on.

**Epoch** — one full pass over the training data.

**F1** — harmonic mean of precision and recall. High only when both are high.

**Fine-tuning** — taking a pretrained model and training it further on your task.

**Gradient accumulation** — summing gradients over several small batches before updating, to simulate a large batch in limited memory.

**Gradient clipping** — capping gradient magnitude so one bad batch cannot wreck the weights.

**Learning rate** — how big each weight nudge is. 1e-3 for our CNN, 2e-5 for fine-tuning.

**Logit** — raw model output before a sigmoid converts it to a probability.

**Loss** — number measuring how wrong the model is. Training minimises it.

**Macro average** — average the metric per class, then mean. Weights every class equally, including rare ones.

**Micro average** — pool all predictions, then compute once. Dominated by frequent classes.

**Multi-class** — exactly one label correct. Uses softmax. **Not this project.**

**Multi-label** — any number of labels correct simultaneously. Uses sigmoids. **This project.**

**Optimizer** — algorithm that updates weights. We use AdamW.

**Overfitting** — memorising training data instead of learning generalisable patterns.

**Parameters / weights** — the adjustable numbers. CNN ~4M, BERT ~110M.

**Pooling** — reducing many values to fewer. *Max-over-time* keeps the strongest response across positions.

**pos_weight** — multiplier making positive examples count more in the loss. Ours run 23×–66×.

**Precision** — of the things flagged, how many were right. Low precision = crying wolf.

**Pretraining** — training on huge general text before any task-specific work.

**Recall** — of the things really there, how many were found. Low recall = missing things.

**Sigmoid** — squashes any number to 0–1. Six independent ones give six independent probabilities.

**Softmax** — makes probabilities sum to 1, forcing a single choice. Wrong for this project.

**Support** — how many true positives of a label exist in the evaluation set. Context for how much to trust a score.

**Threshold** — the cutoff converting a probability to yes/no. Tuned per label here, 0.22–0.83.

**Token** — the unit a model reads. Roughly a word, sometimes a word fragment. Legal English runs ~1.45 tokens per word.

**Tokenization** — splitting text into tokens and mapping them to IDs.

**Transformer** — architecture built on attention. BERT, Legal-BERT and Longformer are all transformers.

**Validation set** — held-out data used to make decisions *about* the model. Never used to report final scores.

**Test set** — held-out data touched once, at the end, to report a number. Never used for any decision.

## This project

**Chunk / window** — a 300-word slice of a contract. The unit the model classifies. 20,048 of them.

**CUAD** — Contract Understanding Atticus Dataset. 510 contracts, lawyer-annotated, 41 categories, ~9,000 hours of work.

**Document level** — scoring whole contracts after pooling window predictions. The product question.

**Window level** — scoring individual windows. The harder, more honest measure.

**Global attention** — Longformer marking `[CLS]` so it attends to everything, since local attention alone cannot pool a window.

**Overlap ratio** — our rule for assigning a label to a window: ≥0.5 of whichever range is shorter.

**Problem 1** — does legal-domain pretraining help? BERT vs Legal-BERT.

**Problem 2** — does longer context help? BERT vs Longformer.

**Span** — a character range in a contract where a lawyer marked a clause. `(41736, 42217)`. The thing v1 threw away.

**`[CLS]`** — special first token whose representation the classifier reads as a summary.

## The six labels

**Audit Rights** — the other party may inspect your books and records.

**Cap on Liability** — a ceiling on damages recoverable.

**Insurance** — obligation to carry and maintain cover.

**License Grant** — intellectual property rights granted.

**Non-Compete** — restriction on competing in a market or with a customer.

**Termination for Convenience** — a party may exit with notice and no cause.

---

[← Decisions](14-decisions.md) · [Index](README.md)
