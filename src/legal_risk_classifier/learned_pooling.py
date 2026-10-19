"""A learned replacement for the hand-chosen aggregation rule.

Every rule in `pooling.py` is a fixed function of a document's chunk
probabilities. Choosing between them means arguing about which fixed function
suits the task. This module learns the function instead.

The parameterisation is deliberately tiny. A document's chunk probabilities for
one label are sorted descending, the strongest K are converted to log-odds, and
the document score is a weighted sum:

    score = b + sum_k w_k * logit(p_(k))

That form already contains the rules we compare against:

    max            w = (1, 0, 0, ...)
    top-k mean     w = (1/k, ..., 1/k, 0, ...)
    noisy-OR       a positive, roughly equal weighting of log(1 - p)

so the fit cannot do worse than the best fixed rule except by overfitting, and
with K + 1 shared parameters there is very little to overfit with. That matters:
selecting among fixed rules on 76 validation documents already overfits
(docs/pooling.md), so a learned rule has to be small enough to resist the same
trap.

Weights are shared across labels. Per-label behaviour is carried by the
thresholds, which are tuned separately anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .labels import LABEL_TO_INDEX, NUM_LABELS

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def top_k_features(probs: np.ndarray, k: int) -> np.ndarray:
    """(n_chunks, n_labels) -> (n_labels, k) log-odds of the k strongest chunks.

    Documents with fewer than k chunks are padded with a near-zero probability,
    so a missing chunk reads as "no further evidence". Repeating the last real
    value instead would fabricate corroboration: a single chunk at 0.9 would
    present as four chunks at 0.9, which under any evidence-combining weighting
    is a much stronger claim than the document supports.
    """
    if probs.shape[0] == 0:
        return np.full((NUM_LABELS, k), float(_logit(np.array(_EPS))), dtype=np.float32)
    ordered = np.sort(probs, axis=0)[::-1][:k]                 # (<=k, labels)
    if ordered.shape[0] < k:
        pad = np.full((k - ordered.shape[0], ordered.shape[1]), _EPS, dtype=ordered.dtype)
        ordered = np.concatenate([ordered, pad], axis=0)
    return _logit(ordered.T.astype(np.float64)).astype(np.float32)


@dataclass
class LearnedAggregator:
    """Shared weights over the K strongest chunks, plus a bias."""

    weights: np.ndarray          # (k,)
    bias: float
    top_k: int

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        features = top_k_features(probs, self.top_k)           # (labels, k)
        scores = features @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-scores))

    def to_dict(self) -> dict:
        return {"weights": self.weights.tolist(), "bias": float(self.bias), "top_k": self.top_k}

    @classmethod
    def from_dict(cls, payload: dict) -> "LearnedAggregator":
        return cls(np.asarray(payload["weights"], dtype=np.float32),
                   float(payload["bias"]), int(payload["top_k"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LearnedAggregator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def describe(self) -> str:
        shape = ", ".join(f"{w:+.2f}" for w in self.weights)
        return f"top_k={self.top_k} weights=({shape}) bias={self.bias:+.2f}"


def _group(chunks: list[Chunk]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for row, chunk in enumerate(chunks):
        grouped.setdefault(chunk.doc_id, []).append(row)
    return grouped


def build_dataset(
    chunks: list[Chunk], chunk_probs: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One training row per (document, label): its K strongest chunk log-odds."""
    grouped = _group(chunks)
    doc_ids = sorted(grouped)
    features, targets = [], []
    for doc_id in doc_ids:
        rows = grouped[doc_id]
        per_label = top_k_features(chunk_probs[rows], top_k)   # (labels, k)
        truth = np.zeros(NUM_LABELS, dtype=np.float32)
        for row in rows:
            for label in chunks[row].labels:
                truth[LABEL_TO_INDEX[label]] = 1.0
        features.append(per_label)
        targets.append(truth)
    return (np.concatenate(features, axis=0),
            np.concatenate(targets, axis=0),
            doc_ids)


def fit(
    chunks: list[Chunk],
    chunk_probs: np.ndarray,
    top_k: int = 5,
    l2: float = 1.0,
    seed: int = 42,
) -> LearnedAggregator:
    """Fit the aggregator by logistic regression on pooled (document, label) rows.

    `l2` is deliberately strong. The fit sees roughly 76 documents times six
    labels, and the whole point of keeping this model tiny is that the same
    corpus already proved large enough to overfit rule selection.
    """
    from sklearn.linear_model import LogisticRegression

    X, y, _ = build_dataset(chunks, chunk_probs, top_k)
    if y.sum() == 0 or y.sum() == len(y):
        # Degenerate split: fall back to max, which is what w=(1,0,...) means.
        weights = np.zeros(top_k, dtype=np.float32)
        weights[0] = 1.0
        return LearnedAggregator(weights, 0.0, top_k)

    model = LogisticRegression(
        C=1.0 / max(l2, _EPS), max_iter=2000, random_state=seed
    ).fit(X, y)
    return LearnedAggregator(
        model.coef_[0].astype(np.float32), float(model.intercept_[0]), top_k
    )


def pool_to_documents(
    chunks: list[Chunk], chunk_probs: np.ndarray, aggregator: LearnedAggregator
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Same contract as pooling.pool_to_documents, with a fitted aggregator."""
    if len(chunks) != len(chunk_probs):
        raise ValueError(f"{len(chunks)} chunks but {len(chunk_probs)} predictions")

    grouped = _group(chunks)
    doc_ids = sorted(grouped)
    doc_prob = np.zeros((len(doc_ids), NUM_LABELS), dtype=np.float32)
    doc_true = np.zeros((len(doc_ids), NUM_LABELS), dtype=np.float32)

    for i, doc_id in enumerate(doc_ids):
        rows = grouped[doc_id]
        doc_prob[i] = aggregator(chunk_probs[rows])
        for row in rows:
            for label in chunks[row].labels:
                doc_true[i, LABEL_TO_INDEX[label]] = 1.0

    return doc_ids, doc_true, doc_prob
