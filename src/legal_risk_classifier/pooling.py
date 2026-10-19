"""Combine chunk-level predictions into a contract-level verdict.

Classification happens per chunk, but the question is about the contract. How
those levels connect is a modelling decision, and the obvious choice is wrong
in a specific way.

Max-pooling keeps the single strongest chunk and discards everything else. A
contract with 0.80 in one chunk and 0.79 in another scores exactly the same as
one with 0.80 and 0.02 — yet the first has two independent pieces of evidence
and the second has one. That corroboration is real information and max throws
it away.

This module therefore offers three families:

  order statistics   max, top_k        — look at the strongest chunk(s) only
  smooth averages    power, logsumexp  — use every chunk, weighted toward strong ones
  evidence combiners noisy_or          — accumulate independent evidence

Every rule is calibrated on validation and compared on the same predictions,
so the default is chosen by measurement rather than by argument.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .chunking import Chunk
from .labels import LABEL_TO_INDEX, NUM_LABELS

# Probabilities are clipped before any log, so a confident 1.0 cannot produce
# an infinity and a 0.0 cannot produce a negative infinity.
_EPS = 1e-6

METHODS = ("max", "mean", "top_k", "power", "logsumexp", "noisy_or", "noisy_or_topk")

# Candidate constants per rule, searched on validation. Kept beside the rules
# so a new rule cannot be added without saying how it is calibrated.
POOLING_GRID: dict[str, list[dict]] = {
    "max": [{}],
    "mean": [{}],
    "top_k": [{"top_k": k} for k in (2, 3, 5, 8)],
    "power": [{"power": r} for r in (2, 3, 4, 6, 8, 12)],
    "logsumexp": [{"temperature": t} for t in (0.03, 0.05, 0.1, 0.2, 0.4)],
    "noisy_or": [{}],
    "noisy_or_topk": [{"top_k": k} for k in (2, 3, 5, 8, 12)],
}


def _clip(probs: np.ndarray) -> np.ndarray:
    return np.clip(probs, _EPS, 1.0 - _EPS)


def aggregate(probs: np.ndarray, method: str = "max", **params) -> np.ndarray:
    """Reduce (n_chunks, n_labels) probabilities to (n_labels,).

    `params` carries the rule's calibration constant, tuned on validation:
      top_k, noisy_or_topk -> top_k   (how many chunks to consider)
      power                -> power   (1 = mean, large = max)
      logsumexp            -> temperature (large = mean, small = max)
    """
    if probs.ndim != 2:
        raise ValueError(f"expected (chunks, labels), got shape {probs.shape}")
    if probs.shape[0] == 0:
        return np.zeros(probs.shape[1], dtype=np.float32)

    if method == "max":
        return probs.max(axis=0)

    if method == "mean":
        return probs.mean(axis=0)

    if method == "top_k":
        k = min(int(params.get("top_k", 3)), probs.shape[0])
        return np.sort(probs, axis=0)[-k:].mean(axis=0)

    if method == "power":
        # Generalised mean. r=1 is the arithmetic mean, r->inf approaches max.
        # Every chunk contributes, but strong ones dominate as r grows.
        r = float(params.get("power", 4.0))
        return np.power(np.mean(np.power(_clip(probs), r), axis=0), 1.0 / r)

    if method == "logsumexp":
        # Smooth maximum. T->0 approaches max, large T approaches the mean.
        # Bounded above by the true max, so it stays a probability.
        t = float(params.get("temperature", 0.1))
        scaled = probs / t
        peak = scaled.max(axis=0, keepdims=True)
        return (t * (peak + np.log(np.mean(np.exp(scaled - peak), axis=0, keepdims=True)))).ravel()

    if method in ("noisy_or", "noisy_or_topk"):
        # Treat each chunk as independent evidence that the clause is present:
        # P(present) = 1 - P(no chunk contains it). Two chunks at 0.80 and 0.79
        # combine to 0.958, which is the corroboration max cannot express.
        selected = probs
        if method == "noisy_or_topk":
            k = min(int(params.get("top_k", 5)), probs.shape[0])
            selected = np.sort(probs, axis=0)[-k:]
        return 1.0 - np.exp(np.sum(np.log1p(-_clip(selected)), axis=0))

    raise ValueError(f"unknown pooling method: {method!r}")


def pool_to_documents(
    chunks: list[Chunk],
    chunk_probs: np.ndarray,
    method: str = "max",
    **params,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (doc_ids, doc_true, doc_prob), all aligned by row."""
    if len(chunks) != len(chunk_probs):
        raise ValueError(f"{len(chunks)} chunks but {len(chunk_probs)} predictions")
    if method not in METHODS:
        raise ValueError(f"unknown pooling method: {method!r}")

    grouped: dict[str, list[int]] = defaultdict(list)
    for row, chunk in enumerate(chunks):
        grouped[chunk.doc_id].append(row)

    doc_ids = sorted(grouped)
    doc_prob = np.zeros((len(doc_ids), NUM_LABELS), dtype=np.float32)
    doc_true = np.zeros((len(doc_ids), NUM_LABELS), dtype=np.float32)

    for i, doc_id in enumerate(doc_ids):
        rows = grouped[doc_id]
        doc_prob[i] = aggregate(chunk_probs[rows], method=method, **params)
        for row in rows:
            for label in chunks[row].labels:
                doc_true[i, LABEL_TO_INDEX[label]] = 1.0

    return doc_ids, doc_true, doc_prob
