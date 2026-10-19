"""Checks for the learned aggregator.

Run with `.venv/bin/python tests/test_learned_pooling.py`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_risk_classifier.chunking import Chunk  # noqa: E402
from legal_risk_classifier.labels import LABEL_NAMES, NUM_LABELS  # noqa: E402
from legal_risk_classifier.learned_pooling import (  # noqa: E402
    LearnedAggregator, build_dataset, fit, pool_to_documents, top_k_features,
)


def _chunk(doc_id: str, positive: bool) -> Chunk:
    return Chunk(doc_id=doc_id, index=0, char_start=0, char_end=1, text="x",
                 labels=("Insurance",) if positive else ())


def _corpus(n_docs: int = 40, chunks_per_doc: int = 10, seed: int = 0):
    """Documents where Insurance is present iff two chunks score high."""
    rng = np.random.default_rng(seed)
    chunks, probs = [], []
    for d in range(n_docs):
        positive = d % 2 == 0
        for c in range(chunks_per_doc):
            chunks.append(_chunk(f"doc{d}", positive and c == 0))
            row = rng.uniform(0.0, 0.1, size=NUM_LABELS)
            if positive and c < 2:
                row[LABEL_NAMES.index("Insurance")] = 0.8
            probs.append(row)
    return chunks, np.array(probs, dtype=np.float32)


def test_top_k_features_are_sorted_log_odds() -> None:
    probs = np.array([[0.1], [0.9], [0.5]], dtype=np.float32)
    feats = top_k_features(probs, k=3)
    assert feats.shape == (1, 3)
    assert feats[0, 0] > feats[0, 1] > feats[0, 2], "strongest chunk first"
    assert abs(feats[0, 1] - 0.0) < 1e-4, "logit(0.5) is 0"


def test_padding_reads_as_absence_not_uncertainty() -> None:
    """A one-chunk document must not be padded with logit(0.5) = 0."""
    feats = top_k_features(np.array([[0.9]], dtype=np.float32), k=4)
    assert feats.shape == (1, 4)
    assert all(v < 0 for v in feats[0, 1:]), "padding is negative evidence"


def test_empty_document_is_handled() -> None:
    feats = top_k_features(np.zeros((0, NUM_LABELS), dtype=np.float32), k=3)
    assert feats.shape == (NUM_LABELS, 3)
    assert np.all(feats < 0)


def test_aggregator_can_represent_max() -> None:
    """w = (1, 0, ...) with zero bias reproduces max-pooling exactly."""
    weights = np.zeros(4, dtype=np.float32)
    weights[0] = 1.0
    agg = LearnedAggregator(weights, 0.0, top_k=4)
    probs = np.array([[0.3], [0.77], [0.1]], dtype=np.float32)
    assert abs(float(agg(probs)[0]) - 0.77) < 1e-4


def test_aggregator_output_is_a_probability() -> None:
    rng = np.random.default_rng(1)
    agg = LearnedAggregator(rng.normal(size=5).astype(np.float32), -0.5, top_k=5)
    out = agg(rng.uniform(0, 1, size=(12, NUM_LABELS)).astype(np.float32))
    assert out.shape == (NUM_LABELS,)
    assert np.all((out >= 0) & (out <= 1))


def test_build_dataset_shapes_and_targets() -> None:
    chunks, probs = _corpus(n_docs=6, chunks_per_doc=4)
    X, y, doc_ids = build_dataset(chunks, probs, top_k=3)
    assert len(doc_ids) == 6
    assert X.shape == (6 * NUM_LABELS, 3)
    assert y.shape == (6 * NUM_LABELS,)
    assert y.sum() == 3, "three of six documents carry Insurance"


def test_fit_learns_to_separate_a_clean_signal() -> None:
    chunks, probs = _corpus()
    agg = fit(chunks, probs, top_k=3, l2=1.0)
    _, truth, pooled = pool_to_documents(chunks, probs, agg)
    index = LABEL_NAMES.index("Insurance")
    positives = pooled[truth[:, index] == 1, index]
    negatives = pooled[truth[:, index] == 0, index]
    assert positives.min() > negatives.max(), "a separable signal must separate"


def test_fit_survives_a_degenerate_split() -> None:
    """All-negative targets must not crash the logistic fit."""
    chunks = [_chunk("d", False) for _ in range(8)]
    probs = np.full((8, NUM_LABELS), 0.1, dtype=np.float32)
    agg = fit(chunks, probs, top_k=3)
    assert agg.weights.shape == (3,)
    assert abs(float(agg.weights[0]) - 1.0) < 1e-6, "falls back to max"


def test_round_trips_through_json() -> None:
    agg = LearnedAggregator(np.array([0.5, -0.2, 0.1], dtype=np.float32), -0.3, 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agg.json"
        agg.save(path)
        back = LearnedAggregator.load(path)
    assert np.allclose(back.weights, agg.weights)
    assert back.bias == agg.bias and back.top_k == agg.top_k


def test_mismatched_lengths_are_rejected() -> None:
    agg = LearnedAggregator(np.ones(3, dtype=np.float32), 0.0, 3)
    try:
        pool_to_documents([_chunk("d", False)] * 3,
                          np.zeros((2, NUM_LABELS), dtype=np.float32), agg)
    except ValueError:
        return
    raise AssertionError("expected ValueError on a length mismatch")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
