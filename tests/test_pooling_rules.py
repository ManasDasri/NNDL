"""Checks for the aggregation rules that combine chunk predictions.

Run with `.venv/bin/python tests/test_pooling_rules.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_risk_classifier.pooling import METHODS, POOLING_GRID, aggregate  # noqa: E402


def _column(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(-1, 1)


def test_max_cannot_see_corroboration() -> None:
    """The defect that motivated this module, stated as a test.

    Two chunks at 0.80 and 0.79 is stronger evidence than one at 0.80 beside
    noise. Max reports the two cases as identical.
    """
    corroborated = _column([0.80, 0.79] + [0.02] * 28)
    single = _column([0.80, 0.02] + [0.02] * 28)
    assert aggregate(corroborated, "max")[0] == aggregate(single, "max")[0]


def test_every_other_rule_separates_those_cases() -> None:
    corroborated = _column([0.80, 0.79] + [0.02] * 28)
    single = _column([0.80, 0.02] + [0.02] * 28)
    for method in METHODS:
        if method == "max":
            continue
        for params in POOLING_GRID[method]:
            a = aggregate(corroborated, method, **params)[0]
            b = aggregate(single, method, **params)[0]
            assert a > b, f"{method} {params} did not reward corroboration"


def test_noisy_or_accumulates_independent_evidence() -> None:
    """1 - (1-0.8)(1-0.79) = 0.958, above either chunk alone."""
    pooled = aggregate(_column([0.80, 0.79]), "noisy_or")[0]
    assert abs(pooled - (1 - 0.20 * 0.21)) < 1e-4
    assert pooled > 0.80, "combined evidence must exceed its strongest part"


def test_every_rule_returns_a_probability() -> None:
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, size=(40, 6)).astype(np.float32)
    for method in METHODS:
        for params in POOLING_GRID[method]:
            out = aggregate(probs, method, **params)
            assert out.shape == (6,)
            assert np.all((out >= 0) & (out <= 1)), f"{method} {params} left [0,1]"


def test_power_and_logsumexp_interpolate_between_mean_and_max() -> None:
    probs = _column([0.9, 0.5, 0.3, 0.1])
    mean, mx = float(probs.mean()), float(probs.max())
    gentle = aggregate(probs, "power", power=1.001)[0]
    assert abs(gentle - mean) < 0.01, "power=1 is the arithmetic mean"

    # The power mean approaches max as max * (1/n)^(1/r), which is slow: with
    # four chunks, r=60 still reads 0.879 against a true max of 0.9. Assert the
    # monotone approach rather than a tolerance that only holds for huge r.
    ladder = [aggregate(probs, "power", power=r)[0] for r in (2, 6, 20, 60, 200)]
    assert ladder == sorted(ladder), "higher power must move toward max"
    assert ladder[-1] < mx + 1e-6, "the power mean never exceeds max"
    assert abs(ladder[-1] - mx) < 0.02, "power=200 is close to max"

    warm = aggregate(probs, "logsumexp", temperature=50)[0]
    cold = aggregate(probs, "logsumexp", temperature=0.005)[0]
    assert abs(warm - mean) < 0.02, "high temperature approaches the mean"
    assert abs(cold - mx) < 0.01, "low temperature approaches max"


def test_noisy_or_saturates_with_many_weak_chunks() -> None:
    """The known failure mode, recorded rather than hidden.

    Sixty chunks at 0.10 each contain no clause, but plain noisy-OR reads them
    as near-certainty. noisy_or_topk exists for exactly this reason.
    """
    weak = _column([0.10] * 60)
    assert aggregate(weak, "noisy_or")[0] > 0.99, "plain noisy-OR saturates"
    assert aggregate(weak, "noisy_or_topk", top_k=3)[0] < 0.3, "top-k bounds it"


def test_top_k_falls_back_for_short_documents() -> None:
    probs = _column([0.4, 0.6])
    assert abs(aggregate(probs, "top_k", top_k=10)[0] - 0.5) < 1e-6


def test_empty_and_malformed_input() -> None:
    assert aggregate(np.zeros((0, 6), dtype=np.float32), "max").shape == (6,)
    for bad in ("median", "softmax", ""):
        try:
            aggregate(_column([0.5]), bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")
    try:
        aggregate(np.zeros(6, dtype=np.float32), "max")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a 1-D input")


def test_every_method_has_a_calibration_grid() -> None:
    assert set(POOLING_GRID) == set(METHODS)
    for method, grid in POOLING_GRID.items():
        assert grid, f"{method} has no candidates"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
