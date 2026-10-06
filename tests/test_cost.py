"""Tests for the cost function and threshold selection."""

import math

import numpy as np

from fraudmon import config as C
from fraudmon import cost

CM = C.CostModel(fn_multiplier=1.0, fn_fixed=20.0, fp_cost=5.0, tp_cost=1.0)


def test_evaluate_matches_a_hand_worked_example():
    y = np.array([1, 1, 0, 0])
    amount = np.array([100.0, 10.0, 50.0, 20.0])
    flagged = np.array([True, False, True, False])
    out = cost.evaluate(y, amount, flagged, CM)
    # caught fraud: 1 review; missed fraud: 10 + 20; blocked legitimate payment: 5
    assert (out["tp"], out["fn"], out["fp"]) == (1, 1, 1)
    assert math.isclose(out["cost"], 1.0 + 30.0 + 5.0)
    assert math.isclose(out["no_model_cost"], 120.0 + 30.0)
    assert math.isclose(out["saving"], 1 - 36.0 / 150.0)
    assert math.isclose(out["cost_per_1k"], 1000 * 36.0 / 4)


def test_sweep_agrees_with_evaluate_at_every_threshold():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    scores = np.round(rng.random(400) * 0.6 + 0.3 * y, 2)  # rounding creates tied scores
    amount = rng.lognormal(3.0, 1.0, 400)
    table = cost.sweep(y, scores, amount, CM)
    for row in table.itertuples():
        direct = cost.evaluate(y, amount, scores >= row.threshold, CM)
        assert math.isclose(row.cost, direct["cost"])
        assert (row.tp, row.fp) == (direct["tp"], direct["fp"])


def test_sweep_starts_by_flagging_nothing():
    y = np.array([1, 0, 0])
    first = cost.sweep(y, np.array([0.9, 0.2, 0.1]), np.array([50.0, 5.0, 5.0]), CM).iloc[0]
    assert (first["tp"], first["fp"]) == (0, 0)
    assert math.isclose(first["cost"], 70.0)


def test_best_threshold_separates_a_clean_ranking():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    threshold = cost.best_threshold(y, scores, np.full(5, 100.0), CM)
    assert 0.3 < threshold < 0.8


def test_a_higher_false_positive_cost_never_lowers_the_threshold():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    scores = np.clip(rng.normal(0.2 + 0.4 * y, 0.15), 0, 1)
    amount = rng.lognormal(3.0, 1.0, 2000)
    thresholds = [
        cost.best_threshold(y, scores, amount, C.CostModel(fp_cost=fp)) for fp in (1.0, 5.0, 25.0, 100.0)
    ]
    assert thresholds == sorted(thresholds)


def test_amount_aware_rule_flags_large_payments_at_lower_probability():
    prob = np.array([0.01, 0.01, 0.5])
    amount = np.array([100.0, 1000.0, 1.0])
    # p=0.01: expected loss 1.2 vs 10.2 against a blocking cost of 4.96
    assert cost.amount_aware_flags(prob, amount, CM).tolist() == [False, True, True]


def test_max_f1_threshold_on_a_separable_sample():
    y = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.7, 0.4, 0.3, 0.2, 0.1])
    assert 0.4 < cost.max_f1_threshold(y, scores) < 0.7
