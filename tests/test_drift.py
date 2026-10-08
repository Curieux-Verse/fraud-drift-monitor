"""Tests for the drift statistics."""

import math

import numpy as np
import pandas as pd

from fraudmon import drift


def test_psi_is_near_zero_for_two_samples_of_one_distribution():
    rng = np.random.default_rng(0)
    assert drift.psi(rng.normal(size=20000), rng.normal(size=20000)) < 0.01


def test_psi_is_large_for_a_one_sd_shift():
    rng = np.random.default_rng(0)
    assert drift.psi(rng.normal(size=20000), rng.normal(size=20000) + 1.0) > 0.25


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20000)
    values = [drift.psi(ref, rng.normal(size=20000) + shift) for shift in (0.0, 0.25, 0.5, 1.0)]
    assert values == sorted(values)


def test_psi_handles_a_constant_reference():
    ref = np.full(1000, 3.0)
    assert drift.psi(ref, ref) == 0.0
    assert drift.psi(ref, np.full(1000, 2.0)) > 1.0


def test_weighted_share_counts_only_importance_on_drifted_features():
    psi_values = pd.Series({"a": 0.30, "b": 0.05, "c": 0.40})
    importance = pd.Series({"a": 0.10, "b": 0.85, "c": 0.05})
    assert math.isclose(drift.weighted_share(psi_values, importance, 0.25), 0.15)


def test_weighted_share_treats_features_without_importance_as_zero():
    psi_values = pd.Series({"a": 0.30, "unused": 0.90})
    assert math.isclose(drift.weighted_share(psi_values, pd.Series({"a": 1.0}), 0.25), 1.0)


def test_score_psi_detects_a_shift_in_model_output():
    rng = np.random.default_rng(0)
    scores = rng.beta(1, 50, 20000)
    assert drift.score_psi(scores, rng.beta(1, 50, 20000)) < 0.01
    assert drift.score_psi(scores, rng.beta(1, 10, 20000)) > 0.25


def test_exceed_ratio_compares_tail_shares():
    ref = np.array([0.1] * 98 + [0.9] * 2)
    cur = np.array([0.1] * 94 + [0.9] * 6)
    assert math.isclose(drift.exceed_ratio(ref, cur, 0.5), 3.0)


def test_exceed_ratio_when_the_reference_has_nothing_above_the_cut():
    cur = np.array([0.1] * 94 + [0.9] * 6)
    assert drift.exceed_ratio(np.zeros(10), cur, 0.5) == math.inf  # alerts where there were none
    assert math.isnan(drift.exceed_ratio(np.zeros(10), np.zeros(10), 0.5))  # none on either side
