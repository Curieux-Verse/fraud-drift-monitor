"""Tests for the model factories and metrics."""

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraudmon import config as C
from fraudmon import data, metrics, models

FAST_LGBM = dict(C.LGBM_PARAMS, n_estimators=30)


@pytest.mark.parametrize("model", ["logreg", "lgbm"])
@pytest.mark.parametrize("strategy", C.STRATEGIES)
def test_every_strategy_fits_and_leaves_evaluation_data_alone(frame, model, strategy):
    train, val, _ = data.split(frame)
    before = val.copy()
    estimator = models.make_supervised(model, strategy, seed=0, lgbm_params=FAST_LGBM)
    estimator.fit(train[C.FEATURES], train["Class"].to_numpy())
    scores = models.predict(estimator, val)
    # Resampling happens inside fit only: one score per evaluation row, rows untouched.
    assert len(scores) == len(val)
    assert val.equals(before)
    assert metrics.average_precision(val["Class"], scores) > 0.5


def test_smote_balances_the_training_data_it_is_given(frame):
    train, _, _ = data.split(frame)
    sampler = models.make_supervised("logreg", "smote", seed=0).named_steps["sample"]
    _, y_resampled = sampler.fit_resample(train[C.FEATURES], train["Class"])
    assert int((y_resampled == 1).sum()) == int((y_resampled == 0).sum())
    assert int((y_resampled == 0).sum()) == int((train["Class"] == 0).sum())


def test_unknown_strategy_and_model_are_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        models.make_supervised("lgbm", "oversample_everything")
    with pytest.raises(ValueError, match="unknown model"):
        models.make_supervised("svm", "none")


def test_out_of_fold_scores_cover_every_row(frame):
    train, _, _ = data.split(frame)
    estimator = models.make_supervised("lgbm", "none", seed=0, lgbm_params=FAST_LGBM)
    oof = models.oof_predict(estimator, train)
    assert len(oof) == len(train)
    assert np.isfinite(oof).all()
    assert metrics.average_precision(train["Class"], oof) > 0.5


def test_gain_importance_sums_to_one_and_finds_the_signal(frame):
    train, _, _ = data.split(frame)
    estimator = models.make_supervised("lgbm", "none", seed=0, lgbm_params=FAST_LGBM)
    estimator.fit(train[C.FEATURES], train["Class"].to_numpy())
    importance = models.gain_importance(estimator)
    assert math.isclose(importance.sum(), 1.0)
    assert importance.index[0] == "V14"


@pytest.mark.parametrize("name", list(models.DETECTORS))
def test_detectors_score_fraud_above_legitimate_rows(frame, name):
    train, val, _ = data.split(frame)
    kwargs = {"epochs": 10} if name == "autoencoder" else {}
    detector = models.DETECTORS[name](seed=0, **kwargs)
    detector.fit(train[C.FEATURES].to_numpy(), train["Class"].to_numpy())
    scores = models.anomaly_score(detector, val)
    assert len(scores) == len(val)
    assert metrics.roc_auc(val["Class"], scores) > 0.8


def test_platt_scaling_pulls_inflated_scores_back_to_the_base_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.01).astype(int)
    inflated = np.clip(rng.normal(0.3 + 0.5 * y, 0.1), 0.01, 0.99)
    calibrated = models.apply_platt(models.fit_platt(inflated, y), inflated)
    assert abs(calibrated.mean() - y.mean()) < 0.002


def test_bootstrap_average_precision_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(float)
    scores = np.round(rng.random(500) * 0.5 + 0.4 * y, 1)
    order = np.argsort(-scores, kind="stable")
    group_end = np.flatnonzero(np.concatenate([np.diff(scores[order]) != 0, [True]]))
    ours = metrics._weighted_ap(y[order], np.ones(500), group_end)
    assert math.isclose(ours, average_precision_score(y, scores))


def test_prior_adjusted_ap_removes_the_effect_of_the_fraud_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(40000) < 0.02).astype(int)
    scores = rng.normal(0.0, 1.0, 40000) + 2.5 * y
    base_rate = y.mean()
    original = metrics.average_precision(y, scores)
    assert math.isclose(metrics.average_precision(y, scores, prior=base_rate), original)

    keep = (y == 1) | (rng.random(40000) < 0.25)  # same model, four times the fraud rate
    richer = metrics.average_precision(y[keep], scores[keep])
    assert richer > original + 0.1
    assert abs(metrics.average_precision(y[keep], scores[keep], prior=base_rate) - original) < 0.03


def test_bootstrap_at_a_fixed_prior_centres_on_the_adjusted_ap():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.02).astype(int)
    scores = rng.normal(0.0, 1.0, 20000) + 2.5 * y
    adjusted = metrics.average_precision(y, scores, prior=0.005)
    draws = metrics.bootstrap_ap(y, scores, n_boot=200, prior=0.005)
    assert adjusted < metrics.average_precision(y, scores)
    assert abs(np.nanmean(draws) - adjusted) < 0.02


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    scores = rng.random(3000) * 0.6 + 0.3 * y
    out = metrics.ap_with_ci(y, scores, n_boot=200)
    assert out["ap_lo"] < out["ap"] < out["ap_hi"]


def test_at_budget_reviews_the_top_share_only():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.linspace(1.0, 0.1, 10)
    out = metrics.at_budget(y, scores, budget=0.2)
    assert (out["n_reviewed"], out["precision"], out["recall"]) == (2, 0.5, 0.5)


def test_at_threshold_counts():
    out = metrics.at_threshold(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 1)
    assert out["alert_rate"] == 0.5
