"""Tests for the pipeline's file output and the threshold its stages share."""

import json

import numpy as np
from sklearn.base import clone

from fraudmon import config as C
from fraudmon import cost, data, models, pipeline

FAST_MODEL = models.make_supervised("lgbm", "none", seed=0, lgbm_params=dict(C.LGBM_PARAMS, n_estimators=30))


def test_metrics_are_rounded_with_null_for_non_finite_values_and_lf_line_endings(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "METRIC_DIR", tmp_path / "metrics")
    pipeline.save_metrics("demo", {"a": np.float64(0.123456789), "b": float("nan"), "c": [np.int64(3), True]})
    raw = (tmp_path / "metrics" / "demo.json").read_bytes()
    assert b"\r" not in raw
    assert json.loads(raw) == {"a": 0.123457, "b": None, "c": [3, True]}


def test_the_monitor_gets_the_exact_threshold_not_a_rounded_copy_from_disk(frame, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "METRIC_DIR", tmp_path)  # nothing to read back: no threshold.json here
    train, val, _ = data.split(frame)
    model = clone(FAST_MODEL).fit(train[C.FEATURES], train["Class"].to_numpy())
    rng = np.random.default_rng(0)
    oof = {"supervised": models.predict(model, train), **{name: rng.random(len(train)) for name in models.DETECTORS}}
    art = {"train": train, "val": val, "model": model, "oof": oof}

    exact = cost.best_threshold(
        val["Class"].to_numpy(), models.predict(model, val), val["Amount"].to_numpy(), C.CostModel()
    )
    reference, _ = pipeline._reference(art)
    assert reference.threshold == exact == pipeline._operating_threshold(art)
