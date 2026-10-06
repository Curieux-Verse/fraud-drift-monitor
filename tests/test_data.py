"""Tests for the chronological split."""

import pytest

from fraudmon import config as C
from fraudmon import data


def test_split_is_chronological_and_disjoint(frame):
    train, val, test = data.split(frame)
    assert len(train) + len(val) + len(test) == len(frame)
    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()


def test_split_follows_the_day_boundary(frame):
    train, val, test = data.split(frame)
    assert train["Time"].max() < 24 * 3600 <= val["Time"].min()
    assert set(train["window"]) == set(C.TRAIN_WINDOWS)
    assert set(test["window"]) == set(C.TEST_WINDOWS)


def test_prepare_sorts_by_time_and_adds_features(raw_factory):
    shuffled = raw_factory(n=500).sample(frac=1.0, random_state=0)
    prepared = data.prepare(shuffled)
    assert prepared["Time"].is_monotonic_increasing
    assert {"log_amount", "window"} <= set(prepared.columns)
    assert prepared["window"].between(0, 7).all()


def test_prepare_rejects_a_file_with_missing_columns(raw_factory):
    with pytest.raises(ValueError, match="missing columns"):
        data.prepare(raw_factory(n=50).drop(columns=["V14"]))


def test_clock_maps_day_two_windows_onto_day_one():
    assert [data.clock(w) for w in (0, 3, 4, 7)] == [0, 3, 0, 3]
