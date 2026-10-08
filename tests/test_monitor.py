"""Tests for the alert policy."""

import math

import numpy as np
import pandas as pd

from fraudmon import config as C
from fraudmon.monitor import FLAG_OPEN, Decision, Monitor, Signals, build_reference, decide

BOUNDS = C.AlertBounds()


def quiet(**overrides):
    """A window on which nothing has moved."""
    base = dict(
        window=4,
        n_rows=10000,
        max_psi=0.01,
        top_feature="V1",
        n_features_drifted=0,
        weighted_share=0.0,
        score_psi=0.01,
        alert_rate=0.002,
        alert_ratio=1.0,
    )
    return Signals(**{**base, **overrides})


def test_quiet_window_is_ok():
    decision = decide(quiet(), bounds=BOUNDS)
    assert (decision.state, decision.rules) == ("OK", [])


def test_feature_drift_alone_only_warns():
    decision = decide(quiet(max_psi=1.5, n_features_drifted=8, weighted_share=0.06), bounds=BOUNDS)
    assert decision.state == "WARN"
    assert decision.rules == ["W_feature_psi"]


def test_prediction_drift_needs_two_consecutive_windows_to_retrain():
    strong = quiet(score_psi=0.40)
    first = decide(strong, history=[], bounds=BOUNDS)
    assert first.state == "WARN"
    second = decide(strong, history=[strong], bounds=BOUNDS)
    assert second.state == "RETRAIN"
    assert "R2_prediction_drift" in second.rules


def test_one_quiet_window_resets_the_persistence_count():
    strong = quiet(score_psi=0.40)
    assert decide(strong, history=[strong, quiet()], bounds=BOUNDS).state == "WARN"


def test_alert_rate_outside_the_retrain_band_counts_as_prediction_drift():
    surge = quiet(alert_ratio=3.0)
    collapse = quiet(alert_ratio=0.3)
    assert decide(collapse, history=[surge], bounds=BOUNDS).state == "RETRAIN"


def test_alerts_appearing_where_the_reference_had_none_are_not_ignored():
    appeared = quiet(alert_ratio=math.inf)
    assert decide(appeared, bounds=BOUNDS).rules == ["W_alert_rate"]
    assert decide(appeared, history=[appeared], bounds=BOUNDS).state == "RETRAIN"
    assert decide(quiet(alert_ratio=math.nan), bounds=BOUNDS).state == "OK"  # no alerts on either side


def test_input_drift_on_important_features_needs_corroboration():
    uncorroborated = decide(quiet(max_psi=0.9, weighted_share=0.53), bounds=BOUNDS)
    assert uncorroborated.state == "WARN"
    corroborated = decide(quiet(max_psi=0.9, weighted_share=0.53, score_psi=0.12), bounds=BOUNDS)
    assert corroborated.state == "RETRAIN"
    assert "R3_corroborated_input_drift" in corroborated.rules


def test_performance_rule_needs_a_real_drop_enough_frauds_and_a_clear_interval():
    degraded = dict(n_frauds=40, ap=0.4, ap_reference=0.8, ap_rel_drop=0.5, ap_diff_hi=-0.1)
    assert decide(quiet(**degraded), bounds=BOUNDS).rules == ["R1_performance"]
    assert decide(quiet(**{**degraded, "n_frauds": 10}), bounds=BOUNDS).state == "OK"
    assert decide(quiet(**{**degraded, "ap_diff_hi": 0.05}), bounds=BOUNDS).state == "OK"
    assert decide(quiet(**{**degraded, "ap_rel_drop": 0.05}), bounds=BOUNDS).state == "OK"


def test_label_free_state_ignores_the_performance_rule():
    decision = decide(
        quiet(n_frauds=40, ap=0.4, ap_reference=0.8, ap_rel_drop=0.5, ap_diff_hi=-0.1), bounds=BOUNDS
    )
    assert decision.state == "RETRAIN"
    assert decision.label_free_state == "OK"


def test_small_windows_skip_the_label_free_rules():
    decision = decide(quiet(n_rows=500, max_psi=2.0, score_psi=0.9), bounds=BOUNDS)
    assert decision.state == "OK"


def test_a_retrain_flag_stays_raised_until_the_monitor_is_replaced():
    monitor = Monitor(reference=None)
    raised = monitor._latch(Decision("RETRAIN", ["R1_performance"], "OK"))
    assert (raised.state, raised.label_free_state) == ("RETRAIN", "OK")

    quiet_window = monitor._latch(Decision("OK", [], "OK"))
    assert quiet_window.state == "RETRAIN"
    assert quiet_window.rules == [FLAG_OPEN]
    assert quiet_window.label_free_state == "OK"  # only labels raised it

    assert Monitor(reference=None)._latch(Decision("OK", [], "OK")).state == "OK"


def test_a_label_free_retrain_flag_also_stays_raised():
    monitor = Monitor(reference=None)
    monitor._latch(Decision("RETRAIN", ["R2_prediction_drift"], "RETRAIN"))
    later = monitor._latch(Decision("WARN", ["W_feature_psi"], "WARN"))
    assert (later.state, later.label_free_state) == ("RETRAIN", "RETRAIN")


def toy_scores(frame):
    return 1 / (1 + np.exp(frame["V14"].to_numpy() + 3.0))


def test_reference_uses_the_latest_occurrence_of_each_clock_position(frame):
    labelled = frame[frame["window"] <= 5]
    ref = build_reference(labelled, toy_scores(labelled), pd.Series({"V14": 1.0}), 0.5)
    assert len(ref.frames[0]) == int((frame["window"] == 4).sum())
    assert len(ref.frames[1]) == int((frame["window"] == 5).sum())
    assert len(ref.frames[2]) == int((frame["window"] == 2).sum())
    assert sum(len(y) for y in ref.labels.values()) == int(frame["window"].between(2, 5).sum())
    assert ref.labels[0].tolist() == frame.loc[frame["window"] == 4, "Class"].tolist()


def test_performance_is_compared_with_the_same_time_of_day(big_frame):
    day1 = big_frame[big_frame["window"] <= 3]
    scores = toy_scores(day1)
    scores[(day1["window"] != 0).to_numpy()] = 0.0  # the model only works in the first window
    ref = build_reference(day1, scores, pd.Series({"V14": 1.0}), 0.5)
    current = big_frame[big_frame["window"] == 4]
    signals, _ = Monitor(ref).check(4, current, toy_scores(current), current["Class"].to_numpy())
    # Window 4 is judged against window 0, where the reference model was good, not the day average.
    assert signals.ap_reference > 0.9
    assert abs(signals.ap_rel_drop) < 0.1


def test_performance_comparison_is_skipped_when_either_side_lacks_a_class(big_frame):
    day1 = big_frame[big_frame["window"] <= 3]
    current = big_frame[big_frame["window"] == 4]
    y = current["Class"].to_numpy()

    fraud_free = build_reference(day1, toy_scores(day1), pd.Series({"V14": 1.0}), 0.5)
    fraud_free.labels[0] = np.zeros_like(fraud_free.labels[0])  # no fraud in the matching reference window
    signals, decision = Monitor(fraud_free).check(4, current, toy_scores(current), y)
    assert signals.recall > 0.9  # the threshold metrics are still reported
    assert signals.ap_rel_drop is None
    assert "R1_performance" not in decision.rules

    ref = build_reference(day1, toy_scores(day1), pd.Series({"V14": 1.0}), 0.5)
    only_frauds = current[current["Class"] == 1]
    signals, _ = Monitor(ref).check(4, only_frauds, toy_scores(only_frauds), only_frauds["Class"].to_numpy())
    assert signals.n_frauds == len(only_frauds)
    assert signals.ap_rel_drop is None


def test_monitor_stays_quiet_without_drift_and_reacts_to_it(big_frame):
    day1 = big_frame[big_frame["window"] <= 3]
    ref = build_reference(day1, toy_scores(day1), pd.Series({"V14": 1.0}), 0.5)
    current = big_frame[big_frame["window"] == 4]

    _, steady = Monitor(ref).check(4, current, toy_scores(current), current["Class"].to_numpy())
    assert steady.state == "OK"

    moved = current.copy()
    moved["V14"] -= 3.0  # the whole population moves toward the fraud region
    signals, alarm = Monitor(ref).check(4, moved, toy_scores(moved), moved["Class"].to_numpy())
    assert signals.top_feature == "V14"
    assert signals.alert_ratio > 2.0
    assert alarm.state == "RETRAIN"
    assert "R3_corroborated_input_drift" in alarm.rules
