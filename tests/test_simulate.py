"""Tests for the drift injectors."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from fraudmon import config as C
from fraudmon import models, simulate
from fraudmon.monitor import build_reference


@pytest.fixture()
def ctx(frame):
    train = frame[frame["window"].isin(C.TRAIN_WINDOWS)]
    order = ["V14"] + [f for f in C.FEATURES if f != "V14"]
    importance = pd.Series(np.linspace(1.0, 0.0, len(order)), index=order)
    return simulate.make_context(train, importance / importance.sum())


@pytest.fixture()
def stream(frame):
    return frame[frame["window"] >= 4]


def unchanged_outside(before, after, mask, columns):
    return before.loc[~mask, columns].equals(after.loc[~mask, columns])


def test_context_knows_which_way_fraud_lies(ctx):
    assert ctx.fraud_direction["V14"] == -1


def test_shift_feature_touches_only_the_masked_rows_of_one_feature(stream):
    mask = stream["window"].isin(C.TEST_WINDOWS)
    out = simulate.shift_feature(stream, "V3", 2.0, mask)
    assert np.allclose(out.loc[mask, "V3"], stream.loc[mask, "V3"] + 2.0)
    assert unchanged_outside(stream, out, mask, C.FEATURES)
    assert out.drop(columns="V3").equals(stream.drop(columns="V3"))


def test_control_scenario_changes_nothing(stream, ctx):
    out = simulate.apply_scenario("S0_control", stream, ctx)
    assert out[stream.columns].equals(stream)
    assert not out["new_tactic"].any()


def test_covariate_scenarios_apply_the_same_shift_to_different_features(stream, ctx):
    mask = stream["window"].isin(ctx.drift_windows)
    unused, key = simulate.shifted_features(ctx)
    assert (unused, key) == ("V28", "V14")

    on_unused = simulate.apply_scenario("S1_unused_feature_shift", stream, ctx)
    on_key = simulate.apply_scenario("S2_key_feature_shift", stream, ctx)
    assert np.allclose(on_unused.loc[mask, "V28"] - stream.loc[mask, "V28"], ctx.sd["V28"])
    assert np.allclose(on_key.loc[mask, "V14"] - stream.loc[mask, "V14"], -ctx.sd["V14"])  # toward fraud
    assert on_unused.drop(columns=["V28", "new_tactic"]).equals(stream.drop(columns="V28"))
    assert on_key.drop(columns=["V14", "new_tactic"]).equals(stream.drop(columns="V14"))
    assert unchanged_outside(stream, on_unused, mask, C.FEATURES)
    assert unchanged_outside(stream, on_key, mask, C.FEATURES)


def test_attack_wave_multiplies_fraud_only_inside_the_drift_windows(stream, ctx):
    out = simulate.apply_scenario("S3_attack_wave", stream, ctx)
    inside = stream["window"].isin(ctx.drift_windows)
    frauds_inside = int(stream.loc[inside, "Class"].sum())
    assert int(out["Class"].sum()) == int(stream["Class"].sum()) + 3 * frauds_inside
    assert int((out["Class"] == 0).sum()) == int((stream["Class"] == 0).sum())
    clean = ~out["window"].isin(ctx.drift_windows)
    assert int(out.loc[clean, "Class"].sum()) == int(stream.loc[~inside, "Class"].sum())
    assert out["Time"].is_monotonic_increasing


def test_new_tactic_rewrites_only_adopting_frauds(stream, ctx):
    hide, signature = simulate.tactic_features(ctx)
    out = simulate.apply_scenario("S4_new_tactic", stream, ctx)
    adopters = out["new_tactic"]
    inside = stream["window"].isin(ctx.drift_windows)

    assert adopters.any()
    assert (out.loc[adopters, "Class"] == 1).all()
    assert inside[adopters].all()
    assert out.loc[~adopters, stream.columns].equals(stream.loc[~adopters])
    untouched = [f for f in C.FEATURES if f not in hide + signature]
    assert out.loc[adopters, untouched].equals(stream.loc[adopters, untouched])
    for feature in signature:
        moved = out.loc[adopters, feature] - stream.loc[adopters, feature]
        assert np.allclose(moved, 4.0 * ctx.sd[feature])


def test_new_tactic_hides_in_important_features_and_moves_to_ignored_ones(ctx):
    hide, signature = simulate.tactic_features(ctx)
    assert hide[0] == "V14"
    assert not set(hide) & set(signature)
    assert ctx.importance[signature].max() < ctx.importance[hide].min()


def test_unknown_scenario_is_rejected(stream, ctx):
    with pytest.raises(ValueError, match="unknown scenario"):
        simulate.apply_scenario("S9_nope", stream, ctx)


FAST_MODEL = models.make_supervised("lgbm", "none", seed=0, lgbm_params=dict(C.LGBM_PARAMS, n_estimators=30))


def test_controlled_bed_is_quiet_until_a_harmful_shift_is_injected(big_frame):
    train = big_frame[big_frame["window"].isin(C.TRAIN_WINDOWS)]
    model, bed, stream = simulate.controlled_setup(train, FAST_MODEL, alert_rate=0.02)
    assert sum(len(f) for f in bed.frames.values()) + len(stream) == len(train)
    assert abs(bed.alert_rate - 0.02) < 0.002

    bed_ctx = simulate.make_context(train, bed.importance, drift_windows=(2, 3))
    states = {}
    for name in simulate.SCENARIOS[:3]:
        injected = simulate.apply_scenario(name, stream, bed_ctx)
        states[name] = [r["state"] for r in simulate.monitor_stream(injected, model, bed, {})]
    assert states["S0_control"] == ["OK"] * 4
    # The same shift warns on a feature the model ignores and flags a retrain on the one it uses.
    assert states["S1_unused_feature_shift"] == ["OK", "OK", "WARN", "WARN"]
    assert states["S2_key_feature_shift"] == ["OK", "OK", "RETRAIN", "RETRAIN"]


def test_replay_policies_retrain_as_described_and_share_refits(frame, ctx):
    train = frame[frame["window"].isin(C.TRAIN_WINDOWS)]
    base = clone(FAST_MODEL).fit(train[C.FEATURES], train["Class"].to_numpy())
    oof = models.oof_predict(FAST_MODEL, train)
    reference = build_reference(train, oof, models.gain_importance(base), float(np.quantile(oof, 0.98)))
    stream = simulate.apply_scenario("S0_control", frame[frame["window"] >= 4], ctx)

    cache = {}
    static = simulate.replay("static", train, stream, base, reference, cache)
    periodic = simulate.replay("periodic", train, stream, base, reference, cache)
    assert [r["window"] for r in static] == [4, 5, 6, 7]
    assert not any(r["retrained_before"] for r in static)
    assert [r["retrained_before"] for r in periodic] == [False, True, True, True]
    assert len(cache) == 3
    simulate.replay("periodic", train, stream, base, reference, cache)
    assert len(cache) == 3  # the second run reuses every refit
