"""Injected drift scenarios and the retrain-policy replay."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import config as C
from . import cost, drift, metrics, models
from .data import clock
from .monitor import Monitor, Reference, build_reference

SCENARIOS = (
    "S0_control",
    "S1_unused_feature_shift",
    "S2_key_feature_shift",
    "S3_attack_wave",
    "S4_new_tactic",
)
POLICIES = ("static", "periodic", "triggered")


@dataclass
class Context:
    """Training-window facts the injectors need."""

    sd: pd.Series  # per-feature standard deviation
    fraud_direction: pd.Series  # +1 if fraud sits above legitimate on the feature, else -1
    importance: pd.Series  # share of model gain, largest first
    drift_windows: tuple = C.TEST_WINDOWS


def make_context(train, importance, drift_windows=C.TEST_WINDOWS):
    fraud = train.loc[train["Class"] == 1, C.FEATURES].mean()
    legit = train.loc[train["Class"] == 0, C.FEATURES].mean()
    return Context(train[C.FEATURES].std(), np.sign(fraud - legit), importance, tuple(drift_windows))


# --- injectors ----------------------------------------------------------------------------


def shift_feature(frame, feature, shift, mask):
    out = frame.copy()
    out.loc[mask, feature] = out.loc[mask, feature] + shift
    return out


def attack_wave(frame, mask, factor, sd, rng, jitter=0.05):
    """Multiply fraud volume inside the mask by `factor`; the copies are jittered slightly."""
    frauds = frame[mask & (frame["Class"] == 1)]
    copies = frauds.loc[frauds.index.repeat(factor - 1)].copy()
    noise = rng.normal(0.0, 1.0, (len(copies), len(C.PCA_FEATURES)))
    copies[C.PCA_FEATURES] = copies[C.PCA_FEATURES].to_numpy() + noise * jitter * sd[C.PCA_FEATURES].to_numpy()
    out = pd.concat([frame, copies], ignore_index=True)
    return out.sort_values("Time", kind="stable").reset_index(drop=True)


def new_tactic(frame, mask, hide, signature, sd, rng, adoption=0.7, strength=4.0):
    """A share of frauds drops the old signature and picks up a new one.

    `hide` features are resampled from legitimate rows, so the fraud no longer stands out
    where the model looks. `signature` features are shifted by `strength` standard deviations.
    """
    out = frame.copy()
    frauds = np.flatnonzero((mask & (out["Class"] == 1)).to_numpy())
    rows = out.index[frauds[rng.random(len(frauds)) < adoption]]
    legit = out[mask & (out["Class"] == 0)]
    for feature in hide:
        out.loc[rows, feature] = rng.choice(legit[feature].to_numpy(), len(rows))
    for feature in signature:
        out.loc[rows, feature] = out.loc[rows, feature] + strength * sd[feature]
    out["new_tactic"] = False
    out.loc[rows, "new_tactic"] = True
    return out


def tactic_features(ctx):
    """Features the new tactic hides in (most used by the model) and moves to (least used)."""
    ranked = ctx.importance[ctx.importance.index.isin(C.PCA_FEATURES)]
    return list(ranked.index[:5]), list(ranked.index[-3:])


def shifted_features(ctx):
    """The model's least-used and most-used feature, for the two covariate scenarios."""
    ranked = ctx.importance[ctx.importance.index.isin(C.PCA_FEATURES)]
    return ranked.index[-1], ranked.index[0]


def apply_scenario(name, stream, ctx, seed=C.SEED):
    """Return a copy of the stream with the scenario injected into the drift windows.

    S1 and S2 are the same one-SD shift on every row; only the feature differs.
    """
    rng = np.random.default_rng(seed)
    out = stream.copy()
    out["new_tactic"] = False
    mask = out["window"].isin(ctx.drift_windows)
    unused, key = shifted_features(ctx)
    if name == "S0_control":
        return out
    if name == "S1_unused_feature_shift":
        return shift_feature(out, unused, ctx.sd[unused], mask)
    if name == "S2_key_feature_shift":
        return shift_feature(out, key, ctx.fraud_direction[key] * ctx.sd[key], mask)
    if name == "S3_attack_wave":
        return attack_wave(out, mask, 4, ctx.sd, rng)
    if name == "S4_new_tactic":
        hide, signature = tactic_features(ctx)
        return new_tactic(out, mask, hide, signature, ctx.sd, rng)
    raise ValueError(f"unknown scenario: {name}")


# --- running the monitor ------------------------------------------------------------------


def monitor_stream(stream, model, reference, detectors, bounds=None, cost_model=None):
    """Run a fresh monitor over each window of a stream, with labels, one record per window."""
    monitor = Monitor(reference, bounds, cost_model)
    records = []
    for window, frame in stream.groupby("window", sort=True):
        scores = models.predict(model, frame)
        anomaly = {name: models.anomaly_score(det, frame) for name, det in detectors.items()}
        signals, decision = monitor.check(window, frame, scores, frame["Class"].to_numpy(), anomaly)
        records.append(
            {
                **asdict(signals),
                "state": decision.state,
                "label_free_state": decision.label_free_state,
                "rules": decision.rules,
            }
        )
    return records


def null_control(train, oof, importance, threshold, reps=10, seed=C.SEED):
    """False-alarm rate: compare random halves of each training window, where nothing drifted."""
    rng = np.random.default_rng(seed)
    windows = train["window"].to_numpy()
    y = train["Class"].to_numpy()
    detectors = [name for name in oof if name != "supervised"]
    states, max_psi, max_score_psi = [], 0.0, 0.0
    for _ in range(reps):
        half = rng.random(len(train)) < 0.5
        ref = Reference({}, {}, {}, importance, threshold)
        for name in detectors:
            ref.anomaly[name] = {}
            ref.anomaly_cut[name] = float(np.quantile(oof[name][half], C.ANOMALY_QUANTILE))
        for w in np.unique(windows):
            mask = (windows == w) & half
            ref.frames[clock(w)] = train.loc[mask, C.FEATURES]
            ref.scores[clock(w)] = oof["supervised"][mask]
            ref.labels[clock(w)] = y[mask]
            for name in detectors:
                ref.anomaly[name][clock(w)] = oof[name][mask]
        monitor = Monitor(ref)
        for w in np.unique(windows):
            mask = (windows == w) & ~half
            anomaly = {name: oof[name][mask] for name in detectors}
            signals, decision = monitor.check(w, train[mask], oof["supervised"][mask], y[mask], anomaly)
            states.append(decision.state)
            max_psi = max(max_psi, signals.max_psi)
            max_score_psi = max(max_score_psi, signals.score_psi)
    return {
        "checks": len(states),
        "warn": states.count("WARN"),
        "retrain": states.count("RETRAIN"),
        "max_feature_psi": max_psi,
        "max_score_psi": max_score_psi,
    }


def refit(base_model, labelled, alert_rate, seed=C.SEED):
    """Refit on all labelled data and rebuild the monitor's reference around the new model.

    The new threshold keeps the alert rate of the original operating point.
    """
    model = clone(base_model).fit(labelled[C.FEATURES], labelled["Class"].to_numpy())
    oof = models.oof_predict(base_model, labelled, seed, n_folds=C.REFIT_FOLDS)
    threshold = float(np.quantile(oof, 1 - alert_rate))
    return model, build_reference(labelled, oof, models.gain_importance(model), threshold)


def controlled_setup(train, base_model, alert_rate, seed=C.SEED):
    """A drift-free test bed: fit on a random half of the training window, stream the other half.

    Both halves cover the same hours, so whatever the monitor reports on the untouched
    stream is a false alarm, and whatever it reports after an injection is the injection.
    """
    in_fit = np.random.default_rng(seed).random(len(train)) < 0.5
    model, reference = refit(base_model, train[in_fit], alert_rate, seed)
    return model, reference, train[~in_fit]


def dose_response(stream, ctx, model, reference, doses):
    """Shift the model's top feature by each dose (in SDs, positive = toward fraud)."""
    top = shifted_features(ctx)[1]
    mask = stream["window"].isin(ctx.drift_windows)
    rows = []
    for dose in doses:
        shifted = shift_feature(stream, top, dose * ctx.fraud_direction[top] * ctx.sd[top], mask)
        records = [r for r in monitor_stream(shifted, model, reference, {}) if r["window"] in ctx.drift_windows]
        drifted = shifted[mask]
        outcome = metrics.at_threshold(drifted["Class"], models.predict(model, drifted), reference.threshold)
        rows.append(
            {
                "dose_sd": dose,
                "feature": top,
                "feature_psi": records[0]["max_psi"],
                "weighted_share": records[0]["weighted_share"],
                "score_psi": records[0]["score_psi"],
                "alert_ratio": records[0]["alert_ratio"],
                "precision": outcome["precision"],
                "recall": outcome["recall"],
                "states": [r["state"] for r in records],
                "label_free_states": [r["label_free_state"] for r in records],
            }
        )
    return rows


def tactic_strength_sweep(stream, ctx, model, reference, detectors, strengths):
    """How visible the new tactic is to the model and to each detector, by signature strength.

    A detector "sees" a fraud if it lands in the top ALERT_BUDGET share of anomaly scores.
    """
    hide, signature = tactic_features(ctx)
    mask = stream["window"].isin(ctx.drift_windows)
    rows = []
    for strength in strengths:
        rng = np.random.default_rng(C.SEED)  # same adopting frauds at every strength
        drifted = new_tactic(stream, mask, hide, signature, ctx.sd, rng, strength=strength)[mask]
        new = drifted["new_tactic"].to_numpy()
        scores = models.predict(model, drifted)
        row = {
            "strength_sd": strength,
            "new_tactic_frauds": int(new.sum()),
            "caught_by_model": int((scores[new] >= reference.threshold).sum()),
        }
        for name, detector in detectors.items():
            anomaly = models.anomaly_score(detector, drifted)
            budget_cut = np.quantile(anomaly, 1 - C.ALERT_BUDGET)
            percentile = np.argsort(np.argsort(anomaly)) / (len(anomaly) - 1)
            row[f"{name}_in_budget"] = int((anomaly[new] >= budget_cut).sum())
            row[f"{name}_median_percentile"] = float(np.median(percentile[new]))
            row[f"{name}_anomaly_ratio"] = max(
                drift.exceed_ratio(
                    reference.anomaly[name][clock(w)],
                    anomaly[(drifted["window"] == w).to_numpy()],
                    reference.anomaly_cut[name],
                )
                for w in ctx.drift_windows
            )
        rows.append(row)
    return rows


# --- retrain policies ---------------------------------------------------------------------


def _fingerprint(labelled):
    return int(pd.util.hash_pandas_object(labelled[C.FEATURES + ["Class"]], index=False).sum())


def replay(policy, train, stream, base_model, base_reference, cache=None, bounds=None, cost_model=None):
    """Score the stream window by window; a window's labels arrive when the next one starts.

    `cache` shares refits between policies and scenarios that reach the same labelled data.
    """
    cost_model = cost_model or C.CostModel()
    cache = {} if cache is None else cache
    model, reference = base_model, base_reference
    alert_rate = base_reference.alert_rate
    monitor = Monitor(reference, bounds, cost_model)
    labelled, previous, records = train, None, []
    for window, frame in stream.groupby("window", sort=True):
        retrained, last_state, last_rules = False, None, []
        if previous is not None:
            _, decision = monitor.check(*previous, previous[1]["Class"].to_numpy())
            last_state, last_rules = decision.state, decision.rules
            labelled = pd.concat([labelled, previous[1]], ignore_index=True)
            if policy == "periodic" or (policy == "triggered" and decision.state == "RETRAIN"):
                key = _fingerprint(labelled)
                if key not in cache:
                    cache[key] = refit(base_model, labelled, alert_rate)
                model, reference = cache[key]
                monitor = Monitor(reference, bounds, cost_model)
                retrained = True
        scores = models.predict(model, frame)
        y = frame["Class"].to_numpy()
        outcome = cost.evaluate(y, frame["Amount"].to_numpy(), scores >= reference.threshold, cost_model)
        records.append(
            {
                "policy": policy,
                "window": int(window),
                "n_rows": len(frame),
                "n_frauds": int(y.sum()),
                "frauds_missed": outcome["fn"],
                "false_alarms": outcome["fp"],
                "cost": outcome["cost"],
                "retrained_before": retrained,
                "previous_window_state": last_state,
                "previous_window_rules": last_rules,
                "ap": metrics.average_precision(y, scores),
                "recall": outcome["recall"],
                "precision": outcome["precision"],
                "cost_per_1k": outcome["cost_per_1k"],
            }
        )
        previous = (window, frame, scores)
    return records
