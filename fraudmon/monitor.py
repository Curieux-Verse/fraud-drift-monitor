"""Per-window drift monitor and the OK / WARN / RETRAIN alert policy."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import cost, drift, metrics
from .data import clock


@dataclass
class Reference:
    """What each new window is compared against."""

    frames: dict  # clock position -> feature rows of the reference window
    scores: dict  # clock position -> out-of-fold model scores on those rows
    labels: dict  # clock position -> labels of those rows
    importance: pd.Series  # share of model gain per feature
    threshold: float  # operating threshold on the model score
    anomaly: dict = field(default_factory=dict)  # detector -> clock position -> anomaly scores
    anomaly_cut: dict = field(default_factory=dict)  # detector -> tail cut-off
    ap_draws: dict = field(default_factory=dict)  # clock position -> bootstrap AP, filled on use

    @property
    def alert_rate(self):
        """Share of reference rows at or above the threshold."""
        return float(np.mean(np.concatenate(list(self.scores.values())) >= self.threshold))


def build_reference(labelled, oof, importance, threshold, anomaly_oof=None):
    """Reference from the most recent occurrence of each clock position in the labelled data."""
    windows = labelled["window"].to_numpy()
    y = labelled["Class"].to_numpy()
    latest = {clock(w): w for w in np.unique(windows)}  # ascending, so the last one wins
    recent = np.isin(windows, list(latest.values()))
    ref = Reference(
        frames={pos: labelled.loc[windows == w, C.FEATURES] for pos, w in latest.items()},
        scores={pos: oof[windows == w] for pos, w in latest.items()},
        labels={pos: y[windows == w] for pos, w in latest.items()},
        importance=importance,
        threshold=float(threshold),
    )
    for name, values in (anomaly_oof or {}).items():
        ref.anomaly[name] = {pos: values[windows == w] for pos, w in latest.items()}
        ref.anomaly_cut[name] = float(np.quantile(values[recent], C.ANOMALY_QUANTILE))
    return ref


@dataclass
class Signals:
    """Everything measured on one window. Performance fields need labels."""

    window: int
    n_rows: int
    max_psi: float
    top_feature: str
    n_features_drifted: int
    weighted_share: float
    score_psi: float
    alert_rate: float
    alert_ratio: float
    anomaly_ratio: dict = field(default_factory=dict)
    n_frauds: int | None = None
    ap: float | None = None
    ap_adjusted: float | None = None  # AP at the reference window's fraud rate
    ap_reference: float | None = None
    ap_rel_drop: float | None = None  # of ap_adjusted against ap_reference
    ap_diff_hi: float | None = None  # upper 95% bound of (ap_adjusted - ap_reference)
    recall: float | None = None
    precision: float | None = None
    cost_per_1k: float | None = None


FLAG_OPEN = "retrain_flag_still_open"


@dataclass
class Decision:
    state: str
    rules: list
    label_free_state: str = "OK"  # the state if labels, and so rule R1, were not available yet


def _state(rules, retrain_prefixes):
    if any(r.startswith(retrain_prefixes) for r in rules):
        return "RETRAIN"
    return "WARN" if any(r.startswith("W_") for r in rules) else "OK"


def _outside(value, band):
    return bool(np.isfinite(value)) and not band[0] <= value <= band[1]


def _strong_prediction_drift(sig, b):
    return sig.score_psi >= b.score_psi_retrain or _outside(sig.alert_ratio, b.alert_ratio_retrain)


def _mild_prediction_drift(sig, b):
    return sig.score_psi >= b.score_psi_warn or _outside(sig.alert_ratio, b.alert_ratio_warn)


def decide(sig, history=(), bounds=None):
    """Turn one window's signals, and the windows before it, into a state and the rules that fired."""
    b = bounds or C.AlertBounds()
    rules = []
    if sig.n_rows >= b.min_rows:
        if sig.max_psi >= b.psi_alert:
            rules.append("W_feature_psi")
        if sig.weighted_share >= b.weighted_share_warn:
            rules.append("W_weighted_share")
        if sig.score_psi >= b.score_psi_warn:
            rules.append("W_score_psi")
        if _outside(sig.alert_ratio, b.alert_ratio_warn):
            rules.append("W_alert_rate")
        if any(r > b.anomaly_ratio_warn for r in sig.anomaly_ratio.values() if np.isfinite(r)):
            rules.append("W_anomaly_rate")

        previous = list(history)[len(history) - (b.persistence - 1) :] if b.persistence > 1 else []
        run = [*previous, sig]
        if len(run) == b.persistence and all(_strong_prediction_drift(s, b) for s in run):
            rules.append("R2_prediction_drift")
        if sig.weighted_share >= b.weighted_share_retrain and _mild_prediction_drift(sig, b):
            rules.append("R3_corroborated_input_drift")

    if (
        sig.n_frauds is not None
        and sig.n_frauds >= b.min_frauds
        and sig.ap_rel_drop >= b.perf_rel_drop
        and sig.ap_diff_hi < 0
    ):
        rules.append("R1_performance")

    return Decision(_state(rules, ("R1", "R2", "R3")), rules, _state(rules, ("R2", "R3")))


class Monitor:
    """Checks successive windows against a fixed reference and remembers what it saw.

    A RETRAIN flag stays raised on later windows until the model is replaced (a new Monitor):
    a request to retrain does not withdraw itself because one window looked quieter.
    """

    def __init__(self, reference, bounds=None, cost_model=None, seed=C.SEED):
        self.reference = reference
        self.bounds = bounds or C.AlertBounds()
        self.cost_model = cost_model or C.CostModel()
        self.seed = seed
        self.history = []
        self._flag_open = False
        self._label_free_flag_open = False

    def check(self, window, frame, scores, y=None, anomaly_scores=None):
        """Measure one window. Pass labels once they are known to add the performance check."""
        ref = self.reference
        pos = clock(window)
        scores = np.asarray(scores, dtype=float)
        psi_values = drift.feature_psi(ref.frames[pos], frame)
        sig = Signals(
            window=int(window),
            n_rows=len(frame),
            max_psi=float(psi_values.max()),
            top_feature=str(psi_values.idxmax()),
            n_features_drifted=int((psi_values >= self.bounds.psi_alert).sum()),
            weighted_share=drift.weighted_share(psi_values, ref.importance, self.bounds.psi_alert),
            score_psi=drift.score_psi(ref.scores[pos], scores),
            alert_rate=float(np.mean(scores >= ref.threshold)),
            alert_ratio=drift.exceed_ratio(ref.scores[pos], scores, ref.threshold),
        )
        for name, values in (anomaly_scores or {}).items():
            sig.anomaly_ratio[name] = drift.exceed_ratio(
                ref.anomaly[name][pos], values, ref.anomaly_cut[name]
            )
        if y is not None:
            self._add_performance(sig, pos, frame, scores, np.asarray(y))
        decision = self._latch(decide(sig, self.history, self.bounds))
        self.history.append(sig)
        return sig, decision

    def _latch(self, decision):
        if self._flag_open and decision.state != "RETRAIN":
            decision.state = "RETRAIN"
            decision.rules = [*decision.rules, FLAG_OPEN]
        if self._label_free_flag_open:
            decision.label_free_state = "RETRAIN"
        self._flag_open = decision.state == "RETRAIN"
        self._label_free_flag_open = decision.label_free_state == "RETRAIN"
        return decision

    def _add_performance(self, sig, pos, frame, scores, y):
        """Compare against the same clock position: accuracy varies with the time of day."""
        ref = self.reference
        flagged = scores >= ref.threshold
        outcome = cost.evaluate(y, frame["Amount"].to_numpy(), flagged, self.cost_model)
        sig.n_frauds = int(y.sum())
        sig.recall, sig.precision = outcome["recall"], outcome["precision"]
        sig.cost_per_1k = outcome["cost_per_1k"]
        if sig.n_frauds == 0:
            return
        ref_y, ref_scores = ref.labels[pos], ref.scores[pos]
        prior = float(np.mean(ref_y))  # judge the window at the reference window's fraud rate
        if pos not in ref.ap_draws:
            ref.ap_draws[pos] = metrics.bootstrap_ap(ref_y, ref_scores, seed=self.seed)
        sig.ap = metrics.average_precision(y, scores)
        sig.ap_adjusted = metrics.average_precision(y, scores, prior=prior)
        sig.ap_reference = metrics.average_precision(ref_y, ref_scores)
        sig.ap_rel_drop = (sig.ap_reference - sig.ap_adjusted) / sig.ap_reference
        # The two samples are independent, so resample each on its own and difference the draws.
        draws = metrics.bootstrap_ap(y, scores, seed=self.seed + 1, prior=prior)
        sig.ap_diff_hi = metrics.interval(draws - ref.ap_draws[pos])[1]
