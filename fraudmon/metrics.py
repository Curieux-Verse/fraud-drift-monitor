"""Ranking metrics, bootstrap confidence intervals and calibration measures."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config as C

RELIABILITY_EDGES = np.array([0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 0.8, 1.0])


def logit(prob, eps=1e-9):
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    return np.log(prob / (1 - prob))


def _prior_factor(n_pos, n_neg, prior):
    """Weight on negatives that makes the fraud rate equal `prior`."""
    return (n_pos / n_neg) * (1 - prior) / prior


def average_precision(y, scores, prior=None):
    """Average precision; with `prior`, as it would be if the fraud rate were `prior`.

    AP rises and falls with the fraud rate, so two windows are only comparable at a common one.
    """
    if prior is None:
        return float(average_precision_score(y, scores))
    y = np.asarray(y)
    n_pos = int(y.sum())
    weights = np.where(y == 1, 1.0, _prior_factor(n_pos, len(y) - n_pos, prior))
    return float(average_precision_score(y, scores, sample_weight=weights))


def roc_auc(y, scores):
    return float(roc_auc_score(y, scores))


def _weighted_ap(y_sorted, weights, group_end):
    """Average precision on rows sorted by descending score, with tied scores grouped."""
    tp = np.cumsum(weights * y_sorted)[group_end]
    flagged = np.cumsum(weights)[group_end]
    if tp[-1] == 0:
        return np.nan
    keep = flagged > 0
    precision = tp[keep] / flagged[keep]
    recall = tp[keep] / tp[-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def bootstrap_ap(y, scores, n_boot=C.N_BOOT, seed=C.SEED, prior=None):
    """Average precision on n_boot resamples of the rows, optionally at a fixed fraud rate."""
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    y_sorted, s_sorted = y[order], scores[order]
    group_end = np.flatnonzero(np.concatenate([np.diff(s_sorted) != 0, [True]]))
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        n_pos = float(weights @ y_sorted)
        if prior is not None and 0 < n_pos < n:
            weights = np.where(y_sorted == 1, weights, weights * _prior_factor(n_pos, n - n_pos, prior))
        draws[i] = _weighted_ap(y_sorted, weights, group_end)
    return draws


def interval(draws, level=0.95):
    tail = (1 - level) / 2
    lo, hi = np.nanquantile(draws, [tail, 1 - tail])
    return float(lo), float(hi)


def ap_with_ci(y, scores, n_boot=C.N_BOOT, seed=C.SEED):
    lo, hi = interval(bootstrap_ap(y, scores, n_boot, seed))
    return {"ap": average_precision(y, scores), "ap_lo": lo, "ap_hi": hi}


def at_threshold(y, scores, threshold):
    """Confusion counts and rates when everything scoring >= threshold is flagged."""
    y = np.asarray(y)
    flagged = np.asarray(scores) >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "alert_rate": float(flagged.mean()),
    }


def at_budget(y, scores, budget=C.ALERT_BUDGET):
    """Precision and recall when only the top `budget` share of rows can be reviewed."""
    y = np.asarray(y)
    k = max(1, int(round(budget * len(y))))
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    hits = int(y[top].sum())
    return {"precision": hits / k, "recall": hits / max(1, int(y.sum())), "n_reviewed": k}


def calibration(y, prob):
    """Brier score and how far the mean prediction sits from the observed fraud rate."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    base_rate = float(y.mean())
    return {
        "brier": float(brier_score_loss(y, prob)),
        "mean_predicted": float(prob.mean()),
        "base_rate": base_rate,
        "predicted_to_observed": float(prob.mean() / base_rate) if base_rate else float("nan"),
    }


def reliability_curve(y, prob, edges=RELIABILITY_EDGES):
    """Mean prediction against observed fraud rate in fixed probability bins."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    bins = np.clip(np.digitize(prob, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.any():
            rows.append(
                {
                    "mean_predicted": float(prob[mask].mean()),
                    "observed": float(y[mask].mean()),
                    "n": int(mask.sum()),
                }
            )
    return rows
