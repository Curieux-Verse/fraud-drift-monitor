"""Cost of false positives and false negatives, and threshold selection."""

import numpy as np
import pandas as pd

from . import config as C


def missed_loss(amount, cm):
    """Loss in EUR when a fraud of this amount is not flagged."""
    return cm.fn_multiplier * np.asarray(amount, dtype=float) + cm.fn_fixed


def evaluate(y, amount, flagged, cm):
    """Total cost of a set of flags, with its breakdown."""
    y = np.asarray(y)
    flagged = np.asarray(flagged, dtype=bool)
    loss = missed_loss(amount, cm)
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    fn_cost = float(loss[~flagged & (y == 1)].sum())
    total = fn_cost + cm.fp_cost * fp + cm.tp_cost * tp
    no_model = float(loss[y == 1].sum())
    return {
        "cost": total,
        "fn_cost": fn_cost,
        "fp_cost": cm.fp_cost * fp,
        "tp_cost": cm.tp_cost * tp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "cost_per_1k": 1000 * total / len(y),
        "no_model_cost": no_model,
        "saving": 1 - total / no_model if no_model else float("nan"),
    }


def sweep(y, scores, amount, cm):
    """Total cost at every distinct threshold. The first row flags nothing."""
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    y_s, s_s, loss_s = y[order], scores[order], missed_loss(amount, cm)[order]
    group_end = np.flatnonzero(np.concatenate([np.diff(s_s) != 0, [True]]))
    tp = np.cumsum(y_s)[group_end]
    fp = np.cumsum(1 - y_s)[group_end]
    total_loss = float((loss_s * y_s).sum())
    caught = np.cumsum(loss_s * y_s)[group_end]
    cost = (total_loss - caught) + cm.fp_cost * fp + cm.tp_cost * tp
    # Halfway to the next lower score, so "score >= threshold" flags exactly this group and above.
    distinct = s_s[group_end]
    threshold = (distinct + np.concatenate([distinct[1:], distinct[-1:]])) / 2
    table = pd.DataFrame({"threshold": threshold, "cost": cost, "tp": tp, "fp": fp})
    nothing = pd.DataFrame({"threshold": [s_s[0] + 1.0], "cost": [total_loss], "tp": [0.0], "fp": [0.0]})
    return pd.concat([nothing, table], ignore_index=True)


def best_threshold(y, scores, amount, cm):
    table = sweep(y, scores, amount, cm)
    return float(table.loc[table["cost"].idxmin(), "threshold"])


def bootstrap_thresholds(y, scores, amount, cm, n_boot=C.N_BOOT, seed=C.SEED):
    """Cost-minimising threshold on resamples of the rows, to show how stable the choice is."""
    y, scores, amount = np.asarray(y), np.asarray(scores), np.asarray(amount)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        out[i] = best_threshold(y[idx], scores[idx], amount[idx], cm)
    return out


def max_f1_threshold(y, scores):
    """The threshold a cost-blind analyst would pick."""
    table = sweep(y, scores, np.zeros(len(y)), C.CostModel())
    f1 = 2 * table["tp"] / (table["tp"] + table["fp"] + float(np.sum(y)))
    return float(table.loc[f1.idxmax(), "threshold"])


def amount_aware_flags(prob, amount, cm):
    """Flag when letting the payment through is expected to cost more than blocking it."""
    prob = np.asarray(prob, dtype=float)
    return prob * missed_loss(amount, cm) > prob * cm.tp_cost + (1 - prob) * cm.fp_cost


def sensitivity(val, test, cost_models):
    """Re-pick the threshold on validation under each cost model and score it on test.

    val and test are (labels, scores, amounts) triples.
    """
    rows = []
    for cm in cost_models:
        threshold = best_threshold(*val, cm)
        outcome = evaluate(test[0], test[2], np.asarray(test[1]) >= threshold, cm)
        rows.append(
            {
                "fp_cost": cm.fp_cost,
                "fn_fixed": cm.fn_fixed,
                "fn_multiplier": cm.fn_multiplier,
                "threshold": threshold,
                "test_precision": outcome["precision"],
                "test_recall": outcome["recall"],
                "test_fp": outcome["fp"],
                "test_saving": outcome["saving"],
            }
        )
    return rows
