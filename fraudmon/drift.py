"""Drift statistics: PSI, KS statistic, importance-weighted drift and score drift."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as C
from .metrics import logit


def psi(ref, cur, bins=10, eps=1e-6):
    """Population stability index with bin edges at the reference quantiles."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    interior = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], interior, [np.inf]])
    ref_share = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    cur_share = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def feature_psi(ref, cur, features=C.FEATURES):
    return pd.Series({f: psi(ref[f].to_numpy(), cur[f].to_numpy()) for f in features})


def feature_drift(ref, cur, features=C.FEATURES):
    """PSI and the two-sample KS statistic and p-value for every feature."""
    rows = []
    for f in features:
        ks = ks_2samp(ref[f].to_numpy(), cur[f].to_numpy())
        rows.append(
            {
                "feature": f,
                "psi": psi(ref[f].to_numpy(), cur[f].to_numpy()),
                "ks": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return pd.DataFrame(rows)


def weighted_share(psi_values, importance, bound):
    """Share of model importance sitting on features whose PSI is at or above the bound."""
    weights = importance.reindex(psi_values.index).fillna(0.0)
    return float(weights[psi_values >= bound].sum())


def score_psi(ref_scores, cur_scores):
    """PSI of the model's output, on the logit scale so the low-score bulk is resolved."""
    return psi(logit(ref_scores), logit(cur_scores))


def exceed_ratio(ref_values, cur_values, cut):
    """Share of current values at or above the cut, relative to the reference share.

    Infinite when the reference has nothing above the cut and the current window does.
    """
    ref_rate = float(np.mean(np.asarray(ref_values) >= cut))
    cur_rate = float(np.mean(np.asarray(cur_values) >= cut))
    if ref_rate > 0:
        return cur_rate / ref_rate
    return float("inf") if cur_rate > 0 else float("nan")


def domain_auc(ref, cur, features=C.FEATURES, seed=C.SEED):
    """ROC-AUC of a classifier told to separate reference rows from current rows."""
    X = pd.concat([ref[features], cur[features]], ignore_index=True)
    origin = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))])
    clf = LGBMClassifier(
        n_estimators=100, num_leaves=15, random_state=seed, n_jobs=C.N_JOBS, verbose=-1
    )
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)
    prob = cross_val_predict(clf, X, origin, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(origin, prob))
