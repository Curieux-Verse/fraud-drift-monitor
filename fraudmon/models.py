"""Supervised classifiers, imbalance strategies and unsupervised anomaly detectors."""

import copy

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from . import config as C
from .metrics import logit


def _sampler(strategy, seed):
    # Library defaults: both resample the training data to a 1:1 class ratio.
    if strategy == "smote":
        return SMOTE(random_state=seed)
    if strategy == "undersample":
        return RandomUnderSampler(random_state=seed)
    return None


def make_supervised(model, strategy, seed=C.SEED, lgbm_params=None):
    """Classifier with one imbalance strategy. Samplers only ever see the data passed to fit."""
    if strategy not in C.STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    class_weight = "balanced" if strategy == "class_weight" else None
    steps = []
    if model == "logreg":
        steps.append(("scale", StandardScaler()))
        clf = LogisticRegression(max_iter=3000, class_weight=class_weight)
    elif model == "lgbm":
        params = dict(C.LGBM_PARAMS if lgbm_params is None else lgbm_params)
        clf = LGBMClassifier(**params, class_weight=class_weight, random_state=seed)
    else:
        raise ValueError(f"unknown model: {model}")
    sampler = _sampler(strategy, seed)
    if sampler is not None:
        steps.append(("sample", sampler))
    steps.append(("clf", clf))
    return Pipeline(steps)


def predict(model, frame):
    return model.predict_proba(frame[C.FEATURES])[:, 1]


def gain_importance(model):
    """Share of total split gain per feature for a LightGBM pipeline, largest first."""
    booster = model.named_steps["clf"].booster_
    gain = pd.Series(booster.feature_importance(importance_type="gain"), index=C.FEATURES, dtype=float)
    return (gain / gain.sum()).sort_values(ascending=False)


def _folds(y, seed, n_folds=C.N_FOLDS):
    return StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def oof_predict(model, frame, seed=C.SEED, n_folds=C.N_FOLDS):
    """Out-of-fold scores: what the model outputs on rows it was not fitted on."""
    X, y = frame[C.FEATURES], frame["Class"].to_numpy()
    out = np.empty(len(frame))
    for fit_idx, hold_idx in _folds(y, seed, n_folds):
        fold = clone(model).fit(X.iloc[fit_idx], y[fit_idx])
        out[hold_idx] = fold.predict_proba(X.iloc[hold_idx])[:, 1]
    return out


def fit_platt(scores, y):
    """Platt scaling: a one-parameter-pair logistic fit on the logit of the raw score."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(logit(scores).reshape(-1, 1), y)


def apply_platt(calibrator, scores):
    return calibrator.predict_proba(logit(scores).reshape(-1, 1))[:, 1]


class IsolationForestDetector:
    """Isolation Forest fitted on every training row; labels are not used."""

    def __init__(self, seed=C.SEED):
        self.seed = seed
        self.model = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.model = IsolationForest(
            n_estimators=300, max_samples=min(4096, len(X)), random_state=self.seed, n_jobs=C.N_JOBS
        ).fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(np.asarray(X, dtype=float))


class AutoencoderDetector:
    """Autoencoder fitted on legitimate rows only; the score is the reconstruction error."""

    def __init__(self, hidden=(20, 10), epochs=40, batch_size=512, lr=1e-3, patience=5, seed=C.SEED):
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seed = seed
        self.scaler = None
        self.net = None

    def _build(self, n_features):
        sizes = [n_features, *self.hidden]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        for a, b in zip(sizes[:0:-1], sizes[-2::-1]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        return nn.Sequential(*layers[:-1])  # linear output layer

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            X = X[np.asarray(y) == 0]
        torch.manual_seed(self.seed)
        torch.set_num_threads(C.N_JOBS)
        self.scaler = StandardScaler().fit(X)
        Z = torch.from_numpy(self.scaler.transform(X).astype(np.float32))
        order = np.random.default_rng(self.seed).permutation(len(Z))
        n_hold = max(1, len(Z) // 10)
        hold, fit = Z[order[:n_hold]], Z[order[n_hold:]]

        self.net = self._build(Z.shape[1])
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        # Huber-style loss: the standardised features have very heavy tails.
        loss_fn = nn.SmoothL1Loss()
        shuffle = torch.Generator().manual_seed(self.seed)
        best, best_state, stale = float("inf"), None, 0
        for _ in range(self.epochs):
            self.net.train()
            idx = torch.randperm(len(fit), generator=shuffle)
            for start in range(0, len(fit), self.batch_size):
                batch = fit[idx[start : start + self.batch_size]]
                optimiser.zero_grad()
                loss_fn(self.net(batch), batch).backward()
                optimiser.step()
            self.net.eval()
            with torch.no_grad():
                hold_loss = loss_fn(self.net(hold), hold).item()
            if hold_loss < best - 1e-5:
                best, best_state, stale = hold_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def score(self, X):
        Z = torch.from_numpy(self.scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32))
        self.net.eval()
        with torch.no_grad():
            error = ((self.net(Z) - Z) ** 2).mean(dim=1)
        return error.numpy().astype(float)


DETECTORS = {"isolation_forest": IsolationForestDetector, "autoencoder": AutoencoderDetector}


def anomaly_score(detector, frame):
    return detector.score(frame[C.FEATURES].to_numpy())


def oof_anomaly(make_detector, frame, seed=C.SEED):
    """Out-of-fold anomaly scores, so the reference is not scored by a model that saw it."""
    X, y = frame[C.FEATURES].to_numpy(), frame["Class"].to_numpy()
    out = np.empty(len(frame))
    for fit_idx, hold_idx in _folds(y, seed):
        detector = make_detector(seed=seed).fit(X[fit_idx], y[fit_idx])
        out[hold_idx] = detector.score(X[hold_idx])
    return out


def save(obj, name):
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, C.MODEL_DIR / f"{name}.joblib")


def load(name):
    path = C.MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m fraudmon train` first.")
    return joblib.load(path)
