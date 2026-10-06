"""Pipeline steps behind the command line interface."""

import json
import math
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from . import config as C
from . import cost, data, drift, metrics, models, simulate
from .monitor import build_reference

LABELS = {
    "supervised": "LightGBM",
    "logreg": "Logistic regression",
    "isolation_forest": "Isolation Forest",
    "autoencoder": "Autoencoder",
}


def _log(message):
    print(message, flush=True)


def _clean(obj):
    """Make a metrics payload JSON-serialisable; non-finite numbers become null."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _clean(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 6) if math.isfinite(obj) else None
    return obj


def save_metrics(name, payload):
    C.METRIC_DIR.mkdir(parents=True, exist_ok=True)
    path = C.METRIC_DIR / f"{name}.json"
    path.write_text(json.dumps(_clean(payload), indent=2) + "\n", encoding="utf-8")
    _log(f"wrote {path.relative_to(C.ROOT)}")


def load_metrics(name):
    path = C.METRIC_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the earlier pipeline steps first.")
    return json.loads(path.read_text(encoding="utf-8"))


# --- data ---------------------------------------------------------------------------------


def step_data():
    data.download()
    df = data.load()
    parts = dict(zip(("train", "validation", "test"), data.split(df)))
    fraud_amount = df.loc[df["Class"] == 1, "Amount"].sort_values(ascending=False)
    windows = df.groupby("window")["Class"].agg(rows="size", frauds="sum").reset_index()
    windows["fraud_rate"] = windows["frauds"] / windows["rows"]
    save_metrics(
        "data_summary",
        {
            "md5": data.file_md5(C.DATA_FILE),
            "rows": len(df),
            "frauds": int(df["Class"].sum()),
            "fraud_rate": float(df["Class"].mean()),
            "hours": float(df["Time"].max() / 3600),
            "duplicate_rows": int(df.duplicated(subset=data.EXPECTED_COLUMNS).sum()),
            "splits": {
                name: {
                    "rows": len(part),
                    "frauds": int(part["Class"].sum()),
                    "fraud_rate": float(part["Class"].mean()),
                    "from_hour": float(part["Time"].min() / 3600),
                    "to_hour": float(part["Time"].max() / 3600),
                }
                for name, part in parts.items()
            },
            "windows": windows,
            "fraud_amount": {
                "median": float(fraud_amount.median()),
                "mean": float(fraud_amount.mean()),
                "total": float(fraud_amount.sum()),
                "share_at_most_1_eur": float((fraud_amount <= 1).mean()),
                "top_decile_share_of_value": float(
                    fraud_amount.iloc[: len(fraud_amount) // 10].sum() / fraud_amount.sum()
                ),
            },
        },
    )


# --- training -----------------------------------------------------------------------------


def _fit_scores(estimator, train, val, test):
    estimator.fit(train[C.FEATURES], train["Class"].to_numpy())
    return models.predict(estimator, val), models.predict(estimator, test)


def _run_record(model, strategy, seed, val, test, s_val, s_test):
    calibration = metrics.calibration(val["Class"], s_val)
    return {
        "model": model,
        "strategy": strategy,
        "seed": seed,
        "val_ap": metrics.average_precision(val["Class"], s_val),
        "test_ap": metrics.average_precision(test["Class"], s_test),
        "val_roc_auc": metrics.roc_auc(val["Class"], s_val),
        "test_roc_auc": metrics.roc_auc(test["Class"], s_test),
        "val_brier": calibration["brier"],
        "val_predicted_to_observed": calibration["predicted_to_observed"],
    }


def choose_strategy(summary, model, val_se):
    """Best validation AP, unless no rebalancing is within one standard error of it.

    Rebalancing distorts predicted probabilities, so it has to earn its place.
    """
    val_ap = summary[summary["model"] == model].set_index("strategy")["val_ap"]
    best = val_ap.idxmax()
    return "none" if val_ap["none"] >= val_ap[best] - val_se else best


def step_train():
    started = time.time()
    df = data.load()
    train, val, test = data.split(df)

    runs, val_se = [], {}
    for name in ("logreg", "lgbm"):
        for strategy in C.STRATEGIES:
            seeded = name == "lgbm" or strategy in ("smote", "undersample")
            for seed in C.SEEDS if seeded else C.SEEDS[:1]:
                estimator = models.make_supervised(name, strategy, seed)
                s_val, s_test = _fit_scores(estimator, train, val, test)
                runs.append(_run_record(name, strategy, seed, val, test, s_val, s_test))
                if strategy == "none" and seed == C.SEEDS[0]:
                    val_se[name] = float(np.nanstd(metrics.bootstrap_ap(val["Class"], s_val)))
            _log(f"fitted {name} / {strategy}")

    summary = (
        pd.DataFrame(runs)
        .groupby(["model", "strategy"], sort=False)
        .agg(
            n_seeds=("seed", "size"),
            val_ap=("val_ap", "mean"),
            val_ap_sd=("val_ap", "std"),
            test_ap=("test_ap", "mean"),
            test_ap_sd=("test_ap", "std"),
            val_roc_auc=("val_roc_auc", "mean"),
            test_roc_auc=("test_roc_auc", "mean"),
            val_brier=("val_brier", "mean"),
            val_predicted_to_observed=("val_predicted_to_observed", "mean"),
        )
        .reset_index()
    )
    chosen = {name: choose_strategy(summary, name, val_se[name]) for name in ("logreg", "lgbm")}
    _log(f"chosen strategies: {chosen}")

    defaults = dict(n_jobs=C.N_JOBS, verbose=-1, deterministic=True, force_row_wise=True)
    untuned = models.make_supervised("lgbm", "none", C.SEEDS[0], lgbm_params=defaults)
    default_params = _run_record(
        "lgbm_default_params", "none", C.SEEDS[0], val, test, *_fit_scores(untuned, train, val, test)
    )

    # The same model on a shuffled split, to show how much a random split flatters it.
    mixed_fit, mixed_hold = train_test_split(
        df, test_size=0.25, stratify=df["Class"], random_state=C.SEED
    )
    leaky = models.make_supervised("lgbm", chosen["lgbm"], C.SEED)
    leaky.fit(mixed_fit[C.FEATURES], mixed_fit["Class"].to_numpy())
    random_split_ap = metrics.average_precision(mixed_hold["Class"], models.predict(leaky, mixed_hold))

    X, y = train[C.FEATURES], train["Class"].to_numpy()
    supervised = models.make_supervised("lgbm", chosen["lgbm"], C.SEED).fit(X, y)
    logreg = models.make_supervised("logreg", chosen["logreg"], C.SEED).fit(X, y)
    models.save(supervised, "supervised")
    models.save(logreg, "logreg")
    oof = {"supervised": models.oof_predict(supervised, train)}
    for name, detector in models.DETECTORS.items():
        models.save(detector(seed=C.SEED).fit(X.to_numpy(), y), name)
        oof[name] = models.oof_anomaly(detector, train)
        _log(f"fitted {name}")
    models.save(oof, "oof")

    save_metrics(
        "imbalance",
        {
            "selection_rule": choose_strategy.__doc__.strip(),
            "val_ap_standard_error": val_se,
            "chosen": chosen,
            "summary": summary,
            "runs": runs,
            "lgbm_default_params": default_params,
            "random_split_ap": random_split_ap,
            "train_oof_ap": metrics.average_precision(y, oof["supervised"]),
        },
    )
    _log(f"train finished in {time.time() - started:.0f}s")


# --- evaluation ---------------------------------------------------------------------------


def _artifacts():
    train, val, test = data.split(data.load())
    return {
        "train": train,
        "val": val,
        "test": test,
        "later": pd.concat([val, test]),
        "model": models.load("supervised"),
        "logreg": models.load("logreg"),
        "detectors": {name: models.load(name) for name in models.DETECTORS},
        "oof": models.load("oof"),
    }


def _all_scores(art, frame):
    scores = {
        LABELS["supervised"]: models.predict(art["model"], frame),
        LABELS["logreg"]: models.predict(art["logreg"], frame),
    }
    for name, detector in art["detectors"].items():
        scores[LABELS[name]] = models.anomaly_score(detector, frame)
    return scores


def _pr_points(y, scores):
    """Precision-recall curve reduced to the best precision at each recall level."""
    precision, recall, _ = precision_recall_curve(y, scores)
    points = pd.DataFrame({"recall": recall, "precision": precision})
    points = points[points["recall"] > 0]  # sklearn appends an artificial (0, 1) end point
    points = points.groupby("recall", as_index=False)["precision"].max()
    return {"recall": points["recall"].tolist(), "precision": points["precision"].tolist()}


def _roc_points(y, scores):
    fpr, tpr, _ = roc_curve(y, scores)
    points = pd.DataFrame({"tpr": tpr, "fpr": fpr}).groupby("tpr", as_index=False)["fpr"].min()
    return {"fpr": [*points["fpr"], 1.0], "tpr": [*points["tpr"], 1.0]}


def step_compare(art=None):
    """Supervised and unsupervised models on the same later-window data."""
    art = art or _artifacts()
    later = art["later"]
    scores = _all_scores(art, later)
    y_all, window = later["Class"].to_numpy(), later["window"].to_numpy()

    results, curves = [], {}
    for split, windows in (("validation", C.VAL_WINDOWS), ("test", C.TEST_WINDOWS)):
        mask = np.isin(window, windows)
        for name, s in scores.items():
            y, s = y_all[mask], s[mask]
            budget = metrics.at_budget(y, s)
            results.append(
                {
                    "model": name,
                    "split": split,
                    **metrics.ap_with_ci(y, s),
                    "roc_auc": metrics.roc_auc(y, s),
                    "precision_at_budget": budget["precision"],
                    "recall_at_budget": budget["recall"],
                }
            )
            if split == "test":
                curves[name] = {"pr": _pr_points(y, s), "roc": _roc_points(y, s)}

    per_window = []
    for w in np.unique(window):
        mask = window == w
        for name, s in scores.items():
            per_window.append(
                {
                    "window": int(w),
                    "model": name,
                    "n_frauds": int(y_all[mask].sum()),
                    "ap": metrics.average_precision(y_all[mask], s[mask]),
                    "roc_auc": metrics.roc_auc(y_all[mask], s[mask]),
                }
            )
    # The same breakdown on the training day, from out-of-fold scores, for a like-for-like read.
    train_y, train_window = art["train"]["Class"].to_numpy(), art["train"]["window"].to_numpy()
    for w in np.unique(train_window):
        mask = train_window == w
        for name, s in art["oof"].items():
            per_window.append(
                {
                    "window": int(w),
                    "model": LABELS[name],
                    "n_frauds": int(train_y[mask].sum()),
                    "ap": metrics.average_precision(train_y[mask], s[mask]),
                    "roc_auc": metrics.roc_auc(train_y[mask], s[mask]),
                }
            )
    save_metrics(
        "models",
        {
            "alert_budget": C.ALERT_BUDGET,
            "test_fraud_rate": float(art["test"]["Class"].mean()),
            "results": results,
            "per_window": sorted(per_window, key=lambda r: (r["window"], r["model"])),
            "curves": curves,
        },
    )


def _cost_curve(y, scores, amount, cm, max_alert_rate=0.03, points=300):
    """Cost per 1,000 transactions against threshold, thinned for plotting."""
    table = cost.sweep(y, scores, amount, cm).iloc[1:]  # drop the flag-nothing row
    table = table[(table["tp"] + table["fp"]) <= max_alert_rate * len(y)]
    keep = np.unique(np.geomspace(1, len(table), points).astype(int) - 1)
    table = table.iloc[keep]
    return {
        "threshold": table["threshold"].tolist(),
        "cost_per_1k": (1000 * table["cost"] / len(y)).tolist(),
    }


def step_threshold(art=None):
    """Pick the operating threshold from the cost model and test how sensitive the choice is."""
    art = art or _artifacts()
    cm = C.CostModel()
    val, test = art["val"], art["test"]
    y_val, a_val = val["Class"].to_numpy(), val["Amount"].to_numpy()
    y_test, a_test = test["Class"].to_numpy(), test["Amount"].to_numpy()
    s_val, s_test = models.predict(art["model"], val), models.predict(art["model"], test)

    threshold = cost.best_threshold(y_val, s_val, a_val, cm)
    resampled = cost.bootstrap_thresholds(y_val, s_val, a_val, cm)
    calibrator = models.fit_platt(s_val, y_val)
    p_test = models.apply_platt(calibrator, s_test)

    rules = {
        "cost_minimising": s_test >= threshold,
        "no_model": np.zeros(len(y_test), dtype=bool),
        "threshold_0.5": s_test >= 0.5,
        "max_f1": s_test >= cost.max_f1_threshold(y_val, s_val),
        "amount_aware": cost.amount_aware_flags(p_test, a_test, cm),
    }
    grid = [
        C.CostModel(fn_multiplier=m, fn_fixed=f, fp_cost=p)
        for m in C.FN_MULTIPLIER_GRID
        for f in C.FN_FIXED_GRID
        for p in C.FP_COST_GRID
    ]
    save_metrics(
        "threshold",
        {
            "cost_model": asdict(cm),
            "threshold": threshold,
            "max_f1_threshold": cost.max_f1_threshold(y_val, s_val),
            "threshold_bootstrap": {
                "p10": float(np.quantile(resampled, 0.10)),
                "median": float(np.median(resampled)),
                "p90": float(np.quantile(resampled, 0.90)),
            },
            "validation": cost.evaluate(y_val, a_val, s_val >= threshold, cm),
            "test": {name: cost.evaluate(y_test, a_test, flags, cm) for name, flags in rules.items()},
            "calibration_test": {
                "raw": metrics.calibration(y_test, s_test),
                "platt": metrics.calibration(y_test, p_test),
                "reliability_raw": metrics.reliability_curve(y_test, s_test),
                "reliability_platt": metrics.reliability_curve(y_test, p_test),
            },
            "sensitivity": cost.sensitivity((y_val, s_val, a_val), (y_test, s_test, a_test), grid),
            "curve_validation": _cost_curve(y_val, s_val, a_val, cm),
            "curve_test": _cost_curve(y_test, s_test, a_test, cm),
        },
    )


def _reference(art):
    """The monitor's reference: day 1, out-of-fold scores, and the chosen threshold."""
    importance = models.gain_importance(art["model"])
    anomaly_oof = {name: art["oof"][name] for name in models.DETECTORS}
    threshold = load_metrics("threshold")["threshold"]
    reference = build_reference(art["train"], art["oof"]["supervised"], importance, threshold, anomaly_oof)
    return reference, importance


def _psi_noise(train, sizes=(500, 2000, 10000), reps=20, seed=C.SEED):
    """Largest feature PSI between a random batch and its own population, by batch size."""
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        worst = [
            drift.feature_psi(train, train.iloc[rng.choice(len(train), size, replace=False)]).max()
            for _ in range(reps)
        ]
        rows.append({"batch_rows": size, "mean_max_psi": float(np.mean(worst)), "max": float(np.max(worst))})
    return rows


def step_drift(art=None):
    """Natural drift between the training window and the later window, and the monitor's view of it."""
    art = art or _artifacts()
    train, later = art["train"], art["later"]
    reference, importance = _reference(art)

    table = drift.feature_drift(train, later)
    table["gain_share"] = importance.reindex(table["feature"]).to_numpy()
    table = table.sort_values("psi", ascending=False).reset_index(drop=True)
    psi_values = table.set_index("feature")["psi"]
    bounds = C.AlertBounds()

    save_metrics(
        "drift",
        {
            "features": table,
            "summary": {
                "n_features": len(table),
                "psi_ge_0.10": int((table["psi"] >= 0.10).sum()),
                "psi_ge_0.25": int((table["psi"] >= bounds.psi_alert).sum()),
                "ks_significant_after_bonferroni": int((table["ks_pvalue"] < 0.05 / len(table)).sum()),
                "ks_ge_0.10": int((table["ks"] >= 0.10).sum()),
                "unweighted_share": float((table["psi"] >= bounds.psi_alert).mean()),
                "weighted_share": drift.weighted_share(psi_values, importance, bounds.psi_alert),
                "gain_share_of_8_most_drifted": float(table["gain_share"].head(8).sum()),
                "domain_auc": drift.domain_auc(train, later),
                "score_psi": drift.score_psi(art["oof"]["supervised"], models.predict(art["model"], later)),
                "time_psi": drift.psi(train["Time"], later["Time"]),
            },
            "psi_noise": _psi_noise(train),
            "null_control": simulate.null_control(train, art["oof"], importance, reference.threshold),
            "natural_windows": simulate.monitor_stream(later, art["model"], reference, art["detectors"]),
            "bounds": asdict(bounds),
        },
    )


def _controlled(art, reference):
    """Injected scenarios on a drift-free stream, so the control really is a control."""
    train = art["train"]
    model, bed, stream = simulate.controlled_setup(train, art["model"], reference.alert_rate)
    ctx = simulate.make_context(train, bed.importance, drift_windows=C.TRAIN_WINDOWS[2:])
    hide, signature = simulate.tactic_features(ctx)

    scenarios = {}
    for name in simulate.SCENARIOS:
        injected = simulate.apply_scenario(name, stream, ctx)
        drifted = injected[injected["window"].isin(ctx.drift_windows)]
        y, s = drifted["Class"].to_numpy(), models.predict(model, drifted)
        scenarios[name] = {
            "windows": simulate.monitor_stream(injected, model, bed, {}),
            "drift_windows": {
                "ap": metrics.average_precision(y, s),
                **metrics.at_threshold(y, s, bed.threshold),
            },
        }
        _log(f"controlled {name}: {[w['state'] for w in scenarios[name]['windows']]}")
    unused, key = simulate.shifted_features(ctx)
    return {
        "fit_rows": int(sum(len(f) for f in bed.frames.values())),
        "stream_rows": len(stream),
        "threshold": bed.threshold,
        "drift_windows": list(ctx.drift_windows),
        "unused_feature": {"name": unused, "gain_share": float(bed.importance[unused])},
        "key_feature": {"name": key, "gain_share": float(bed.importance[key])},
        "new_tactic": {"hidden_features": hide, "signature_features": signature},
        "scenarios": scenarios,
        "dose_response": simulate.dose_response(
            stream, ctx, model, bed, doses=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
        ),
    }


def step_simulate(art=None):
    """Injected drift on a drift-free test bed, then the new tactic and retrain policies on day 2."""
    art = art or _artifacts()
    train, later, model = art["train"], art["later"], art["model"]
    reference, importance = _reference(art)
    controlled = _controlled(art, reference)

    ctx = simulate.make_context(train, importance)
    hide, signature = simulate.tactic_features(ctx)
    visibility = simulate.tactic_strength_sweep(
        later, ctx, model, reference, art["detectors"], strengths=(2.0, 4.0, 8.0, 16.0)
    )
    _log("swept new-tactic strength")

    replays, totals, cache = {}, {}, {}
    for name in ("S0_control", "S4_new_tactic"):
        stream = simulate.apply_scenario(name, later, ctx)
        replays[name] = [
            record
            for policy in simulate.POLICIES
            for record in simulate.replay(policy, train, stream, model, reference, cache)
        ]
        totals[name] = (
            pd.DataFrame(replays[name])
            .groupby("policy", sort=False)
            .agg(
                retrains=("retrained_before", "sum"),
                cost=("cost", "sum"),
                frauds=("n_frauds", "sum"),
                frauds_missed=("frauds_missed", "sum"),
                false_alarms=("false_alarms", "sum"),
            )
            .reset_index()
        )
        _log(f"replayed {name}")

    save_metrics(
        "simulation",
        {
            "controlled": controlled,
            "day2": {
                "drift_windows": list(ctx.drift_windows),
                "new_tactic": {"hidden_features": hide, "signature_features": signature},
                "tactic_visibility": visibility,
                "replay": replays,
                "replay_totals": totals,
            },
        },
    )


def step_plots():
    from . import plots

    plots.make_all()


def step_summary():
    from . import summary

    summary.write()


def step_evaluate():
    """Recompute every metric and figure from the saved models, without refitting them."""
    art = _artifacts()
    for step in (step_compare, step_threshold, step_drift, step_simulate):
        started = time.time()
        step(art)
        _log(f"{step.__name__} took {time.time() - started:.0f}s")
    step_plots()
    step_summary()


def step_all():
    step_data()
    step_train()
    step_evaluate()
