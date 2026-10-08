"""Markdown tables of every result, generated from the JSON files in reports/metrics."""

import json

from . import config as C

SUMMARY_FILE = C.ROOT / "reports" / "summary.md"
MODELS = ("LightGBM", "Logistic regression", "Isolation Forest", "Autoencoder")
STRATEGIES = {
    "none": "no rebalancing",
    "class_weight": "class weights",
    "smote": "SMOTE",
    "undersample": "undersampling",
}
SCENARIOS = {
    "S0_control": "S0 control, nothing injected",
    "S1_unused_feature_shift": "S1 one-SD shift on the model's least-used feature",
    "S2_key_feature_shift": "S2 one-SD shift on the model's key feature, toward fraud",
    "S3_attack_wave": "S3 attack wave, fraud volume x4",
    "S4_new_tactic": "S4 new fraud tactic, 70% of frauds change signature",
}
RULES = {
    "W_feature_psi": "feature PSI",
    "W_weighted_share": "weighted share",
    "W_score_psi": "score PSI",
    "W_alert_rate": "alert rate",
    "W_anomaly_rate": "anomaly rate",
    "R1_performance": "R1 performance",
    "R2_prediction_drift": "R2 prediction drift",
    "R3_corroborated_input_drift": "R3 corroborated input drift",
    "retrain_flag_still_open": "flag still open",
}


def _load(name):
    return json.loads((C.METRIC_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _f(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}%}"


def _hours(window):
    return f"{6 * window}-{6 * window + 6} h"


def _rules(names):
    return ", ".join(RULES.get(n, n) for n in names) or "none"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _data():
    d = _load("data_summary")
    rows = [
        (name, f"{s['from_hour']:.0f}-{s['to_hour']:.0f}", f"{s['rows']:,}", s["frauds"], _pct(s["fraud_rate"], 3))
        for name, s in d["splits"].items()
    ]
    windows = [
        (_hours(w["window"]), f"{w['rows']:,}", w["frauds"], _pct(w["fraud_rate"], 3)) for w in d["windows"]
    ]
    amount = d["fraud_amount"]
    return "\n\n".join(
        [
            "## Data",
            f"{d['rows']:,} transactions, {d['frauds']} frauds ({_pct(d['fraud_rate'], 3)}), "
            f"{d['hours']:.0f} hours, {d['duplicate_rows']:,} duplicate rows kept. File md5 `{d['md5']}`.",
            _table(("Split", "Hours", "Rows", "Frauds", "Fraud rate"), rows),
            _table(("Window", "Rows", "Frauds", "Fraud rate"), windows),
            f"Fraud amounts: median EUR {amount['median']:.2f}, mean EUR {amount['mean']:.2f}, "
            f"{_pct(amount['share_at_most_1_eur'])} are EUR 1 or less, and the largest tenth of frauds "
            f"holds {_pct(amount['top_decile_share_of_value'])} of the fraud value.",
        ]
    )


def _imbalance():
    d = _load("imbalance")
    rows = [
        (
            "LightGBM" if r["model"] == "lgbm" else "Logistic regression",
            STRATEGIES[r["strategy"]],
            r["n_seeds"],
            _f(r["val_ap"]),
            _f(r["test_ap"]) + ("" if r["test_ap_sd"] is None else f" (SD {_f(r['test_ap_sd'])})"),
            _f(r["test_roc_auc"]),
            _f(r["val_predicted_to_observed"], 1),
        )
        for r in d["summary"]
    ]
    default = d["lgbm_default_params"]
    return "\n\n".join(
        [
            "## Imbalance strategies",
            "Fitted on day 1. Validation is hours 24-36, test is hours 36-48. The last column is the mean "
            "predicted probability divided by the observed fraud rate on validation; 1.0 is calibrated.",
            _table(
                ("Model", "Strategy", "Seeds", "Validation AP", "Test AP", "Test ROC-AUC", "Predicted / observed"),
                rows,
            ),
            f"Chosen: LightGBM with {STRATEGIES[d['chosen']['lgbm']]}, logistic regression with "
            f"{STRATEGIES[d['chosen']['logreg']]}. Rule: {d['selection_rule'].splitlines()[0]} "
            f"(standard error of validation AP: {_f(d['val_ap_standard_error']['lgbm'])}).",
            f"LightGBM with library-default parameters: validation AP {_f(default['val_ap'])}, "
            f"test AP {_f(default['test_ap'])}, test ROC-AUC {_f(default['test_roc_auc'])}.",
            f"The chosen LightGBM on a shuffled 75/25 split scores AP {_f(d['random_split_ap'])}; its "
            f"out-of-fold AP on day 1 is {_f(d['train_oof_ap'])}.",
        ]
    )


def _models():
    d = _load("models")
    rows = [
        (
            r["model"],
            r["split"],
            f"{_f(r['ap'])} ({_f(r['ap_lo'])}-{_f(r['ap_hi'])})",
            _f(r["roc_auc"]),
            _f(r["precision_at_budget"]),
            _f(r["recall_at_budget"]),
        )
        for r in d["results"]
    ]
    by_window = {}
    for r in d["per_window"]:
        by_window.setdefault(r["window"], {"n_frauds": r["n_frauds"]})[r["model"]] = r["ap"]
    windows = [
        (_hours(w), "day 1, out-of-fold" if w < 4 else "day 2", v["n_frauds"], *(_f(v.get(m)) for m in MODELS))
        for w, v in sorted(by_window.items())
    ]
    return "\n\n".join(
        [
            "## Supervised vs unsupervised",
            f"AP with a bootstrap 95% interval. Budget columns review the top {_pct(d['alert_budget'])} of "
            f"transactions by score. Test fraud rate {_pct(d['test_fraud_rate'], 3)}, which is the AP of a random ranker.",
            _table(("Model", "Split", "AP (95% CI)", "ROC-AUC", "Precision at budget", "Recall at budget"), rows),
            "AP by 6-hour window:",
            _table(("Window", "Source", "Frauds", *MODELS), windows),
        ]
    )


def _threshold():
    d = _load("threshold")
    cm = d["cost_model"]
    names = {
        "cost_minimising": f"Cost-minimising threshold ({d['threshold']:.4f})",
        "threshold_0.5": "Threshold 0.5",
        "max_f1": f"Max-F1 threshold ({d['max_f1_threshold']:.3f})",
        "amount_aware": "Amount-aware rule on calibrated probabilities",
        "no_model": "No model",
    }
    rows = [
        (
            names[key],
            r["tp"],
            r["fn"],
            r["fp"],
            _f(r["precision"]),
            _f(r["recall"]),
            f"{r['cost']:,.0f}",
            _pct(r["saving"]),
        )
        for key in names
        for r in [d["test"][key]]
    ]
    base = [
        (f"{r['fp_cost']:g}", f"{r['threshold']:.4f}", _f(r["test_precision"]), _f(r["test_recall"]), r["test_fp"], _pct(r["test_saving"]))
        for r in d["sensitivity"]
        if r["fn_multiplier"] == cm["fn_multiplier"] and r["fn_fixed"] == cm["fn_fixed"]
    ]
    boot, cal = d["threshold_bootstrap"], d["calibration_test"]
    return "\n\n".join(
        [
            "## Threshold and cost",
            f"Assumed costs (EUR): a missed fraud costs {cm['fn_multiplier']:g} x its amount plus {cm['fn_fixed']:g}; "
            f"a blocked legitimate payment costs {cm['fp_cost']:g}; a correctly blocked fraud costs {cm['tp_cost']:g}.",
            f"The threshold is the cost minimum on validation. Across bootstrap resamples of the validation rows the "
            f"minimum sits at {boot['p10']:.4f} (10th percentile), {boot['median']:.4f} (median) and {boot['p90']:.4f} (90th).",
            "Decision rules on the test window:",
            _table(
                ("Rule", "Caught", "Missed", "False alarms", "Precision", "Recall", "Cost (EUR)", "Saving vs no model"),
                rows,
            ),
            "The threshold re-chosen on validation as the cost of a false alarm changes, other costs at base values:",
            _table(("False-alarm cost (EUR)", "Threshold", "Test precision", "Test recall", "Test false alarms", "Saving"), base),
            f"Calibration on test, mean predicted / observed fraud rate: raw scores {_f(cal['raw']['predicted_to_observed'], 2)}, "
            f"after Platt scaling {_f(cal['platt']['predicted_to_observed'], 2)}.",
        ]
    )


def _window_rows(windows):
    return [
        (
            _hours(w["window"]),
            _f(w["max_psi"], 2),
            w["n_features_drifted"],
            _f(w["weighted_share"], 2),
            _f(w["score_psi"]),
            _f(w["alert_ratio"], 2),
            w["n_frauds"],
            _f(w["precision"], 2),
            _f(w["recall"], 2),
            _f(w["ap_adjusted"]),
            _f(w["ap_reference"]),
            w["label_free_state"],
            w["state"],
            _rules(w["rules"]),
        )
        for w in windows
    ]


WINDOW_HEADERS = (
    "Window",
    "Max feature PSI",
    "Features over 0.25",
    "Weighted share",
    "Score PSI",
    "Alert ratio",
    "Frauds",
    "Precision",
    "Recall",
    "AP at reference fraud rate",
    "Reference AP",
    "State without labels",
    "State with labels",
    "Rules fired",
)


def _drift():
    d = _load("drift")
    s, null = d["summary"], d["null_control"]
    top = sorted(d["features"], key=lambda r: -r["psi"])[:8]
    key = sorted(d["features"], key=lambda r: -r["gain_share"])[:5]
    feature_rows = [(r["feature"], _f(r["psi"]), _f(r["ks"]), _pct(r["gain_share"])) for r in top + key]
    anomaly = [
        (_hours(w["window"]), *(_f(v, 2) for v in w["anomaly_ratio"].values())) for w in d["natural_windows"]
    ]
    noise = ", ".join(f"{r['mean_max_psi']:.3f} at {r['batch_rows']:,} rows" for r in d["psi_noise"])
    return "\n\n".join(
        [
            "## Natural drift, day 1 against day 2",
            f"Of {s['n_features']} features, {s['psi_ge_0.10']} have PSI of 0.10 or more and {s['psi_ge_0.25']} have 0.25 or more. "
            f"All {s['ks_significant_after_bonferroni']} are significant on a KS test after Bonferroni correction, so p-values "
            f"separate nothing at this sample size. Features over the 0.25 bound are {_pct(s['unweighted_share'])} of the "
            f"feature count but {_pct(s['weighted_share'])} of model gain. A classifier separates day 1 from day 2 with "
            f"ROC-AUC {_f(s['domain_auc'])}. Whole-day score PSI is {_f(s['score_psi'])}. Raw `Time` has PSI {_f(s['time_psi'], 1)}.",
            "The eight most-drifted features, then the five the model uses most:",
            _table(("Feature", "PSI", "KS statistic", "Share of model gain"), feature_rows),
            f"Sampling noise, mean largest feature PSI of a random batch against its own population: {noise}.",
            "### Monitor on the real later window",
            "Each window is compared with the same hours of day 1.",
            _table(WINDOW_HEADERS, _window_rows(d["natural_windows"])),
            "Anomaly-rate ratio by detector:",
            _table(("Window", *d["natural_windows"][0]["anomaly_ratio"]), anomaly),
            f"False-alarm control, random halves of each day-1 window compared with each other: {null['checks']} checks, "
            f"{null['warn']} WARN, {null['retrain']} RETRAIN.",
        ]
    )


def _simulation():
    d = _load("simulation")
    bed, day2 = d["controlled"], d["day2"]
    parts = [
        "## Injected drift on a drift-free test bed",
        f"The model is refitted on a random half of day 1 ({bed['fit_rows']:,} rows) and monitors the other half "
        f"({bed['stream_rows']:,} rows). Scenarios are injected into hours "
        f"{6 * bed['drift_windows'][0]}-{6 * bed['drift_windows'][-1] + 6}. Least-used feature: "
        f"{bed['unused_feature']['name']} ({_pct(bed['unused_feature']['gain_share'])} of gain). Key feature: "
        f"{bed['key_feature']['name']} ({_pct(bed['key_feature']['gain_share'])} of gain). The new tactic hides in "
        f"{', '.join(bed['new_tactic']['hidden_features'])} and moves four SDs on "
        f"{', '.join(bed['new_tactic']['signature_features'])}.",
        "The reference AP here comes from out-of-fold models that each saw two-thirds of the fitting half, "
        "so it sits below what the deployed model achieves on the untouched stream.",
    ]
    for name, scenario in bed["scenarios"].items():
        injected = [w for w in scenario["windows"] if w["window"] in bed["drift_windows"]]
        parts += [f"**{SCENARIOS[name]}**", _table(WINDOW_HEADERS, _window_rows(injected))]

    dose = [
        (
            f"{r['dose_sd']:+g}",
            _f(r["feature_psi"], 2),
            _f(r["score_psi"]),
            _f(r["alert_ratio"], 2),
            _f(r["precision"], 2),
            _f(r["recall"], 2),
            " then ".join(r["label_free_states"]),
        )
        for r in bed["dose_response"]
    ]
    parts += [
        f"Dose response, shifting {bed['key_feature']['name']} by a number of SDs (positive is toward fraud). "
        "Signals are from the first injected window:",
        _table(("Shift (SD)", "Feature PSI", "Score PSI", "Alert ratio", "Precision", "Recall", "States without labels"), dose),
    ]

    detectors = [k[: -len("_in_budget")] for k in day2["tactic_visibility"][0] if k.endswith("_in_budget")]
    visibility = [
        (
            f"{r['strength_sd']:g}",
            r["new_tactic_frauds"],
            r["caught_by_model"],
            *(r[f"{det}_in_budget"] for det in detectors),
            *(_f(r[f"{det}_anomaly_ratio"], 2) for det in detectors),
        )
        for r in day2["tactic_visibility"]
    ]
    parts += [
        "## New tactic on day 2: who sees it",
        f"Injected into hours 36-48 of the real stream. A detector sees a fraud when it ranks in the top "
        f"{_pct(C.ALERT_BUDGET)} of anomaly scores.",
        _table(
            (
                "Signature (SD)",
                "New-tactic frauds",
                "Caught by LightGBM",
                *(f"Seen by {det}" for det in detectors),
                *(f"Anomaly-rate ratio, {det}" for det in detectors),
            ),
            visibility,
        ),
        "## Retrain policies on day 2",
        "Labels for a window arrive when the next one starts. Static never retrains, periodic retrains every "
        "window, triggered retrains when the monitor says RETRAIN.",
    ]
    for name, heading in (("S0_control", "Real stream"), ("S4_new_tactic", "Real stream with the new tactic from hour 36")):
        totals = [
            (r["policy"], r["retrains"], f"{r['cost']:,.0f}", r["frauds_missed"], r["frauds"], r["false_alarms"])
            for r in day2["replay_totals"][name]
        ]
        by_window = [
            (
                r["policy"],
                _hours(r["window"]),
                "yes" if r["retrained_before"] else "",
                _f(r["ap"]),
                _f(r["precision"], 2),
                _f(r["recall"], 2),
                _f(r["cost_per_1k"], 1),
            )
            for r in day2["replay"][name]
        ]
        parts += [
            f"**{heading}**",
            _table(("Policy", "Retrains", "Total cost (EUR)", "Frauds missed", "Frauds", "False alarms"), totals),
            _table(("Policy", "Window", "Retrained before", "AP", "Precision", "Recall", "EUR per 1,000"), by_window),
        ]
    return "\n\n".join(parts)


def write():
    sections = [
        "# Results summary",
        "Generated by `python -m fraudmon evaluate` from the JSON files in `reports/metrics/`. "
        "Do not edit by hand; rerun the step instead.",
        _data(),
        _imbalance(),
        _models(),
        _threshold(),
        _drift(),
        _simulation(),
    ]
    SUMMARY_FILE.write_text("\n\n".join(sections) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {SUMMARY_FILE.relative_to(C.ROOT)}", flush=True)
