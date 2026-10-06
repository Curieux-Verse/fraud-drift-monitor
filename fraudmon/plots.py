"""Figures for the report, drawn only from the JSON files in reports/metrics."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, BLUE_LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#86b6ef"

# Colour follows the entity in every figure.
MODEL_COLOURS = {
    "LightGBM": BLUE,
    "Logistic regression": ORANGE,
    "Isolation Forest": AQUA,
    "Autoencoder": YELLOW,
}
POLICY_COLOURS = {"static": BLUE, "periodic": ORANGE, "triggered": AQUA}
STRATEGY_NAMES = {
    "none": "No rebalancing",
    "class_weight": "Class weights",
    "smote": "SMOTE",
    "undersample": "Undersampling",
}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.constrained_layout.use": True,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_2,
}
MARKER = dict(marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5)


def _load(name):
    return json.loads((C.METRIC_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path.relative_to(C.ROOT)}", flush=True)


def _plain(value, _position=None):
    return f"{value:g}"


def _bound(ax, value, label, vertical=False, align="left", at_top=True):
    """A threshold line with a label that sits on it."""
    line = ax.axvline if vertical else ax.axhline
    line(value, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    if vertical:
        ax.annotate(label, (value, 1.0 if at_top else 0.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -3 if at_top else 3), textcoords="offset points",
                    va="top" if at_top else "bottom", ha="left", color=INK_2, fontsize=8)
    else:
        x = 0.0 if align == "left" else 1.0
        ax.annotate(label, (x, value), xycoords=("axes fraction", "data"),
                    xytext=(3 if align == "left" else -3, 3), textcoords="offset points",
                    va="bottom", ha=align, color=INK_2, fontsize=8)


def _window_label(window):
    return f"{6 * window}-{6 * window + 6}h"


def pr_vs_roc():
    m = _load("models")
    test = {r["model"]: r for r in m["results"] if r["split"] == "test"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for name, colour in MODEL_COLOURS.items():
        curve = m["curves"][name]
        left.plot(curve["pr"]["recall"], curve["pr"]["precision"], color=colour,
                  label=f"{name} (AP {test[name]['ap']:.2f})")
        right.plot(curve["roc"]["fpr"], curve["roc"]["tpr"], color=colour,
                   label=f"{name} (AUC {test[name]['roc_auc']:.2f})")
    left.set(title="Precision-recall: the models are far apart", xlabel="Recall",
             ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03))
    right.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8)
    right.set(title="ROC: all four look strong", xlabel="False positive rate",
              ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.03))
    # The middle-left of the PR panel and the lower right of the ROC panel hold no data.
    left.legend(loc="center left", bbox_to_anchor=(0.02, 0.42))
    right.legend(loc="lower right")
    fig.suptitle(
        f"Test window (36-48 h), {m['test_fraud_rate']:.2%} fraud. A random ranker scores AP "
        f"{m['test_fraud_rate']:.4f} and AUC 0.50.",
        x=0.01, ha="left", fontsize=8.5, color=INK_2,
    )
    _save(fig, "pr_vs_roc")


def imbalance():
    summary = _load("imbalance")["summary"]
    strategies = list(STRATEGY_NAMES)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True)
    for model, label, colour, offset in (
        ("lgbm", "LightGBM", BLUE, -0.15),
        ("logreg", "Logistic regression", ORANGE, 0.15),
    ):
        rows = {r["strategy"]: r for r in summary if r["model"] == model}
        ys = [i + offset for i in range(len(strategies))]
        ap = [rows[s]["test_ap"] for s in strategies]
        sd = [rows[s]["test_ap_sd"] or 0.0 for s in strategies]
        left.errorbar(ap, ys, xerr=sd, fmt="none", ecolor=colour, elinewidth=1.6)
        left.plot(ap, ys, linestyle="none", color=colour, label=label, **MARKER)
        right.plot([rows[s]["val_predicted_to_observed"] for s in strategies], ys,
                   linestyle="none", color=colour, **MARKER)
    left.set_yticks(range(len(strategies)), [STRATEGY_NAMES[s] for s in strategies])
    left.invert_yaxis()
    left.set(title="Test average precision", xlabel="AP (mean and SD over seeds)", xlim=(0, 1))
    right.set_xscale("log")
    right.set_xlim(0.7, 150)
    right.xaxis.set_major_formatter(_plain)
    right.set(title="Mean predicted probability / observed fraud rate",
              xlabel="Ratio on validation, log scale (1 = calibrated)")
    _bound(right, 1.0, "calibrated", vertical=True, at_top=False)
    for ax in (left, right):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelcolor=INK_2, length=0)
    left.legend(loc="upper left")
    _save(fig, "imbalance")


def cost_threshold():
    t = _load("threshold")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(t["curve_validation"]["threshold"], t["curve_validation"]["cost_per_1k"], color=BLUE,
            label="Validation (threshold chosen here)")
    ax.plot(t["curve_test"]["threshold"], t["curve_test"]["cost_per_1k"], color=BLUE_LIGHT,
            label="Test")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(_plain)
    ax.set_ylim(bottom=0)
    _bound(ax, t["threshold"], f"chosen threshold {t['threshold']:.4f}", vertical=True)
    ax.set(title="Cost of errors against the alert threshold",
           xlabel="Score threshold (log scale); lower flags more payments",
           ylabel="EUR per 1,000 transactions")
    ax.legend(loc="lower right")
    _save(fig, "cost_threshold")


def drift_vs_importance():
    d = _load("drift")
    rows = d["features"]
    floor = 1e-4
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot([max(r["gain_share"], floor) for r in rows], [r["psi"] for r in rows],
            linestyle="none", color=BLUE, **MARKER)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(lambda value, _position: f"{value:.0%}" if value >= 0.01 else f"{value:.1%}")
    ax.yaxis.set_major_formatter(_plain)
    _bound(ax, d["bounds"]["psi_alert"], "PSI 0.25: conventional alert bound", align="right")
    by_gain = sorted(rows, key=lambda r: -r["gain_share"])[:3]
    by_psi = sorted(rows, key=lambda r: -r["psi"])[:4]
    for r in {r["feature"]: r for r in by_gain + by_psi}.values():
        ax.annotate(r["feature"], (max(r["gain_share"], floor), r["psi"]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=8)
    ax.set(title="The features that drifted are not the ones the model relies on",
           xlabel="Share of model gain (log scale)",
           ylabel="PSI, day 1 vs day 2 (log scale)")
    _save(fig, "drift_vs_importance")


def dose_response():
    bed = _load("simulation")["controlled"]
    bounds = _load("drift")["bounds"]
    rows = sorted(bed["dose_response"], key=lambda r: r["dose_sd"])
    dose = [r["dose_sd"] for r in rows]
    # (title, key, bounds as (value, label, label side), y ticks on a log axis or None for linear)
    panels = (
        ("PSI of the shifted feature (log)", "feature_psi",
         [(bounds["psi_alert"], "warn", "left")], [0.001, 0.01, 0.1, 1, 10]),
        ("Score PSI (log)", "score_psi",
         [(bounds["score_psi_warn"], "warn", "right"), (bounds["score_psi_retrain"], "retrain", "left")],
         [0.01, 0.1, 1, 10]),
        ("Alert rate / reference (log)", "alert_ratio",
         [(bounds["alert_ratio_retrain"][0], "retrain below", "right"),
          (bounds["alert_ratio_retrain"][1], "retrain above", "left")], [0.25, 0.5, 1, 2, 4, 8]),
        ("Precision at the threshold", "precision", [], None),
    )
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for ax, (title, key, lines, ticks) in zip(axes, panels):
        floor = ticks[0] if ticks else 0.0  # keep near-zero values on the log axis
        ax.plot(dose, [max(r[key], floor) for r in rows], color=BLUE, **MARKER)
        if ticks:
            ax.set_yscale("log")
            ax.set_yticks(ticks, [_plain(t) for t in ticks])
            ax.minorticks_off()
            ax.set_ylim(ticks[0], ticks[-1])
        else:
            ax.set_ylim(0, 1)
        for value, label, side in lines:
            _bound(ax, value, label, align=side)
        ax.set(title=title, xlabel=f"Shift of {bed['key_feature']['name']} (SD)")
        ax.set_xticks([-2, -1, 0, 1, 2])
    fig.suptitle(
        "Negative shifts move every payment away from the fraud region, positive shifts toward it. "
        "Feature PSI is symmetric; the alert rate and precision show which direction hurts.",
        x=0.01, ha="left", fontsize=8.5, color=INK_2,
    )
    _save(fig, "dose_response")


def retrain_replay():
    sim = _load("simulation")["day2"]
    onset = min(sim["drift_windows"])
    columns = (("S0_control", "No injected drift"), ("S4_new_tactic", "New fraud tactic from 36 h"))
    measures = (("ap", "Average precision"), ("cost_per_1k", "EUR per 1,000 transactions"))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4), sharex=True, sharey="row")
    # Policies often coincide, so each is nudged sideways to keep all three visible.
    dodge = dict(zip(POLICY_COLOURS, (-0.08, 0.0, 0.08)))
    for col, (scenario, heading) in enumerate(columns):
        records = sim["replay"][scenario]
        windows = sorted({r["window"] for r in records})
        for row, (key, label) in enumerate(measures):
            ax = axes[row, col]
            for policy, colour in POLICY_COLOURS.items():
                series = sorted((r for r in records if r["policy"] == policy), key=lambda r: r["window"])
                retrains = sum(r["retrained_before"] for r in series)
                ax.plot([r["window"] + dodge[policy] for r in series], [r[key] for r in series],
                        color=colour, label=f"{policy} ({retrains} retrain{'s' if retrains != 1 else ''})",
                        **MARKER)
            if scenario != "S0_control":
                _bound(ax, onset - 0.5, "tactic starts" if row == 0 else "", vertical=True)
            ax.set_xticks(windows, [_window_label(w) for w in windows])
            if row == 0:
                ax.set_title(heading)
                ax.legend(loc="lower left")
            if col == 0:
                ax.set_ylabel(label)
    axes[0, 0].set_ylim(0, 1)
    highest_cost = max(r["cost_per_1k"] for records in sim["replay"].values() for r in records)
    axes[1, 0].set_ylim(0, 1.08 * highest_cost)
    _save(fig, "retrain_replay")


FIGURES = (pr_vs_roc, imbalance, cost_threshold, drift_vs_importance, dose_response, retrain_replay)


def make_all():
    with plt.rc_context(STYLE):
        for figure in FIGURES:
            figure()
