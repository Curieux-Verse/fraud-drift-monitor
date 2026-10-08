# Fraud detection that knows when it is going stale

Every number here comes from [reports/summary.md](reports/summary.md); [README.md](README.md) explains how to regenerate it.

LightGBM with no rebalancing is the best detector, and the monitor separates drift that only moves features from drift that hurts the model. A change in fraud tactics stayed invisible until labels arrived; retraining on the monitor's flag then recovered it in one window.

## Approach

The data is 48 hours of card payments: 284,807 rows, 492 frauds. I train on day 1, choose the threshold on hours 24 to 36 and report on hours 36 to 48, so nothing from the future reaches training. A shuffled split would have flattered the same model, 0.857 against 0.785. Features are the 28 PCA components and log amount. The monitor compares each 6-hour window with the same hours of day 1, because fraud follows the clock: it is 0.45% to 0.60% of payments in the first six hours of each day and 0.08% to 0.26% afterwards.

## Key decisions

**Imbalance: leave the data alone and constrain the model.** On test, LightGBM scored a PR-AUC of 0.789 with no rebalancing, 0.794 with class weights, 0.746 with SMOTE and 0.749 with undersampling, and the last three overstate fraud probability by 3, 7 and 54 times. Logistic regression went from 0.574 to 0.677 with class weights. So rebalancing helps the weak model and not the strong one, and because the cost rule needs honest probabilities I kept the unweighted model. The imbalance bites elsewhere: LightGBM with library defaults collapses to 0.056, so its learning rate and leaf size are held down.

**PR-AUC, not ROC-AUC, and supervised over unsupervised.** With 0.12% fraud, a 1% false-positive rate means about 925 false alarms against 115 frauds. ROC-AUC barely registers that.

| Model, test window | PR-AUC (95% interval) | ROC-AUC |
|---|---|---|
| LightGBM | 0.785 (0.703 to 0.854) | 0.979 |
| Logistic regression, class-weighted | 0.677 (0.590 to 0.767) | 0.937 |
| Isolation Forest | 0.057 (0.042 to 0.077) | 0.949 |
| Autoencoder, fitted on legitimate payments | 0.050 (0.039 to 0.065) | 0.956 |

**Threshold from costs, with the fragility stated.** I assumed a missed fraud costs its amount plus EUR 20, a blocked legitimate payment EUR 5, and a correctly blocked fraud EUR 1. Minimising cost on validation gives a threshold of 0.0085. On test it catches 95 of 115 frauds with 456 false alarms and removes 66% of the cost of having no model (EUR 6,289 against 18,713). The choice is fragile: pricing a false alarm at EUR 10 moves the optimum to 0.62. Weighing each payment's amount against its calibrated fraud probability does better than the single threshold: 72% saved with 111 false alarms.

**Alerting on what reaches the model, not on what moved.** Each window the monitor reads feature PSI with the share of model importance on drifted features, drift in the model's score and alert rate, the unsupervised anomaly rate and, once labels arrive, PR-AUC. Feature drift alone only warns. A retrain flag needs one of three things: a PR-AUC drop of 15% or more on at least 25 frauds, with an interval that excludes zero (R1); score PSI of 0.25 or more, or an alert rate outside half to double the reference, two windows running (R2); or 25% or more of model importance on drifted features plus a milder shift in the output (R3). Comparing random halves of day 1 with each other, 160 drift-free checks drew 5 warnings and one false retrain flag.

## Results

**The real later window.** Eight of 29 features cross the usual PSI bound of 0.25, and they hold 6% of the model's importance ([figure](reports/figures/drift_vs_importance.png)). The output still moved:

| Hours | Features over 0.25 | Score PSI | Alerts / reference | Precision | Recall | Without labels | With labels |
|---|---|---|---|---|---|---|---|
| 24 to 30 | 11 | 0.107 | 1.68 | 0.44 | 0.93 | WARN | WARN |
| 30 to 36 | 8 | 0.205 | 1.15 | 0.11 | 0.81 | WARN | RETRAIN (R1) |
| 36 to 42 | 8 | 0.197 | 2.32 | 0.17 | 0.86 | WARN | RETRAIN (flag open) |
| 42 to 48 | 10 | 0.178 | 2.70 | 0.18 | 0.79 | RETRAIN (R2) | RETRAIN |

Replaying day 2 says the flag was right, modestly. Retraining every window cut total cost by 9% (EUR 10,763 against 11,882). Retraining only when flagged cut it by 6% with one retrain instead of three. The gain came from fewer false alarms (389 against 719), not from catching more fraud.

**Injected drift.** Day 2 is not a clean control, so the scenarios run on a drift-free test bed: a model refitted on a random half of day 1 monitors the other half. The table shows the first injected window.

| Scenario | Max feature PSI | Score PSI | Alerts / reference | Precision | Recall | Without labels | With labels |
|---|---|---|---|---|---|---|---|
| S0 nothing injected | 0.00 | 0.023 | 1.05 | 0.49 | 0.84 | OK | OK |
| S1 one-SD shift, least-used feature | 4.04 | 0.020 | 1.09 | 0.49 | 0.86 | WARN | WARN |
| S2 one-SD shift toward fraud, key feature | 2.23 | 1.065 | 2.27 | 0.23 | 0.86 | RETRAIN | RETRAIN |
| S3 fraud volume times four | 0.00 | 0.024 | 2.61 | 0.80 | 0.85 | WARN, then RETRAIN | WARN, then RETRAIN |
| S4 new tactic for 70% of frauds | 0.00 | 0.023 | 0.73 | 0.27 | 0.32 | OK | RETRAIN |

S1 and S2 are the same shift on different features: one warns and one flags a retrain, which is what weighting by importance is for. S3 looks like a false-alarm surge until labels show precision of 0.80, an attack to act on. S4 moves no label-free signal: injected into day 2, LightGBM caught 1 of 83 such frauds, Isolation Forest ranked 21 in its top 0.5% and the autoencoder none.

**Retrain loop.** With labels arriving one window late, retraining on the flag took PR-AUC from 0.27 back to 0.93 in the next window and cut total cost by 37% against never retraining (EUR 11,901 against 18,769), with two retrains against three for the periodic policy ([figure](reports/figures/retrain_replay.png)).

**Limits.** Without labels the monitor sees that the output moved, not whether that was good: shifting the key feature one SD away from fraud raised precision from 0.46 to 0.78 and still flagged a retrain. Labels are assumed to arrive within six hours, where real chargebacks take days or weeks. The data covers two days, costs are assumptions, the scenarios are constructed, and the alert bounds were checked against controls, not optimised.
