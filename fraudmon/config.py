"""Paths, seeds, split boundaries, cost assumptions and alert bounds."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
METRIC_DIR = ROOT / "reports" / "metrics"

DATA_FILE = DATA_DIR / "creditcard.csv"
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_MD5 = "e90efcb83d69faf99fcab8b0255024de"
N_ROWS = 284_807
N_FRAUDS = 492

SEED = 42
SEEDS = (0, 1, 2, 3, 4)
N_JOBS = 4  # fixed so results do not depend on the machine's core count
N_BOOT = 300
N_FOLDS = 5
REFIT_FOLDS = 3  # out-of-fold passes when a model is refitted during a simulation

# The data covers 48 hours. Day 1 is the training window, day 2 the later window.
WINDOW_SECONDS = 6 * 3600
WINDOWS_PER_DAY = 4
TRAIN_WINDOWS = (0, 1, 2, 3)
VAL_WINDOWS = (4, 5)
TEST_WINDOWS = (6, 7)

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = PCA_FEATURES + ["log_amount"]

STRATEGIES = ("none", "class_weight", "smote", "undersample")

# LightGBM's defaults diverge on this data (AP near the fraud rate), so the learning
# rate and leaf size are held down.
LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    deterministic=True,
    force_col_wise=True,
    n_jobs=N_JOBS,
    verbose=-1,
)

# Share of transactions an analyst team could review; used for precision-at-budget.
ALERT_BUDGET = 0.005
# Tail of the anomaly score treated as "anomalous" by the anomaly-rate monitor.
ANOMALY_QUANTILE = 0.995


@dataclass(frozen=True)
class CostModel:
    """Cost assumptions in EUR. These are assumptions, not measurements."""

    fn_multiplier: float = 1.0  # share of the amount lost when fraud is missed
    fn_fixed: float = 20.0  # chargeback handling and card reissue per missed fraud
    fp_cost: float = 5.0  # support contact and friction per blocked legitimate payment
    tp_cost: float = 1.0  # handling cost per correctly blocked fraud


FP_COST_GRID = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
FN_FIXED_GRID = (0.0, 20.0, 50.0)
FN_MULTIPLIER_GRID = (1.0, 3.0, 5.0)


@dataclass(frozen=True)
class AlertBounds:
    """Bounds for the OK / WARN / RETRAIN policy in monitor.decide."""

    min_rows: int = 2000  # below this PSI is dominated by sampling noise
    psi_alert: float = 0.25
    weighted_share_warn: float = 0.10
    weighted_share_retrain: float = 0.25
    score_psi_warn: float = 0.10
    score_psi_retrain: float = 0.25
    alert_ratio_warn: tuple = (0.67, 1.5)
    alert_ratio_retrain: tuple = (0.5, 2.0)
    anomaly_ratio_warn: float = 2.0
    perf_rel_drop: float = 0.15
    min_frauds: int = 25
    persistence: int = 2
