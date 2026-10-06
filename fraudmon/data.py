"""Dataset download, integrity check, feature preparation and chronological split."""

import hashlib
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

EXPECTED_COLUMNS = ["Time"] + C.PCA_FEATURES + ["Amount", "Class"]

MANUAL_HINT = (
    "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
    f"and place it at {C.DATA_FILE}."
)


def file_md5(path, chunk_size=1 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force=False):
    """Fetch the CSV from the public mirror and verify its checksum."""
    if C.DATA_FILE.exists() and not force:
        return C.DATA_FILE
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    partial = C.DATA_FILE.with_suffix(".part")
    print(f"downloading {C.DATA_URL} (about 150 MB)")
    try:
        urllib.request.urlretrieve(C.DATA_URL, partial)
    except OSError as err:
        raise RuntimeError(f"download failed: {err}. {MANUAL_HINT}") from err
    found = file_md5(partial)
    if found != C.DATA_MD5:
        partial.unlink()
        raise RuntimeError(f"checksum mismatch: expected {C.DATA_MD5}, got {found}. {MANUAL_HINT}")
    partial.replace(C.DATA_FILE)
    return C.DATA_FILE


def prepare(raw):
    """Order by time and add the model feature and the 6-hour window index."""
    missing = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")
    df = raw.sort_values("Time", kind="stable").reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)
    df["log_amount"] = np.log1p(df["Amount"])
    df["window"] = (df["Time"] // C.WINDOW_SECONDS).astype(int)
    return df


def load():
    if not C.DATA_FILE.exists():
        raise FileNotFoundError(f"{C.DATA_FILE} not found. Run `python -m fraudmon data` first.")
    df = prepare(pd.read_csv(C.DATA_FILE))
    if len(df) != C.N_ROWS or int(df["Class"].sum()) != C.N_FRAUDS:
        raise ValueError(
            f"unexpected dataset: {len(df)} rows and {int(df['Class'].sum())} frauds, "
            f"expected {C.N_ROWS} and {C.N_FRAUDS}"
        )
    return df


def split(df):
    """Chronological train / validation / test split on window boundaries."""
    train = df[df["window"].isin(C.TRAIN_WINDOWS)]
    val = df[df["window"].isin(C.VAL_WINDOWS)]
    test = df[df["window"].isin(C.TEST_WINDOWS)]
    return train, val, test


def clock(window):
    """Position of a window within its day, used to compare like with like."""
    return int(window) % C.WINDOWS_PER_DAY
