"""Shared synthetic data for the tests. No test needs the real dataset."""

import numpy as np
import pandas as pd
import pytest

from fraudmon import config as C
from fraudmon import data


def make_raw(n=8000, fraud_rate=0.02, seed=0):
    """Transactions with the real schema; fraud sits five SDs low on V14."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.normal(size=(n, 28)), columns=C.PCA_FEATURES)
    raw.insert(0, "Time", np.sort(rng.uniform(0, 48 * 3600 - 1, n)))
    raw["Amount"] = rng.lognormal(3.0, 1.0, n).round(2)
    raw["Class"] = (rng.random(n) < fraud_rate).astype(int)
    raw.loc[raw["Class"] == 1, "V14"] -= 5.0
    return raw


@pytest.fixture(scope="session")
def raw_factory():
    return make_raw


@pytest.fixture(scope="session")
def frame():
    return data.prepare(make_raw())


@pytest.fixture(scope="session")
def big_frame():
    return data.prepare(make_raw(n=40000, seed=1))
