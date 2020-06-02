import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state

from .constants import *

__all__ = [
    # General functions
    "add_gaussian_noise",
    "label_encode",
    # Specidifc functions
    "detrend",
]


def add_gaussian_noise(df, cols, sigma=0.01, random_state=None):
    n, _ = df.shape
    random_state = check_random_state(random_state)

    for col in cols:
        df[col] *= 1.0 + sigma * random_state.randn(n)


def label_encode(df, cols):
    for col in cols:
        codes, _ = pd.factorize(df[col], sort=True)
        df[col] = codes


def detrend(df):
    model = LinearRegression()

    X = np.arange(train_days + 2 * test_days)
    X = X.reshape(-1, 1)

    Y = df.pivot_table(columns=level_ids[11], dropna=False, index="date", values=target)

    is_selled = df.pivot_table(
        columns=level_ids[11], dropna=False, index="date", values="sell_price"
    )
    is_selled = is_selled.notnull()

    Y_mean = Y[is_selled].mean()

    Y.where(is_selled, Y_mean, axis=1, inplace=True)

    model.fit(X[:train_days], Y.iloc[:train_days])

    tmp = model.predict(X)
    tmp = pd.DataFrame(tmp, columns=Y.columns, index=Y.index)
    tmp = tmp.unstack()
    on = ["item_id", "store_id", "date"]

    tmp.rename("linear_trend", inplace=True)

    tmp = df[on].merge(tmp, copy=False, on=on)

    df["linear_trend"] = tmp["linear_trend"]
    df[f"detrended_{target}"] = df[target] - tmp["linear_trend"]
