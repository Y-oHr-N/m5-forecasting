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
    "trim_outliers",
]


def clip_based_on_quantile(s, low=None, high=None):
    if low is None:
        lower = None
    else:
        lower = s.quantile(q=low)

    if high is None:
        upper = None
    else:
        upper = s.quantile(q=high)

    return s.clip(lower=lower, upper=upper)


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

    tmp.rename("trend", inplace=True)

    tmp = df[on].merge(tmp, copy=False, on=on)

    df["trend"] = tmp["trend"]
    df[f"detrended_{target}"] = df[target] - tmp["trend"]


def trim_outliers(df, low=None, high=0.99):
    grouped = df.groupby(level_ids[11])

    trimmed_target = f"trimmed_{target}"
    is_not_selled = df["sell_price"].isnull()

    df[trimmed_target] = df[target]
    df.loc[is_not_selled, trimmed_target] = np.nan

    df[trimmed_target] = grouped[trimmed_target].apply(
        clip_based_on_quantile, low=low, high=high
    )
