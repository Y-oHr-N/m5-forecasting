import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state

from .constants import *

__all__ = [
    # General functions
    "add_gaussian_noise",
    "label_encode",
    "trim_outliers",
    # Specidifc functions
    "create_ids",
    "detrend",
]


def clip_based_on_quantile(s, low="auto", high="auto"):
    q1 = s.quantile(q=0.25)
    q3 = s.quantile(q=0.75)
    iqr = q3 - q1

    if low is None:
        lower = None
    elif low == "auto":
        lower = q1 - 1.5 * iqr
    else:
        lower = s.quantile(q=low)

    if high is None:
        upper = None
    elif high == "auto":
        upper = q3 + 1.5 * iqr
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


def trim_outliers(df, by_col, cols, low="auto", high="auto"):
    grouped = df.groupby(by_col)

    for col in cols:
        df[col] = grouped[col].apply(clip_based_on_quantile, low=low, high=high)


def create_ids(df):
    if "id" in df:
        df[["item_id", "store_id"]] = df["id"].str.extract(
            r"(\w+_\d+_\d+)_(\w+_\d+)_\w+"
        )

    df["dept_id"] = df["item_id"].str.extract(r"(\w+_\d+)_\d+")
    df["cat_id"] = df["dept_id"].str.extract(r"(\w+)_\d+")
    df["state_id"] = df["store_id"].str.extract(r"(\w+)_\d+")


def detrend(df):
    model = LinearRegression()

    X = np.arange(train_days + 2 * evaluation_days)
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
