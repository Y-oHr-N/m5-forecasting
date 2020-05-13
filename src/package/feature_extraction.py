import itertools

import numpy as np
from workalendar.usa import California
from workalendar.usa import Texas
from workalendar.usa import Wisconsin

from .constants import *
from .preprocessing import *

__all__ = [
    "create_aggregated_features",
    "create_calendar_features",
    "create_combined_features",
    "create_lag_features",
    "create_pct_change_features",
]


def weekofmonth(dt):
    dt_first = dt.replace(day=1)

    return (dt.day + dt_first.weekday() - 1) // 7


def create_aggregated_features(df, col):
    grouped = df.groupby(["store_id", "item_id"])

    for agg_func in agg_funcs:
        df[f"{col}_{agg_func}"] = grouped[col].transform(agg_func)


def create_calendar_features(df, col):
    for attr in calendar_features:
        if attr == "weekofmonth":
            df[attr] = df[col].apply(weekofmonth)
        else:
            df[attr] = getattr(df[col].dt, attr)

    cals = [
        California(),
        Texas(),
        Wisconsin(),
    ]

    for cal in cals:
        df[f"is_{cal.__class__.__name__.lower()}_holiday"] = df[col].apply(
            cal.is_holiday
        )


def create_combined_features(df, cols):
    func = np.vectorize(lambda x1, x2: "{}*{}".format(x1, x2))

    for col1, col2 in itertools.combinations(cols, 2):
        values = func(df[col1].values, df[col2].values)
        df[f"{col1}*{col2}"] = label_encode(values)


def create_lag_features(df, col):
    grouped = df.groupby(["store_id", "item_id"])

    for i in periods:
        df[f"{col}_shift_{i}"] = grouped[col].shift(i)

        for j in windows:
            df[f"{col}_shift_{i}_rolling_{j}_mean"] = grouped[
                f"{col}_shift_{i}"
            ].transform(lambda x: x.rolling(j).mean())


def create_pct_change_features(df, col):
    grouped = df.groupby(["store_id", "item_id"])

    for i in periods:
        df[f"{col}_pct_change_{i}"] = grouped[col].pct_change(i)
