import itertools

import numpy as np
import pandas as pd
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
    "create_elapsed_days",
    "create_event_name",
    "create_event_type",
    "create_sell_price_ending",
    "create_snap",
]


def weekofmonth(dt):
    dt_first = dt.replace(day=1)

    return (dt.day + dt_first.weekday() - 1) // 7


def create_aggregated_features(df, col):
    grouped = df.groupby(by)

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
    grouped = df.groupby(by)

    for i in periods:
        df[f"{col}_shift_{i}"] = grouped[col].shift(i)

        for j in windows:
            df[f"{col}_shift_{i}_rolling_{j}_mean"] = grouped[
                f"{col}_shift_{i}"
            ].transform(lambda x: x.rolling(j).mean())


def create_pct_change_features(df, col):
    grouped = df.groupby(by)

    for i in periods:
        df[f"{col}_pct_change_{i}"] = grouped[col].pct_change(i)


def create_elapsed_days(df, col):
    grouped = df.groupby(by)

    is_not_selled = df["sell_price"].isnull()
    df["elapsed_days"] = df[col]
    df.loc[is_not_selled, "elapsed_days"] = np.nan

    df["elapsed_days"] = grouped["elapsed_days"].transform("min")
    df["elapsed_days"] = (df[col] - df["elapsed_days"]).dt.days


def create_event_name(df):
    event_name_1 = pd.get_dummies(df["event_name_1"])
    event_name_2 = pd.get_dummies(df["event_name_2"])

    for col in event_name_2:
        event_name_1[col] |= event_name_2[col]

    df["event_name"] = one_hot_decode(event_name_1)

    df.drop(columns=["event_name_1", "event_name_2"], inplace=True)


def create_event_type(df):
    event_type_1 = pd.get_dummies(df["event_type_1"])
    event_type_2 = pd.get_dummies(df["event_type_2"])

    for col in event_type_2:
        event_type_1[col] |= event_type_2[col]

    df["event_type"] = one_hot_decode(event_type_1)

    df.drop(columns=["event_type_1", "event_type_2"], inplace=True)


def create_sell_price_ending(df, col):
    df["sell_price_ending"] = df[col].astype("str")
    df["sell_price_ending"] = df["sell_price_ending"].str[-1]
    df["sell_price_ending"] = df["sell_price_ending"].astype("int")


def create_snap(df):
    for state_id in ["TX", "WI"]:
        is_state = df["state_id"] == state_id
        df.loc[is_state, "snap_CA"] = df.loc[is_state, f"snap_{state_id}"]

        df.drop(columns=f"snap_{state_id}", inplace=True)

    df.rename(columns={"snap_CA": "snap"}, inplace=True)
