import itertools

import numpy as np
import pandas as pd
from workalendar.usa import California
from workalendar.usa import Texas
from workalendar.usa import Wisconsin

from .constants import *

__all__ = [
    # Functions for general features
    "create_aggregated_features",
    "create_calendar_features",
    "create_combined_features",
    "create_encoded_features",
    "create_ewm_features",
    "create_expanding_features",
    "create_pct_change_features",
    "create_rolling_features",
    "create_shift_features",
    "create_scaled_features",
    # Functions for specific features
    "create_days_since_release",
    "create_days_until_event",
    "create_event_name",
    "create_event_type",
    "create_is_weekend",
    "create_nearest_event_name",
    "create_nearest_event_type",
    "create_sell_price_ending",
    "create_snap",
]


def weekofmonth(dt):
    dt_first = dt.replace(day=1)

    return (dt.day + dt_first.weekday() - 1) // 7


def days_until_one_day(s):
    out = np.empty_like(s, dtype="float32")
    state = np.nan

    for i, elm in enumerate(s.iloc[::-1]):
        if elm:
            state = 0
        else:
            state += 1

        out[-i - 1] = state

    return out


def create_aggregated_features(df, cols):
    grouped = df.groupby(by)

    for col in cols:
        for agg_func in agg_funcs:
            df[f"{col}_{agg_func}"] = grouped[col].transform(agg_func)


def create_calendar_features(df, col):
    for attr in calendar_features:
        if attr == "weekofmonth":
            df[attr] = df[col].apply(weekofmonth)
        else:
            df[attr] = getattr(df[col].dt, attr)


def create_combined_features(df, cols):
    func = np.vectorize(lambda x1, x2: "{}*{}".format(x1, x2))

    for col1, col2 in itertools.combinations(cols, 2):
        df[f"{col1}*{col2}"] = func(df[col1].values, df[col2].values)


def create_encoded_features(df, cols):
    for col in cols:
        grouped = df.groupby(col)
        df[f"encoded_{col}"] = grouped[target].cumsum() / (
            grouped[target].cumcount() + 1
        )
        df[f"encoded_{col}"] = grouped[f"encoded_{col}"].ffill()


def create_ewm_features(df, cols, windows):
    grouped = df.groupby(by)

    for col in cols:
        for j in windows:
            for agg_func in agg_funcs_for_ewm:
                df[f"{col}_ewm_{j}_{agg_func}"] = grouped[col].apply(
                    lambda s: s.ewm(span=j).aggregate(agg_func)
                )


def create_expanding_features(df, cols):
    grouped = df.groupby(by)

    for col in cols:
        for agg_func in agg_funcs_for_expanding:
            if agg_func == "min":
                df[f"{col}_expanding_{agg_func}"] = grouped[col].cummin()
            elif agg_func == "max":
                df[f"{col}_expanding_{agg_func}"] = grouped[col].cummax()
            else:
                feature = grouped[col].expanding().aggregate(agg_func)

                feature.sort_index(level=-1, inplace=True)

                df[f"{col}_expanding_{agg_func}"] = feature.values


def create_pct_change_features(df, cols, periods):
    grouped = df.groupby(by)

    for col in cols:
        for i in periods:
            df[f"{col}_pct_change_{i}"] = grouped[col].pct_change(i)


def create_rolling_features(df, cols, windows):
    grouped = df.groupby(by)

    for col in cols:
        for j in windows:
            for agg_func in agg_funcs_for_rolling:
                df[f"{col}_rolling_{j}_{agg_func}"] = grouped[col].apply(
                    lambda s: s.rolling(j).aggregate(agg_func)
                )


def create_scaled_features(df, cols):
    grouped = df.groupby(by)

    for col in cols:
        df[f"scaled_{col}"] = df[col] / grouped[col].transform("max")


def create_shift_features(df, cols, periods):
    for col in cols:
        for i in periods:
            grouped = df.groupby(by)
            df[f"{col}_shift_{i}"] = grouped[col].shift(i)


def create_days_since_release(df):
    grouped = df.groupby(by)

    is_not_selled = df["sell_price"].isnull()
    df["days_since_release"] = df["date"]
    df.loc[is_not_selled, "days_since_release"] = np.nan

    df["days_since_release"] = grouped["days_since_release"].transform("min")
    df["days_since_release"] = (df["date"] - df["days_since_release"]).dt.days


def create_days_until_event(df):
    is_event = df["event_name_1"].notnull()
    df["days_until_event"] = days_until_one_day(is_event)


def create_event_name(df):
    event_name_1 = pd.get_dummies(df["event_name_1"])
    event_name_2 = pd.get_dummies(df["event_name_2"])

    for col in event_name_2:
        event_name_1[col] |= event_name_2[col]

    # See https://stackoverflow.com/questions/39638403/valueerror-cannot-use-inplace-with-categoricalindex
    # event_name_1.columns = event_name_1.columns.add_categories("NBAFinals")

    # event_name_1["NBAFinals"] = df["date"].isin(nba_finals_dates)

    event_name_1 = event_name_1.astype("str")
    df["event_name"] = event_name_1.apply(lambda s: "".join(s), axis=1)


def create_event_type(df):
    event_type_1 = pd.get_dummies(df["event_type_1"])
    event_type_2 = pd.get_dummies(df["event_type_2"])

    for col in event_type_2:
        event_type_1[col] |= event_type_2[col]

    # is_nba_finals = df["date"].isin(nba_finals_dates)
    # event_type_1.loc[is_nba_finals, "Sporting"] = 1

    event_type_1 = event_type_1.astype("str")
    df["event_type"] = event_type_1.apply(lambda s: "".join(s), axis=1)


def create_is_weekend(df):
    df["is_weekend"] = False

    cals = {
        "CA": California(),
        "TX": Texas(),
        "WI": Wisconsin(),
    }

    for state_id, cal in cals.items():
        is_state = df["state_id"] == state_id
        df.loc[is_state, "is_weekend"] = ~df.loc[is_state, "date"].apply(
            cal.is_working_day
        )


def create_nearest_event_name(df):
    df["nearest_event_name"] = df["event_name_1"].bfill()


def create_nearest_event_type(df):
    df["nearest_event_type"] = df["event_type_1"].bfill()


def create_sell_price_ending(df):
    df["sell_price_ending"] = df["sell_price"].astype("str")
    df["sell_price_ending"] = df["sell_price_ending"].str[-1]
    df["sell_price_ending"] = df["sell_price_ending"].astype("int")


def create_snap(df):
    df["snap"] = False

    for state_id in ["TX", "WI"]:
        is_state = df["state_id"] == state_id
        df.loc[is_state, "snap"] = df.loc[is_state, f"snap_{state_id}"]
