import datetime
import decimal

import numpy as np
import pandas as pd
from workalendar.usa import California
from workalendar.usa import Texas
from workalendar.usa import Wisconsin

from .constants import *

__all__ = [
    # Functions for general features
    "create_aggregate_features",
    "create_calendar_features",
    "create_count_up_until_nonzero_features",
    "create_diff_features",
    "create_ewm_features",
    "create_expanding_features",
    "create_pct_change_features",
    "create_rolling_features",
    "create_shift_features",
    "create_scaled_features",
    # Functions for specific features
    "create_days_since_release",
    "create_days_until_event",
    "create_days_until_non_working_day",
    "create_event_name",
    "create_event_type",
    "create_is_working_day",
    "create_moon_phase",
    "create_nearest_event_name",
    "create_nearest_event_type",
    "create_sell_price_ending",
    "create_snap",
]


def count_up_until_nonzero(s):
    out = np.full_like(s, np.nan, dtype="float32")
    state = np.nan

    for i, elm in enumerate(s):
        if np.isnan(elm):
            continue

        if elm:
            state = 0
        else:
            state += 1

        out[i] = state

    return out


def count_down_until_nonzero(s):
    out = count_up_until_nonzero(s.iloc[::-1])

    return out[::-1]


def moon_phase(dt):
    diff = dt - datetime.datetime(2001, 1, 1)
    days = decimal.Decimal(diff.days) + (
        decimal.Decimal(diff.seconds) / decimal.Decimal(86400)
    )
    lunations = decimal.Decimal("0.20439731") + (
        days * decimal.Decimal("0.03386319269")
    )
    position = lunations % decimal.Decimal(1)
    index = np.floor((position * decimal.Decimal(8)) + decimal.Decimal("0.5"))

    return int(index) % 8


def weekofmonth(dt):
    dt_first = dt.replace(day=1)

    return (dt.day + dt_first.weekday() - 1) // 7


def create_aggregate_features(df, by_cols, cols):
    for by_col in by_cols:
        if isinstance(by_col, list):
            id_name = "_&_".join(by_col)
        else:
            id_name = by_col

        grouped = df.groupby(by_col)

        for agg_func_name, agg_func in agg_funcs.items():
            new_cols = [
                aggregate_feature_name_format(id_name, col, agg_func_name)
                for col in cols
            ]
            df[new_cols] = grouped[cols].transform(agg_func)


def create_calendar_features(df, cols):
    for col in cols:
        for attr in attrs:
            new_col = calendar_feature_name_format(col, attr)

            if attr == "weekofmonth":
                df[new_col] = df[col].apply(weekofmonth)
            else:
                df[new_col] = getattr(df[col].dt, attr)


def create_count_up_until_nonzero_features(df, by_col, cols):
    grouped = df.groupby(by_col)

    for col in cols:
        new_col = count_up_until_nonzero_feature_format(col)
        df[new_col] = grouped[col].transform(count_up_until_nonzero)


def create_diff_features(df, by_col, cols, periods):
    grouped = df.groupby(by_col)

    for i in periods:
        new_cols = [diff_feature_name_format(col, i) for col in cols]
        df[new_cols] = grouped[cols].diff(i)


def create_ewm_features(df, by_cols, cols, windows):
    for by_col in by_cols:
        if isinstance(by_col, list):
            id_name = "_&_".join(by_col)
        else:
            id_name = by_col

        grouped = df.groupby(by_col)

        for col in cols:
            for j in windows:
                for agg_func_name, agg_func in agg_funcs_for_ewm.items():
                    new_col = ewm_feature_name_format(id_name, col, j, agg_func_name)
                    df[new_col] = grouped[col].apply(
                        lambda s: s.ewm(span=j).aggregate(agg_func)
                    )


def create_expanding_features(df, by_cols, cols):
    for by_col in by_cols:
        if isinstance(by_col, list):
            id_name = "_&_".join(by_col)
        else:
            id_name = by_col

        grouped = df.groupby(by_col)

        for col in cols:
            for agg_func_name, agg_func in agg_funcs_for_expanding.items():
                new_col = expanding_feature_name_format(id_name, col, agg_func_name)

                if agg_func_name == "min":
                    df[new_col] = grouped[col].cummin()
                elif agg_func_name == "max":
                    df[new_col] = grouped[col].cummax()
                else:
                    feature = grouped[col].expanding().aggregate(agg_func)

                    feature.reset_index(drop=True, inplace=True, level=by_col)

                    df[new_col] = feature


def create_pct_change_features(df, by_col, cols, periods):
    grouped = df.groupby(by_col)

    for i in periods:
        new_cols = [pct_change_feature_name_format(col, i) for col in cols]
        df[new_cols] = grouped[cols].pct_change(i)


def create_rolling_features(df, by_cols, cols, windows, min_periods=None):
    for by_col in by_cols:
        if isinstance(by_col, list):
            id_name = "_&_".join(by_col)
        else:
            id_name = by_col

        grouped = df.groupby(by_col)

        for col in cols:
            for j in windows:
                rolling = grouped[col].rolling(j, min_periods=min_periods)

                for agg_func_name, agg_func in agg_funcs_for_rolling.items():
                    new_col = rolling_feature_name_format(
                        id_name, col, j, agg_func_name
                    )
                    feature = rolling.aggregate(agg_func)

                    feature.reset_index(drop=True, inplace=True, level=by_col)

                    df[new_col] = feature


def create_scaled_features(df, by_cols, cols):
    for by_col in by_cols:
        if isinstance(by_col, list):
            id_name = "_&_".join(by_col)
        else:
            id_name = by_col

        grouped = df.groupby(by_col)

        new_cols = [scaled_feature_name_format(id_name, col) for col in cols]
        df[new_cols] = df[cols] / grouped[cols].transform("max")


def create_shift_features(df, by_col, cols, periods):
    grouped = df.groupby(by_col)

    for i in periods:
        new_cols = [shift_feature_name_format(col, i) for col in cols]
        df[new_cols] = grouped[cols].shift(i)


def create_days_since_release(df, offset=0):
    grouped = df.groupby(level_ids[11])

    is_not_selled = df["sell_price"].isnull()
    df["days_since_release"] = df["date"]
    df.loc[is_not_selled, "days_since_release"] = np.nan

    tmp = grouped["days_since_release"].transform("min")
    df["days_since_release"] = (df["date"] - tmp).dt.days

    if offset > 0:
        is_oldest = tmp == train_start_date
        df.loc[is_oldest, "days_since_release"] += offset


def create_days_until_event(df):
    is_event = df["event_name_1"].notnull()
    df["days_until_event"] = count_down_until_nonzero(is_event)


def create_days_until_non_working_day(df):
    tmp = df["date"].unique()
    tmp = pd.DataFrame(index=tmp)

    cals = {
        "CA": California(),
        "TX": Texas(),
        "WI": Wisconsin(),
    }

    for state_id, cal in cals.items():
        tmp[state_id] = tmp.index.map(cal.is_working_day)
        tmp[state_id] = tmp[state_id].astype("bool")
        tmp[state_id] = count_down_until_nonzero(~tmp[state_id])

    tmp = tmp.stack()
    on = ["date", "state_id"]

    tmp.index.rename(on, inplace=True)
    tmp.rename("days_until_non_working_day", inplace=True)

    tmp = df[on].merge(tmp, copy=False, on=on)
    df["days_until_non_working_day"] = tmp["days_until_non_working_day"]


def create_event_name(df):
    event_name_1 = pd.get_dummies(df["event_name_1"])
    event_name_2 = pd.get_dummies(df["event_name_2"])

    for col in event_name_2:
        event_name_1[col] |= event_name_2[col]

    for event in events:
        event_name_1.columns = event_name_1.columns.add_categories(event["event_name"])
        event_name_1[event["event_name"]] = df["date"].isin(event["dates"])

    _, n_event_names = event_name_1.shape
    tmp = 2 ** np.arange(n_event_names)

    df["event_name"] = event_name_1 @ tmp[::-1]


def create_event_type(df):
    event_type_1 = pd.get_dummies(df["event_type_1"])
    event_type_2 = pd.get_dummies(df["event_type_2"])

    for col in event_type_2:
        event_type_1[col] |= event_type_2[col]

    for event in events:
        is_event = df["date"].isin(event["dates"])
        event_type_1.loc[is_event, event["event_type"]] = 1

    _, n_event_types = event_type_1.shape
    tmp = 2 ** np.arange(n_event_types)

    df["event_type"] = event_type_1 @ tmp[::-1]


def create_is_working_day(df):
    tmp = df["date"].unique()
    tmp = pd.DataFrame(index=tmp)

    cals = {
        "CA": California(),
        "TX": Texas(),
        "WI": Wisconsin(),
    }

    for state_id, cal in cals.items():
        tmp[state_id] = tmp.index.map(cal.is_working_day)

    tmp = tmp.astype("bool")
    tmp = tmp.stack()
    on = ["date", "state_id"]

    tmp.index.rename(on, inplace=True)
    tmp.rename("is_working_day", inplace=True)

    tmp = df[on].merge(tmp, copy=False, on=on)
    df["is_working_day"] = tmp["is_working_day"]


def create_moon_phase(df):
    df["moon_phase"] = df["date"].apply(moon_phase)


def create_nearest_event_name(df, limit=None):
    # TODO: handle event_name_2
    df["nearest_event_name"] = df["event_name_1"].astype("object")
    df["nearest_event_name"] = df["nearest_event_name"].bfill(limit=limit)
    df["nearest_event_name"] = df["nearest_event_name"].astype("category")


def create_nearest_event_type(df, limit=None):
    # TODO: handle event_type_2
    df["nearest_event_type"] = df["event_type_1"].astype("object")
    df["nearest_event_type"] = df["nearest_event_type"].bfill(limit=limit)
    df["nearest_event_type"] = df["nearest_event_type"].astype("category")


def create_sell_price_ending(df):
    df["sell_price_ending"] = df["sell_price"].astype("str")
    df["sell_price_ending"] = df["sell_price_ending"].str[-1]
    df["sell_price_ending"] = df["sell_price_ending"].astype("int")


def create_snap(df):
    df["snap"] = df["snap_CA"]

    for state_id in ["TX", "WI"]:
        is_state = df["state_id"] == state_id
        df.loc[is_state, "snap"] = df.loc[is_state, f"snap_{state_id}"]
