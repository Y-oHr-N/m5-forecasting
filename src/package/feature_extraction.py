from workalendar.usa import California
from workalendar.usa import Texas
from workalendar.usa import Wisconsin

from .constants import *

__all__ = [
    "create_aggregated_features",
    "create_calendar_features",
    "create_lag_features",
]


def weekofmonth(dt):
    dt_first = dt.replace(day=1)

    return (dt.day + dt_first.weekday() - 1) // 7


def create_aggregated_features(df):
    grouped = df.groupby(["store_id", "item_id"])

    for agg_func in agg_funcs:
        df[f"sell_price_{agg_func}"] = grouped["sell_price"].transform(agg_func)


def create_calendar_features(df):
    for col in calendar_features:
        if col == "weekofmonth":
            df[col] = df["date"].apply(weekofmonth)
        else:
            df[col] = getattr(df["date"].dt, col)

    cals = [
        California(),
        Texas(),
        Wisconsin(),
    ]

    for cal in cals:
        df[f"is_{cal.__class__.__name__.lower()}_holiday"] = df["date"].apply(
            cal.is_holiday
        )


def create_lag_features(df):
    grouped = df.groupby(["store_id", "item_id"])

    for i in periods:
        df[f"demand_shift_{i}"] = grouped["demand"].shift(i)

        for j in windows:
            df[f"demand_shift_{i}_rolling_{j}_mean"] = grouped[
                f"demand_shift_{i}"
            ].transform(lambda x: x.rolling(j).mean())
