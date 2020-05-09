from workalendar.usa import California
from workalendar.usa import Texas
from workalendar.usa import Wisconsin

from .constants import *

__all__ = ["create_calendar_features", "create_lag_features"]


def create_calendar_features(df):
    for col in calendar_features:
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
