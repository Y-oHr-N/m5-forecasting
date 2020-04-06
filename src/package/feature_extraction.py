import numpy as np
import pandas as pd
from workalendar.usa import California
from workalendar.usa import Texas
from workalendar.usa import Wisconsin

from .constants import calendar_features
from .utils import reduce_memory_usage


def create_no_demand_period(is_zero):
    out = np.empty_like(is_zero, dtype="int32")
    state = 0

    for i, elm in enumerate(is_zero):
        if elm:
            state += 1
        else:
            state = 0

        out[i] = state

    return out


def extract_features(df):
    cals = [
        California(),
        Texas(),
        Wisconsin(),
    ]

    intermediate = df["date"].unique()
    intermediate = pd.DataFrame(intermediate, columns=["date"])

    for cal in cals:
        intermediate[f"is_{cal.__class__.__name__.lower()}_holiday"] = intermediate["date"].apply(cal.is_holiday)

    df = df.merge(intermediate, how="left", on="date")

    # Create missing indicators
    df["sell_price_isnull"] = df["sell_price"].isnull()

    # Create calendar features
    for col in calendar_features:
        df[col] = getattr(df["date"].dt, col)

    grouped = df.groupby(["store_id", "item_id"])

    # Create lag features
    for i in [1, 22, 28]:
        df[f"demand_shift_{i}"] = grouped["demand"].shift(i)
        df[f"sell_price_shift_{i}"] = grouped["sell_price"].shift(i)

    # Create arithmetical features
    df["sell_price_day_over_day"] = df["sell_price"] / df["sell_price_shift_1"]
    df["sell_price_month_over_month"] = df["sell_price"] / df["sell_price_shift_28"]

    # Create aggregated features
    for i in [1, 22]:
        for j in [7, 28]:
            df[f"demand_shift_{i}_rolling_{j}_mean"] = (
                grouped[f"demand_shift_{i}"].transform(lambda x: x.rolling(j, min_periods=1).mean())
            )

    df["sell_price_notnull_and_demand_shift_28_is_zero"] = (~df["sell_price_isnull"]) & (df["demand_shift_28"] == 0)
    df["no_demand_period_shift_28"] = (
        grouped["sell_price_notnull_and_demand_shift_28_is_zero"].transform(create_no_demand_period)
    )

    df.drop(columns="sell_price_notnull_and_demand_shift_28_is_zero", inplace=True)

    reduce_memory_usage(df)

    return df