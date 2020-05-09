from .constants import *

__all__ = ["create_lag_features"]


def create_lag_features(df):
    grouped = df.groupby(["store_id", "item_id"])

    for i in periods:
        df[f"demand_shift_{i}"] = grouped["demand"].shift(i)

        for j in windows:
            df[f"demand_shift_{i}_rolling_{j}_mean"] = grouped[
                f"demand_shift_{i}"
            ].transform(lambda x: x.rolling(j).mean())
