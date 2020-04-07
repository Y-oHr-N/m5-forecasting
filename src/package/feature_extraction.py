from .constants import *

__all__ = ["extract_features"]


def extract_features(df):
    # Create calendar features
    for col in calendar_features:
        df[col] = getattr(df["date"].dt, col)

    grouped = df.groupby(["store_id", "item_id"])

    for i in periods:
        # Create lag features
        df[f"demand_shift_{i}"] = grouped["demand"].shift(i)

        # Create aggregated features
        for j in windows:
            df[f"demand_shift_{i}_rolling_{j}_mean"] = grouped[
                f"demand_shift_{i}"
            ].transform(lambda x: x.rolling(j).mean())
