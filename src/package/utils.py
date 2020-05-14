import numpy as np
import pandas as pd

__all__ = ["create_scale", "create_weight_12", "reduce_memory_usage"]


def create_scale(df):
    grouped = df.groupby(["store_id", "item_id"])

    is_not_selled = df["sell_price"].isnull()
    df["scale"] = grouped["demand"].diff()
    df.loc[is_not_selled, "scale"] = np.nan

    df["scale"] **= 2
    df["scale"] = grouped["scale"].transform("mean")
    df["scale"] = pd.to_numeric(df["scale"], downcast="integer")


def create_weight_12(df, start_date="2016-03-28", end_date="2016-04-24"):
    grouped = df.groupby(["store_id", "item_id"])

    is_valid = (df["date"] >= start_date) & (df["date"] <= end_date)
    df["weight_12"] = np.nan
    df.loc[is_valid, "weight_12"] = (
        df.loc[is_valid, "sell_price"] * df.loc[is_valid, "demand"]
    )

    df["weight_12"] = grouped["weight_12"].transform("sum")
    df["weight_12"] = pd.to_numeric(df["weight_12"], downcast="integer")


def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == "datetime64[ns]":
            continue

        try:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        except ValueError:
            continue

        col_type = df[col].dtype

        if col_type in ["float16", "float32", "float64"]:
            c_min = df[col].min()
            c_max = df[col].max()

            if c_min > np.finfo(np.float32).min and c_max < np.finfo("float32").max:
                df[col] = df[col].astype("float32")
