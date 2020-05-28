import numpy as np
import pandas as pd

from .constants import *

__all__ = [
    "create_sample_weight",
    "reduce_memory_usage",
]


def compute_scale(df):
    grouped = df.groupby(by)

    is_not_selled = df["sell_price"].isnull()
    df["scale"] = grouped[target].diff()
    df.loc[is_not_selled, "scale"] = np.nan

    df["scale"] **= 2

    return grouped["scale"].mean()


def compute_weight_12(df, start_date=validation_start_date, end_date=train_end_date):
    grouped = df.groupby(by)

    is_valid = (df["date"] >= start_date) & (df["date"] <= end_date)
    df["weight_12"] = np.nan
    df.loc[is_valid, "weight_12"] = (
        df.loc[is_valid, "sell_price"] * df.loc[is_valid, target]
    )

    weight_12 = grouped["weight_12"].sum()
    weight_12 /= weight_12.sum()

    return weight_12


def compute_scaled_weight_12(df):
    scale = compute_scale(df)
    weight_12 = compute_weight_12(df)
    weight_12 /= np.sqrt(scale)

    weight_12.rename("scaled_weight_12", inplace=True)

    return weight_12


def create_sample_weight(df):
    date_min = df["date"].min()
    date_max = df["date"].max()
    span = (date_max - date_min).days + 1
    alpha = 2 / (span + 1)
    elapsed_days = (df["date"] - date_min).dt.days
    df["sample_weight"] = (1 - alpha) ** (span - elapsed_days - 1)


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
