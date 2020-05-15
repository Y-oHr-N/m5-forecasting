import numpy as np
import pandas as pd

from .constants import *

__all__ = [
    "create_scale",
    "create_weight_12",
    "create_sample_weight",
    "reduce_memory_usage",
]


def create_scale(df):
    grouped = df.groupby(by)

    is_not_selled = df["sell_price"].isnull()
    df["scale"] = grouped[target].diff()
    df.loc[is_not_selled, "scale"] = np.nan

    df["scale"] **= 2
    df["scale"] = grouped["scale"].transform("mean")
    df["scale"] = pd.to_numeric(df["scale"], downcast="integer")


def create_weight_12(df, start_date="2016-03-28", end_date="2016-04-24"):
    grouped = df.groupby(by)

    is_valid = (df["date"] >= start_date) & (df["date"] <= end_date)
    df["weight_12"] = np.nan
    df.loc[is_valid, "weight_12"] = (
        df.loc[is_valid, "sell_price"] * df.loc[is_valid, target]
    )

    df["weight_12"] = grouped["weight_12"].transform("sum")
    df["weight_12"] = pd.to_numeric(df["weight_12"], downcast="integer")


def create_sample_weight(df):
    create_scale(df)
    create_weight_12(df)

    df["sample_weight"] = df["weight_12"] ** 2 / df["scale"]

    df.drop(columns=["scale", "weight_12"], inplace=True)


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
