import numpy as np

from .constants import *


def compute_scale_12(df):
    grouped = df.groupby(by)

    is_not_selled = df["sell_price"].isnull()
    df["scale_12"] = grouped[target].diff()
    df.loc[is_not_selled, "scale"] = np.nan

    df["scale_12"] **= 2

    return grouped["scale_12"].mean()


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
    scale = compute_scale_12(df)
    weight_12 = compute_weight_12(df)
    weight_12 /= np.sqrt(scale)

    weight_12.rename("scaled_weight_12", inplace=True)

    return weight_12
