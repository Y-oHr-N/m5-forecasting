import lightgbm as lgb
import numpy as np
import pandas as pd

__all__ = [
    "create_dataset",
    "create_exponential_sample_weight",
    "reduce_memory_usage",
]


def create_dataset(data, is_train, feature_name, label, **kwargs):
    X = data.loc[is_train, feature_name]
    X = X.astype("float32")
    X = X.values
    y = data.loc[is_train, label]
    y = y.values

    kwargs["feature_name"] = feature_name

    return lgb.Dataset(X, y, **kwargs)


def create_exponential_sample_weight(df, date_col, denom=1):
    date_min = df[date_col].min()
    date_max = df[date_col].max()
    span = (date_max - date_min).days // denom + 1
    elapsed_time = (df[date_col] - date_min).dt.days // denom
    alpha = 2 / (span + 1)
    df["sample_weight"] = (1 - alpha) ** (span - elapsed_time - 1)


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

            if c_min > np.finfo("float32").min and c_max < np.finfo("float32").max:
                df[col] = df[col].astype("float32")
