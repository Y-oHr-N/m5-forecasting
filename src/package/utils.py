import numpy as np
import pandas as pd

__all__ = ["reduce_memory_usage"]


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

            if (
                c_min > np.finfo(np.float32).min
                and c_max < np.finfo("float32").max
            ):
                df[col] = df[col].astype("float32")
