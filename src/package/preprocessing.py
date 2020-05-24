import pandas as pd

__all__ = ["label_encode"]


def label_encode(df, cols):
    for col in cols:
        codes, _ = pd.factorize(df[col], sort=True)
        df[col] = codes
