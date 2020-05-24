import pandas as pd

__all__ = ["label_encode", "one_hot_decode"]


def label_encode(df, cols):
    for col in cols:
        codes, _ = pd.factorize(df[col], sort=True)
        df[col] = codes


def one_hot_decode(df):
    df = df.astype("str")

    return df.apply(lambda s: "".join(s), axis=1)
