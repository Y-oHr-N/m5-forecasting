import pandas as pd

__all__ = ["label_encode", "one_hot_decode"]


def label_encode(s):
    codes, _ = pd.factorize(s, sort=True)

    return codes


def one_hot_decode(df):
    df = df.astype("str")
    s = df.apply(lambda s: "".join(s), axis=1)

    return label_encode(s)
