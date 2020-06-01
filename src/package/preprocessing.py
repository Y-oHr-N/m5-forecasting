import pandas as pd
from sklearn.utils import check_random_state

__all__ = ["add_gaussian_noise", "label_encode"]


def add_gaussian_noise(df, cols, sigma=0.01, random_state=None):
    n, _ = df.shape
    random_state = check_random_state(random_state)

    for col in cols:
        df[col] *= 1.0 + sigma * random_state.randn(n)


def label_encode(df, cols):
    for col in cols:
        codes, _ = pd.factorize(df[col], sort=True)
        df[col] = codes
