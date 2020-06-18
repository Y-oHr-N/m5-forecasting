from .constants import *

import numpy as np

__all__ = ["Folds"]


class Folds(object):
    def __init__(self, s, n_folds=5, period=evaluation_days):
        self.s = s
        self.n_folds = n_folds
        self.period = period

        self.i = n_folds - 1
        self.date_min = s.min()
        self.date_max = s.max()
        self.offset = np.timedelta64(period, "D")

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < 0:
            raise StopIteration()

        train_end_date = self.date_max - (self.i + 1) * self.offset
        test_start_date = self.date_max - ((self.i + 1) * self.offset - 1)
        test_end_date = self.date_max - self.i * self.offset

        train_index = (self.date_min <= self.s) & (self.s <= train_end_date)
        train_index = np.where(train_index)[0]
        test_index = (test_start_date <= self.s) & (self.s <= test_end_date)
        test_index = np.where(test_index)[0]

        self.i -= 1

        return train_index, test_index
