from .constants import *

import numpy as np

__all__ = ["Folds"]


class Folds(object):
    def __init__(self, s, n_folds=5, period=evaluation_days):
        self.s = s
        self.n_folds = n_folds
        self.period = period

        self.i = 0
        self.date_min = s.min()
        self.date_max = s.max()
        self.offset = np.timedelta64(period, "D")

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_folds

    def __next__(self):
        if self.i == self.n_folds:
            self.i = 0

            raise StopIteration()

        train_end_date = self.date_max - (self.n_folds - self.i) * self.offset
        test_end_date = self.date_max - (self.n_folds - self.i - 1) * self.offset

        train_index = (self.date_min <= self.s) & (self.s <= train_end_date)
        train_index = np.where(train_index)[0]
        test_index = (train_end_date < self.s) & (self.s <= test_end_date)
        test_index = np.where(test_index)[0]

        self.i += 1

        return train_index, test_index
