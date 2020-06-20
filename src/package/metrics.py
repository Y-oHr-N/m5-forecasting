import numpy as np
import pandas as pd

from .constants import *

__all__ = ["WRMSSEEvaluator"]


class WRMSSEEvaluator(object):
    def __init__(self, valid_start_d, weight_start_d=None, target_transform=False):
        calendar = pd.read_csv(calendar_path)
        sales_train_evaluation = pd.read_csv(sales_train_evaluation_path)
        sell_prices = pd.read_csv(sell_prices_path)

        sales_train_evaluation.insert(0, "all_id", 0)

        if weight_start_d is None:
            weight_start_d = valid_start_d - evaluation_days

        train_start = 7
        valid_start = train_start + valid_start_d - 1
        weight_start = train_start + weight_start_d - 1

        id_columns = sales_train_evaluation.columns[:train_start].tolist()
        train_columns = sales_train_evaluation.columns[train_start:valid_start].tolist()
        valid_columns = sales_train_evaluation.columns[
            valid_start : valid_start + evaluation_days
        ].tolist()
        weight_columns = sales_train_evaluation.columns[
            weight_start : weight_start + evaluation_days
        ].tolist()

        id_df = sales_train_evaluation[id_columns]
        train_df = sales_train_evaluation[id_columns + train_columns]
        valid_df = sales_train_evaluation[id_columns + valid_columns]

        self.target_transform = target_transform
        self.id_df = id_df
        self.valid_columns = valid_columns

        day_to_week = calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = sales_train_evaluation[
            ["item_id", "store_id"] + weight_columns
        ].set_index(["item_id", "store_id"])
        weight_df = (
            weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: target})
        )
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)

        weight_df = weight_df.merge(
            sell_prices, copy=False, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_df[target] = weight_df[target] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[
            target
        ]
        weight_df = weight_df.loc[
            zip(train_df.item_id, train_df.store_id), :
        ].reset_index(drop=True)
        weight_df = pd.concat([id_df, weight_df], axis=1, sort=False)

        for i, level_id in enumerate(level_ids):
            train_y = train_df.groupby(level_id)[train_columns].sum()
            scale = []

            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0) :]

                scale.append(((series[1:] - series[:-1]) ** 2).mean())

            setattr(self, f"level_{i + 1}_scale", np.array(scale))
            setattr(
                self,
                f"level_{i + 1}_valid_df",
                valid_df.groupby(level_id)[valid_columns].sum(),
            )

            level_weight = weight_df.groupby(level_id)[weight_columns].sum().sum(axis=1)

            setattr(self, f"level_{i + 1}_weight", level_weight / level_weight.sum())

    def rmsse(self, valid_preds, level):
        valid_y = getattr(self, f"level_{level}_valid_df")
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f"level_{level}_scale")

        return (score / scale).map(np.sqrt)

    def score(self, valid_preds):
        valid_preds = pd.DataFrame(valid_preds, columns=self.valid_columns)
        valid_preds = pd.concat([self.id_df, valid_preds], axis=1, sort=False)

        all_scores = []

        for i, level_id in enumerate(level_ids):
            level_scores = self.rmsse(
                valid_preds.groupby(level_id)[self.valid_columns].sum(), i + 1
            )
            weight = getattr(self, f"level_{i + 1}_weight")
            level_scores = pd.concat([weight, level_scores], axis=1, sort=False).prod(
                axis=1
            )

            all_scores.append(level_scores.sum())

        return np.mean(all_scores)

    def feval(self, preds, dtrain):
        if self.target_transform:
            preds /= dtrain.get_weight()

        preds = preds.reshape((-1, evaluation_days), order="F")

        return "wrmsse", self.score(preds), False
