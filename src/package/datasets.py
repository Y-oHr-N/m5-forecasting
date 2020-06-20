import numpy as np
import pandas as pd

from .constants import *
from .feature_extraction import *
from .preprocessing import *
from .utils import *

__all__ = ["load_interim", "load_processed"]


def load_interim(overwrite=False):
    if interim_path.is_file() and not overwrite:
        return pd.read_parquet(interim_path)

    interim_dir_path.mkdir(parents=True, exist_ok=True)

    calendar = pd.read_csv(calendar_path, dtype=dtype, parse_dates=parse_dates)
    sales_train_evaluation = pd.read_csv(sales_train_evaluation_path, dtype=dtype)
    sell_prices = pd.read_csv(sell_prices_path, dtype=dtype)

    for i in range(
        train_days + evaluation_days + 1, train_days + 2 * evaluation_days + 1,
    ):
        sales_train_evaluation[f"d_{i}"] = np.nan

    interim = sales_train_evaluation.melt(
        id_vars=["id", "item_id", "store_id", "dept_id", "cat_id", "state_id"],
        var_name="d",
        value_name=target,
    )
    interim = interim.merge(calendar, copy=False, how="left", on="d")
    interim = interim.merge(
        sell_prices, copy=False, how="left", on=["store_id", "item_id", "wm_yr_wk"]
    )

    interim.reset_index(drop=True, inplace=True)

    reduce_memory_usage(interim, allow_float16=False)

    interim.to_parquet(interim_path)

    return interim


def load_processed(overwrite=False):
    if processed_path.is_file() and not overwrite:
        return pd.read_parquet(processed_path)

    processed_dir_path.mkdir(parents=True, exist_ok=True)

    interim = load_interim(overwrite=overwrite)

    interim.dropna(inplace=True, subset=raw_numerical_features)

    trim_outliers(interim, level_ids[11], raw_numerical_features, low=None)

    # Extract features
    create_aggregate_features(interim, level_ids[1:11], raw_numerical_features)
    create_calendar_features(interim, parse_dates)
    create_expanding_features(interim, level_ids[11:], raw_numerical_features)
    create_pct_change_features(interim, level_ids[11], raw_numerical_features, periods)
    create_scaled_features(interim, level_ids[11:], raw_numerical_features)
    create_shift_features(interim, level_ids[11], [target], periods)
    create_rolling_features(
        interim, level_ids[11:], shift_features, windows, min_periods=1
    )
    create_days_since_release(interim)
    create_event_name(interim)
    create_event_type(interim)
    create_is_working_day(interim)
    create_snap(interim)
    label_encode(interim, categorical_features)

    # Transform target
    interim[transformed_target] = interim[target] * interim["sell_price"]

    interim.reset_index(drop=True, inplace=True)

    interim.drop(
        columns=[
            "event_name_1",
            "event_name_2",
            "event_type_1",
            "event_type_2",
            "month",
            "snap_CA",
            "snap_TX",
            "snap_WI",
            "wday",
            "weekday",
            "year",
            "wm_yr_wk",
        ],
        inplace=True,
    )

    reduce_memory_usage(interim, allow_float16=False)

    interim.to_parquet(processed_path)

    return interim
