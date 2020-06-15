import numpy as np
import pandas as pd

from .constants import *
from .utils import *


def load_interim():
    if interim_path.is_file():
        return pd.read_parquet(interim_path)

    sales_train_validation = pd.read_csv(sales_train_validation_path, dtype=dtype)

    for i in range(train_days + 1, train_days + 1 + 2 * test_days):
        sales_train_validation[f"d_{i}"] = np.nan

    reduce_memory_usage(sales_train_validation)

    interim = sales_train_validation.melt(
        id_vars=["id", "item_id", "store_id", "dept_id", "cat_id", "state_id"],
        var_name="d",
        value_name=target,
    )

    interim_dir_path.mkdir(parents=True, exist_ok=True)
    interim.to_parquet(interim_path)

    return interim
