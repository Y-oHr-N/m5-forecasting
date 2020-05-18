dtype = {
    "wm_yr_wk": "int16",
    "year": "int16",
    "month": "int8",
    "wday": "int8",
    "event_name_1": "category",
    "event_name_2": "category",
    "event_type_1": "category",
    "event_type_2": "category",
    "snap_CA": "bool",
    "snap_TX": "bool",
    "snap_WI": "bool",
    "state_id": "category",
    "store_id": "category",
    "cat_id": "category",
    "dept_id": "category",
    "item_id": "category",
    "sell_price": "float32",
}

parse_dates = ["date"]

train_start_date = "2011-01-29"
train_end_date = "2016-03-28"
validation_start_date = "2016-04-24"

by = ["store_id", "item_id"]
target = "sales"

agg_funcs = ["min", "max", "mean", "std", "nunique"]

periods = [7, 28]
windows = [7, 28]
max_lags = max(periods) + max(windows) - 1

aggregated_features = [f"sell_price_{agg_func}" for agg_func in agg_funcs]

calendar_features = [
    "year",
    "dayofyear",
    "weekofyear",
    "month",
    "quarter",
    "day",
    "weekofmonth",
    "weekday",
    # "is_year_start",
    # "is_year_end",
    # "is_quarter_start",
    # "is_quarter_end",
    # "is_month_start",
    # "is_month_end",
]

lag_features = [f"{target}_shift_{i}" for i in periods] + [
    f"{target}_shift_{i}_rolling_{j}_mean" for i in periods for j in windows
]

pct_change_features = [f"sell_price_pct_change_{i}" for i in periods]

categorical_features = [
    "store_id",
    "item_id",
    "dept_id",
    "cat_id",
    "state_id",
    "event_name",
    "event_type",
]

numerical_features = (
    [
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "sell_price",
        "is_california_holiday",
        "is_texas_holiday",
        "is_wisconsin_holiday",
    ]
    + aggregated_features
    + calendar_features
    + lag_features
    + pct_change_features
)

features = categorical_features + numerical_features
