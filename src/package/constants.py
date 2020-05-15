by = ["store_id", "item_id"]
target = "sales"

agg_funcs = ["min", "max", "mean", "std", "nunique"]

periods = [7, 28]
windows = [7, 28]

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

categorical_features = [
    "store_id",
    "item_id",
    "dept_id",
    "cat_id",
    "state_id",
    "event_name",
    "event_type",
]

pct_change_features = [f"sell_price_pct_change_{i}" for i in periods]

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
