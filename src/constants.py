target = "demand"

categorical_features = [
    "store_id",
    "item_id",
    "dept_id",
    "cat_id",
    "state_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

calendar_features = [
    "year",
    "dayofyear",
    "weekofyear",
    "quarter",
    "month",
    "day",
    "weekday",
    "is_year_start",
    "is_year_end",
    "is_month_start",
    "is_month_end",
]
lag_features = [f"{target}_shift_{i}" for i in [28]]
arithmerical_features = [
    "sell_price_day_over_day",
]
aggregated_features = [f"{target}_shift_1_rolling_{i}_mean" for i in [28]]
numerical_features = [
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sell_price",
] + calendar_features + lag_features + arithmerical_features + aggregated_features

features = categorical_features + numerical_features
