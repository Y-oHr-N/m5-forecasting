periods = [7, 28]
windows = [7, 28]

calendar_features = [
    "year",
    # "dayofyear",
    "weekofyear",
    "quarter",
    "month",
    "day",
    "weekday",
    # "is_year_start",
    # "is_year_end",
    # "is_quarter_start",
    # "is_quarter_end",
    # "is_month_start",
    # "is_month_end",
]
lag_features = [f"demand_shift_{i}" for i in periods]
aggregated_features = [
    f"demand_shift_{i}_rolling_{j}_mean"
    for i in periods
    for j in windows
]

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

numerical_features = (
    [
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "sell_price",
    ]
    + calendar_features
    + lag_features
    + aggregated_features
)

features = categorical_features + numerical_features
