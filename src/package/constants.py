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
lag_features = [f"{target}_shift_{i}" for i in [28,]] + [
    # f"sell_price_shift_{i}" for i in [
    #     1,
    #     22,
    #     28,
    # ]
]
arithmerical_features = [
    "sell_price_day_over_day",
    # "sell_price_month_over_month",
]
aggregated_features = [
    f"{target}_shift_{i}_rolling_{j}_mean"
    for (i, j) in [
        (1, 28),
        (22, 7),
        # (22, 28),
    ]
]
numerical_features = (
    [
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "sell_price",
        # "is_california_holiday",
        # "is_texas_holiday",
        # "is_wisconsin_holiday",
    ]
    + calendar_features
    + lag_features
    + arithmerical_features
    + aggregated_features
)

features = categorical_features + numerical_features
