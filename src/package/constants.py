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

train_days = 1913
test_days = 28

train_start_date = "2011-01-29"
train_end_date = "2016-04-24"
validation_start_date = "2016-03-28"

by = ["store_id", "item_id"]
target = "sales"

nba_finals_dates = [
    "2011-05-31",
    "2011-06-02",
    "2011-06-05",
    "2011-06-07",
    "2011-06-09",
    "2011-06-12",
    "2012-06-12",
    "2012-06-14",
    "2012-06-17",
    "2012-06-19",
    "2012-06-21",
    "2013-06-06",
    "2013-06-09",
    "2013-06-11",
    "2013-06-13",
    "2013-06-16",
    "2013-06-18",
    "2013-06-20",
    "2014-06-05",
    "2014-06-08",
    "2014-06-10",
    "2014-06-12",
    "2014-06-15",
    "2015-06-04",
    "2015-06-07",
    "2015-06-09",
    "2015-06-11",
    "2015-06-14",
    "2015-06-16",
    "2016-06-02",
    "2016-06-05",
    "2016-06-08",
    "2016-06-10",
    "2016-06-13",
    "2016-06-16",
    "2016-06-19",
]

agg_funcs = ["min", "max", "mean", "std", "nunique"]
agg_funcs_for_expanding = ["min", "max", "mean", "std"]
agg_funcs_for_rolling = ["mean", "std"]

periods_batch = [28]
periods_online = [7]
periods = periods_online + periods_batch
windows = [7, 28]
max_lags = max(periods_online) + max(windows) - 1

binary_features = [
    "snap",
    "is_holiday",
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
]

pct_change_features = [f"sell_price_pct_change_{i}" for i in periods]

rolling_features = [
    f"{target}_shift_{i}_rolling_{j}_{agg_func}"
    for i in periods
    for j in windows
    for agg_func in agg_funcs_for_rolling
]

shift_features_batch = [f"{target}_shift_{i}" for i in periods_batch]
shift_features_online = [f"{target}_shift_{i}" for i in periods_online]

numerical_features = (
    ["sell_price"]
    + aggregated_features
    + calendar_features
    + pct_change_features
    + rolling_features
    + shift_features_online
    + shift_features_batch
)

features = binary_features + categorical_features + numerical_features

random_state = 0
