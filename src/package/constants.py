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

chinese_new_year_dates = [
    "2011-02-03",
    "2012-01-23",
    "2013-02-10",
    "2014-01-31",
    "2015-02-19",
    "2016-02-08",
]

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

ids = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
level_ids = [
    None,
    "state_id",
    "store_id",
    "cat_id",
    "dept_id",
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    "item_id",
    ["item_id", "state_id"],
    ["item_id", "store_id"],
]
target = "sales"
transformed_target = "dollar_sales"

agg_funcs = {
    "min": "min",
    "max": "max",
    "mean": "mean",
    "std": "std",
    "nunique": "nunique",
}
agg_funcs_for_ewm = {
    "mean": "mean",
}
agg_funcs_for_expanding = {
    "min": "min",
    "max": "max",
    "mean": "mean",
    "std": "std",
}
agg_funcs_for_rolling = {
    "mean": "mean",
    "std": "std",
}

periods_batch = [28]
periods_online = [7]
periods = periods_online + periods_batch
windows = [7, 28]
max_lags = max(periods_online) + max(windows) - 1

binary_features = [
    "snap",
    "is_working_day",
]

categorical_features = ids + [
    "event_name",
    "event_type",
]


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

expanding_features = [
    f"groupby_item_id_store_id_sell_price_expanding_{agg_func_name}"
    for agg_func_name in agg_funcs_for_expanding
]

pct_change_features = [f"sell_price_pct_change_{i}" for i in periods]

scaled_features = ["scaled_sell_price"]

shift_features_batch = [f"{target}_shift_{i}" for i in periods_batch]
shift_features_online = [f"{target}_shift_{i}" for i in periods_online]
shift_features = shift_features_online + shift_features_batch

rolling_features = [
    f"{shift_feature}_rolling_{j}_{agg_func_name}"
    for shift_feature in shift_features
    for j in windows
    for agg_func_name in agg_funcs_for_rolling
]

numerical_features = (
    ["sell_price"]
    + calendar_features
    + expanding_features
    + pct_change_features
    + rolling_features
    + scaled_features
    + shift_features_online
    + shift_features_batch
)

features = binary_features + categorical_features + numerical_features

random_state = 0
