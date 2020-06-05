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

level_id_cols = [
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

attrs = [
    "year",
    "dayofyear",
    "weekofyear",
    "month",
    "quarter",
    "day",
    "weekofmonth",
    "weekday",
]

agg_funcs = {
    # "min": "min",
    # "max": "max",
    "mean": "mean",
    # "std": "std",
    # "nunique": "nunique",
}
agg_funcs_for_ewm = {
    "mean": "mean",
    "std": "std",
}
agg_funcs_for_expanding = {
    "min": "min",
    "max": "max",
    "mean": "mean",
    "std": "std",
}
agg_funcs_for_rolling = {
    # "min": "min",
    # "max": "max",
    "mean": "mean",
    "std": "std",
}

periods_batch = [28]
periods_online = [7]
periods = periods_online + periods_batch
windows = [7, 28]
max_lags = max(periods_online) + max(windows) - 1

aggregate_feature_name_format = "groupby_{}_{}_{}".format
calendar_feature_name_format = "{}_{}".format
count_up_until_nonzero_feature_format = "{}_count_up_until_nonzero".format
expanding_feature_name_format = "groupby_{}_{}_expanding_{}".format
ewm_feature_name_format = "groupby_{}_{}_ewm_{}_{}".format
pct_change_feature_name_format = "{}_pct_change_{}".format
scaled_feature_name_format = "scaled_{}".format
shift_feature_name_format = "{}_shift_{}".format
rolling_feature_name_format = "groupby_{}_{}_rolling_{}_{}".format

binary_features = [
    "snap",
    "is_working_day",
]

categorical_features = [
    "state_id",
    "store_id",
    "cat_id",
    "dept_id",
    "item_id",
    "event_name",
    "event_type",
]

raw_numerical_features = ["sell_price"]

aggregate_features = [
    aggregate_feature_name_format(
        "_&_".join(id_col), raw_numerical_feature, agg_func_name
    )
    if isinstance(id_col, list)
    else aggregate_feature_name_format(id_col, raw_numerical_feature, agg_func_name)
    for id_col in level_id_cols[1:11]
    for raw_numerical_feature in raw_numerical_features
    for agg_func_name in agg_funcs
]

calendar_features = [f"{col}_{attr}" for col in parse_dates for attr in attrs]

expanding_features = [
    expanding_feature_name_format(
        "_&_".join(id_col), raw_numerical_feature, agg_func_name
    )
    if isinstance(id_col, list)
    else expanding_feature_name_format(id_col, raw_numerical_feature, agg_func_name)
    for id_col in level_id_cols[11:]
    for raw_numerical_feature in raw_numerical_features
    for agg_func_name in agg_funcs_for_expanding
]

pct_change_features = [
    pct_change_feature_name_format(raw_numerical_feature, i)
    for raw_numerical_feature in raw_numerical_features
    for i in periods
]

scaled_features = [
    scaled_feature_name_format(raw_numerical_feature)
    for raw_numerical_feature in raw_numerical_features
]

shift_features_batch = [shift_feature_name_format(target, i) for i in periods_batch]
shift_features_online = [shift_feature_name_format(target, i) for i in periods_online]
shift_features = shift_features_online + shift_features_batch

rolling_features = [
    rolling_feature_name_format("_&_".join(id_col), shift_feature, j, agg_func_name)
    if isinstance(id_col, list)
    else rolling_feature_name_format(id_col, shift_feature, j, agg_func_name)
    for id_col in level_id_cols[11:]
    for shift_feature in shift_features
    for j in windows
    for agg_func_name in agg_funcs_for_rolling
]

numerical_features = (
    raw_numerical_features
    + ["days_since_release"]
    + aggregate_features
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

lgb_params = {
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l2": 0.1,
    "learning_rate": 0.075,
    "metric": "rmse",
    "min_data_in_leaf": 104,
    "n_jobs": -1,
    "num_boost_round": 1_250,
    "num_leaves": 128,
    "objective": "tweedie",
    "seed": random_state,
    "tweedie_variance_power": 1.2,
}
