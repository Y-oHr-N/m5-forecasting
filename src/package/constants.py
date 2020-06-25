import pathlib

from .utils import *

module_path = pathlib.Path(__file__)
package_dir_path = module_path.parent
src_dir_path = package_dir_path.parent
root_dir_path = src_dir_path.parent

data_dir_path = root_dir_path / "data"

raw_dir_path = data_dir_path / "raw"
calendar_path = raw_dir_path / "calendar.csv"
sales_train_validation_path = raw_dir_path / "sales_train_validation.csv"
sales_train_evaluation_path = raw_dir_path / "sales_train_evaluation.csv"
sample_submission_path = raw_dir_path / "sample_submission.csv"
sell_prices_path = raw_dir_path / "sell_prices.csv"

interim_dir_path = data_dir_path / "interim"
interim_path = interim_dir_path / "interim.parquet"

processed_dir_path = data_dir_path / "processed"
processed_path = processed_dir_path / "processed.parquet"

models_dir_path = root_dir_path / "models"
removed_features_path = models_dir_path / "removed_features.joblib"
cb_reg_path = models_dir_path / "cb_reg.joblib"
lgbm_clf_path = models_dir_path / "lgbm_clf.joblib"
lgbm_reg_path = models_dir_path / "lgbm_reg.joblib"
coef_path = models_dir_path / "coef.joblib"
prediction_path = models_dir_path / "prediction.parquet"
submission_accuracy_path = models_dir_path / "submission_accuracy.csv.gz"
submission_uncertainty_path = models_dir_path / "submission_uncertainty.csv.gz"

notebooks_dir_path = root_dir_path / "notebooks"
inputs_dir_path = notebooks_dir_path / "inputs"
outputs_dir_path = notebooks_dir_path / "outputs"

train_days = 1913
evaluation_days = 28

train_start_date = "2011-01-29"
train_end_date = "2016-04-24"
validation_start_date = "2016-04-25"
validation_end_date = "2016-05-22"
evaluation_start_date = "2016-05-23"
evaluation_end_date = "2016-06-19"

events = [
    # {
    #     "event_name": "ChineseNewYear",
    #     "event_type": "Religious",
    #     "dates": [
    #         "2011-02-03",
    #         "2012-01-23",
    #         "2013-02-10",
    #         "2014-01-31",
    #         "2015-02-19",
    #         "2016-02-08",
    #     ],
    # },
    # {
    #     "event_name": "NBAFinals",
    #     "event_type": "Sporting",
    #     "dates": [
    #         "2011-05-31",
    #         "2011-06-02",
    #         "2011-06-05",
    #         "2011-06-07",
    #         "2011-06-09",
    #         "2011-06-12",
    #         "2012-06-12",
    #         "2012-06-14",
    #         "2012-06-17",
    #         "2012-06-19",
    #         "2012-06-21",
    #         "2013-06-06",
    #         "2013-06-09",
    #         "2013-06-11",
    #         "2013-06-13",
    #         "2013-06-16",
    #         "2013-06-18",
    #         "2013-06-20",
    #         "2014-06-05",
    #         "2014-06-08",
    #         "2014-06-10",
    #         "2014-06-12",
    #         "2014-06-15",
    #         "2015-06-04",
    #         "2015-06-07",
    #         "2015-06-09",
    #         "2015-06-11",
    #         "2015-06-14",
    #         "2015-06-16",
    #         "2016-06-02",
    #         "2016-06-05",
    #         "2016-06-08",
    #         "2016-06-10",
    #         "2016-06-13",
    #         "2016-06-16",
    #         "2016-06-19",
    #     ],
    # },
    # {
    #     "event_name": "OrthodoxPentecost",
    #     "event_type": "Religious",
    #     "dates": [
    #         "2011-06-12",
    #         "2012-06-03",
    #         "2013-06-23",
    #         "2014-06-08",
    #         "2015-05-31",
    #         "2016-06-19",
    #     ],
    # },
    # {
    #     "event_name": "Pentecost",
    #     "event_type": "Cultural",
    #     "dates": [
    #         "2011-06-12",
    #         "2012-05-27",
    #         "2013-05-19",
    #         "2014-06-08",
    #         "2015-05-24",
    #         "2016-05-15",
    #     ],
    # },
    # {
    #     "event_name": "PesachStart",
    #     "event_type": "Religious",
    #     "dates": [
    #         "2011-04-18",
    #         "2012-04-06",
    #         "2013-03-25",
    #         "2014-04-14",
    #         "2015-04-03",
    #         "2016-04-22",
    #     ],
    # },
    # {
    #     "event_name": "RamadanEnd",
    #     "event_type": "Religious",
    #     "dates": [
    #         "2011-08-29",
    #         "2012-08-18",
    #         "2013-08-07",
    #         "2014-07-27",
    #         "2015-07-16",
    #         "2016-07-05",
    #     ],
    # },
]

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
    "sell_price": "float16",
}

for i in range(1, train_days + 1):
    dtype[f"d_{i}"] = "int16"

parse_dates = ["date"]

level_ids = [
    ["all_id"],
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["item_id", "state_id"],
    ["item_id", "store_id"],
]

level_targets = [f"level_{i + 1}_sales" for i in range(12)]

target = level_targets[-1]
transformed_target = "revenue"

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
    "std": "std",
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
windows = [7, 14, 28, 56]
max_lags = max(periods_online) + max(windows) - 1

aggregate_feature_name_format = "groupby_{}_{}_{}".format
calendar_feature_name_format = "{}_{}".format
count_up_until_nonzero_feature_format = "{}_count_up_until_nonzero".format
diff_feature_name_format = "{}_diff_{}".format
expanding_feature_name_format = "groupby_{}_{}_expanding_{}".format
ewm_feature_name_format = "groupby_{}_{}_ewm_{}_{}".format
pct_change_feature_name_format = "{}_pct_change_{}".format
scaled_feature_name_format = "groupby_{}_scaled_{}".format
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
    "event_name_1",
    "event_name_2",
    "event_type_1",
    "event_type_2",
]

raw_numerical_features = ["sell_price"]

aggregate_features = [
    aggregate_feature_name_format(to_str(by_col), raw_numerical_feature, agg_func_name)
    for by_col in level_ids[1:11]
    for raw_numerical_feature in raw_numerical_features
    for agg_func_name in agg_funcs
]

calendar_features = [f"{col}_{attr}" for col in parse_dates for attr in attrs]

expanding_features = [
    expanding_feature_name_format(to_str(by_col), raw_numerical_feature, agg_func_name)
    for by_col in level_ids[11:]
    for raw_numerical_feature in raw_numerical_features
    for agg_func_name in agg_funcs_for_expanding
]

pct_change_features = [
    pct_change_feature_name_format(raw_numerical_feature, i)
    for raw_numerical_feature in raw_numerical_features
    for i in periods
]

scaled_features = [
    scaled_feature_name_format(to_str(by_col), raw_numerical_feature)
    for by_col in level_ids[11:]
    for raw_numerical_feature in raw_numerical_features
]

shift_features_batch = [
    shift_feature_name_format(level_target, i)
    for level_target in level_targets[9:]
    for i in periods_batch
]
shift_features_online = [
    shift_feature_name_format(level_target, i)
    for level_target in level_targets[9:]
    for i in periods_online
]
shift_features = shift_features_online + shift_features_batch

rolling_features = [
    rolling_feature_name_format(to_str(by_col), shift_feature, j, agg_func_name)
    for by_col in level_ids[11:]
    for shift_feature in shift_features
    for j in windows
    for agg_func_name in agg_funcs_for_rolling
]

numerical_features = (
    ["days_since_release"]
    + raw_numerical_features
    + aggregate_features
    + calendar_features
    + expanding_features
    + pct_change_features
    + rolling_features
    + scaled_features
    + shift_features
)

features = binary_features + categorical_features + numerical_features

random_state = 0

lgb_params = {
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "force_row_wise": True,
    "lambda_l2": 0.1,
    "learning_rate": 0.03,
    "metric": "None",
    "min_data_in_leaf": 1_000,
    "n_jobs": -1,
    "num_leaves": 128,
    "objective": "tweedie",
    "seed": random_state,
    "tweedie_variance_power": 1.2,
}

magic_multiplier = 1.023
