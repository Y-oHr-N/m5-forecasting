import argparse
import datetime
import pathlib
import sys

import papermill as pm

# See https://github.com/jupyter/notebook/issues/4613#issuecomment-548992047
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

now = datetime.datetime.now()

root_dir_path = pathlib.Path(".")
notebooks_dir_path = root_dir_path / "notebooks"
inputs_dir_path = notebooks_dir_path / "inputs"
outputs_dir_path = notebooks_dir_path / "outputs"
work_dir_path = outputs_dir_path / now.strftime("%Y_%m_%d_%H_%M_%S")

work_dir_path.mkdir()

parser = argparse.ArgumentParser()

parser.add_argument(
    "-a", "--accuracy", action="store_true", help="submit to m5-forecasting-accuracy",
)

parser.add_argument(
    "-u",
    "--uncertainty",
    action="store_true",
    help="submit to m5-forecasting-uncertainty",
)

args = parser.parse_args()

tasks = [
    "preprocess",
    "train_lgbm_reg",
    "predict",
]

if args.accuracy:
    tasks.append("submit_accuracy")

if args.uncertainty:
    tasks.append("submit_uncertainty")

for task in tasks:
    pm.execute_notebook(
        str(inputs_dir_path / (f"{task}.ipynb")),
        str(work_dir_path / (f"{task}.ipynb")),
        parameters={"root_dir": str(root_dir_path)},
    )
