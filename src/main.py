import datetime
import pathlib

import papermill as pm

now = datetime.datetime.now()

root_dir_path = pathlib.Path(".")
notebooks_dir_path = root_dir_path / "notebooks"
inputs_dir_path = notebooks_dir_path / "inputs"
outputs_dir_path = notebooks_dir_path / "outputs"
work_dir_path = outputs_dir_path / now.strftime("%Y_%m_%d_%H_%M_%S")

work_dir_path.mkdir()

tasks = [
    "preprocess",
    "engineer",
    "train_reg",
    "predict",
    "postprocess",
    "submit",
]

for task in tasks:
    pm.execute_notebook(
        str(inputs_dir_path / (f"{task}.ipynb")),
        str(work_dir_path / (f"{task}.ipynb")),
        parameters={"root_dir": str(root_dir_path)},
    )
