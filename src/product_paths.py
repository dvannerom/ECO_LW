from pathlib import Path
import json


SELECTION_FILE = Path("data/models/selected_n_components.json")


def selected_n_components(selection_file=SELECTION_FILE):
    with Path(selection_file).open() as handle:
        selection = json.load(handle)
    n_components = selection.get("n_components")
    if n_components is None:
        raise ValueError(f"{selection_file} does not contain a selected n_components")
    return int(n_components)


def scene_id_path(day, resolution, n_components):
    return f"data/scene_id/scene_id_{day}_res{resolution}km_{n_components}comp.nc"


def model_path(resolution, n_components, n_training_files=1):
    return f"data/models/gmm_pipeline_merged_{n_training_files}files_res{resolution}km_{n_components}comp.joblib"