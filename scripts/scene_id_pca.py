"""Classify every BT-valid ABI pixel with the shared spectral-only scene GMM.

The model is the reviewed full-population streaming GMM
(``spectral_gmm.joblib``: scaler/PCA/GMM Pipeline on the 10 spectral features
of ``scene_features.spectral_feature_names``). Features are streamed by row
chunk, so peak RAM is the output grids (int16 + float32 per pixel) plus one
chunk; the cost is O(n_pixels * K * n_pca) for prediction.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import xarray as xr

from netcdf_io import write_dataset
from product_paths import scene_id_path
from scene_features import iter_abi_spectral_features, scene_label_mapping, spectral_feature_names
from scene_taxonomy import FEATURE_SET


def load_scene_model(path):
	"""Return the scaler/PCA/GMM Pipeline stored in ``path``."""
	loaded = joblib.load(path)
	return loaded["model"] if isinstance(loaded, dict) else loaded


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("-f", "--input_file", type=str, required=True)
	parser.add_argument("--model", type=str, required=True,
		help="Shared spectral_gmm.joblib from the nominal full-population run")
	parser.add_argument("--gate", type=str, required=True,
		help="Verified scene-taxonomy gate JSON (scripts/check_scene_taxonomy.py)")
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	parser.add_argument("--chunk-rows", type=int, default=512)
	parser.add_argument(
		"--label-order",
		choices=("c14", "c14_btd14_08"),
		default="c14",
		help="Ordering of sorted_labels from cold/icy to hot/clear",
	)
	args = parser.parse_args()

	gate = json.loads(Path(args.gate).read_text())
	pipeline = load_scene_model(args.model)
	pca = pipeline.named_steps["pca"]
	gmm = pipeline.named_steps["gmm"]
	n_components = int(gmm.n_components)
	if n_components != int(gate["n_components"]):
		raise ValueError(f"Model K={n_components} differs from reviewed K={gate['n_components']}")
	if int(pipeline.named_steps["scaler"].n_features_in_) != len(spectral_feature_names()):
		raise ValueError(f"{args.model} is not a {FEATURE_SET} model")

	stem_parts = Path(args.input_file).stem.split("_")
	day, res = stem_parts[1], stem_parts[2]
	with xr.open_dataset(args.input_file, engine="netcdf4") as dataset:
		lat = dataset["lat_interp_grid"].values
		lon = dataset["lon_interp_grid"].values
	n_pixels = lat.size

	# labels: raw GMM component; 0 also marks invalid pixels (legacy convention).
	labels = np.zeros(n_pixels, dtype=np.int16)
	valid = np.zeros(n_pixels, dtype=bool)
	for features, positions in iter_abi_spectral_features(
			args.input_file, chunk_rows=args.chunk_rows, pixel_step=1):
		labels[positions] = pipeline.predict(features)
		valid[positions] = True

	label_map = scene_label_mapping(pca.inverse_transform(gmm.means_), args.label_order)
	lookup = np.array([label_map[component] for component in range(n_components)], dtype=np.float32)
	sorted_labels = np.full(n_pixels, np.nan, dtype=np.float32)
	sorted_labels[valid] = lookup[labels[valid]]

	out_path = scene_id_path(day, int(res.removeprefix("res").removesuffix("km")), n_components)
	write_dataset(
		out_path,
		{
			"labels": labels,
			"labels_nozeros": labels[valid],
			"sorted_labels": sorted_labels,
			"lat": lat,
			"lon": lon,
		},
		{
			"labels": ("pixel",),
			"labels_nozeros": ("valid_pixel",),
			"sorted_labels": ("pixel",),
			"lat": ("y", "x"),
			"lon": ("y", "x"),
		},
		attrs={
			"product": "ECO scene identification",
			"n_components": n_components,
			"feature_set": FEATURE_SET,
			"model": str(args.model),
			"scene_taxonomy_gate": str(args.gate),
			"diagnostics_csv_sha256": str(gate["diagnostics_csv_sha256"]),
			"validity": "all six G16 and G18 BTs finite and in (0, 1000) K",
		},
	)


if __name__ == "__main__":
	main()
