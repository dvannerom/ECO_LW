import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
import time
import math
import argparse
import joblib
from netcdf_io import write_dataset
from scene_features import build_scene_features, scene_label_mapping
from product_paths import model_path, selected_n_components

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="GOES16/18 Scene ID with optional PCA+GMM")
	parser.add_argument("-f", "--input_file", type=str, default="data/preprocessed_files/abi_pix1000_step5.nc")
	parser.add_argument("--model", type=str, default=None)
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	parser.add_argument(
		"--label-order",
		choices=("c14", "c14_btd14_08"),
		default="c14_btd14_08",
		help="Ordering from cold/icy to hot/clear (default: c14_btd14_08)",
	)
	args = parser.parse_args()
	
	input_file = args.input_file
	model = args.model or model_path(2, selected_n_components())
	lambda_center = args.lambda_center
	
	day = input_file.split("/")[-1].split(".")[0].split("_")[1]
	res = input_file.split("/")[-1].split(".")[0].split("_")[2]
		
	# Retrieve data
	BT_nozeros, valid_flat, mask, lat_interp_grid, lon_interp_grid = build_scene_features(input_file)

	# Retrieve model
	pipeline = joblib.load(model)
	scaler = pipeline.named_steps["scaler"]
	pca = pipeline.named_steps["pca"]
	gmm = pipeline.named_steps["gmm"]
	n_components = gmm.n_components

	labels_nozeros = pipeline.predict(BT_nozeros)
	
	# Map labels back to full grid while keeping memory use bounded.
	n_pixels = mask.size
	labels = np.zeros(n_pixels, dtype=np.int16)
	labels[valid_flat] = labels_nozeros

	# -----------------------------
	# Physically meaningful label ordering
	# -----------------------------
	# Sort by standardized C14 plus BTD14-08 (indices 3 and 8).
	# gmm.means_ are in Z-space; map to original feature space:
	#  (1) if PCA used: inverse PCA, then inverse scale
	means_scaled = pca.inverse_transform(gmm.means_)
	
	label_map = scene_label_mapping(means_scaled, args.label_order)
	sorted_labels_nozeros = np.array([label_map[l] for l in labels_nozeros], dtype=np.int16)
	
	sorted_labels = np.full(n_pixels, fill_value=np.nan, dtype=np.float32)
	sorted_labels[valid_flat] = sorted_labels_nozeros
	
	scene_map = sorted_labels.reshape(lat_interp_grid.shape[0], lat_interp_grid.shape[1])
	suffix = f"{day}_{res}_{int(n_components)}comp"
	
	# -----------------------------
	# Save outputs (labels + settings)
	# -----------------------------
	out_path = f"data/scene_id/scene_id_{suffix}.nc"
	write_dataset(
		out_path,
		{
			"labels": labels,
			"labels_nozeros": labels_nozeros,
			"sorted_labels": sorted_labels,
			"lat": lat_interp_grid,
			"lon": lon_interp_grid,
		},
		{
			"labels": ("pixel",),
			"labels_nozeros": ("valid_pixel",),
			"sorted_labels": ("pixel",),
			"lat": ("y", "x"),
			"lon": ("y", "x"),
		},
		attrs={"product": "ECO scene identification", "n_components": int(n_components)},
	)
