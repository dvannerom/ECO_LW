import os
import json
import numpy as np
import argparse
import joblib
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scene_features import build_scene_features

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="GOES16/18 Scene ID with optional PCA+GMM")
	parser.add_argument("-f", "--input_file", type=str, nargs="+", default="data/preprocessed_files/abi_pix1000_step5.nc")
	parser.add_argument("-l", "--lambda_center", type=float, default=-106)
	parser.add_argument("-n", "--n_components", type=int, default=10, help="Number of GMM components")
	parser.add_argument("--n_components_file", type=str, default=None,
		help="JSON file from find_nComponents.py; overrides -n with its selected n_components")
	parser.add_argument("--use_pca", action="store_true", help="Enable PCA before GMM")
	parser.add_argument("--pca_var", type=float, default=0.98, help="Cumulative variance to keep (e.g., 0.98)")
	parser.add_argument("--points_per_file", type=int, default=2_000_000,
		help="Max valid-pixel rows sampled per input file before fitting (caps total EM cost)")
	parser.add_argument("--seed", type=int, default=42, help="RNG seed for per-file subsampling")
	args = parser.parse_args()
	
	input_file = args.input_file
	if isinstance(input_file, str):
		input_file = [input_file]
	lambda_center = args.lambda_center
	n_components = args.n_components
	if args.n_components_file:
		selection = json.load(open(args.n_components_file))
		if selection.get("n_components") is None:
			raise SystemExit(f"{args.n_components_file} has no selected n_components (find_nComponents.py found no passing candidate)")
		n_components = int(selection["n_components"])
		print(f"Using n_components={n_components} from {args.n_components_file}")
	use_pca = args.use_pca
	pca_var = args.pca_var
	points_per_file = args.points_per_file

	# Independent per-file RNG substreams so the sample is identical regardless
	# of input_file ordering (same pattern as find_nComponents.py).
	seed_sequences = np.random.SeedSequence(args.seed).spawn(len(input_file))

	BT_arrays = []

	for f, seed_sequence in zip(input_file, seed_sequences):
		features = build_scene_features(f, pixel_step=3)[0]
		rng = np.random.default_rng(seed_sequence)
		selection = rng.choice(features.shape[0], min(points_per_file, features.shape[0]), replace=False)
		BT_arrays.append(features[selection])
		print(f"{f}: sampled {selection.size} of {features.shape[0]} valid rows")

	# float64 avoids spurious non-positive-definite covariances in GaussianMixture's Cholesky step
	# (the 6 channel-average + 4 BTD feature columns are exact linear combinations of each other).
	BT_merged = np.concatenate(BT_arrays, axis=0).astype(np.float64, copy=False)

	print(BT_merged.shape)
	
	# ---------------------------------------
	# Build pipeline Scale -> PCA (optional) -> GMM
	# ---------------------------------------
	
	steps = [
		("scaler", StandardScaler())
	]
	
	if use_pca:
		# 1) Fit a full PCA for diagnostics
		pca_full = PCA(svd_solver='full')
		pca_full.fit(StandardScaler().fit_transform(BT_merged))
		
		# Auto-select number of PCs to reach target variance
		cvar = np.cumsum(pca_full.explained_variance_ratio_)
		n_pc = int(np.searchsorted(cvar, pca_var) + 1)
		print(f"Running PCA with {n_pc} PCs")

		steps.append(
			("pca",
				PCA(
					n_components=n_pc,
					svd_solver="full",
					whiten=False)
			)
		)

	steps.append(
		("gmm",
			GaussianMixture(
				n_components=n_components,
				n_init=5,
				covariance_type="full",
				random_state=42,
				reg_covar=1e-6)
		)
	)

	pipeline = Pipeline(steps)
	pipeline.fit(BT_merged)

	suffix = f"merged_{len(input_file)}files_res2km_{int(n_components)}comp"

	# -----------------------------
	# Save model
	# -----------------------------
	model_path = f"data/models/gmm_pipeline_{suffix}.joblib"
	os.makedirs("data/models", exist_ok=True)
	joblib.dump(pipeline, model_path)
	print(f"Model saved to {model_path}")
	
#	# -----------------------------
#	# Save outputs (labels + settings)
#	# -----------------------------
#	out_npz_path = f"data/scene_id/scene_id_{suffix}.npz"
#	np.savez(out_npz_path,
#	         labels=labels,
#	         labels_nozeros=labels_nozeros,
#	         sorted_labels=sorted_labels,
#	         use_pca=use_pca,
#	         n_pc=(int(n_pc) if use_pca else None),
#	         pca_var=(float(pca_var) if use_pca else None),
#	         n_components=int(n_components),
#	         random_state=42)
