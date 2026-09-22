#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from adm import radiance_linear, radiance_linear_ratio
from netcdf_io import load_data
from radiometry import radiance_to_brightness_temperature
from scene_features import build_scene_features


CHANNEL_INDICES = (0, 3, 4, 6, 7, 8)


def fit_candidate(features, n_components, use_pca, pca_var, random_state):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(features)
	pca = None
	transformed = scaled
	if use_pca:
		full_pca = PCA(svd_solver="full")
		full_pca.fit(scaled)
		n_pc = int(np.searchsorted(np.cumsum(full_pca.explained_variance_ratio_), pca_var) + 1)
		pca = PCA(n_components=n_pc, svd_solver="full", whiten=False)
		transformed = pca.fit_transform(scaled)

	model = GaussianMixture(
		n_components=n_components,
		n_init=5,
		covariance_type="full",
		random_state=random_state,
		reg_covar=1e-6,
	)
	model.fit(transformed)
	return scaler, pca, model, transformed


def transform_predict(features, scaler, pca, model):
	transformed = scaler.transform(features)
	if pca is not None:
		transformed = pca.transform(transformed)
	return transformed, model.predict(transformed)


def cluster_stability_metrics(
	train_features,
	test_features,
	transformed_test,
	baseline_labels,
	n_components,
	use_pca,
	pca_var,
	random_state,
	repeats,
	silhouette_sample_size,
):
	"""Measure geometric separation and assignment stability on held-out points."""
	if np.unique(baseline_labels).size < 2:
		silhouette = np.nan
	else:
		silhouette = float(
			silhouette_score(
				transformed_test,
				baseline_labels,
				sample_size=min(silhouette_sample_size, transformed_test.shape[0]),
				random_state=random_state,
			)
		)

	rng = np.random.default_rng(random_state)
	ari_scores = []
	for repeat in range(repeats):
		bootstrap_indices = rng.integers(0, train_features.shape[0], train_features.shape[0])
		scaler, pca, model, _ = fit_candidate(
			train_features[bootstrap_indices],
			n_components,
			use_pca,
			pca_var,
			random_state + repeat + 1,
		)
		_, labels = transform_predict(test_features, scaler, pca, model)
		ari_scores.append(adjusted_rand_score(baseline_labels, labels))

	return (
		silhouette,
		float(np.mean(ari_scores)) if ari_scores else np.nan,
		float(np.min(ari_scores)) if ari_scores else np.nan,
		float(np.std(ari_scores)) if ari_scores else np.nan,
	)


def load_observations(path, valid_flat, sample_indices):
	data = load_data(
		path,
		variable_names=(
			"lza_G16_interp_corr",
			"lza_G18_interp_corr",
			"rad_G16_interp_corr",
			"rad_G18_interp_corr",
			"planck_G16",
			"planck_G18",
		),
	)
	flat_indices = valid_flat[sample_indices]
	return (
		data["lza_G16_interp_corr"].ravel()[flat_indices],
		data["lza_G18_interp_corr"].ravel()[flat_indices],
		data["rad_G16_interp_corr"].reshape(-1, 9)[flat_indices],
		data["rad_G18_interp_corr"].reshape(-1, 9)[flat_indices],
		data["planck_G16"],
		data["planck_G18"],
	)


def fit_adm_and_score(train_obs, train_labels, test_obs, test_labels, n_components):
	train_lza16, train_lza18, train_rad16, train_rad18, planck16, planck18 = train_obs
	test_lza16, test_lza18, test_rad16, test_rad18, _, _ = test_obs
	ratio_errors = []
	ratio_errors_by_channel = {channel: [] for channel in CHANNEL_INDICES}
	ratio_errors_by_scene = {scene: [] for scene in range(n_components)}
	corrected_bt_diffs = []
	uncorrected_bt_diffs = []
	corrected_flux_diffs = []
	uncorrected_flux_diffs = []

	for channel in CHANNEL_INDICES:
		corrected16 = np.full(test_labels.size, np.nan, dtype=np.float32)
		corrected18 = np.full(test_labels.size, np.nan, dtype=np.float32)
		valid_test = (
			(test_rad16[:, channel] > 0)
			& (test_rad18[:, channel] > 0)
			& np.isfinite(test_rad16[:, channel])
			& np.isfinite(test_rad18[:, channel])
		)
		for scene in range(n_components):
			train_mask = train_labels == scene
			test_mask = (test_labels == scene) & valid_test
			if train_mask.sum() < 50 or test_mask.sum() < 10:
				continue
			train_ratio = np.divide(
				train_rad16[train_mask, channel],
				train_rad18[train_mask, channel],
			)
			finite_train = (
				(train_ratio > 0)
				& np.isfinite(train_ratio)
				& np.isfinite(train_lza16[train_mask])
				& np.isfinite(train_lza18[train_mask])
			)
			if finite_train.sum() < 50:
				continue
			try:
				b_value, _ = curve_fit(
					radiance_linear_ratio,
					(train_lza16[train_mask][finite_train], train_lza18[train_mask][finite_train]),
					train_ratio[finite_train],
					maxfev=2000,
				)
			except (RuntimeError, ValueError):
				continue

			expected_ratio = radiance_linear_ratio(
				(test_lza16[test_mask], test_lza18[test_mask]), b_value[0]
			)
			observed_ratio = np.divide(
				test_rad16[test_mask, channel], test_rad18[test_mask, channel]
			)
			ratio_errors.extend((observed_ratio - expected_ratio).tolist())
			ratio_errors_by_channel[channel].extend((observed_ratio - expected_ratio).tolist())
			ratio_errors_by_scene[scene].extend((observed_ratio - expected_ratio).tolist())

			train_base16 = radiance_linear(train_lza16[train_mask][finite_train], b_value[0])
			train_base18 = radiance_linear(train_lza18[train_mask][finite_train], b_value[0])
			norm16 = np.dot(train_base16, train_rad16[train_mask, channel][finite_train]) / np.dot(train_base16, train_base16)
			norm18 = np.dot(train_base18, train_rad18[train_mask, channel][finite_train]) / np.dot(train_base18, train_base18)
			base16 = radiance_linear(test_lza16[test_mask], b_value[0])
			base18 = radiance_linear(test_lza18[test_mask], b_value[0])
			corrected16[test_mask] = test_rad16[test_mask, channel] / (norm16 * base16)
			corrected18[test_mask] = test_rad18[test_mask, channel] / (norm18 * base18)

		valid_corrected = np.isfinite(corrected16) & np.isfinite(corrected18)
		if valid_corrected.any():
			bt16 = radiance_to_brightness_temperature(corrected16[valid_corrected], planck16[channel])
			bt18 = radiance_to_brightness_temperature(corrected18[valid_corrected], planck18[channel])
			corrected_bt_diffs.extend((bt16 - bt18).tolist())
			corrected_flux_diffs.extend((5.670374e-8 * (bt16**4 - bt18**4)).tolist())
			raw_bt16 = radiance_to_brightness_temperature(test_rad16[valid_corrected, channel], planck16[channel])
			raw_bt18 = radiance_to_brightness_temperature(test_rad18[valid_corrected, channel], planck18[channel])
			uncorrected_bt_diffs.extend((raw_bt16 - raw_bt18).tolist())
			uncorrected_flux_diffs.extend((5.670374e-8 * (raw_bt16**4 - raw_bt18**4)).tolist())

	if not ratio_errors:
		return (np.nan,) * 9
	raw_std = float(np.nanstd(uncorrected_bt_diffs))
	corrected_std = float(np.nanstd(corrected_bt_diffs))
	raw_flux_std = float(np.nanstd(uncorrected_flux_diffs))
	corrected_flux_std = float(np.nanstd(corrected_flux_diffs))
	channel_rmse = [
		float(np.sqrt(np.mean(np.square(errors))))
		for errors in ratio_errors_by_channel.values()
		if errors
	]
	scene_rmse = [
		float(np.sqrt(np.mean(np.square(errors))))
		for errors in ratio_errors_by_scene.values()
		if errors
	]
	angle_bins = np.arange(0, 90, 10)
	coverage = []
	for scene in range(n_components):
		mask = test_labels == scene
		if mask.any():
			angles = test_lza16[mask]
			angles = angles[np.isfinite(angles) & (angles >= 0) & (angles <= 90)]
			if angles.size:
				occupied = np.unique(np.minimum(np.floor(angles / 10).astype(int), 8))
				coverage.append(occupied.size / 9)
	return (
		float(np.sqrt(np.mean(np.square(ratio_errors)))),
		raw_std,
		corrected_std,
		(raw_flux_std - corrected_flux_std) / raw_flux_std if raw_flux_std > 0 else np.nan,
		raw_flux_std,
		corrected_flux_std,
		max(channel_rmse) if channel_rmse else np.nan,
		max(scene_rmse) if scene_rmse else np.nan,
		min(coverage) if coverage else np.nan,
	)


def occupancy_metrics(labels, n_components):
	counts = np.bincount(labels, minlength=n_components)
	fractions = counts / labels.size
	return float(fractions.min()), int(np.count_nonzero(fractions < 0.005)), fractions


def evaluate_candidate(
	n_components, train_features, test_features, train_obs, test_obs,
	validation_sets, use_pca, pca_var, random_state,
	silhouette_sample_size, stability_repeats,
):
	scaler, pca, model, transformed_train = fit_candidate(
		train_features, n_components, use_pca, pca_var, random_state
	)
	transformed_test, test_labels = transform_predict(test_features, scaler, pca, model)
	train_labels = model.predict(transformed_train)
	silhouette, ari_mean, ari_min, ari_std = cluster_stability_metrics(
		train_features,
		test_features,
		transformed_test,
		test_labels,
		n_components,
		use_pca,
		pca_var,
		random_state,
		stability_repeats,
		silhouette_sample_size,
	)
	train_min, train_sparse, _ = occupancy_metrics(train_labels, n_components)
	test_min, test_sparse, test_fraction = occupancy_metrics(test_labels, n_components)
	responsibilities = model.predict_proba(transformed_test)
	entropy = -np.sum(responsibilities * np.log(np.maximum(responsibilities, 1e-12)))
	metrics = fit_adm_and_score(train_obs, train_labels, test_obs, test_labels, n_components)
	occupancy_js = []
	centroid_shift = []
	for validation_features, _ in validation_sets:
		transformed_validation, validation_labels = transform_predict(validation_features, scaler, pca, model)
		validation_fraction = np.bincount(validation_labels, minlength=n_components) / validation_labels.size
		occupancy_js.append(float(jensenshannon(test_fraction, validation_fraction) ** 2))
		shifts = []
		for scene in range(n_components):
			mask = validation_labels == scene
			if mask.any():
				shifts.append(np.linalg.norm(transformed_validation[mask].mean(axis=0) - model.means_[scene]))
		centroid_shift.append(float(np.mean(shifts)) if shifts else np.nan)
	(
		adm_rmse,
		raw_std,
		corrected_std,
		flux_improvement,
		raw_flux_std,
		corrected_flux_std,
		worst_channel_rmse,
		worst_scene_rmse,
		angular_coverage_min,
	) = metrics
	return {
		"n_components": n_components,
		"bic": float(model.bic(transformed_train)),
		"aic": float(model.aic(transformed_train)),
		"heldout_loglik": float(model.score(transformed_test)),
		"icl": float(model.bic(transformed_test) + 2 * entropy),
		"train_min_fraction": train_min,
		"test_min_fraction": test_min,
		"train_sparse_components": train_sparse,
		"test_sparse_components": test_sparse,
		"occupancy_js": float(np.mean(occupancy_js)) if occupancy_js else np.nan,
		"centroid_shift": float(np.mean(centroid_shift)) if centroid_shift else np.nan,
		"adm_ratio_rmse": adm_rmse,
		"uncorrected_bt_std": raw_std,
		"corrected_bt_std": corrected_std,
		"corrected_bt_improvement": (raw_std - corrected_std) / raw_std if raw_std > 0 else np.nan,
		"uncorrected_flux_std": raw_flux_std,
		"corrected_flux_std": corrected_flux_std,
		"corrected_flux_improvement": flux_improvement,
		"silhouette": silhouette,
		"ari_mean": ari_mean,
		"ari_min": ari_min,
		"ari_std": ari_std,
		"worst_channel_adm_rmse": worst_channel_rmse,
		"worst_scene_adm_rmse": worst_scene_rmse,
		"minimum_angular_coverage": angular_coverage_min,
	}


def main():
	parser = argparse.ArgumentParser(
		description="Evaluate GMM component counts with statistical and ADM diagnostics."
	)
	parser.add_argument("-f", "--input-file", nargs="+", required=True)
	parser.add_argument("--validation-inputs", nargs="*", default=[])
	parser.add_argument("--use-pca", action="store_true")
	parser.add_argument("--pca-var", type=float, default=0.98)
	parser.add_argument("--min-components", type=int, default=2)
	parser.add_argument("--max-components", type=int, default=19)
	parser.add_argument("--max-points", type=int, default=100000)
	parser.add_argument("--silhouette-sample-size", type=int, default=5000)
	parser.add_argument("--stability-repeats", type=int, default=3)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output-csv", default="figures/gmm_component_diagnostics.csv")
	parser.add_argument("--output-plot", default="figures/gmm_component_diagnostics.png")
	parser.add_argument("--adm-plateau-tolerance", type=float, default=0.01, help="Maximum relative ADM RMSE improvement considered a plateau")
	parser.add_argument("--min-scene-fraction", type=float, default=0.005, help="Minimum held-out fraction for every scene")
	parser.add_argument("--max-occupancy-js", type=float, default=0.10, help="Maximum Jensen-Shannon divergence of scene occupancy")
	parser.add_argument("--max-centroid-shift", type=float, default=1.0, help="Maximum mean standardized feature-centroid shift")
	parser.add_argument("--min-silhouette", type=float, default=0.0, help="Minimum held-out silhouette score")
	parser.add_argument("--min-ari", type=float, default=0.80, help="Minimum repeated-fit adjusted Rand index")
	parser.add_argument("--min-angular-coverage", type=float, default=0.70, help="Minimum fraction of 10-degree viewing-angle bins occupied by every scene")
	parser.add_argument("--max-worst-scene-adm-rmse", type=float, default=np.inf, help="Maximum worst-scene held-out ADM ratio RMSE")
	parser.add_argument("--min-corrected-improvement", type=float, default=0.0, help="Minimum fractional reduction in G16-G18 flux spread")
	args = parser.parse_args()

	if args.min_components < 2 or args.max_components < args.min_components:
		raise ValueError("component range must satisfy 2 <= min-components <= max-components")
	rng = np.random.default_rng(args.seed)

	# Build one reproducible train/test pool across all training days.
	features_parts = []
	observation_parts = []
	points_per_file = max(2, args.max_points // len(args.input_file))
	for path in args.input_file:
		features, valid_flat, _, _, _ = build_scene_features(path, pixel_step=3)
		selection = rng.choice(features.shape[0], min(points_per_file, features.shape[0]), replace=False)
		features_parts.append(features[selection])
		observation_parts.append(load_observations(path, valid_flat, selection))
	features = np.concatenate(features_parts, axis=0)
	observations = tuple(
		(np.concatenate([part[index] for part in observation_parts], axis=0) if index < 4 else observation_parts[0][index])
		for index in range(6)
	)
	order = rng.permutation(features.shape[0])
	split = max(1, int(0.8 * order.size))
	train_selection, test_selection = order[:split], order[split:]
	if test_selection.size == 0:
		raise ValueError("max-points must provide at least two points")
	train_features = features[train_selection]
	test_features = features[test_selection]
	train_obs = tuple(value[train_selection] if index < 4 else value for index, value in enumerate(observations))
	test_obs = tuple(value[test_selection] if index < 4 else value for index, value in enumerate(observations))

	validation_sets = []
	for path in args.validation_inputs:
		validation_features, _, _, _, _ = build_scene_features(path, pixel_step=3)
		selection = rng.choice(validation_features.shape[0], min(args.max_points, validation_features.shape[0]), replace=False)
		validation_sets.append((validation_features[selection], path))

	rows = []
	for n_components in range(args.min_components, args.max_components + 1):
		print(f"Evaluating {n_components} components")
		rows.append(evaluate_candidate(
			n_components, train_features, test_features, train_obs, test_obs,
			validation_sets, args.use_pca, args.pca_var, args.seed,
			args.silhouette_sample_size, args.stability_repeats,
		))

	previous_rmse = np.nan
	for row in rows:
		if np.isfinite(previous_rmse) and np.isfinite(row["adm_ratio_rmse"]):
			row["adm_rmse_relative_improvement"] = (previous_rmse - row["adm_ratio_rmse"]) / previous_rmse
		else:
			row["adm_rmse_relative_improvement"] = np.nan
		row["criterion_adm_plateau"] = bool(
			np.isfinite(row["adm_rmse_relative_improvement"])
			and row["adm_rmse_relative_improvement"] < args.adm_plateau_tolerance
		)
		row["criterion_occupancy"] = bool(row["test_min_fraction"] >= args.min_scene_fraction)
		row["criterion_stability"] = bool(
			args.validation_inputs
			and np.isfinite(row["occupancy_js"])
			and row["occupancy_js"] <= args.max_occupancy_js
			and np.isfinite(row["centroid_shift"])
			and row["centroid_shift"] <= args.max_centroid_shift
			and np.isfinite(row["ari_min"])
			and row["ari_min"] >= args.min_ari
		)
		row["criterion_geometry"] = bool(
			np.isfinite(row["silhouette"])
			and row["silhouette"] >= args.min_silhouette
			and np.isfinite(row["minimum_angular_coverage"])
			and row["minimum_angular_coverage"] >= args.min_angular_coverage
		)
		row["criterion_adm_coverage"] = bool(
			np.isfinite(row["worst_scene_adm_rmse"])
			and row["worst_scene_adm_rmse"] <= args.max_worst_scene_adm_rmse
		)
		row["criterion_corrected_improvement"] = bool(
			np.isfinite(row["corrected_flux_improvement"])
			and row["corrected_flux_improvement"] >= args.min_corrected_improvement
		)
		row["all_criteria"] = all(
			row[name]
			for name in (
				"criterion_adm_plateau",
				"criterion_occupancy",
				"criterion_stability",
				"criterion_geometry",
				"criterion_adm_coverage",
				"criterion_corrected_improvement",
			)
		)
		if np.isfinite(row["adm_ratio_rmse"]):
			previous_rmse = row["adm_ratio_rmse"]

	output_path = Path(args.output_csv)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
		writer.writeheader()
		writer.writerows(rows)

	counts = [row["n_components"] for row in rows]
	metric_groups = (
		(
			"Statistical fit",
			(
				("bic", "BIC (lower)"),
				("aic", "AIC (lower)"),
				("icl", "ICL (lower)"),
				("heldout_loglik", "Held-out log likelihood (higher)"),
			),
		),
		(
			"Occupancy and stability",
			(
				("train_min_fraction", "Train minimum fraction (higher)"),
				("test_min_fraction", "Test minimum fraction (higher)"),
				("train_sparse_components", "Train sparse scenes (lower)"),
				("test_sparse_components", "Test sparse scenes (lower)"),
				("occupancy_js", "Occupancy JS (lower)"),
				("centroid_shift", "Centroid shift (lower)"),
				("silhouette", "Silhouette (higher)"),
				("ari_mean", "ARI mean (higher)"),
				("ari_min", "ARI minimum (higher)"),
				("ari_std", "ARI variability (lower)"),
				("minimum_angular_coverage", "Minimum angular coverage (higher)"),
			),
		),
		(
			"ADM and flux physics",
			(
				("adm_ratio_rmse", "ADM ratio RMSE (lower)"),
				("worst_channel_adm_rmse", "Worst-channel ADM RMSE (lower)"),
				("worst_scene_adm_rmse", "Worst-scene ADM RMSE (lower)"),
				("adm_rmse_relative_improvement", "ADM RMSE improvement (lower = plateau)"),
				("uncorrected_bt_std", "Uncorrected BT spread (lower)"),
				("corrected_bt_std", "Corrected BT spread (lower)"),
				("corrected_bt_improvement", "BT improvement (higher)"),
				("uncorrected_flux_std", "Uncorrected flux spread (lower)"),
				("corrected_flux_std", "Corrected flux spread (lower)"),
				("corrected_flux_improvement", "Flux improvement (higher)"),
			),
		),
	)
	markers = ("o", "s", "^", "D", "v", "<", ">", "P", "X", "*", "h", "8", "p", "d", "H", "+", "x", "1")
	fig, axes = plt.subplots(len(metric_groups), 1, figsize=(15, 18), sharex=True)
	for axis, (group_name, metrics) in zip(axes, metric_groups):
		for marker, (name, label) in zip(markers, metrics):
			values = np.asarray([row[name] for row in rows], dtype=float)
			if not np.isfinite(values).any():
				continue
			finite_values = values[np.isfinite(values)]
			value_range = np.nanmax(finite_values) - np.nanmin(finite_values)
			if value_range > 0:
				values = (values - np.nanmin(finite_values)) / value_range
			else:
				values = np.zeros_like(values)
			axis.plot(counts, values, marker=marker, label=label, linewidth=1.5)
		axis.set_title(group_name)
		axis.set_ylabel("Normalized value")
		axis.grid(True, alpha=0.3)
		axis.legend(loc="best", fontsize="small", ncol=2)
	axes[-1].set_xlabel("Number of GMM components")
	fig.suptitle("GMM component diagnostics", fontsize=14)
	fig.tight_layout()
	plot_path = Path(args.output_plot)
	plot_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(plot_path, dpi=150)
	plt.close()

	print(f"Diagnostics written to {output_path}")
	print(f"Plot written to {plot_path}")
	print("Interpretation: lower ADM RMSE, lower occupancy JS, lower centroid shift, and higher corrected flux improvement are preferred.")


if __name__ == "__main__":
	main()
