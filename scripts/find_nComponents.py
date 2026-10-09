#!/usr/bin/env python3
import argparse
import csv
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon
from threadpoolctl import threadpool_limits

from adm import radiance_linear, radiance_linear_ratio
from adm_fitting import correct_radiance, fit_adm_scene
from broadband import cubic_regression
from gmm_stability import cluster_stability_metrics, fit_candidate, transform_predict
from netcdf_io import load_data
from radiometry import radiance_to_brightness_temperature
from scene_features import build_scene_features


CHANNEL_INDICES = (0, 3, 4, 6, 7, 8)
FLUX_SIGMA = 5.670374e-08
MIN_TRAIN_POINTS = 50
MIN_EVAL_POINTS = 10
# Conservative peak RSS per build_scene_features() worker (chunk buffers at the default
# chunk_rows/pixel_step=3), used to cap --load-jobs so it can't OOM-kill the pool.
LOAD_WORKER_MEMORY_BUDGET_BYTES = 8 * 1024**3


def _available_memory_bytes():
	"""Linux-only best-effort read of MemAvailable; returns None if unavailable (e.g. non-Linux)."""
	try:
		with open("/proc/meminfo") as handle:
			for line in handle:
				if line.startswith("MemAvailable:"):
					return int(line.split()[1]) * 1024
	except OSError:
		pass
	return None


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


def _load_input_file(path, points_per_file, seed):
	"""Sample one --input-file path (process-pool entry point, order-independent RNG)."""
	rng = np.random.default_rng(seed)
	features, valid_flat, _, _, _ = build_scene_features(path, pixel_step=3)
	selection = rng.choice(features.shape[0], min(points_per_file, features.shape[0]), replace=False)
	return features[selection], load_observations(path, valid_flat, selection)


def _load_validation_file(path, max_points, seed):
	"""Sample one --validation-inputs path (process-pool entry point, order-independent RNG)."""
	rng = np.random.default_rng(seed)
	validation_features, valid_flat, _, _, _ = build_scene_features(path, pixel_step=3)
	selection = rng.choice(validation_features.shape[0], min(max_points, validation_features.shape[0]), replace=False)
	return validation_features[selection], path, load_observations(path, valid_flat, selection)


def fit_adm_and_score(train_obs, train_labels, test_obs, test_labels, n_components):
	"""Cheap per-candidate ADM diagnostic: least-squares normalization, not the production fit."""
	train_lza16, train_lza18, train_rad16, train_rad18, _, _ = train_obs
	test_lza16, test_lza18, test_rad16, test_rad18, _, _ = test_obs
	ratio_errors = []
	ratio_errors_by_channel = {channel: [] for channel in CHANNEL_INDICES}
	ratio_errors_by_scene = {scene: [] for scene in range(n_components)}
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
			if train_mask.sum() < MIN_TRAIN_POINTS or test_mask.sum() < MIN_EVAL_POINTS:
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
			if finite_train.sum() < MIN_TRAIN_POINTS:
				continue
			try:
				popt, _ = curve_fit(
					radiance_linear_ratio,
					(train_lza16[train_mask][finite_train], train_lza18[train_mask][finite_train]),
					train_ratio[finite_train],
					maxfev=2000,
				)
			except (RuntimeError, ValueError):
				continue
			b_value = popt[0]

			expected_ratio = radiance_linear_ratio(
				(test_lza16[test_mask], test_lza18[test_mask]), b_value
			)
			observed_ratio = np.divide(
				test_rad16[test_mask, channel], test_rad18[test_mask, channel]
			)
			ratio_errors.extend((observed_ratio - expected_ratio).tolist())
			ratio_errors_by_channel[channel].extend((observed_ratio - expected_ratio).tolist())
			ratio_errors_by_scene[scene].extend((observed_ratio - expected_ratio).tolist())

			train_base16 = radiance_linear(train_lza16[train_mask][finite_train], b_value)
			train_base18 = radiance_linear(train_lza18[train_mask][finite_train], b_value)
			norm16 = np.dot(train_base16, train_rad16[train_mask, channel][finite_train]) / np.dot(train_base16, train_base16)
			norm18 = np.dot(train_base18, train_rad18[train_mask, channel][finite_train]) / np.dot(train_base18, train_base18)
			base16 = radiance_linear(test_lza16[test_mask], b_value)
			base18 = radiance_linear(test_lza18[test_mask], b_value)
			corrected16[test_mask] = test_rad16[test_mask, channel] / (norm16 * base16)
			corrected18[test_mask] = test_rad18[test_mask, channel] / (norm18 * base18)

		valid_corrected = np.isfinite(corrected16) & np.isfinite(corrected18)
		if valid_corrected.any():
			bt16 = radiance_to_brightness_temperature(corrected16[valid_corrected], test_obs[4][channel])
			bt18 = radiance_to_brightness_temperature(corrected18[valid_corrected], test_obs[5][channel])
			corrected_flux_diffs.extend((FLUX_SIGMA * (bt16**4 - bt18**4)).tolist())
			raw_bt16 = radiance_to_brightness_temperature(test_rad16[valid_corrected, channel], test_obs[4][channel])
			raw_bt18 = radiance_to_brightness_temperature(test_rad18[valid_corrected, channel], test_obs[5][channel])
			uncorrected_flux_diffs.extend((FLUX_SIGMA * (raw_bt16**4 - raw_bt18**4)).tolist())

	if not ratio_errors:
		return (np.nan,) * 7
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


def fit_exact_adm_params(train_obs, train_labels, n_components):
	"""Fit (b, norm) per channel/scene using the exact production ADM routine (adm_fitting.fit_adm_scene)."""
	lza16, lza18, rad16, rad18, _, _ = train_obs
	adm_params = {}
	for channel in CHANNEL_INDICES:
		for scene in range(n_components):
			mask = train_labels == scene
			if mask.sum() < MIN_TRAIN_POINTS:
				continue
			valid = (
				(rad16[mask, channel] > 0) & (rad18[mask, channel] > 0)
				& np.isfinite(rad16[mask, channel]) & np.isfinite(rad18[mask, channel])
				& np.isfinite(lza16[mask]) & np.isfinite(lza18[mask])
			)
			if valid.sum() < MIN_TRAIN_POINTS:
				continue
			try:
				b, norm = fit_adm_scene(
					lza16[mask][valid], lza18[mask][valid],
					rad16[mask][valid, channel], rad18[mask][valid, channel],
				)
			except (RuntimeError, ValueError):
				continue
			adm_params[(channel, scene)] = (b, norm)
	return adm_params


def exact_broadband_flux_diffs(obs, labels, adm_params, n_components):
	"""Compute exact corrected/uncorrected broadband flux differences (production formulas)."""
	lza16, lza18, rad16, rad18, planck16, planck18 = obs
	n_points = labels.size
	n_channels = len(CHANNEL_INDICES)
	bt16_raw = np.full((n_points, n_channels), np.nan)
	bt18_raw = np.full((n_points, n_channels), np.nan)
	bt16_corr = np.full((n_points, n_channels), np.nan)
	bt18_corr = np.full((n_points, n_channels), np.nan)

	for idx, channel in enumerate(CHANNEL_INDICES):
		valid_channel = (
			(rad16[:, channel] > 0) & (rad18[:, channel] > 0)
			& np.isfinite(rad16[:, channel]) & np.isfinite(rad18[:, channel])
		)
		bt16_raw[valid_channel, idx] = radiance_to_brightness_temperature(rad16[valid_channel, channel], planck16[channel])
		bt18_raw[valid_channel, idx] = radiance_to_brightness_temperature(rad18[valid_channel, channel], planck18[channel])

		for scene in range(n_components):
			params = adm_params.get((channel, scene))
			if params is None:
				continue
			b, norm = params
			mask = (labels == scene) & valid_channel
			if not mask.any():
				continue
			corrected16 = correct_radiance(rad16[mask, channel], lza16[mask], b, norm)
			corrected18 = correct_radiance(rad18[mask, channel], lza18[mask], b, norm)
			bt16_corr[mask, idx] = radiance_to_brightness_temperature(corrected16, planck16[channel])
			bt18_corr[mask, idx] = radiance_to_brightness_temperature(corrected18, planck18[channel])

	def flux_diff(bt16, bt18):
		valid = np.all(np.isfinite(bt16), axis=1) & np.all(np.isfinite(bt18), axis=1)
		if not valid.any():
			return np.array([])
		flux16 = FLUX_SIGMA * np.power(cubic_regression(bt16[valid]), 4)
		flux18 = FLUX_SIGMA * np.power(cubic_regression(bt18[valid]), 4)
		flux16 = np.where((flux16 >= 50) & (flux16 <= 1e3), flux16, np.nan)
		flux18 = np.where((flux18 >= 50) & (flux18 <= 1e3), flux18, np.nan)
		diff = flux16 - flux18
		return diff[np.isfinite(diff)]

	return flux_diff(bt16_raw, bt18_raw), flux_diff(bt16_corr, bt18_corr)


def exact_adm_ratio_metrics(eval_sets, adm_params, n_components):
	"""Evaluate held-out ratio residuals using the exact production ADM parameters."""
	all_errors = []
	errors_by_channel = {channel: [] for channel in CHANNEL_INDICES}
	errors_by_scene = {scene: [] for scene in range(n_components)}

	for obs, labels in eval_sets:
		lza16, lza18, rad16, rad18, _, _ = obs
		for channel in CHANNEL_INDICES:
			valid_channel = (
				(rad16[:, channel] > 0) & (rad18[:, channel] > 0)
				& np.isfinite(rad16[:, channel]) & np.isfinite(rad18[:, channel])
				& np.isfinite(lza16) & np.isfinite(lza18)
			)
			for scene in range(n_components):
				params = adm_params.get((channel, scene))
				if params is None:
					continue
				mask = (labels == scene) & valid_channel
				if mask.sum() < MIN_EVAL_POINTS:
					continue
				b, _ = params
				observed = np.divide(rad16[mask, channel], rad18[mask, channel])
				predicted = radiance_linear_ratio((lza16[mask], lza18[mask]), b)
				errors = observed - predicted
				all_errors.extend(errors.tolist())
				errors_by_channel[channel].extend(errors.tolist())
				errors_by_scene[scene].extend(errors.tolist())

	if not all_errors:
		return np.nan, np.nan, np.nan

	def rmse(errors):
		return float(np.sqrt(np.mean(np.square(errors))))

	channel_rmse = [rmse(errors) for errors in errors_by_channel.values() if errors]
	scene_rmse = [rmse(errors) for errors in errors_by_scene.values() if errors]
	return rmse(all_errors), max(channel_rmse), max(scene_rmse)


def evaluate_exact_candidate(n_components, train_obs, train_labels, eval_sets):
	"""Run the exact ADM fit and cubic broadband regression on a shortlisted candidate.

	``eval_sets`` is a list of ``(obs, labels)`` pairs: the held-out test split of
	the primary training file plus any validation-day observations and their
	candidate-predicted labels. All differences are pooled before computing the
	final standard deviation, matching how the production workflow evaluates a
	full day rather than averaging per-day statistics.
	"""
	adm_params = fit_exact_adm_params(train_obs, train_labels, n_components)
	raw_diffs, corrected_diffs = [], []
	for obs, labels in eval_sets:
		raw, corrected = exact_broadband_flux_diffs(obs, labels, adm_params, n_components)
		raw_diffs.append(raw)
		corrected_diffs.append(corrected)
	raw_diffs = np.concatenate(raw_diffs) if raw_diffs else np.array([])
	corrected_diffs = np.concatenate(corrected_diffs) if corrected_diffs else np.array([])
	exact_adm_rmse, exact_worst_channel_rmse, exact_worst_scene_rmse = exact_adm_ratio_metrics(
		eval_sets, adm_params, n_components
	)
	if raw_diffs.size == 0 or corrected_diffs.size == 0:
		return (np.nan, np.nan, np.nan, exact_adm_rmse, exact_worst_channel_rmse, exact_worst_scene_rmse)
	raw_std = float(np.std(raw_diffs))
	corrected_std = float(np.std(corrected_diffs))
	improvement = (raw_std - corrected_std) / raw_std if raw_std > 0 else np.nan
	return (
		raw_std,
		corrected_std,
		improvement,
		exact_adm_rmse,
		exact_worst_channel_rmse,
		exact_worst_scene_rmse,
	)


def evaluate_candidate_worker(n_components, threads_per_worker, args):
	"""Process-pool entry point: caps BLAS threads so concurrent workers don't oversubscribe cores."""
	with threadpool_limits(limits=threads_per_worker):
		return n_components, evaluate_candidate(n_components, *args)


_worker_shared_args = None
_worker_threads_per_worker = 1


def _init_worker(threads_per_worker, shared_args):
	"""Runs once per spawned worker process, so shared_args is only pickled/sent `jobs` times, not once per task."""
	global _worker_shared_args, _worker_threads_per_worker
	_worker_threads_per_worker = threads_per_worker
	_worker_shared_args = shared_args


def _evaluate_candidate_in_worker(n_components):
	return evaluate_candidate_worker(n_components, _worker_threads_per_worker, _worker_shared_args)


def evaluate_candidate(
	n_components, train_features, test_features, train_obs, test_obs,
	validation_sets, use_pca, pca_var, random_state, stability_repeats, report_n_pc_candidate,
):
	scaler, pca, model, transformed_train = fit_candidate(
		train_features, n_components, use_pca, pca_var, random_state,
		report_n_pc=n_components == report_n_pc_candidate,
	)
	transformed_test, test_labels = transform_predict(test_features, scaler, pca, model)
	train_labels = model.predict(transformed_train)
	ari_min, ari_std = cluster_stability_metrics(
		train_features, test_features, test_labels,
		n_components, use_pca, pca_var, random_state, stability_repeats,
	)
	_, train_sparse, _ = occupancy_metrics(train_labels, n_components)
	test_min, test_sparse, test_fraction = occupancy_metrics(test_labels, n_components)

	# ICL is computed on the same (training) set as BIC, per the standard definition.
	train_responsibilities = model.predict_proba(transformed_train)
	train_entropy = -np.sum(train_responsibilities * np.log(np.maximum(train_responsibilities, 1e-12)))
	icl = float(model.bic(transformed_train) + 2 * train_entropy)

	metrics = fit_adm_and_score(train_obs, train_labels, test_obs, test_labels, n_components)
	(
		adm_rmse, flux_improvement, raw_flux_std, corrected_flux_std,
		worst_channel_rmse, worst_scene_rmse, angular_coverage_min,
	) = metrics

	occupancy_js = []
	centroid_shift = []
	validation_labels_by_path = {}
	for validation_features, path in validation_sets:
		transformed_validation, validation_labels = transform_predict(validation_features, scaler, pca, model)
		validation_labels_by_path[path] = validation_labels
		validation_fraction = np.bincount(validation_labels, minlength=n_components) / validation_labels.size
		occupancy_js.append(float(jensenshannon(test_fraction, validation_fraction) ** 2))
		shifts = []
		for scene in range(n_components):
			mask = validation_labels == scene
			if mask.any():
				shifts.append(np.linalg.norm(transformed_validation[mask].mean(axis=0) - model.means_[scene]))
		centroid_shift.append(float(np.mean(shifts)) if shifts else np.nan)

	row = {
		"n_components": n_components,
		"heldout_loglik": float(model.score(transformed_test)),
		"icl": icl,
		"test_min_fraction": test_min,
		"train_sparse_components": train_sparse,
		"test_sparse_components": test_sparse,
		"occupancy_js": float(np.mean(occupancy_js)) if occupancy_js else np.nan,
		"centroid_shift": float(np.mean(centroid_shift)) if centroid_shift else np.nan,
		"adm_ratio_rmse": adm_rmse,
		"uncorrected_flux_std": raw_flux_std,
		"corrected_flux_std": corrected_flux_std,
		"corrected_flux_improvement": flux_improvement,
		"ari_min": ari_min,
		"ari_std": ari_std,
		"worst_channel_adm_rmse": worst_channel_rmse,
		"worst_scene_adm_rmse": worst_scene_rmse,
		"minimum_angular_coverage": angular_coverage_min,
	}
	state = {"train_labels": train_labels, "test_labels": test_labels, "validation_labels": validation_labels_by_path}
	return row, state


def main():
	parser = argparse.ArgumentParser(
		description="Evaluate GMM component counts with statistical, ADM, and exact broadband-flux diagnostics."
	)
	parser.add_argument("-f", "--input-file", nargs="+", required=True)
	parser.add_argument("--validation-inputs", nargs="*", default=[])
	parser.add_argument("--use-pca", action="store_true")
	parser.add_argument("--pca-var", type=float, default=0.98)
	parser.add_argument("--min-components", type=int, default=2)
	parser.add_argument("--max-components", type=int, default=15)
	parser.add_argument("--max-points", type=int, default=250000)
	parser.add_argument("--stability-repeats", type=int, default=3)
	parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes, one per n_components candidate")
	parser.add_argument("--load-jobs", type=int, default=1, help="Parallel worker processes for loading/featurizing input+validation files")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output-csv", default="figures/diagnostics/gmm/production/gmm_component_diagnostics.csv")
	parser.add_argument("--selection-output", default="data/models/selected_n_components.json",
		help="JSON file written with the selected n_components, consumable by train_GMM.py --n-components-file")
	parser.add_argument("--exact-shortlist-size", type=int, default=5,
		help="Number of candidates evaluated with the exact ADM/broadband workflow")
	parser.add_argument("--adm-plateau-tolerance", type=float, default=0.01, help="Maximum relative ADM RMSE improvement considered a plateau")
	parser.add_argument("--min-scene-fraction", type=float, default=0.005, help="Minimum held-out fraction for every scene")
	parser.add_argument("--max-occupancy-js", type=float, default=0.10, help="Maximum Jensen-Shannon divergence of scene occupancy")
	parser.add_argument("--max-centroid-shift", type=float, default=1.0, help="Maximum mean standardized feature-centroid shift")
	parser.add_argument("--min-ari", type=float, default=0.80, help="Minimum repeated-fit adjusted Rand index")
	parser.add_argument("--min-angular-coverage", type=float, default=0.70, help="Minimum fraction of 10-degree viewing-angle bins occupied by every scene")
	parser.add_argument("--max-worst-scene-adm-rmse", type=float, default=np.inf, help="Maximum worst-scene held-out ADM ratio RMSE")
	parser.add_argument("--min-exact-flux-improvement", type=float, default=0.0, help="Minimum fractional reduction in exact broadband G16-G18 flux spread")
	args = parser.parse_args()

	if args.min_components < 2 or args.max_components < args.min_components:
		raise ValueError("component range must satisfy 2 <= min-components <= max-components")
	rng = np.random.default_rng(args.seed)

	# Build one reproducible train/test pool across all training days.
	all_paths = list(args.input_file) + list(args.validation_inputs)
	# Independent per-file seeds (not one shared advancing RNG) so results don't depend on
	# load order/concurrency -- required for --load-jobs to be reproducible.
	file_seeds = np.random.SeedSequence(args.seed).spawn(len(all_paths))
	points_per_file = max(2, args.max_points // len(args.input_file))
	# Cap by cpu_count and by available RAM, so a large --load-jobs can't OOM-kill the pool
	# (each worker peaks at roughly LOAD_WORKER_MEMORY_BUDGET_BYTES while chunking a file).
	load_jobs = max(1, min(args.load_jobs, len(all_paths), os.cpu_count() or len(all_paths)))
	available_memory = _available_memory_bytes()
	if available_memory is not None:
		memory_capped_jobs = max(1, available_memory // LOAD_WORKER_MEMORY_BUDGET_BYTES)
		if memory_capped_jobs < load_jobs:
			print(
				f"Capping --load-jobs from {load_jobs} to {memory_capped_jobs} "
				f"(~{available_memory / 1024**3:.0f} GiB available / "
				f"~{LOAD_WORKER_MEMORY_BUDGET_BYTES / 1024**3:.0f} GiB per worker)"
			)
			load_jobs = memory_capped_jobs

	if load_jobs == 1:
		input_results = [
			_load_input_file(path, points_per_file, file_seeds[index])
			for index, path in enumerate(args.input_file)
		]
		validation_results = [
			_load_validation_file(path, args.max_points, file_seeds[len(args.input_file) + index])
			for index, path in enumerate(args.validation_inputs)
		]
	else:
		print(f"Loading {len(all_paths)} files across {load_jobs} worker processes")
		# spawn (not fork) avoids deadlocks from inheriting the parent's BLAS/OpenMP thread pools.
		mp_context = multiprocessing.get_context("spawn")
		with ProcessPoolExecutor(max_workers=load_jobs, mp_context=mp_context) as executor:
			input_futures = [
				executor.submit(_load_input_file, path, points_per_file, file_seeds[index])
				for index, path in enumerate(args.input_file)
			]
			validation_futures = [
				executor.submit(_load_validation_file, path, args.max_points, file_seeds[len(args.input_file) + index])
				for index, path in enumerate(args.validation_inputs)
			]
			input_results = [future.result() for future in input_futures]
			validation_results = [future.result() for future in validation_futures]

	features_parts, observation_parts = zip(*input_results)
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

	validation_sets = [(validation_features, path) for validation_features, path, _ in validation_results]
	validation_obs_by_path = {path: obs for _, path, obs in validation_results}

	candidates = list(range(args.min_components, args.max_components + 1))
	# Cap by cpu_count too, so a future wider --min/--max-components range can't oversubscribe cores.
	jobs = max(1, min(args.jobs, len(candidates), os.cpu_count() or len(candidates)))
	shared_args = (
		train_features, test_features, train_obs, test_obs,
		validation_sets, args.use_pca, args.pca_var, args.seed, args.stability_repeats,
		candidates[0],
	)
	# Full-covariance GMM fits already use BLAS threading, so cap threads per worker to avoid oversubscribing cores.
	threads_per_worker = max(1, (os.cpu_count() or 1) // jobs)
	results = {}
	if jobs == 1:
		for n_components in candidates:
			print(f"Evaluating {n_components} components")
			_, result = evaluate_candidate_worker(n_components, threads_per_worker, shared_args)
			results[n_components] = result
	else:
		print(f"Evaluating {len(candidates)} component counts across {jobs} worker processes")
		# spawn (not fork) avoids deadlocks from inheriting the parent's BLAS/OpenMP thread pools.
		mp_context = multiprocessing.get_context("spawn")
		with ProcessPoolExecutor(
			max_workers=jobs, mp_context=mp_context,
			initializer=_init_worker, initargs=(threads_per_worker, shared_args),
		) as executor:
			futures = [
				executor.submit(_evaluate_candidate_in_worker, n_components)
				for n_components in candidates
			]
			for future in futures:
				n_components, result = future.result()
				print(f"Finished {n_components} components")
				results[n_components] = result

	rows, states = [], {}
	for n_components in candidates:
		row, state = results[n_components]
		rows.append(row)
		states[n_components] = state

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
			bool(args.validation_inputs)
			and np.isfinite(row["occupancy_js"]) and row["occupancy_js"] <= args.max_occupancy_js
			and np.isfinite(row["centroid_shift"]) and row["centroid_shift"] <= args.max_centroid_shift
			and np.isfinite(row["ari_min"]) and row["ari_min"] >= args.min_ari
		)
		row["criterion_angular_coverage"] = bool(
			np.isfinite(row["minimum_angular_coverage"])
			and row["minimum_angular_coverage"] >= args.min_angular_coverage
		)
		row["criterion_adm_coverage"] = bool(
			np.isfinite(row["worst_scene_adm_rmse"])
			and row["worst_scene_adm_rmse"] <= args.max_worst_scene_adm_rmse
		)
		if np.isfinite(row["adm_ratio_rmse"]):
			previous_rmse = row["adm_ratio_rmse"]

	# Shortlist candidates for the exact ADM/broadband stage using only the cheap criteria.
	cheap_pass = [
		row for row in rows
		if row["criterion_adm_plateau"] and row["criterion_occupancy"]
		and row["criterion_stability"] and row["criterion_angular_coverage"]
		and row["criterion_adm_coverage"]
	]
	if cheap_pass:
		shortlist_rows = sorted(cheap_pass, key=lambda r: r["n_components"])[: args.exact_shortlist_size]
	else:
		print("No candidate passed the cheap diagnostics; falling back to lowest worst-scene ADM RMSE for exact evaluation.")
		ranked = sorted(rows, key=lambda r: r["worst_scene_adm_rmse"] if np.isfinite(r["worst_scene_adm_rmse"]) else np.inf)
		shortlist_rows = ranked[: args.exact_shortlist_size]
	shortlist_components = {row["n_components"] for row in shortlist_rows}

	for row in rows:
		n_components = row["n_components"]
		row["shortlisted"] = n_components in shortlist_components
		if not row["shortlisted"]:
			row["exact_raw_flux_std"] = np.nan
			row["exact_corrected_flux_std"] = np.nan
			row["exact_flux_improvement"] = np.nan
			row["exact_adm_ratio_rmse"] = np.nan
			row["exact_worst_channel_adm_rmse"] = np.nan
			row["exact_worst_scene_adm_rmse"] = np.nan
			row["criterion_exact_improvement"] = False
			row["criterion_exact_adm_coverage"] = False
			continue

		print(f"Running exact ADM/broadband evaluation for {n_components} components")
		state = states[n_components]
		eval_sets = [(test_obs, state["test_labels"])]
		for _, path in validation_sets:
			eval_sets.append((validation_obs_by_path[path], state["validation_labels"][path]))

		(
			raw_std,
			corrected_std,
			improvement,
			exact_adm_rmse,
			exact_worst_channel_rmse,
			exact_worst_scene_rmse,
		) = evaluate_exact_candidate(
			n_components, train_obs, state["train_labels"], eval_sets
		)
		row["exact_raw_flux_std"] = raw_std
		row["exact_corrected_flux_std"] = corrected_std
		row["exact_flux_improvement"] = improvement
		row["exact_adm_ratio_rmse"] = exact_adm_rmse
		row["exact_worst_channel_adm_rmse"] = exact_worst_channel_rmse
		row["exact_worst_scene_adm_rmse"] = exact_worst_scene_rmse
		row["criterion_exact_improvement"] = bool(
			np.isfinite(improvement) and improvement >= args.min_exact_flux_improvement
		)
		row["criterion_exact_adm_coverage"] = bool(
			np.isfinite(exact_worst_scene_rmse)
			and exact_worst_scene_rmse <= args.max_worst_scene_adm_rmse
		)

	for row in rows:
		row["all_criteria"] = bool(
			row["shortlisted"]
			and row["criterion_adm_plateau"]
			and row["criterion_occupancy"]
			and row["criterion_stability"]
			and row["criterion_angular_coverage"]
			and row["criterion_adm_coverage"]
			and row["criterion_exact_adm_coverage"]
			and row["criterion_exact_improvement"]
		)

	output_path = Path(args.output_csv)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
		writer.writeheader()
		writer.writerows(rows)

	# Automate the selection output consumed by train_GMM.py --n-components-file.
	passing = sorted(row["n_components"] for row in rows if row["all_criteria"])
	selection = {
		"n_components": passing[0] if passing else None,
		"source": "find_nComponents.py",
		"candidates_passing_all_criteria": passing,
		"shortlist_evaluated": sorted(shortlist_components),
		"thresholds": {
			"adm_plateau_tolerance": args.adm_plateau_tolerance,
			"min_scene_fraction": args.min_scene_fraction,
			"max_occupancy_js": args.max_occupancy_js,
			"max_centroid_shift": args.max_centroid_shift,
			"min_ari": args.min_ari,
			"min_angular_coverage": args.min_angular_coverage,
			"max_worst_scene_adm_rmse": args.max_worst_scene_adm_rmse,
			"min_exact_flux_improvement": args.min_exact_flux_improvement,
		},
	}
	selection_path = Path(args.selection_output)
	selection_path.parent.mkdir(parents=True, exist_ok=True)
	with selection_path.open("w") as handle:
		json.dump(selection, handle, indent=2)
	if passing:
		print(f"Selected n_components={passing[0]}; written to {selection_path}")
	else:
		print(f"No candidate passed all criteria; wrote null selection to {selection_path}")

	print(f"Diagnostics written to {output_path}")
	print("Interpretation: lower ADM RMSE, higher ARI, higher angular coverage, and higher exact flux improvement are preferred.")


if __name__ == "__main__":
	main()
