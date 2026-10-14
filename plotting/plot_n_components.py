#!/usr/bin/env python3
"""Plot component-count diagnostics from a find_nComponents CSV output."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _number(row, name):
	value = row[name]
	return float(value) if value not in (None, "") else np.nan


def _is_true(row, name):
	return row[name].strip().lower() in {"true", "1", "yes"}


def _plot_full_diagnostics(rows, details, plot_path):
	"""Four categories with aggregate curves and individual bootstrap agreements."""
	counts = np.array([int(row["n_components"]) for row in rows])
	if len(set(counts)) != len(counts):
		raise ValueError("Duplicate component counts in diagnostic table")
	if set(details) != set(map(str, counts)):
		raise ValueError("Detailed diagnostics must cover every plotted component count")
	for row in rows:
		for name in (
			"full_heldout_loglik", "full_training_icl", "ari_min",
			"full_minimum_occupancy", "minimum_day_occupancy",
		):
			if not np.isfinite(_number(row, name)):
				raise ValueError(f"K={row['n_components']}: non-finite {name}")
	days = sorted(details[str(counts[0])]["by_day"], key=int)
	if not days or any(set(details[str(k)]["by_day"]) != set(days) for k in counts):
		raise ValueError("Component diagnostics must use the same nonempty held-out days")
	for k in counts:
		record = details[str(k)]
		values = [
			*[record["by_day"][day][metric] for day in days for metric in ("sd", "bias", "rmse")],
			*[repeat[metric] for repeat in record["bootstrap"]
			  for metric in ("ari", "minimum_scene_jaccard")],
		]
		if not record["bootstrap"] or not np.isfinite(values).all():
			raise ValueError(f"K={k}: invalid detailed diagnostic values")

	fig = plt.figure(figsize=(17, 13), layout="constrained")
	grid = fig.add_gridspec(2, 2)
	fit = fig.add_subplot(grid[0, 0])
	robust = fig.add_subplot(grid[0, 1])
	physical_grid = grid[1, 0].subgridspec(2, 1)
	adm = fig.add_subplot(physical_grid[0])
	flux = fig.add_subplot(physical_grid[1], sharex=adm)
	occupancy = fig.add_subplot(grid[1, 1])
	main_axes = (fit, robust, adm, flux, occupancy)
	for axis in main_axes:
		axis.set_xticks(counts)
		axis.grid(True, alpha=.25)
		axis.set_xlabel("Number of GMM components")

	likelihood_line, = fit.plot(
		counts, [_number(row, "full_heldout_loglik") for row in rows],
		color="tab:blue", marker="o", label="Full held-out mean log likelihood",
	)
	icl_axis = fit.twinx()
	icl_line, = icl_axis.plot(
		counts, [_number(row, "full_training_icl") for row in rows],
		color="tab:orange", marker="s", label="Full-training ICL",
	)
	fit.set_title("Statistical fit (all eligible ABI pixels)")
	fit.set_ylabel("Mean log likelihood", color="tab:blue")
	icl_axis.set_ylabel("ICL (lower is better)", color="tab:orange")
	fit.legend(handles=[likelihood_line, icl_line], fontsize=8, loc="best")

	for index, k in enumerate(counts):
		repeats = details[str(k)]["bootstrap"]
		if not repeats:
			raise ValueError(f"K={k}: no bootstrap repeat details")
		offsets = np.linspace(-.12, .12, len(repeats))
		robust.scatter(k + offsets, [repeat["ari"] for repeat in repeats],
			color="tab:purple", marker="o", label="Bootstrap ARI" if index == 0 else None)
		robust.scatter(k + offsets, [repeat["minimum_scene_jaccard"] for repeat in repeats],
			color="tab:green", marker="x", label="Worst matched-scene Jaccard" if index == 0 else None)
	robust.plot(counts, [_number(row, "ari_min") for row in rows],
		color="tab:purple", linewidth=1, alpha=.5, label="Minimum bootstrap ARI")
	robust.set_title("Partition robustness (sample bootstrap refits)")
	robust.set_ylabel("Assignment agreement")
	robust.set_ylim(-.05, 1.05)
	robust.legend(fontsize=8, loc="lower left")

	for name, label, color, marker in (
		("exact_adm_ratio_rmse", "Global", "tab:red", "D"),
		("exact_worst_channel_adm_rmse", "Worst channel", "tab:purple", "P"),
		("exact_worst_scene_adm_rmse", "Worst scene", "tab:brown", "X"),
	):
		adm.plot(counts, [_number(row, name) for row in rows],
			color=color, marker=marker, label=label)
	adm.set_title("Physical retrieval (ABI; Sunny-calibrated broadband conversion)")
	adm.set_ylabel("Exact ADM ratio RMSE")
	adm.legend(fontsize=8, ncol=3)
	for metric, label, color in (
		("sd", "SD", "tab:purple"),
		("bias", "Signed bias", "tab:blue"),
		("rmse", "RMSE", "tab:red"),
	):
		baseline = details[str(counts[0])]["by_day"]
		changes = np.array([
			[details[str(k)]["by_day"][day][metric] - baseline[day][metric] for day in days]
			for k in counts
		])
		flux.plot(counts, changes.mean(axis=1), color=color,
			marker="o", label=f"{label}: equal-day mean")
	flux.axhline(0, color="black", linewidth=.8)
	flux.set_ylabel(f"Flux difference metric change\nvs K={counts[0]} (W/m^2)")
	flux.legend(fontsize=7)

	occupancy.plot(counts, [100*_number(row, "full_minimum_occupancy") for row in rows],
		marker="o", label="Minimum pooled scene fraction")
	occupancy.plot(counts, [100*_number(row, "minimum_day_occupancy") for row in rows],
		marker="s", label="Minimum scene fraction over days")
	occupancy.axhline(.5, color="black", linestyle="--", label="Existing 0.5% occupancy criterion")
	occupancy.set_title("Scene occupancy")
	occupancy.set_ylabel("Scene occupancy (%)")
	occupancy.legend(fontsize=8)
	fig.suptitle(
		"GMM component selection: fit, robustness, retrieval and support\n"
		"ICL uses nominal pixel count (spatial dependence uncorrected)",
		fontsize=14,
	)
	fig.savefig(plot_path, dpi=150)
	plt.close(fig)


def _plot_criteria(rows, counts, plot_path):
	criterion_names = (
		("shortlisted", "Shortlisted"),
		("criterion_adm_plateau", "ADM plateau"),
		("criterion_occupancy", "Occupancy"),
		("criterion_stability", "Stability"),
		("criterion_angular_coverage", "Angular coverage"),
		("criterion_adm_coverage", "ADM coverage"),
		("criterion_exact_adm_coverage", "Exact ADM coverage"),
		("criterion_exact_improvement", "Exact flux gain"),
	)
	criteria = np.asarray(
		[[_is_true(row, name) for name, _ in criterion_names] for row in rows], dtype=int,
	)
	first_passing = next((index for index, row in enumerate(rows) if _is_true(row, "all_criteria")), None)
	heatmap_path = plot_path.with_name(plot_path.stem + "_criteria.png")
	fig, axis = plt.subplots(figsize=(12, max(4, len(rows) * 0.35)))
	axis.imshow(criteria, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
	axis.set_xticks(range(len(criterion_names)), [label for _, label in criterion_names], rotation=30, ha="right")
	axis.set_yticks(range(len(counts)), counts)
	axis.set_xlabel("Selection criterion")
	axis.set_ylabel("Number of GMM components")
	axis.set_title("Component-selection criteria: green = pass, red = fail")
	if first_passing is not None:
		axis.axhline(first_passing, color="black", linewidth=2, linestyle="--")
		axis.text(len(criterion_names) - 0.5, first_passing, " first all-criteria pass", va="bottom", ha="right")
	fig.tight_layout()
	fig.savefig(heatmap_path, dpi=150)
	plt.close(fig)
	return heatmap_path


def plot_n_components(
    input_csv,
    output_plot=None,
    reliability_label="Bootstrap assignment stability",
):
	"""Plot diagnostics, showing only exact ADM curves when every K is evaluated."""
	input_csv = Path(input_csv)
	with input_csv.open(newline="") as handle:
		rows = list(csv.DictReader(handle))
	if not rows:
		raise ValueError(f"No diagnostic rows found in {input_csv}")
	rows.sort(key=lambda row: int(row["n_components"]))

	plot_path = Path(output_plot) if output_plot else input_csv.with_suffix(".png")
	plot_path.parent.mkdir(parents=True, exist_ok=True)
	counts = [int(row["n_components"]) for row in rows]
	shortlisted = [_is_true(row, "shortlisted") for row in rows]
	exact_counts = [count for count, is_shortlisted in zip(counts, shortlisted) if is_shortlisted]
	exact_metrics = (
		("exact_adm_ratio_rmse", "Exact global ADM ratio RMSE", "D", "tab:red"),
		("exact_worst_channel_adm_rmse", "Exact worst-channel ADM RMSE", "P", "tab:purple"),
		("exact_worst_scene_adm_rmse", "Exact worst-scene ADM RMSE", "X", "tab:brown"),
	)
	exact_for_all = all(shortlisted) and all(
		np.isfinite(_number(row, name)) for row in rows for name, _, _, _ in exact_metrics
	)
	metadata_path = Path(str(input_csv) + ".json")
	if metadata_path.exists():
		metadata = json.loads(metadata_path.read_text())
		if metadata.get("diagnostic_schema_version") == 2:
			if not exact_for_all:
				raise ValueError("Full diagnostic overview requires exact ADM evaluation for every K")
			_plot_full_diagnostics(rows, metadata["component_details"], plot_path)
			return plot_path, _plot_criteria(rows, counts, plot_path)

	fig, axes = plt.subplots(2, 2, figsize=(14, 12))

	axis = axes[0][0]
	loglik = [_number(row, "heldout_loglik") for row in rows]
	icl = [_number(row, "icl") for row in rows]
	line1, = axis.plot(counts, loglik, marker="o", color="tab:blue", label="Held-out log likelihood")
	axis.set_ylabel("Held-out log likelihood", color="tab:blue")
	axis.tick_params(axis="y", labelcolor="tab:blue")
	twin = axis.twinx()
	line2, = twin.plot(counts, icl, marker="s", color="tab:orange", label="ICL")
	twin.set_ylabel("ICL", color="tab:orange")
	twin.tick_params(axis="y", labelcolor="tab:orange")
	axis.set_title("Likelihood")
	axis.grid(True, alpha=0.3)
	axis.legend(handles=[line1, line2], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize="x-small")

	axis = axes[0][1]
	ari_min = [_number(row, "ari_min") for row in rows]
	ari_std = [_number(row, "ari_std") for row in rows]
	axis.errorbar(counts, ari_min, yerr=ari_std, marker="o", capsize=4, label="ARI minimum +/- 1 std")
	axis.set_title(reliability_label)
	axis.set_ylabel("Adjusted Rand index")
	axis.grid(True, alpha=0.3)
	axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize="x-small")

	axis = axes[1][0]
	if not exact_for_all:
		for name, label, marker in (
			("adm_ratio_rmse", "Global ADM ratio RMSE", "o"),
			("worst_channel_adm_rmse", "Worst-channel ADM RMSE", "s"),
			("worst_scene_adm_rmse", "Worst-scene ADM RMSE", "^"),
		):
			values = [_number(row, name) for row in rows]
			if np.isfinite(values).any():
				axis.plot(counts, values, marker=marker, linewidth=1.5, label=label)
	for name, label, marker, color in exact_metrics:
		values = [_number(row, name) for row, is_shortlisted in zip(rows, shortlisted) if is_shortlisted]
		if exact_for_all:
			axis.plot(counts, values, marker=marker, color=color, linewidth=1.5, label=label)
		else:
			axis.scatter(exact_counts, values, marker=marker, color=color, s=60, label=label, zorder=3)
	axis.set_title("Physical ADM quality")
	axis.set_ylabel("Radiance-ratio RMSE")
	axis.set_xlabel("Number of GMM components")
	axis.grid(True, alpha=0.3)
	axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize="x-small")

	axis = axes[1][1]
	coverage = [_number(row, "minimum_angular_coverage") for row in rows]
	line1, = axis.plot(counts, coverage, marker="o", color="tab:green", label="Minimum angular coverage")
	exact_improvement = [
		_number(row, "exact_flux_improvement")
		for row, is_shortlisted in zip(rows, shortlisted) if is_shortlisted
	]
	line2 = axis.scatter(exact_counts, exact_improvement, marker="D", color="tab:red", label="Exact flux improvement", zorder=3)
	axis.set_ylabel("Fraction (angular coverage / flux improvement)")
	axis.set_ylim(-0.05, 1.05)
	twin = axis.twinx()
	exact_spread = [
		_number(row, "exact_corrected_flux_std")
		for row, is_shortlisted in zip(rows, shortlisted) if is_shortlisted
	]
	line3 = twin.scatter(exact_counts, exact_spread, marker="P", color="tab:purple", label="Exact corrected G16-G18 flux spread", zorder=3)
	twin.set_ylabel("Exact corrected flux spread (W/m^2)", color="tab:purple")
	twin.tick_params(axis="y", labelcolor="tab:purple")
	axis.set_title(
		"Physical coverage and production impact (exact, all component counts)"
		if exact_for_all else "Physical coverage and production impact (exact, shortlisted only)"
	)
	axis.set_xlabel("Number of GMM components")
	axis.grid(True, alpha=0.3)
	axis.legend(handles=[line1, line2, line3], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize="x-small")

	fig.suptitle("GMM component diagnostics (native units)", fontsize=15)
	fig.subplots_adjust(left=0.09, right=0.94, top=0.91, bottom=0.18, hspace=0.85, wspace=0.35)
	fig.savefig(plot_path, dpi=150)
	plt.close(fig)

	return plot_path, _plot_criteria(rows, counts, plot_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input-csv", type=Path, required=True, help="CSV written by find_nComponents.py")
	parser.add_argument("--output-plot", type=Path, help="Main plot path (default: input CSV with .png suffix)")
	args = parser.parse_args()
	plot_path, heatmap_path = plot_n_components(args.input_csv, args.output_plot)
	print(f"Plot written to {plot_path}")
	print(f"Criteria heatmap written to {heatmap_path}")