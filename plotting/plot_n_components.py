#!/usr/bin/env python3
"""Plot component-count diagnostics from a find_nComponents CSV output."""
import argparse
import csv
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


def plot_n_components(
    input_csv,
    output_plot=None,
    reliability_label="Bootstrap assignment stability",
):
	"""Create diagnostic plots from a CSV written by find_nComponents.py."""
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
	for name, label, marker in (
		("adm_ratio_rmse", "Global ADM ratio RMSE", "o"),
		("worst_channel_adm_rmse", "Worst-channel ADM RMSE", "s"),
		("worst_scene_adm_rmse", "Worst-scene ADM RMSE", "^"),
	):
		values = [_number(row, name) for row in rows]
		if np.isfinite(values).any():
			axis.plot(counts, values, marker=marker, linewidth=1.5, label=label)
	for name, label, marker, color in (
		("exact_adm_ratio_rmse", "Exact global ADM ratio RMSE", "D", "tab:red"),
		("exact_worst_channel_adm_rmse", "Exact worst-channel ADM RMSE", "P", "tab:purple"),
		("exact_worst_scene_adm_rmse", "Exact worst-scene ADM RMSE", "X", "tab:brown"),
	):
		values = [_number(row, name) for row, is_shortlisted in zip(rows, shortlisted) if is_shortlisted]
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
	axis.set_title("Physical coverage and production impact (exact, shortlisted only)")
	axis.set_xlabel("Number of GMM components")
	axis.grid(True, alpha=0.3)
	axis.legend(handles=[line1, line2, line3], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize="x-small")

	fig.suptitle("GMM component diagnostics (native units)", fontsize=15)
	fig.subplots_adjust(left=0.09, right=0.94, top=0.91, bottom=0.18, hspace=0.85, wspace=0.35)
	fig.savefig(plot_path, dpi=150)
	plt.close(fig)

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
		[[_is_true(row, name) for name, _ in criterion_names] for row in rows],
		dtype=int,
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
	return plot_path, heatmap_path


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input-csv", type=Path, required=True, help="CSV written by find_nComponents.py")
	parser.add_argument("--output-plot", type=Path, help="Main plot path (default: input CSV with .png suffix)")
	args = parser.parse_args()
	plot_path, heatmap_path = plot_n_components(args.input_csv, args.output_plot)
	print(f"Plot written to {plot_path}")
	print(f"Criteria heatmap written to {heatmap_path}")