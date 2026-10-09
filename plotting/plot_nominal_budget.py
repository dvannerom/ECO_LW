"""Canonical source/error-class table; supporting diagnostics in separate roots."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"plotting"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from uncertainty import check_provenance
from plot_n_components import plot_n_components


def plot_gmm_diagnostics(selection, diagnostics_csv, settings):
    """Combine per-seed score variation and bootstrap stability by component count."""
    with Path(diagnostics_csv).open(newline="") as handle:
        diagnostic_rows = {
            int(row["n_components"]): row for row in csv.DictReader(handle)
        }
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    rows = selection["candidates"]
    for seed in sorted({row["seed"] for row in rows}):
        candidates = sorted((row for row in rows if row["seed"] == seed),
                            key=lambda row: row["components"])
        counts = [row["components"] for row in candidates]
        errors = [np.std(row["heldout_day_log_likelihood"], ddof=1)/
                  np.sqrt(len(row["heldout_day_log_likelihood"])) for row in candidates]
        axes[0].errorbar(counts, [row["heldout_mean_log_likelihood"] for row in candidates],
                         yerr=errors, marker="o", label=f"Seed {seed}")
        axes[1].plot(counts, [100*row["minimum_occupancy"] for row in candidates],
                     marker="o", label=f"Seed {seed}")
    for axis in axes:
        axis.axvline(selection["components"], color="black", linestyle="--",
                     label=f"Manual nominal: {selection['components']}")
        axis.set_xlabel("Number of GMM components")
        axis.legend()
    axes[0].set_ylabel("Held-out mean log likelihood (day-based SE)")
    axes[1].set_ylabel("Minimum held-out component population (%)")
    counts = sorted(diagnostic_rows)
    ari_min = [float(diagnostic_rows[count]["ari_min"]) for count in counts]
    ari_std = [float(diagnostic_rows[count]["ari_std"]) for count in counts]
    axes[2].errorbar(counts, ari_min, yerr=ari_std, marker="o", capsize=4,
                     color="tab:purple")
    axes[2].axvline(selection["components"], color="black", linestyle="--",
                    label=f"Manual nominal: {selection['components']}")
    axes[2].set_xlabel("Number of GMM components")
    axes[2].set_ylabel("Minimum bootstrap ARI (error bars: repeat SD)")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()
    figure.suptitle(
        "Spectral-only ABI classifier: seed variability and bootstrap stability"
    )
    figure.savefig(Path(settings["diagnostic_figure_dir"])/"gmm_stability.png", dpi=160)
    plt.close(figure)


def plot(report):
    settings = report["settings"]
    output = Path(settings["figure_dir"])
    diagnostics = Path(settings["diagnostic_figure_dir"])
    output.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    names = {
        "spectral": "N2BC reconstruction (within-library)",
        "angular": "Scene-ADM approximation / within-scene diversity",
        "radiometric": "Radiometric noise (scene label held fixed)",
        "assignment": "Noise-mediated scene assignment (ABI-to-ECO assumption)",
        "total": "Conditional joint subtotal (not full ECO uncertainty)",
        "spatial": "Spatial processing: ABI 2-to-10 km proxy",
        "combined": "Combined budget (spatial independent; not full ECO uncertainty)",
    }
    records = []
    for row in report["rows"]:
        records.append([
            names[row["id"]], f"{row['bias_w_m2']:+.4f}", f"{row['scene_sd_w_m2']:.4f}",
            f"{row['random_sd_w_m2']:.4f}", f"{row['ensemble_rmse_w_m2']:.4f}",
        ])
        if row["negative_corrected_scene_variance"]:
            records[-1][2] = "Below MC resolution"
    proxy = report["assignment_transfer"]
    records.extend([
        ["Training / library estimation variability", "Unquantified", "Unquantified", "Unquantified", "--"],
        ["ECO spatial PSF / spatial-noise interactions", "Unquantified", "Unquantified", "Unquantified", "--"],
        ["Calibration, SRF, registration, evolution, library discrepancy",
         "Unquantified", "Unquantified", "Unquantified", "--"],
    ])
    headers = ["Error source", "Bias\nW/m2", "Scene-dependent SD\nW/m2", "Random SD\nW/m2",
               "RMSE\nW/m2"]
    with (Path(settings["output_dir"])/"numerical_budget.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([text.replace("\n", " ") for text in headers])
        writer.writerows(records)
    figure, axis = plt.subplots(figsize=(18, 10))
    axis.axis("off")
    table = axis.table(cellText=records, colLabels=headers, bbox=[0, .15, 1, .75],
                       cellLoc="center", colWidths=[.48, .11, .15, .13, .13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dbe9f4")
        if column == 0:
            cell.set_text_props(ha="left")
    axis.set_title(
        "ABI-informed ECO broadband uncertainty: bias / scene-dependent / random\n"
        "4-100 um; goal channels; 15 views 0-70 deg; original modified-log scene ADM",
        fontsize=15, pad=20,
    )
    figure.text(.03, .04,
        f"ABI population covered by Sunny: {report['support']['covered_population']:.1%}; "
        f"noise-induced ABI pixel-label change: {proxy['weighted_change_probability']:.2%}\n"
        "ABI box-footprint mixing included assuming transfer to ECO; spatial variance independent of Sunny-source errors. "
        "No monthly extrapolation or CERES.\n"
        "Sunny covariance retained; combined RMSE from independent second moments and added signed biases. "
        "Finite-Monte-Carlo-corrected class SDs need not exactly reconstruct ensemble RMSE.\n"
        f"Weighted effective Sunny sample size: {report['support']['effective_sample_size']:.0f}; "
        f"low-ABI-density views: {report['transfer_diagnostics']['weighted_fraction_views_below_ABI_density_p01']:.1%}. "
        "Coverage and population-transfer limitations remain.",
        fontsize=10)
    figure.savefig(output/"eco_uncertainty_numerical_budget.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    transition = np.asarray(proxy["transition_probabilities"], dtype=float)
    image = axes[0].imshow(transition, vmin=0, vmax=1)
    figure.colorbar(image, ax=axes[0], label="Conditional transition probability")
    axes[0].set_xlabel("Added-noise ABI pixel scene")
    axes[0].set_ylabel("Baseline ABI pixel scene")
    axes[0].set_title("Held-out ABI pixels; fixed GMM; observed ABI as truth")
    rows = report["rows"][:4]
    axes[1].bar(np.arange(4)-.15, [r["ensemble_rmse_w_m2"] for r in rows], width=.3, label="ABI-weighted")
    axes[1].bar(np.arange(4)+.15, [r["ensemble_rmse_w_m2"] for r in report["unweighted_rows"][:4]],
                width=.3, label="Unweighted")
    axes[1].set_xticks(np.arange(4), ["Spectral", "Angular", "Noise", "Assignment"])
    axes[1].set_ylabel("RMSE (W/m2)")
    axes[1].legend()
    figure.savefig(diagnostics/"population_and_assignment.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    spatial = report["spatial"]
    rows = spatial["rows"]
    for key, label in (("bias_w_m2", "Signed mean"), ("scene_sd_w_m2", "Scene SD"),
                       ("ensemble_rmse_w_m2", "RMSE")):
        axes[0].plot([row["resolution_km"] for row in rows], [row[key] for row in rows],
                     marker="o", label=label)
    axes[0].set_xlabel("ABI box-footprint resolution (km)")
    axes[0].set_xticks([row["resolution_km"] for row in rows])
    axes[0].set_ylabel("Coarse-minus-mean-fine broadband flux (W/m2)")
    axes[0].legend()
    nominal = str(settings["spatial"]["nominal_block_size"])
    days = list(spatial["by_day"])
    axes[1].bar(days, [spatial["by_day"][day][nominal]["ensemble_rmse_w_m2"] for day in days])
    axes[1].set_xlabel("Held-out ABI day")
    axes[1].set_ylabel("Nominal spatial processing RMSE (W/m2)")
    figure.suptitle("Fixed spectral GMM / ABI scene ADMs / ABI N2BC; complete matched footprints")
    figure.savefig(diagnostics/"spatial_processing.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    freshness = check_provenance(report["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale nominal budget: {freshness}")
    plot(report)
    selection = json.loads((Path(report["settings"]["diagnostic_dir"])/"gmm_selection.json").read_text())
    freshness = check_provenance(selection["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale GMM diagnostics: {freshness}")
    diagnostic_csv = Path(report["settings"]["diagnostic_dir"])/"gmm_component_diagnostics.csv"
    metadata = json.loads(Path(str(diagnostic_csv)+".json").read_text())
    freshness = check_provenance(metadata["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale four-panel GMM diagnostics: {freshness}")
    plot_gmm_diagnostics(selection, diagnostic_csv, report["settings"])
    plot_n_components(
        diagnostic_csv,
        Path(report["settings"]["diagnostic_figure_dir"]) / "gmm_diagnostics.png",
    )


if __name__ == "__main__":
    main()
