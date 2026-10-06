"""Figures for equal-weight ECO geometry and nested assignment stress."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uncertainty import check_provenance


def plot_reports(geometry, assignment, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    names = list(geometry["settings"]["geometry_sets"])
    for form, color in (("regularized", "#0072B2"), ("quadratic", "#D55E00")):
        for noisy, marker in ((False, "o"), (True, "s")):
            x, values, low, high = [], [], [], []
            for index, name in enumerate(names):
                rows = [row for row in geometry["rows"] if row["geometry"] == name
                        and row["form"] == form and row["status"] == "quantified"
                        and (row["noise_seed"] is not None) == noisy]
                if not rows:
                    continue
                metrics = [row["truth_error_w_m2"]["rmse"] for row in rows]
                x.append(index)
                values.append(np.mean(metrics))
                low.append(min(metrics))
                high.append(max(metrics))
            axes[0].plot(x, values, marker=marker, color=color,
                         linestyle="--" if noisy else "-",
                         label=f"{form}: {'noisy' if noisy else 'noise-free'}")
            axes[0].vlines(x, low, high, color=color, alpha=.5)
        rows = [row for row in geometry["rows"] if row["form"] == form
                and row["status"] == "quantified" and row["noise_seed"] is None]
        axes[1].plot([names.index(row["geometry"]) for row in rows],
                     [row["diagnostics"]["normalized_design_condition"] for row in rows],
                     "o-", color=color, label=form)
        axes[2].plot([names.index(row["geometry"]) for row in rows],
                     [row["diagnostics"]["unconstrained_flux_noise_gain"] for row in rows],
                     "o-", color=color, label=form)
    axes[0].axhline(geometry["spectral_only_truth_error_w_m2"]["rmse"],
                    color="grey", linestyle=":", label="Spectral-only reference")
    axes[0].set_ylabel("Broadband error vs Sunny truth: RMSE (W m$^{-2}$)")
    axes[1].set_ylabel("Column-normalized amplitude/shape design condition")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("Unconstrained band-flux noise gain\nper unit independent radiance SD")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Idealized ECO geometry: direct amplitude/shape fit, equal weights\n"
        "Noise bars: range of five realization RMSEs, not confidence intervals; quadratic 2-view excluded"
    )
    figure.savefig(directory / "eco_geometry.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for seed in assignment["settings"]["assignment_seeds"]:
        rows = sorted([row for row in assignment["rows"] if row["seed"] == seed],
                      key=lambda row: row["fraction"])
        x = [100*row["fraction"] for row in rows]
        for axis, metric in zip(axes, ("rmse", "bias")):
            axis.plot(x, [row["w_m2"]["estimate"][metric] for row in rows],
                      "o-", label=f"Seed {seed}")
            bounds = np.array([row["w_m2"]["intervals"][metric] for row in rows])
            axis.vlines(x, bounds[:, 0], bounds[:, 1], alpha=.25)
            axis.set_ylabel(f"Paired broadband {metric.upper()} (W m$^{{-2}}$)")
            axis.set_xlabel("Native pixels assigned second GMM choice (%)")
            axis.set_xticks(x)
            axis.grid(alpha=.2)
            axis.legend()
    figure.suptitle(
        "ABI assignment stress response: fixed models and held-out footprint population\n"
        "Nested switch sets; exploratory 95% day/tile intervals; not misclassification probabilities"
    )
    figure.savefig(directory / "assignment_curve.png", dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--assignment", required=True)
    args = parser.parse_args()
    reports = []
    for path in (args.geometry, args.assignment):
        with Path(path).open() as source:
            report = json.load(source)
        freshness = check_provenance(report["provenance"], ROOT)
        if freshness["status"] != "current":
            raise ValueError(f"Stale extension report: {freshness}")
        reports.append(report)
    from figure_layout import figure_path
    plot_reports(*reports, figure_path(reports[0]["settings"]["figure_dir"]))


if __name__ == "__main__":
    main()
