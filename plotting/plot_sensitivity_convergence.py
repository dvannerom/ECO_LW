"""Plot independently controlled ABI convergence and regime diagnostics."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from sensitivity import angular_basis
from uncertainty import check_provenance

VARIANTS = {
    "spatial": ("Spatial processing order", "#0072B2"),
    "adm": ("ADM model form", "#D55E00"),
    "components": ("GMM 6 versus 7", "#009E73"),
    "assignment": ("10% second-choice stress", "#CC79A7"),
}


def plot_all(report, model, output_dir):
    """Write three figures; do not replace the original numerical-budget figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings, runs = report["settings"], report["runs"]
    max_tiles = max(settings["tile_counts"])
    point_count = settings["fixed_gmm_points"]
    initialization = settings["initialization_seeds"][0]
    sampling_seed = settings["sampling_seeds"][0]
    panels = [
        ("Evaluation tiles/day (fixed model)", "evaluation_tiles", [
            run for run in runs if run["training_tiles"] == max_tiles
            and run["gmm_points"] == point_count
            and run["initialization_seed"] == initialization
        ]),
        ("Training tiles/day (fixed evaluation)", "training_tiles", [
            run for run in runs if run["sampling_seed"] == sampling_seed
            and run["evaluation_tiles"] == max_tiles
            and run["gmm_points"] == point_count
            and run["initialization_seed"] == initialization
        ]),
        ("GMM training records (fixed tiles)", "gmm_points", [
            run for run in runs if run["sampling_seed"] == sampling_seed
            and run["evaluation_tiles"] == max_tiles
            and run["training_tiles"] == max_tiles
            and run["initialization_seed"] == initialization
        ]),
        ("GMM initialization seed (fixed data)", "initialization_seed", [
            run for run in runs if run["sampling_seed"] == sampling_seed
            and run["evaluation_tiles"] == max_tiles
            and run["training_tiles"] == max_tiles
            and run["gmm_points"] == point_count
        ]),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, (title, key, selected) in zip(axes.flat, panels):
        for variant, (label, color) in VARIANTS.items():
            groups = (settings["sampling_seeds"] if key == "evaluation_tiles" else [sampling_seed])
            for group_index, seed in enumerate(groups):
                records = sorted(
                    [run for run in selected if run["sampling_seed"] == seed],
                    key=lambda run: run[key],
                )
                x = [run[key] for run in records]
                y = [run["results"][variant]["w_m2"]["estimate"]["rmse"] for run in records]
                bounds = np.array([
                    run["results"][variant]["w_m2"]["intervals"]["rmse"]
                    for run in records
                ])
                axis.plot(x, y, marker=("o", "s", "^")[group_index], color=color,
                          linestyle="none" if key == "initialization_seed" else "-",
                          alpha=1 if group_index == 0 else .5,
                          label=label if group_index == 0 else None)
                # Percentile bounds need not bracket the finite-sample estimate.
                axis.vlines(x, bounds[:, 0], bounds[:, 1], color=color, alpha=.25)
        axis.set_title(title)
        axis.set_ylabel("Paired broadband RMSE (W m$^{-2}$)")
        axis.set_xticks(sorted({run[key] for run in selected}))
        axis.grid(alpha=.2)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles.extend([
        Line2D([], [], color="grey", marker=marker, linestyle="none",
               label=f"Sampling seed {seed}")
        for marker, seed in zip(("o", "s", "^"), settings["sampling_seeds"])
    ])
    axes[0, 0].legend(handles=handles, fontsize=8, ncol=2)
    figure.suptitle(
        "ABI proxy sensitivity convergence\n"
        "Lines: nested/controlled runs; bars: exploratory 95% day/tile intervals (4 days)",
        fontsize=14,
    )
    figure.savefig(output_dir / "convergence.png", dpi=170)
    plt.close(figure)

    regimes = [
        ("scene", "Baseline modal GMM component"),
        ("vza_g16", "G16 VZA bin"),
        ("vza_g18", "G18 VZA bin"),
        ("angular_separation", "View-angle separation bin"),
        ("heterogeneity_K", "Native C14 BT heterogeneity bin"),
        ("positivity_boundary", "Any baseline positivity-boundary pixel"),
        ("day", "Held-out day"),
        ("scene_purity", "Dominant-scene fraction bin"),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(15, 17), constrained_layout=True)
    for axis, (regime, title) in zip(axes.flat, regimes):
        for variant, (label, color) in VARIANTS.items():
            rows = [row for row in report["regime_diagnostics"][variant][regime]
                    if row["status"] == "quantified"]
            axis.plot([row["label"] for row in rows],
                      [row["pooled"]["rmse"] for row in rows],
                      "o-", color=color, label=label)
        axis.set_title(title)
        axis.set_ylabel("RMSE (W m$^{-2}$)")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle("Fixed-baseline ABI regime diagnostics (largest evaluation pool)", fontsize=15)
    figure.supxlabel(
        "VZA bins: 0/20/40/50/60/70 deg; separation: 0/5/10/20/30/70 deg\n"
        "BT SD bins: 0/1/3/5/10/inf K; scene purity: 0/0.5/0.75/0.9/1\n"
        "Sparse regimes omitted; scene IDs are model-local, not validated scene truth.",
        fontsize=10,
    )
    figure.savefig(output_dir / "regime_diagnostics.png", dpi=170)
    plt.close(figure)

    libraries = model["libraries"]
    components = len(libraries["regularized"])
    figure, axes = plt.subplots(components, 2, figsize=(14, 3 * components),
                                constrained_layout=True)
    angles = np.linspace(0, 90, 901)
    for scene in range(components):
        for column, form in enumerate(("regularized", "quadratic")):
            axis = axes[scene, column]
            for channel, params in enumerate(libraries[form][scene]):
                diagnostic = model["diagnostics"][form][scene][channel]
                boundary = "*" if diagnostic["positivity_boundary"] else ""
                axis.plot(angles, 1 + angular_basis(angles, form) @ params,
                          label=f"C{(8,11,12,14,15,16)[channel]:02}{boundary}")
            condition = max(
                diagnostic["normalized_jacobian_condition"]
                for diagnostic in model["diagnostics"][form][scene]
            )
            axis.set_title(f"Scene {scene} / {form}: max normalized condition {condition:.1f}")
            axis.axhline(0, color="black", linewidth=.5)
            axis.axvline(55, color="grey", linewidth=.5)
            axis.set_xlim(0, 90)
            axis.set_ylabel("Radiance / radiance at 55 deg")
            axis.grid(alpha=.2)
            if scene == 0:
                axis.legend(ncol=3, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("View zenith angle (deg)")
    figure.suptitle(
        "Fixed-baseline pooled scene ADM profiles: * indicates positivity boundary\n"
        "Quadratic shape conditioning is normalized for basis scale, not an uncertainty estimate",
        fontsize=14,
    )
    figure.savefig(output_dir / "adm_profiles.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    with args.report.open() as source:
        report = json.load(source)
    with Path(str(args.model) + ".json").open() as source:
        metadata = json.load(source)
    for evidence in (report, metadata):
        freshness = check_provenance(evidence["provenance"], ROOT)
        if freshness["status"] != "current":
            raise ValueError(f"Cannot plot stale convergence evidence: {freshness}")
    from figure_layout import figure_path
    plot_all(report, joblib.load(args.model), figure_path(report["settings"]["figure_dir"]))


if __name__ == "__main__":
    main()
