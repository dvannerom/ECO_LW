"""Plot ECO spectral-regression performance and held-out residuals."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "data" / "uncertainty" / "eco_spectral_reconstruction.json"
DEFAULT_RESIDUALS = ROOT / "data" / "uncertainty" / "eco_spectral_residuals.csv"
DEFAULT_ERROR_BUDGET = ROOT / "config" / "error_budget.yaml"
TRAINING_MODE = "abi_population_weighted"
RESIDUAL_DEGREE = "3"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "uncertainty" / "spectral_angular"

SCENARIO_LABELS = {
    "rfma_goal_6": "RfMA goal (6 bands)",
    "rfma_threshold_6": "RfMA threshold (6 bands)",
}


def load_assessment(metrics_path, residuals_path, error_budget_path):
    with Path(metrics_path).open() as handle:
        metrics = json.load(handle)
    with Path(error_budget_path).open() as handle:
        error_budget = yaml.safe_load(handle)
    phase_zero_k2 = error_budget["phase_0_evidence"][
        "lw_narrowband_to_broadband_fit_noise"
    ]["value"]
    requirement = metrics.get("requirement_mappings", {}).get(
        "n2bc_only_true_band_flux", metrics.get("requirement")
    )
    abi_adm_evidence = metrics["additional_proxy_evidence"]["abi_adm_variability"]
    residual_groups = {}
    with Path(residuals_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["scenario_id"] == "rfma_goal_6"
                and row["training_mode"] == TRAINING_MODE
                and row["regression_degree"] == RESIDUAL_DEGREE
                and row["assessment_stage"] == "end_to_end"
            ):
                group = residual_groups.setdefault(
                    int(row["abi_scene"]), {"errors": [], "weights": []}
                )
                group["errors"].append(float(row["relative_error_percent"]))
                group["weights"].append(float(row["population_weight"]))
    if not residual_groups:
        raise ValueError("Residual CSV lacks ABI-weighted goal-baseline residuals")
    return metrics, requirement, phase_zero_k2, residual_groups, abi_adm_evidence


def plot_scenario_comparison(metrics, requirement, phase_zero_k2, output_path):
    scenario_ids = list(metrics["scenarios"])
    labels = [SCENARIO_LABELS.get(name, name) for name in scenario_ids]
    series = [
        ("Quadratic N2BC-only", "degree_2", "n2bc_only", "#267a8a"),
        ("Quadratic end-to-end", "degree_2", "end_to_end", "#83bdc5"),
        ("Cubic N2BC-only", "degree_3", "n2bc_only", "#dc7b42"),
        ("Cubic end-to-end", "degree_3", "end_to_end", "#e8ae8a"),
    ]
    series_values = [
        [
            metrics["scenarios"][name]["regressions"][TRAINING_MODE][degree][stage][
                "overall"
            ]["k2_scatter_percent"]
            for name in scenario_ids
        ]
        for _, degree, stage, _ in series
    ]

    positions = np.arange(len(scenario_ids))
    bar_height = 0.19
    offsets = np.linspace(-1.5 * bar_height, 1.5 * bar_height, len(series))
    figure, axis = plt.subplots(figsize=(11, 2.0 + 1.6 * len(scenario_ids)))
    for (label, _, _, color), offset, values in zip(series, offsets, series_values):
        axis.barh(
            positions + offset,
            values,
            height=bar_height,
            color=color,
            label=label,
        )
    axis.axvline(
        phase_zero_k2,
        color="#555555",
        linestyle=":",
        linewidth=1.4,
        label=f"RfMA Phase 0 reported: {phase_zero_k2:.1f}%",
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("2 x ABI-population-weighted SD of grouped out-of-fold relative error (%)")
    axis.set_title("ECO L2c broadband OLR: ABI-weighted global N2BC, N2BC-only vs ADM-inclusive")
    axis.grid(axis="x", color="#d7d7d7", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    requirement_id = requirement.get("id", "ObsReq_16")
    figure.text(
        0.01,
        0.01,
        f"{requirement_id} maps to N2BC-only; ObsReq 15 maps to end-to-end. "
        "Plotted 2 x SD is scatter excluding bias, not accuracy or compliance. "
        "Sunny files weighted by ABI-GMM scene population. "
        "15-view, noise-free ADM simulation and SRFs remain conditional.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_goal_residuals(residual_groups, output_path):
    """Plot each ABI scene's mean end-to-end residual and its two-SD spread."""
    scenes = sorted(residual_groups)
    total_weight = sum(np.sum(residual_groups[scene]["weights"]) for scene in scenes)
    labels = [
        f"Scene {scene}\n{100 * np.sum(residual_groups[scene]['weights']) / total_weight:.1f}%"
        f", n={len(residual_groups[scene]['errors'])}"
        for scene in scenes
    ]
    scene_errors = [np.asarray(residual_groups[scene]["errors"]) for scene in scenes]
    scene_means = np.asarray([np.mean(errors) for errors in scene_errors])
    scene_two_sigma = np.asarray(
        [2.0 * np.std(errors, ddof=1) if errors.size > 1 else 0.0
         for errors in scene_errors]
    )
    all_errors = np.concatenate(scene_errors)
    all_weights = np.concatenate([residual_groups[scene]["weights"] for scene in scenes])
    weighted_mean = float(all_weights @ all_errors / all_weights.sum())
    figure, axis = plt.subplots(figsize=(max(9, 1.3 * len(scenes)), 5))
    positions = np.arange(1, len(scenes) + 1)
    axis.errorbar(
        positions,
        scene_means,
        yerr=scene_two_sigma,
        fmt="o",
        color="#267a8a",
        ecolor="#267a8a",
        elinewidth=1.6,
        capsize=5,
        markersize=6,
        label="Scene mean ± 2 SD",
    )
    axis.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
    axis.axhline(
        weighted_mean,
        color="#dc7b42",
        linewidth=1.2,
        label=f"ABI-weighted mean: {weighted_mean:+.3f}%",
    )
    axis.set_xticks(positions, labels)
    axis.set_xlabel("ABI-GMM scene (population share, Sunny files)")
    axis.set_ylabel("Out-of-fold relative OLR error (%)")
    axis.set_title("RfMA goal bands: end-to-end cubic OLR error by ABI scene")
    axis.grid(axis="y", color="#d7d7d7", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False, fontsize=8)
    figure.text(
        0.01,
        0.01,
        "Error bars: ±2 within-scene sample SD. Points: scene means. "
        "Horizontal line: ABI-population-weighted mean.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_adm_narrowband_flux_error(metrics, output_path):
    channels = metrics["scenarios"]["rfma_goal_6"]["angular_retrieval"]
    names = list(channels)
    scatter = [
        channels[name]["narrowband_flux_error_percent"]["overall"][
            "k2_scatter_percent"
        ]
        for name in names
    ]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.barh(positions, scatter, color="#267a8a")
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlabel("2 x sample SD of ADM-retrieved band-flux error (%, unweighted)")
    axis.set_title("ECO simulated L1b radiance to L2b narrowband flux")
    axis.grid(axis="x", color="#d7d7d7", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.text(
        0.01,
        0.01,
        "Idealized 15-view, noise-free Sunny simulation; provisional k=2 convention.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_abi_adm_proxy_variability(abi_adm_evidence, output_path):
    configuration = abi_adm_evidence["configuration"]
    groups = abi_adm_evidence["groups"]
    scenes = sorted({entry["proxy_scene"] for entry in groups})
    channels = configuration["abi_channels"]
    spread = np.full((len(scenes), len(channels)), np.nan, dtype=np.float64)
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    channel_index = {channel: index for index, channel in enumerate(channels)}
    for entry in groups:
        profile = entry["angular_profile"]
        angles = np.asarray(profile["viewing_angles_deg"])
        selected_angles = (angles >= 50.0) & (angles <= 60.0)
        spread[scene_index[entry["proxy_scene"]], channel_index[entry["abi_channel"]]] = (
            float(
                np.mean(
                    np.asarray(
                        profile["k2_relative_between_day_spread_percent"]
                    )[selected_angles]
                )
            )
        )
    figure, axis = plt.subplots(figsize=(9, 5))
    image = axis.imshow(spread, aspect="auto", cmap="magma", interpolation="nearest")
    axis.set_xticks(np.arange(len(channels)), channels)
    axis.set_yticks(np.arange(len(scenes)), [f"Scene {scene}" for scene in scenes])
    axis.set_xlabel("GOES ABI proxy channel")
    axis.set_ylabel("ABI GMM scene class")
    axis.set_title("Empirical ABI ADM shape spread across configured days")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("k=2 relative ADM-factor spread at VZA 50-60 deg (%)")
    figure.text(
        0.01,
        0.01,
        "ABI empirical proxy evidence, not an ECO uncertainty estimate; transfer to ECO channels/scenes is unvalidated.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--error-budget", type=Path, default=DEFAULT_ERROR_BUDGET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, requirement, phase_zero_k2, residual_groups, abi_adm_evidence = load_assessment(
        args.metrics, args.residuals, args.error_budget
    )
    comparison_path = args.output_dir / "eco_spectral_k2_scatter.png"
    residual_path = args.output_dir / "eco_goal_residuals_by_abi_scene.png"
    adm_path = args.output_dir / "eco_adm_narrowband_flux_error.png"
    abi_adm_path = args.output_dir / "abi_adm_proxy_variability.png"
    plot_scenario_comparison(metrics, requirement, phase_zero_k2, comparison_path)
    plot_goal_residuals(residual_groups, residual_path)
    plot_adm_narrowband_flux_error(metrics, adm_path)
    plot_abi_adm_proxy_variability(abi_adm_evidence, abi_adm_path)
    print(f"Wrote {comparison_path}")
    print(f"Wrote {residual_path}")
    print(f"Wrote {adm_path}")
    print(f"Wrote {abi_adm_path}")


if __name__ == "__main__":
    main()
