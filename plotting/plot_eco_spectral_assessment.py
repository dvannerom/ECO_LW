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
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "uncertainty"

SCENARIO_LABELS = {
    "rfma_goal_6": "RfMA goal (6 bands)",
    "rfma_threshold_6": "RfMA threshold (6 bands)",
    "supplied_square_6_14": "Square 6-14 um (6 bands)",
    "supplied_sloped_6_14": "Sloped 6-14 um (6 bands)",
    "supplied_tails_6_14": "Tails 6-14 um (6 bands)",
    "supplied_extended_4p5_18": "Extended 4.5-18 um (6 bands)",
}


def load_assessment(metrics_path, residuals_path, error_budget_path):
    with Path(metrics_path).open() as handle:
        metrics = json.load(handle)
    with Path(error_budget_path).open() as handle:
        error_budget = yaml.safe_load(handle)
    phase_zero_k2 = error_budget["phase_0_evidence"][
        "lw_narrowband_to_broadband_fit_noise"
    ]["value"]
    requirement = metrics["requirement"]
    abi_adm_evidence = metrics["additional_proxy_evidence"]["abi_adm_variability"]
    residual_groups = {
        ("unstratified", "clear_sky"): [],
        ("unstratified", "cloudy"): [],
        ("clear_cloud_stratified", "clear_sky"): [],
        ("clear_cloud_stratified", "cloudy"): [],
    }
    with Path(residuals_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["scenario_id"] == "rfma_goal_6"
                and row["regression_degree"] == "3"
                and row["assessment_stage"] == "end_to_end"
            ):
                key = (row["training_mode"], row["regime"])
                if key in residual_groups:
                    residual_groups[key].append(float(row["relative_error_percent"]))
    if any(not values for values in residual_groups.values()):
        raise ValueError("Residual CSV is missing one or more goal-baseline groups")
    return metrics, requirement, phase_zero_k2, residual_groups, abi_adm_evidence


def plot_scenario_comparison(metrics, requirement, phase_zero_k2, output_path):
    scenario_ids = list(metrics["scenarios"])
    labels = [SCENARIO_LABELS.get(name, name) for name in scenario_ids]
    series = [
        (
            "Global N2BC-only",
            "unstratified",
            "n2bc_only",
            "#267a8a",
        ),
        (
            "Global end-to-end",
            "unstratified",
            "end_to_end",
            "#83bdc5",
        ),
        (
            "Clear/cloud N2BC-only",
            "clear_cloud_stratified",
            "n2bc_only",
            "#dc7b42",
        ),
        (
            "Clear/cloud end-to-end",
            "clear_cloud_stratified",
            "end_to_end",
            "#e8ae8a",
        ),
    ]
    series_values = [
        [
            metrics["scenarios"][name]["regressions"][training_mode]["degree_3"][
                stage
            ]["overall"]["k2_scatter_percent"]
            for name in scenario_ids
        ]
        for _, training_mode, stage, _ in series
    ]

    positions = np.arange(len(scenario_ids))
    bar_height = 0.19
    offsets = np.linspace(-1.5 * bar_height, 1.5 * bar_height, len(series))
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for (label, _, _, color), offset, values in zip(series, offsets, series_values):
        axis.barh(
            positions + offset,
            values,
            height=bar_height,
            color=color,
            label=label,
        )
    axis.axvline(
        requirement["maximum_percent"],
        color="#a83232",
        linestyle="--",
        linewidth=1.6,
        label=f"ObsReq 16: {requirement['maximum_percent']:.2f}%",
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
    axis.set_xlabel("2 x sample SD of grouped out-of-fold relative error (%)")
    axis.set_title("ECO L2c broadband OLR: N2BC-only vs ADM-inclusive")
    axis.grid(axis="x", color="#d7d7d7", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    figure.text(
        0.01,
        0.01,
        "Provisional k=2 convention; RfMA Phase 0 metric may not be directly comparable. "
        "N2BC-only uses true integrated band flux; end-to-end uses ADM-retrieved band flux. "
        "15-view, noise-free ADM simulation; k=2 convention and SRFs remain provisional.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_goal_residuals(residual_groups, output_path):
    group_keys = [
        ("unstratified", "clear_sky"),
        ("unstratified", "cloudy"),
        ("clear_cloud_stratified", "clear_sky"),
        ("clear_cloud_stratified", "cloudy"),
    ]
    labels = [
        "Global fit\nClear-sky",
        "Global fit\nCloudy",
        "Separate fits\nClear-sky",
        "Separate fits\nCloudy",
    ]
    values = [residual_groups[key] for key in group_keys]
    figure, axis = plt.subplots(figsize=(9, 5))
    boxplot = axis.boxplot(
        values,
        tick_labels=labels,
        whis=(5, 95),
        showfliers=False,
        showmeans=True,
        patch_artist=True,
    )
    for patch, color in zip(
        boxplot["boxes"], ("#267a8a", "#267a8a", "#dc7b42", "#dc7b42")
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axis.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
    axis.set_ylabel("Out-of-fold relative OLR error (%)")
    axis.set_title("RfMA goal bands: end-to-end cubic N2BC residuals")
    axis.grid(axis="y", color="#d7d7d7", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.text(
        0.01,
        0.01,
        "Boxes show the interquartile range; whiskers show the 5th-95th percentiles; "
        "means are marked.",
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
    axis.set_xlabel("2 x sample SD of ADM-retrieved band-flux error (%)")
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
    residual_path = args.output_dir / "eco_goal_residuals_by_regime.png"
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
