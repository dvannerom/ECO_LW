#!/usr/bin/env python3
"""Plot GOES fitted or ECO held-out narrowband-to-broadband cubic errors."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compute_temperature_SBDART import (
    DEFAULT_CHANNEL_A,
    DEFAULT_CHANNEL_B,
    SIGMA,
    load_sunny_scene,
)
from spectral_response import CHANNELS, load_goes_filters, load_power_law


DEFAULT_COEFFICIENTS = ROOT / "data" / "models" / "narrowband_to_broadband_coeffs.json"
DEFAULT_SUNNY_DIR = ROOT / "data" / "Sunny"
DEFAULT_FILTER_DIR = ROOT / "data" / "goes_channels"
DEFAULT_POWER_LAW = ROOT / "data" / "models" / "channel_radiance_power_law.json"
DEFAULT_ECO_RESIDUALS = ROOT / "data" / "uncertainty" / "eco_spectral_residuals.csv"
DEFAULT_ECO_SCENARIO = "rfma_goal_6"
DEFAULT_TRAINING_MODE = "unstratified"
TRAINING_MODES = ("unstratified", "clear_cloud_stratified")


def load_goes_flux_predictions(coefficients_file, sunny_dir, filter_dir, power_law_file):
    """Return reference and fitted flux arrays (scene,), in W/m2."""
    with Path(coefficients_file).open() as handle:
        models = json.load(handle)

    lambdas_goes, response_goes = load_goes_filters(filter_dir, CHANNELS)
    lambdas_goes = [lambdas / 1000.0 for lambdas in lambdas_goes]
    if Path(power_law_file).exists():
        _, channel_a, channel_b = load_power_law(power_law_file)
    else:
        channel_a, channel_b = DEFAULT_CHANNEL_A, DEFAULT_CHANNEL_B

    scene_files = sorted((Path(sunny_dir) / "radiance_lw_cs").glob("*"))
    scene_files += sorted((Path(sunny_dir) / "radiance_lw_cl").glob("*"))
    if not scene_files:
        raise FileNotFoundError(f"No Sunny scene files found under {sunny_dir}")

    flux_bb = np.empty(len(scene_files))
    radiance_nb = np.empty((len(scene_files), len(CHANNELS)))
    for index, scene_file in enumerate(scene_files):
        flux_bb[index], radiance_nb[index] = load_sunny_scene(
            scene_file, lambdas_goes, response_goes
        )

    t_narrowband = np.power(radiance_nb, 1.0 / channel_b) / channel_a
    cubic_poly = PolynomialFeatures(degree=3, include_bias=False)
    t_narrowband_poly = cubic_poly.fit_transform(t_narrowband)
    cubic = models["cubic"]
    t_broadband_pred = (
        t_narrowband_poly @ np.asarray(cubic["coefficients"])
        + cubic["intercept"]
    )
    flux_pred = SIGMA * np.power(t_broadband_pred, 4)
    return flux_bb, flux_pred


def load_eco_flux_predictions(residuals_file, scenario, training_mode):
    """Stream saved cubic N2BC-only held-out fluxes (scene,), in W/m2."""
    reference_flux = []
    predicted_flux = []
    with Path(residuals_file).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "scenario_id", "assessment_stage", "training_mode", "regression_degree",
            "reference_olr_w_m2", "predicted_olr_w_m2",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ECO residual CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if (
                row["scenario_id"] == scenario
                and row["assessment_stage"] == "n2bc_only"
                and row["training_mode"] == training_mode
                and row["regression_degree"] == "3"
            ):
                reference_flux.append(float(row["reference_olr_w_m2"]))
                predicted_flux.append(float(row["predicted_olr_w_m2"]))
    if not reference_flux:
        raise ValueError(
            f"No cubic N2BC-only ECO residuals for scenario {scenario!r} "
            f"and training mode {training_mode!r} in {residuals_file}"
        )
    return np.asarray(reference_flux), np.asarray(predicted_flux)


def plot_flux_errors(flux_bb, flux_pred, output_path, title):
    """Plot mean-centered (reference - prediction)/reference for (scene,) fluxes."""
    if (
        flux_bb.ndim != 1
        or flux_bb.shape != flux_pred.shape
        or flux_bb.size == 0
        or not np.all(np.isfinite(flux_bb))
        or not np.all(np.isfinite(flux_pred))
        or np.any(flux_bb <= 0)
    ):
        raise ValueError("Flux arrays must be matching finite 1D arrays with positive references")
    flux_err = (flux_bb - flux_pred) / flux_bb

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(flux_pred, 100 * (flux_err - np.mean(flux_err)), ".")
    axis.set_xlabel("Estimated broadband irradiance (W/m2)")
    axis.set_ylabel("Mean-centered error (%)")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_narrowband_to_broadband(
    coefficients_file=DEFAULT_COEFFICIENTS,
    sunny_dir=DEFAULT_SUNNY_DIR,
    filter_dir=DEFAULT_FILTER_DIR,
    power_law_file=DEFAULT_POWER_LAW,
    output_dir=ROOT / "figures" / "diagnostics" / "n2bc",
    *,
    sensor="goes",
    residuals_file=DEFAULT_ECO_RESIDUALS,
    scenario=DEFAULT_ECO_SCENARIO,
    training_mode=DEFAULT_TRAINING_MODE,
):
    """Prepare sensor-specific predictions and use one common error plot.

    GOES recomputes saved-coefficient predictions from Sunny scenes; ECO reads
    cubic N2BC-only grouped out-of-fold predictions from its assessment CSV.
    ECO streams the CSV and retains only the selected O(N) scalar flux pairs;
    neither path loads daily satellite grids or repeats ECO spectral fitting.
    """
    if sensor == "goes":
        flux_bb, flux_pred = load_goes_flux_predictions(
            coefficients_file, sunny_dir, filter_dir, power_law_file
        )
        filename = "narrowband_to_broadband_cubic_fit_error.png"
        title = "GOES ABI cubic N2BC: fitted Sunny scenes"
    elif sensor == "eco":
        if not scenario or any(
            not (character.isascii() and (character.isalnum() or character in "_-"))
            for character in scenario
        ):
            raise ValueError(
                "ECO scenario must be a nonempty alphanumeric/underscore/hyphen identifier"
            )
        if training_mode not in TRAINING_MODES:
            raise ValueError(f"Unsupported ECO training mode: {training_mode!r}")
        flux_bb, flux_pred = load_eco_flux_predictions(
            residuals_file, scenario, training_mode
        )
        filename = f"narrowband_to_broadband_cubic_fit_error_eco_{scenario}_{training_mode}.png"
        title = f"ECO cubic N2BC: held-out {scenario}, {training_mode}"
    else:
        raise ValueError(f"Unsupported sensor: {sensor!r}")
    return plot_flux_errors(flux_bb, flux_pred, Path(output_dir) / filename, title)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=("goes", "eco"), default="goes")
    parser.add_argument(
        "--coefficients-file",
        type=Path,
        help="GOES only: saved narrowband-to-broadband coefficient JSON",
    )
    parser.add_argument(
        "--sunny-dir",
        type=Path,
        help="GOES only: directory holding SBDART Sunny scene folders",
    )
    parser.add_argument(
        "--filter-dir",
        type=Path,
        help="GOES only: directory holding GOES ABI spectral response files",
    )
    parser.add_argument(
        "--power-law-file",
        type=Path,
        help="GOES only: saved channel power-law JSON",
    )
    parser.add_argument(
        "--residuals-file", type=Path,
        help="ECO only: saved assessment residual CSV (default: data/uncertainty/eco_spectral_residuals.csv)",
    )
    parser.add_argument("--scenario", help="ECO only: channel scenario (default: rfma_goal_6)")
    parser.add_argument(
        "--training-mode", choices=TRAINING_MODES,
        help="ECO only: training mode (default: unstratified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "diagnostics" / "n2bc",
        help="Directory for the diagnostic plot",
    )
    args = parser.parse_args(argv)
    exclusive_options = (
        ("coefficients_file", "sunny_dir", "filter_dir", "power_law_file")
        if args.sensor == "eco"
        else ("residuals_file", "scenario", "training_mode")
    )
    for option in exclusive_options:
        if getattr(args, option) is not None:
            parser.error(f"--{option.replace('_', '-')} is not supported for --sensor {args.sensor}")
    output_path = plot_narrowband_to_broadband(
        args.coefficients_file or DEFAULT_COEFFICIENTS,
        args.sunny_dir or DEFAULT_SUNNY_DIR,
        args.filter_dir or DEFAULT_FILTER_DIR,
        args.power_law_file or DEFAULT_POWER_LAW,
        args.output_dir,
        sensor=args.sensor,
        residuals_file=args.residuals_file or DEFAULT_ECO_RESIDUALS,
        scenario=args.scenario if args.scenario is not None else DEFAULT_ECO_SCENARIO,
        training_mode=args.training_mode or DEFAULT_TRAINING_MODE,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
