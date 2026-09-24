#!/usr/bin/env python3
"""Plot the saved narrowband-to-broadband cubic-fit error diagnostic."""
import argparse
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


def plot_narrowband_to_broadband(
    coefficients_file=ROOT / "data" / "models" / "narrowband_to_broadband_coeffs.json",
    sunny_dir=ROOT / "data" / "Sunny",
    filter_dir=ROOT / "data" / "goes_channels",
    power_law_file=ROOT / "data" / "models" / "channel_radiance_power_law.json",
    output_dir=ROOT / "figures" / "broadband_flux",
):
    """Recompute the Sunny diagnostics and plot cubic flux error."""
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

    t_broadband = np.power(flux_bb / SIGMA, 1.0 / 4)
    t_narrowband = np.power(radiance_nb, 1.0 / channel_b) / channel_a
    cubic_poly = PolynomialFeatures(degree=3, include_bias=False)
    t_narrowband_poly = cubic_poly.fit_transform(t_narrowband)
    cubic = models["cubic"]
    t_broadband_pred = (
        t_narrowband_poly @ np.asarray(cubic["coefficients"])
        + cubic["intercept"]
    )
    flux_pred = SIGMA * np.power(t_broadband_pred, 4)
    flux_err = (flux_bb - flux_pred) / flux_bb

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(flux_pred, 100 * (flux_err - np.mean(flux_err)), ".")
    axis.set_xlabel("Estimated broadband irradiance (W/m2)")
    axis.set_ylabel("Error (%)")
    fig.tight_layout()
    output_path = output_dir / "narrowband_to_broadband_cubic_fit_error.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficients-file",
        type=Path,
        default=ROOT / "data" / "models" / "narrowband_to_broadband_coeffs.json",
        help="Saved narrowband-to-broadband coefficient JSON",
    )
    parser.add_argument(
        "--sunny-dir",
        type=Path,
        default=ROOT / "data" / "Sunny",
        help="Directory holding SBDART Sunny scene folders",
    )
    parser.add_argument(
        "--filter-dir",
        type=Path,
        default=ROOT / "data" / "goes_channels",
        help="Directory holding GOES ABI spectral response files",
    )
    parser.add_argument(
        "--power-law-file",
        type=Path,
        default=ROOT / "data" / "models" / "channel_radiance_power_law.json",
        help="Saved channel power-law JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "broadband_flux",
        help="Directory for the diagnostic plot",
    )
    args = parser.parse_args()
    output_path = plot_narrowband_to_broadband(
        args.coefficients_file,
        args.sunny_dir,
        args.filter_dir,
        args.power_law_file,
        args.output_dir,
    )
    print(f"Wrote {output_path}")
