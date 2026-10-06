#!/usr/bin/env python3
"""Plot saved channel radiance power-law fits."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fit_irradiance import band_radiance, power_law
from spectral_response import CHANNELS, load_goes_filters


def plot_irradiance_fit(
    model_file=ROOT / "data" / "models" / "channel_radiance_power_law.json",
    filter_dir=ROOT / "data" / "goes_channels",
    output_dir=ROOT / "figures" / "diagnostics" / "n2bc",
    t_min=200.0,
    t_max=320.0,
    t_step=1.0,
):
    """Recompute and plot the Planck curves and saved power-law fits."""
    with Path(model_file).open() as handle:
        model = json.load(handle)

    temperatures = np.arange(t_min, t_max + t_step, t_step)
    lambdas_goes, response_goes = load_goes_filters(filter_dir, CHANNELS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for channel, lambdas_nm, response, a, b in zip(
        CHANNELS,
        lambdas_goes,
        response_goes,
        model["a"],
        model["b"],
    ):
        radiance = np.array(
            [
                band_radiance(lambdas_nm, response, temperature)
                for temperature in temperatures
            ]
        )
        fig, axis = plt.subplots()
        axis.plot(temperatures, radiance, label=f"Planck blackbody narrowband radiance ({channel})")
        axis.plot(
            temperatures,
            power_law(temperatures, a, b),
            "g--",
            label=f"Power-law fit: a={a:5.2e}, b={b:5.3f}",
        )
        axis.set_xlabel("Temperature (K)")
        axis.set_ylabel(r"Radiance (W/$\mathrm{m}^{2}$/sr)")
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"irradiance_fit_goes-r_abi_{channel}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "data" / "models" / "channel_radiance_power_law.json",
        help="Saved channel power-law JSON",
    )
    parser.add_argument(
        "--filter-dir",
        type=Path,
        default=ROOT / "data" / "goes_channels",
        help="Directory holding GOES ABI spectral response files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "diagnostics" / "n2bc",
        help="Directory for diagnostic plots",
    )
    parser.add_argument("--t-min", type=float, default=200.0)
    parser.add_argument("--t-max", type=float, default=320.0)
    parser.add_argument("--t-step", type=float, default=1.0)
    args = parser.parse_args()
    plot_irradiance_fit(
        args.model,
        args.filter_dir,
        args.output_dir,
        args.t_min,
        args.t_max,
        args.t_step,
    )
