"""Plot configured ECO channel responses and compare active temporary bands."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eco_spectral_response import channel_response_matrix, load_channel_scenarios


DEFAULT_SCENARIOS = ROOT / "config" / "eco_channel_scenarios.yaml"
DEFAULT_OUTPUT = ROOT / "figures" / "diagnostics" / "instrument" / "eco_channel_responses.png"


def plot_responses(scenario_path, output_path, wavelength_min, wavelength_max):
    with Path(scenario_path).open() as handle:
        catalog = yaml.safe_load(handle)
    scenarios = load_channel_scenarios(scenario_path)
    default_slope = float(catalog["default_edge_slope_per_um"])
    wavelengths = np.linspace(wavelength_min, wavelength_max, 2000)

    columns = 3
    rows = math.ceil(len(scenarios) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 3.9 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, (scenario_name, scenario) in zip(axes.flat, scenarios.items()):
        responses = channel_response_matrix(wavelengths, scenario, default_slope)
        channel_names = scenario["channel_names"]
        for channel_index, channel_name in enumerate(channel_names):
            color = f"C{channel_index % 10}"
            axis.plot(
                wavelengths,
                responses[:, channel_index],
                color=color,
                linewidth=1.6,
                label=channel_name,
            )

        axis.set_title(scenario_name.replace("_", " "), fontsize=10)
        axis.set_xlim(wavelength_min, wavelength_max)
        axis.set_ylim(-0.03, 1.08)
        axis.grid(color="#d9d9d9", linewidth=0.7)
        axis.legend(fontsize=7, loc="lower center", ncol=2, frameon=False)

    for axis in axes.flat[len(scenarios) :]:
        axis.set_visible(False)

    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Wavelength (µm)")
    for axis in axes[:, 0]:
        if axis.get_visible():
            axis.set_ylabel("Relative response")

    figure.suptitle("ECO LW channel spectral responses", y=0.995, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wavelength-min", type=float, default=3.0)
    parser.add_argument("--wavelength-max", type=float, default=20.0)
    args = parser.parse_args()
    if args.wavelength_min >= args.wavelength_max:
        parser.error("--wavelength-min must be less than --wavelength-max")

    plot_responses(
        args.scenarios,
        args.output,
        args.wavelength_min,
        args.wavelength_max,
    )


if __name__ == "__main__":
    main()