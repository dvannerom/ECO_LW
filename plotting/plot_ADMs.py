#!/usr/bin/env python3
"""Compare the fitted ADM curves of every scene for one day/resolution/channel."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from adm import radiance_linear
from netcdf_io import load_data


def plot_adm_scenes(day, resolution, channel, n_components, output_dir="figures/ADM"):
	viewing_angle = np.linspace(0, 89, 100)

	fig = plt.figure(figsize=(10, 6))
	for scene in range(n_components):
		adm_file = f"data/ADM/ADM_{day}_res{resolution}km_C{channel}_scene{scene}.nc"
		adm_data = load_data(adm_file, variable_names=("b",))
		plt.plot(viewing_angle, radiance_linear(viewing_angle, adm_data["b"]), label=f"Scene {scene}")

	plt.ylabel(r"Radiance (mW/m$^{2}$.sr.$\mu$m)")
	plt.xlabel(r"Viewing zenith angle ($\degree$)")
	plt.xticks(np.arange(0, 100, 10))
	plt.legend(frameon=False)
	fig.tight_layout()

	Path(output_dir).mkdir(parents=True, exist_ok=True)
	out_path = Path(output_dir) / f"ADMs_{day}_res{resolution}km_C{channel}.png"
	plt.savefig(out_path)
	plt.close(fig)
	return out_path


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Compare the fitted ADMs of every scene for one day/resolution/channel."
	)
	parser.add_argument("-d", "--day", type=int, required=True)
	parser.add_argument("-r", "--resolution", type=int, default=2)
	parser.add_argument("-c", "--channel", type=int, required=True, help="ADM channel index (0-5)")
	parser.add_argument("--n-components", type=int, required=True, help="Number of scene components")
	parser.add_argument("--output-dir", type=str, default="figures/ADM")
	args = parser.parse_args()

	out_path = plot_adm_scenes(args.day, args.resolution, args.channel, args.n_components, args.output_dir)
	print(f"Saved {out_path}")
