#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netcdf_io import load_data


def plot_scene_id(scene_file, output_dir="figures/scene_id", lambda_center=-106):
    dataset = load_data(scene_file)
    lat = dataset["lat"]
    lon = dataset["lon"]
    labels = dataset.get("sorted_labels", dataset["labels"])
    n_components = int(np.nanmax(labels)) + 1

    scene_map = labels.reshape(lat.shape[0], lat.shape[1])
    cmap = mpl.cm.turbo
    bounds = [i - 0.5 for i in range(n_components + 1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(
        1,
        1,
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.Sinusoidal(central_longitude=lambda_center)},
    )
    pc = axs.pcolormesh(lon, lat, scene_map, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    axs.set_global()
    axs.coastlines()
    axs.set_xlabel("Longitude")
    axs.set_ylabel("Latitude")
    cbar = fig.colorbar(pc, ax=axs, orientation="vertical")
    cbar.set_label("Scene ID")
    cbar.set_ticks(range(n_components))
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(scene_file).replace(".nc", ".png"))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a scene-ID NetCDF output offline.")
    parser.add_argument("--input", type=str, required=True, help="Scene ID NetCDF file")
    parser.add_argument("--output-dir", type=str, default="figures/scene_id", help="Directory for PNG output")
    parser.add_argument("--lambda-center", type=float, default=-106, help="Central longitude for the map projection")
    args = parser.parse_args()

    plot_scene_id(args.input, args.output_dir, args.lambda_center)
