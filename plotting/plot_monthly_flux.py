#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr


def plot_monthly_flux(
    data_file,
    output_dir="figures/products/monthly_flux",
    lambda_center=-106,
):
    """Plot monthly G16, G18, and standard deviation of their difference."""
    with xr.open_dataset(data_file, engine="netcdf4", cache=False) as dataset:
        lat = dataset["lat"].values
        lon = dataset["lon"].values
        flux_g16 = dataset["flux_G16_monthly"].values
        flux_g18 = dataset["flux_G18_monthly"].values
        std_diff = dataset["std_diff"].values

    projection = ccrs.Sinusoidal(central_longitude=lambda_center)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 8),
        subplot_kw={"projection": projection},
    )
    plot_data = (
        (flux_g16, "Monthly GOES-16 broadband flux", 150, 350),
        (flux_g18, "Monthly GOES-18 broadband flux", 150, 350),
        (std_diff, "Standard deviation of G16 - G18", 0, 20),
    )
    for axis, (values, title, vmin, vmax) in zip(axes, plot_data):
        image = axis.pcolormesh(
            lon,
            lat,
            values,
            cmap="jet",
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_global()
        axis.set_extent([-153, -59, -90, 90], crs=ccrs.PlateCarree())
        axis.coastlines()
        axis.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1,
            color="black",
            linestyle="--",
            xlocs=range(-180, 180, 30),
            ylocs=range(-90, 90, 30),
        )
        axis.set_title(title)
        fig.colorbar(image, ax=axis, orientation="vertical")

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"monthly_flux_{Path(data_file).stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a saved monthly broadband flux product."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/monthly/monthly_flux_res2km.nc",
        help="Monthly broadband flux NetCDF input file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/products/monthly_flux",
        help="Directory for the PNG output",
    )
    parser.add_argument(
        "--lambda-center",
        type=float,
        default=-106,
        help="Central longitude for the map projection",
    )
    args = parser.parse_args()

    plot_monthly_flux(args.input, args.output_dir, args.lambda_center)
