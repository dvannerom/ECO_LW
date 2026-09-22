#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr


def plot_broadband_flux(
    data_file,
    output_dir="figures/broadband_flux",
    lambda_center=-106,
):
    """Plot G16, G18, and G16-G18 broadband flux from a saved product."""
    with xr.open_dataset(data_file, engine="netcdf4", cache=False) as dataset:
        lat = dataset["lat"].values
        lon = dataset["lon"].values
        flux_g16 = dataset["flux_G16"].values
        flux_g18 = dataset["flux_G18"].values

    difference = flux_g16 - flux_g18
    projection = ccrs.Sinusoidal(central_longitude=lambda_center)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 8),
        subplot_kw={"projection": projection},
    )
    plot_data = (
        (flux_g16, "GOES-16 broadband flux", mpl.cm.jet, 80, 380),
        (flux_g18, "GOES-18 broadband flux", mpl.cm.jet, 80, 380),
        (difference, "G16 - G18 broadband flux", mpl.cm.bwr, -40, 40),
    )
    for axis, (values, title, cmap, vmin, vmax) in zip(axes, plot_data):
        image = axis.pcolormesh(
            lon,
            lat,
            values,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_global()
        axis.set_extent([-163, -49, -90, 90], crs=ccrs.PlateCarree())
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
    out_path = output_dir / f"broadband_flux_{Path(data_file).stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot saved G16 and G18 broadband flux products."
    )
    parser.add_argument("-d", "--day", type=int, required=True, help="Day index")
    parser.add_argument("-r", "--resolution", type=int, default=2, help="Grid resolution in km")
    parser.add_argument("--input", type=str, help="Broadband flux NetCDF input file")
    parser.add_argument("--output-dir", type=str, default="figures/broadband_flux")
    parser.add_argument("--lambda-center", type=float, default=-106)
    args = parser.parse_args()

    data_file = args.input or (
        f"data/broadband_flux/broadband_flux_{args.day}_res{args.resolution}km.nc"
    )
    plot_broadband_flux(data_file, args.output_dir, args.lambda_center)
