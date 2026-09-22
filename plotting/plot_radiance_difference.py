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


ABI_CHANNELS = (0, 3, 4, 6, 7, 8)


def plot_radiance_difference(
    data_file,
    channel,
    output_dir="figures/radiance_difference",
    lambda_center=-106,
):
    """Plot G16-G18 radiance differences before and after ADM correction."""
    if not 0 <= channel < len(ABI_CHANNELS):
        raise ValueError(f"channel must be between 0 and {len(ABI_CHANNELS) - 1}")

    abi_channel = ABI_CHANNELS[channel]
    with xr.open_dataset(data_file, engine="netcdf4", cache=False) as dataset:
        lat = dataset["lat_interp_grid"].values
        lon = dataset["lon_interp_grid"].values
        rad_g16 = dataset["rad_G16_interp"].isel(channel=abi_channel).values
        rad_g18 = dataset["rad_G18_interp"].isel(channel=abi_channel).values
        rad_g16_corr = dataset["rad_G16_interp_corr"].isel(channel=abi_channel).values
        rad_g18_corr = dataset["rad_G18_interp_corr"].isel(channel=abi_channel).values

    diff = rad_g16 - rad_g18
    diff_corr = rad_g16_corr - rad_g18_corr

    projection = ccrs.Sinusoidal(central_longitude=lambda_center)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 8),
        subplot_kw={"projection": projection},
    )
    for axis, values, title in zip(
        axes,
        (diff, diff_corr),
        ("G16 - G18 radiance", "G16 - G18 radiance after ADM correction"),
    ):
        image = axis.pcolormesh(
            lon,
            lat,
            values,
            cmap=mpl.cm.bwr,
            transform=ccrs.PlateCarree(),
            vmin=-40,
            vmax=40,
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
        fig.colorbar(image, ax=axis, orientation="vertical", shrink=0.7)

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"radiance_difference_{Path(data_file).stem}_C{channel}.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot G16-G18 radiance differences before and after ADM correction."
    )
    parser.add_argument("-d", "--day", type=int, required=True, help="Day index")
    parser.add_argument("-r", "--resolution", type=int, default=2, help="Grid resolution in km")
    parser.add_argument("-c", "--channel", type=int, required=True, help="Logical channel index (0-5)")
    parser.add_argument("--input", type=str, help="Preprocessed NetCDF input file")
    parser.add_argument("--output-dir", type=str, default="figures/radiance_difference")
    parser.add_argument("--lambda-center", type=float, default=-106)
    args = parser.parse_args()

    data_file = args.input or (
        f"data/preprocessed_files/abi_{args.day}_res{args.resolution}km_step1.nc"
    )
    plot_radiance_difference(
        data_file,
        args.channel,
        args.output_dir,
        args.lambda_center,
    )
