#!/usr/bin/env python3
"""Plot ADM-corrected narrowband spectral irradiance from one NetCDF file."""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PLOT_STEP = 4


def plot_narrowband_irradiance(
    data_file,
    output_dir="figures/abi_proxy/products/narrowband_flux",
    lambda_center=-106,
):
    """Plot G16, G18, and their difference in spectral irradiance."""
    with xr.open_dataset(data_file, engine="netcdf4", cache=False) as dataset:
        required = ("spectral_irradiance_G16", "spectral_irradiance_G18")
        missing = [name for name in required if name not in dataset]
        if missing:
            raise ValueError(
                f"{data_file} lacks {', '.join(missing)}; regenerate the "
                "narrowband product with scripts/radiance_to_flux.py"
            )
        sampled = dataset.isel(
            y=slice(None, None, PLOT_STEP),
            x=slice(None, None, PLOT_STEP),
        )
        lat = sampled["lat"].values
        lon = sampled["lon"].values
        flux_g16 = sampled["spectral_irradiance_G16"].values
        flux_g18 = sampled["spectral_irradiance_G18"].values

    if not np.isfinite(flux_g16).any() or not np.isfinite(flux_g18).any():
        raise ValueError(f"{data_file} contains no finite spectral-irradiance values")
    difference = flux_g16 - flux_g18
    if not np.isfinite(difference).any():
        raise ValueError(
            f"{data_file} has no grid cells with finite values for both satellites"
        )
    flux_min = min(float(np.nanmin(flux_g16)), float(np.nanmin(flux_g18)))
    flux_max = max(float(np.nanmax(flux_g16)), float(np.nanmax(flux_g18)))
    if flux_min == flux_max:
        flux_max = flux_min + 1e-12
    difference_limit = max(
        abs(float(np.nanmin(difference))), abs(float(np.nanmax(difference)))
    )
    if difference_limit == 0:
        difference_limit = 1e-12
    projection = ccrs.Sinusoidal(central_longitude=lambda_center)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(22, 8),
        subplot_kw={"projection": projection},
    )
    plot_data = (
        (
            flux_g16,
            "GOES-16 ADM-corrected spectral irradiance",
            mpl.cm.turbo,
            flux_min,
            flux_max,
        ),
        (
            flux_g18,
            "GOES-18 ADM-corrected spectral irradiance",
            mpl.cm.turbo,
            flux_min,
            flux_max,
        ),
        (
            difference,
            "G16 - G18 spectral irradiance",
            mpl.cm.bwr,
            -difference_limit,
            difference_limit,
        ),
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
        figure.colorbar(
            image,
            ax=axis,
            orientation="vertical",
            label=r"W m$^{-2}$ (cm$^{-1}$)$^{-1}$",
        )

    figure.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"narrowband_spectral_irradiance_{Path(data_file).stem}.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Narrowband NetCDF input file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/abi_proxy/products/narrowband_flux",
    )
    parser.add_argument("--lambda-center", type=float, default=-106)
    args = parser.parse_args(argv)
    plot_narrowband_irradiance(args.input, args.output_dir, args.lambda_center)


if __name__ == "__main__":
    main()
