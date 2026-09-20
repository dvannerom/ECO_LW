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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from adm import (
    radiance_linear as radiance_lin,
    radiance_linear_normalized as radiance_lin_norm,
    radiance_linear_ratio as radiance_lin_ratio,
)
from netcdf_io import load_data


def plot_fit_adm(data_file, scene_file, channel, scene, output_dir="figures/ADM"):
    preprocessed_data = load_data(data_file)
    rad_G16 = preprocessed_data['rad_G16_interp_corr'][:, :, [0, 3, 4, 6, 7, 8]]
    rad_G18 = preprocessed_data['rad_G18_interp_corr'][:, :, [0, 3, 4, 6, 7, 8]]
    scene_data = load_data(scene_file)
    labels = scene_data['labels']

    mask_rad = np.ones(rad_G16.reshape(-1, rad_G16.shape[-1]).shape[0], dtype=bool)
    rad_G16_flat = rad_G16.reshape(-1, rad_G16.shape[-1])
    rad_G18_flat = rad_G18.reshape(-1, rad_G18.shape[-1])
    for arr16, arr18 in zip(rad_G16_flat.T, rad_G18_flat.T):
        mask_rad &= (arr16 > 0) & (arr16 < 1e3) & np.isfinite(arr16)
        mask_rad &= (arr18 > 0) & (arr18 < 1e3) & np.isfinite(arr18)

    rad_G16_sel = rad_G16_flat[:, channel][mask_rad]
    rad_G18_sel = rad_G18_flat[:, channel][mask_rad]
    lza_G16 = preprocessed_data['lza_G16_interp_corr'].ravel()[mask_rad]
    lza_G18 = preprocessed_data['lza_G18_interp_corr'].ravel()[mask_rad]
    labels_sel = labels.ravel()[mask_rad]

    unique_labels = np.unique(labels_sel)
    scene_label = unique_labels[scene]
    scene_mask = (labels_sel == scene_label)

    rad_ratio_scene = np.divide(rad_G16_sel[scene_mask], rad_G18_sel[scene_mask], out=np.zeros_like(rad_G16_sel[scene_mask]), where=rad_G18_sel[scene_mask] != 0)
    x_var = lza_G16[scene_mask]
    y_var = rad_G16_sel[scene_mask]
    popt, _ = curve_fit(radiance_lin_ratio, (lza_G16[scene_mask], lza_G18[scene_mask]), rad_ratio_scene)
    popt_norm, _ = curve_fit(radiance_lin_norm, (x_var, np.full_like(x_var, popt[0])), y_var)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(x_var, y_var, s=0.1, c='b', label='GOES East data')
    plt.scatter(x_var, y_var / radiance_lin(x_var, *popt), s=0.1, c='r', label='Corrected GOES East data')
    plt.plot(np.linspace(0, 89, 100), radiance_lin_norm((np.linspace(0, 89, 100), *popt), popt_norm), label='Fit')
    plt.ylabel(r'Radiance (mW/m$^{2}$.sr.$\mu$m)')
    plt.xlabel(r'Viewing zenith angle ($\degree$)')
    plt.xticks(np.arange(0, 100, 10))
    plt.legend(frameon=False)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'fit_ADM_{os.path.basename(data_file).split("_")[1]}_res{os.path.basename(data_file).split("_")[2].split(".")[0]}km_C{channel}_scene{scene}.png')
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create ADM diagnostic plots offline from saved scene outputs.')
    parser.add_argument('-d', '--day', type=int, required=True, help='Day index')
    parser.add_argument('-r', '--resolution', type=int, default=2, help='Grid resolution in km')
    parser.add_argument('-c', '--channel', type=int, required=True, help='Channel index (0-5)')
    parser.add_argument('-s', '--scene', type=int, required=True, help='Scene class index')
    parser.add_argument('--output-dir', type=str, default='figures/ADM', help='Directory for output plots')
    args = parser.parse_args()

    data_file = f'data/preprocessed_files/abi_{args.day}_res{args.resolution}km_step1.nc'
    scene_file = f'data/scene_id/scene_id_{args.day}_res{args.resolution}km_10comp.nc'
    plot_fit_adm(data_file, scene_file, args.channel, args.scene, args.output_dir)
