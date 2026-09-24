"""Shared GOES ABI spectral-response-function (SRF) loading, used by
scripts/fit_irradiance.py and scripts/compute_temperature_SBDART.py."""

import json

import numpy as np

# GOES ABI channels used for the LW broadband estimate, in the fixed order
# that all downstream narrowband arrays (C0..C5) rely on.
CHANNELS = ["ch08", "ch11", "ch12", "ch14", "ch15", "ch16"]


def load_goes_filters(filter_dir, channels=CHANNELS):
    """Load GOES ABI spectral response functions.

    :param filter_dir: (Path) directory holding one file per channel, columns
        wavelength[nm] and relative response, two header lines.
    :param channels: (list[str]) channel file suffixes, e.g. "ch08".
    :return: (list[ndarray], list[ndarray]) per-channel wavelength[nm] and response arrays.
    """
    lambdas, responses = [], []
    for channel in channels:
        path = filter_dir / f"goes-r_abi_{channel}"
        data = np.loadtxt(path, skiprows=2)
        lambdas.append(data[:, 0])
        responses.append(data[:, 1])
    return lambdas, responses


def load_power_law(path):
    """Load per-channel radiance-to-BT power law coefficients (radiance = (a*BT)**b).

    :param path: (str or Path) JSON file with "channels", "a", "b" arrays,
        as written by scripts/fit_irradiance.py.
    :return: (list[str], ndarray, ndarray) channels, a, b
    """
    with open(path) as handle:
        data = json.load(handle)
    return data["channels"], np.asarray(data["a"]), np.asarray(data["b"])
