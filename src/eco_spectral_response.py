"""Load ECO channel scenarios and evaluate their spectral responses."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.special import expit


def load_channel_scenarios(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate the named ECO channel scenarios from YAML."""
    with Path(path).open() as handle:
        catalog = yaml.safe_load(handle)

    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported ECO channel scenario schema")
    scenarios = catalog.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("channel scenario catalog must contain scenarios")

    for name, scenario in scenarios.items():
        channel_names = scenario.get("channel_names")
        channel_bands = scenario.get("channel_bands_um")
        if not channel_names or len(channel_names) != len(channel_bands):
            raise ValueError(f"scenario {name!r} has mismatched channel definitions")
        if scenario.get("response_model") != "smooth_bands":
            raise ValueError(f"scenario {name!r} has an unsupported response model")
        for band in channel_bands:
            if len(band) != 2 or any(edge is None for edge in band):
                raise ValueError(f"scenario {name!r} has an invalid band")
            if band[0] >= band[1]:
                raise ValueError(f"scenario {name!r} has non-increasing band edges")
    return scenarios


def channel_response_matrix(
    wavelength_um: np.ndarray, scenario: dict[str, Any], default_edge_slope_per_um: float
) -> np.ndarray:
    """Return responses[wavelength, channel] for one RfMA passband scenario.

    Each passband [lower, upper] (um) gets idealized logistic edges with slope
    edge_slope_per_um (um-1); responses are dimensionless in [0, 1].
    """
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    if wavelength.ndim != 1 or not np.all(np.isfinite(wavelength)):
        raise ValueError("wavelength_um must be a finite one-dimensional array")

    edge_slope = float(scenario.get("edge_slope_per_um", default_edge_slope_per_um))
    if edge_slope <= 0:
        raise ValueError("edge_slope_per_um must be positive")
    bands = np.asarray(scenario["channel_bands_um"], dtype=np.float64)
    responses = expit(edge_slope * (wavelength[:, None] - bands[None, :, 0]))
    responses *= expit(edge_slope * (bands[None, :, 1] - wavelength[:, None]))

    if not np.all(np.isfinite(responses)) or np.any(responses < 0):
        raise ValueError("channel responses must be finite and non-negative")
    if np.any(np.all(responses == 0, axis=0)):
        raise ValueError("channel response is zero over the supplied wavelength grid")
    return responses
