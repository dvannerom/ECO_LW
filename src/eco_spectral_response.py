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
        if scenario.get("response_model") not in {"smooth_bands", "triangular_three"}:
            raise ValueError(f"scenario {name!r} has an unsupported response model")
        if scenario["response_model"] == "smooth_bands":
            for band in channel_bands:
                if len(band) != 2 or all(edge is None for edge in band):
                    raise ValueError(f"scenario {name!r} has an invalid band")
                lower, upper = band
                if lower is not None and upper is not None and lower >= upper:
                    raise ValueError(f"scenario {name!r} has non-increasing band edges")
    return scenarios


def channel_response_matrix(
    wavelength_um: np.ndarray, scenario: dict[str, Any], default_edge_slope_per_um: float
) -> np.ndarray:
    """Return a (wavelength, channel) response matrix for one scenario."""
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    if wavelength.ndim != 1 or not np.all(np.isfinite(wavelength)):
        raise ValueError("wavelength_um must be a finite one-dimensional array")

    channel_bands = scenario["channel_bands_um"]
    response_model = scenario["response_model"]
    if response_model == "triangular_three":
        if len(channel_bands) != 3:
            raise ValueError("triangular_three requires exactly three channels")
        (rise_start, rise_end), (plateau_start, plateau_end), (fall_start, fall_end) = channel_bands
        responses = np.column_stack(
            (
                np.where(
                    (wavelength >= rise_start) & (wavelength < rise_end),
                    (wavelength - rise_start) / (rise_end - rise_start),
                    0.0,
                ),
                np.where(
                    (wavelength >= plateau_start) & (wavelength < plateau_end),
                    1.0,
                    0.0,
                ),
                np.where(
                    (wavelength >= fall_start) & (wavelength < fall_end),
                    (fall_end - wavelength) / (fall_end - fall_start),
                    0.0,
                ),
            )
        )
    else:
        edge_slope = float(scenario.get("edge_slope_per_um", default_edge_slope_per_um))
        if edge_slope <= 0:
            raise ValueError("edge_slope_per_um must be positive")
        channels = []
        for lower, upper in channel_bands:
            response = np.ones_like(wavelength)
            if lower is not None:
                response *= expit(edge_slope * (wavelength - float(lower)))
            if upper is not None:
                response *= expit(edge_slope * (float(upper) - wavelength))
            channels.append(response)
        responses = np.column_stack(channels)

        envelope_band = scenario.get("envelope_band_um")
        if envelope_band is not None:
            envelope_slope = float(
                scenario.get("envelope_edge_slope_per_um", default_edge_slope_per_um)
            )
            envelope = expit(envelope_slope * (wavelength - envelope_band[0]))
            envelope *= expit(envelope_slope * (envelope_band[1] - wavelength))
            responses *= envelope[:, None]

    if not np.all(np.isfinite(responses)) or np.any(responses < 0):
        raise ValueError("channel responses must be finite and non-negative")
    if np.any(np.all(responses == 0, axis=0)):
        raise ValueError("channel response is zero over the supplied wavelength grid")
    return responses
