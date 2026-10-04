"""Summarize empirical ABI ADM variability as proxy evidence for ECO."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adm import radiance_linear
from netcdf_io import load_data
from spectral_response import CHANNELS

FILENAME_PATTERN = re.compile(
    r"^ADM_(?P<day>\d+)_res(?P<resolution>\d+)km_C(?P<channel>\d+)_scene(?P<scene>\d+)\.nc$"
)


def summarize_adm_files(input_files, expected_days, n_components, resolution_km):
    expected_days = tuple(sorted(int(day) for day in expected_days))
    grouped = defaultdict(list)
    excluded_records = 0
    for input_file in input_files:
        path = Path(input_file)
        match = FILENAME_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected ABI ADM filename: {path.name}")
        day = int(match.group("day"))
        file_resolution = int(match.group("resolution"))
        channel = int(match.group("channel"))
        scene = int(match.group("scene"))
        if day not in expected_days or file_resolution != resolution_km:
            continue
        if scene >= n_components:
            excluded_records += 1
            continue
        if not 0 <= channel < len(CHANNELS) or scene < 0:
            raise ValueError(f"ABI ADM index outside configured range: {path.name}")
        data = load_data(str(path), variable_names=("b", "norm"))
        b_value = float(np.asarray(data["b"]))
        norm_value = float(np.asarray(data["norm"]))
        if not np.isfinite(b_value) or not np.isfinite(norm_value) or norm_value <= 0:
            raise ValueError(f"Invalid ABI ADM parameters in {path}")
        grouped[(channel, scene)].append(
            {"day": day, "b": b_value, "norm": norm_value}
        )

    viewing_angles_deg = np.arange(0.0, 70.0 + 5.0, 5.0)
    summaries = []
    for channel in range(len(CHANNELS)):
        for scene in range(n_components):
            records = sorted(grouped.get((channel, scene), []), key=lambda item: item["day"])
            record_days = [record["day"] for record in records]
            if record_days != list(expected_days):
                raise ValueError(
                    f"Incomplete ABI ADM coverage for C{channel}/scene{scene}: "
                    f"expected {len(expected_days)} days, found {len(record_days)}"
                )

            shape_parameters = np.asarray([record["b"] for record in records])
            normalizations = np.asarray([record["norm"] for record in records])
            angular_profiles = normalizations[:, None] * radiance_linear(
                viewing_angles_deg[None, :], shape_parameters[:, None]
            )
            mean_profile = np.mean(angular_profiles, axis=0)
            profile_spread = np.std(angular_profiles, axis=0, ddof=1)
            relative_profile_spread = np.divide(
                profile_spread,
                np.abs(mean_profile),
                out=np.full_like(profile_spread, np.nan),
                where=mean_profile != 0,
            )
            summaries.append(
                {
                    "abi_channel_index": channel,
                    "abi_channel": CHANNELS[channel],
                    "proxy_scene": scene,
                    "n_days": len(records),
                    "days": record_days,
                    "shape_parameter_b": {
                        "mean": float(np.mean(shape_parameters)),
                        "between_day_standard_deviation": float(
                            np.std(shape_parameters, ddof=1)
                        ),
                        "minimum": float(np.min(shape_parameters)),
                        "maximum": float(np.max(shape_parameters)),
                    },
                    "normalization": {
                        "mean": float(np.mean(normalizations)),
                        "between_day_standard_deviation": float(
                            np.std(normalizations, ddof=1)
                        ),
                    },
                    "angular_profile": {
                        "viewing_angles_deg": viewing_angles_deg.tolist(),
                        "mean": mean_profile.tolist(),
                        "between_day_standard_deviation": profile_spread.tolist(),
                        "relative_between_day_spread_percent": (
                            100.0 * relative_profile_spread
                        ).tolist(),
                        "k2_relative_between_day_spread_percent": (
                            200.0 * relative_profile_spread
                        ).tolist(),
                    },
                }
            )

    return {
        "evidence_source": "GOES-16/GOES-18 ABI fitted ADM products",
        "evidence_type": "empirical_proxy_between_day_ADM_variability",
        "quantity": "between-day spread of scene/channel fitted ADM shape and normalized angular profile",
        "not_estimated": [
            "curve_fit parameter covariance (not retained in existing ADM files)",
            "ECO-specific ADM uncertainty without channel/scene/geometry transfer validation",
            "independent radiance measurement noise",
        ],
        "requirement_traceability": {
            "ObsReq_15": "Proxy evidence about angular radiance-to-flux sensitivity; not direct ECO compliance evidence.",
            "ObsReq_17": "Proxy scene-conditioned ADM behavior; does not measure ECO scene misclassification or co-registration error.",
            "ObsReq_12": "Not directly assessed: GOES fixed two-view sampling is not ECO's dense quasi-simultaneous view sequence.",
        },
        "transferability": "Keep separate from Sunny/ECO-channel estimates. Use as empirical scene/day variability evidence or a prior-range candidate only after the ABI-to-ECO channel, scene, and geometry mapping is justified.",
        "configuration": {
            "resolution_km": int(resolution_km),
            "n_components": int(n_components),
            "days": list(expected_days),
            "abi_channels": CHANNELS,
            "viewing_angles_deg": viewing_angles_deg.tolist(),
        },
        "excluded_stale_scene_records": excluded_records,
        "groups": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--days", nargs="+", type=int, required=True)
    parser.add_argument("--n-components", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument(
        "--error-budget", type=Path, default=ROOT / "config" / "error_budget.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = summarize_adm_files(
        args.inputs, args.days, args.n_components, args.resolution
    )
    with args.error_budget.open() as handle:
        error_budget = yaml.safe_load(handle)
    result["requirement_traceability"] = error_budget["evidence_mappings"][
        "abi_adm_proxy_variability"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Wrote ABI ADM proxy evidence summary: {args.output}")


if __name__ == "__main__":
    main()
