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

from abi_product import load_adm_library
from sensitivity import angular_basis, angular_integral
from spectral_response import CHANNELS
from uncertainty import build_provenance

FILENAME_PATTERN = re.compile(r"^adm_k(?P<components>\d+)_day(?P<day>\d+)\.joblib$")


def summarize_adm_files(input_files, expected_days, n_components, resolution_km):
    """Between-day spread of per-day in-sample scene ADM libraries.

    Each input is component_adms/daily/adm_k{K}_day{d}.joblib (K, 6, 1),
    the regularized form shape(theta) = 1 + b*(Elmer(theta)-1) fitted on one
    day's paired G16/G18 ratios with the production scene taxonomy. Reported
    per channel/scene: b, the hemispheric integral I ("normalization") and
    the anisotropy profile shape(theta)/I used to correct radiances.
    """
    expected_days = tuple(sorted(int(day) for day in expected_days))
    grouped = defaultdict(list)
    for input_file in input_files:
        path = Path(input_file)
        match = FILENAME_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected ABI daily ADM library filename: {path.name}")
        day = int(match.group("day"))
        if int(match.group("components")) != n_components:
            raise ValueError(f"{path.name} does not use K={n_components}")
        if day not in expected_days:
            continue
        library, form, _ = load_adm_library(path, n_components)
        if form != "regularized":
            raise ValueError(f"Daily ADM proxy summary expects the regularized form: {path}")
        integral = angular_integral(library, form)
        for channel in range(len(CHANNELS)):
            for scene in range(n_components):
                grouped[(channel, scene)].append({
                    "day": day, "b": float(library[scene, channel, 0]),
                    "norm": float(integral[scene, channel])})
    excluded_records = 0

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
            basis = angular_basis(viewing_angles_deg, "regularized")[:, 0]
            angular_profiles = (1 + shape_parameters[:, None] * basis[None, :]) / normalizations[:, None]
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
        "evidence_source": "GOES-16/GOES-18 ABI per-day in-sample scene ADM libraries (production taxonomy)",
        "adm_parameterization": (
            "shape(theta)=1+b*(Elmer(theta)-1); normalization is the hemispheric integral I; "
            "angular_profile is shape(theta)/I (radiance anisotropy relative to F/pi)"),
        "evidence_type": "empirical_proxy_between_day_ADM_variability",
        "quantity": "between-day spread of scene/channel fitted ADM shape and normalized angular profile",
        "not_estimated": [
            "least-squares parameter covariance (not retained in the daily libraries)",
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
    result["provenance"] = build_provenance(
        root=ROOT,
        code_paths=(
            Path(__file__).resolve(),
            ROOT / "src" / "abi_product.py",
            ROOT / "src" / "sensitivity.py",
            ROOT / "src" / "spectral_response.py",
        ),
        configuration_paths=(
            ROOT / "config.yaml",
            args.error_budget.resolve(),
        ),
        input_paths={"abi_adm_directory": args.inputs[0].resolve().parent},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Wrote ABI ADM proxy evidence summary: {args.output}")


if __name__ == "__main__":
    main()
