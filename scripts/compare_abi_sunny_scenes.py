"""Fit a separate spectral GMM on ABI means and classify each Sunny view directly."""

import argparse
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from abi_sunny_comparison import (
    SUNNY_VIEW_ANGLES_DEG, abi_response_on_grid, blackbody_roundtrip,
    classify_spectral_features, convolve_abi_radiances, fit_spectral_model,
    sample_abi_file, summarize_classification, sunny_scene_files,
    synthetic_brightness_temperature,
)
from scene_features import (
    SPECTRAL_CHANNEL_INDICES, spectral_feature_names, spectral_features_from_bt,
)
from spectral_response import CHANNELS


def write_classifications(
    path, identifier_names, identifiers, features, labels, probabilities, scores
):
    """Write one row per classified sample/view; posterior columns use ordered scene IDs."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            identifier_names + spectral_feature_names()
            + ["scene_id", "max_posterior", "log_density_pca"]
            + [f"p_scene_{scene}" for scene in range(probabilities.shape[1])]
        )
        for identity, feature, label, posterior, score in zip(
            identifiers, features, labels, probabilities, scores
        ):
            writer.writerow(
                list(identity.values()) + feature.tolist()
                + [int(label), float(posterior.max()), float(score)] + posterior.tolist()
            )


def run_comparison(args):
    """Execute ABI-only fitting and direct Sunny-view transfer without production changes."""
    if args.points_per_file < 1 or args.components < 1 or args.gmm_jobs < 1:
        raise ValueError("Sampling cap, component count and GMM jobs must be positive")
    files = sorted(Path(path) for path in args.abi_files)
    if not files or len(set(files)) != len(files):
        raise ValueError("Provide nonempty, unique ABI input paths")
    samples, sampled_positions, inventory = [], [], []
    seeds = np.random.SeedSequence(args.seed).spawn(len(files))
    for path, seed in zip(files, seeds):
        features, positions, count = sample_abi_file(
            path, args.points_per_file, seed, args.chunk_rows, args.pixel_step
        )
        samples.append(features)
        sampled_positions.append(positions)
        inventory.append({
            "path": str(path), "valid_strided_pixels": count, "sampled_pixels": len(features),
        })
        print(f"{path}: sampled {len(features)} of {count} valid strided pixels", flush=True)
    abi_features = np.concatenate(samples, axis=0)
    del samples
    model, order = fit_spectral_model(
        abi_features, args.components, args.seed, args.gmm_jobs
    )
    abi_labels, abi_probabilities, abi_scores = classify_spectral_features(
        model, order, abi_features
    )
    threshold = float(np.quantile(abi_scores, 0.01))
    with xr.open_dataset(files[0], engine="netcdf4") as dataset:
        planck = dataset["planck_G16"].values[list(SPECTRAL_CHANNEL_INDICES)]

    sunny_features, sunny_radiances, sunny_identities = [], [], []
    previous_grid = None
    max_roundtrip_error = 0.0
    spectrum_files = sunny_scene_files(args.sunny_dir)
    for path, regime, scene_index in spectrum_files:
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] != 2 + len(SUNNY_VIEW_ANGLES_DEG):
            raise ValueError(f"Expected wavelength, flux and 18 directional columns in {path}")
        wavelength = data[:, 0]
        if previous_grid is None or not np.array_equal(wavelength, previous_grid):
            responses, normalization = abi_response_on_grid(wavelength, args.filter_dir)
            error = blackbody_roundtrip(wavelength, responses, normalization, planck)
            if error > args.max_roundtrip_error_K:
                raise ValueError(
                    f"ABI blackbody round-trip error {error:.4f} K exceeds "
                    f"{args.max_roundtrip_error_K} K on {path}; check SRF/calibration/grid"
                )
            max_roundtrip_error = max(max_roundtrip_error, error)
            previous_grid = wavelength.copy()
        radiances = convolve_abi_radiances(
            wavelength, data[:, 2:], responses, normalization
        )
        bt = synthetic_brightness_temperature(radiances, planck)
        sunny_radiances.append(radiances)
        sunny_features.append(spectral_features_from_bt(bt))
        sunny_identities.extend({
            "source_file": str(path), "scene_index": scene_index, "regime": regime,
            "view_angle_deg": int(angle),
        } for angle in SUNNY_VIEW_ANGLES_DEG)
    sunny_features = np.concatenate(sunny_features, axis=0)
    sunny_radiances = np.concatenate(sunny_radiances, axis=0)
    sunny_labels, sunny_probabilities, sunny_scores = classify_spectral_features(
        model, order, sunny_features
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": model, "scene_order": order, "feature_names": spectral_feature_names(),
        "training_source": "ABI only", "reference_planck_G16": planck,
        "abi_log_density_p01": threshold,
    }, output / "spectral_gmm.joblib")
    write_classifications(
        output / "abi_sample_scenes.csv", ["source_file", "flat_pixel_index"],
        (
            {"source_file": str(path), "flat_pixel_index": int(index)}
            for path, positions in zip(files, sampled_positions) for index in positions
        ), abi_features,
        abi_labels, abi_probabilities, abi_scores,
    )
    write_classifications(
        output / "sunny_scenes.csv",
        ["source_file", "scene_index", "regime", "view_angle_deg"],
        sunny_identities, sunny_features,
        sunny_labels, sunny_probabilities, sunny_scores,
    )
    np.savez(
        output / "sunny_abi_radiances.npz",
        radiance=sunny_radiances.reshape(len(spectrum_files), 18, 6),
        brightness_temperature=sunny_features[:, :6].reshape(len(spectrum_files), 18, 6),
        source_file=np.asarray([str(path) for path, _, _ in spectrum_files]),
        scene_index=np.asarray([index for _, _, index in spectrum_files]),
        regime=np.asarray([regime for _, regime, _ in spectrum_files]),
        view_angle_deg=SUNNY_VIEW_ANGLES_DEG, channels=np.asarray(CHANNELS),
        radiance_units="mW m-2 sr-1 (cm-1)-1",
    )

    summary = {
        "assessment": "Spectral-only scene coverage; not flux accuracy or ECO uncertainty",
        "feature_names": spectral_feature_names(),
        "training_source": "ABI only; summaries use the in-sample ABI training subset",
        "abi_feature_convention": "Arithmetic mean of GOES-16/18 BTs, then BT differences",
        "sunny_feature_convention": "Each provided view directly replaces ABI mean BT; no paired VZA emulation",
        "radiometry": {
            "radiance_units": "mW m-2 sr-1 (cm-1)-1",
            "normalization": "1000 * integral(L_lambda R d_lambda) / integral(R d_wavenumber)",
            "planck_reference": str(files[0]),
            "planck_variable": "planck_G16",
            "planck_coefficients": planck.tolist(),
            "srf_files": [str(Path(args.filter_dir) / f"goes-r_abi_{c}") for c in CHANNELS],
            "max_blackbody_roundtrip_error_K": max_roundtrip_error,
            "roundtrip_limit_K": args.max_roundtrip_error_K,
        },
        "settings": {
            "n_components": args.components, "seed": args.seed,
            "points_per_file": args.points_per_file, "pixel_step": args.pixel_step,
            "chunk_rows": args.chunk_rows, "pca_variance_retained": 0.98,
        },
        "abi_inputs": inventory,
        "sunny_spectrum_count": len(spectrum_files),
        "sunny_shared_index_count": len(spectrum_files) // 2,
        "abi_log_density_p01": threshold,
        "abi": summarize_classification(
            abi_features, abi_labels, abi_probabilities, abi_scores, threshold
        ),
        "sunny": summarize_classification(
            sunny_features, sunny_labels, sunny_probabilities, sunny_scores, threshold
        ),
        "sunny_by_view": {},
        "sunny_by_regime": {},
        "limitations": [
            "Single Sunny view is treated as ABI mean-view input by design, not physical angular equivalence.",
            "PFM SRFs and G16 reference Planck coefficients do not reproduce distinct G16/G18 instrument responses.",
            "The separate spectral GMM scene IDs are not production texture-GMM scene IDs.",
            "Sunny angles and clear/cloud pairs are not independent scenes; shared indices are grouping keys.",
            "Library occupancy is not climatology; ABI baseline is capped and equally sampled per file when caps bind.",
            "Posterior confidence and PCA-space density are diagnostics, not calibrated correctness or complete out-of-domain tests.",
            "No instrumental noise, spatial texture, footprint heterogeneity or flux accuracy is assessed here.",
        ],
    }
    angles = np.tile(SUNNY_VIEW_ANGLES_DEG, len(spectrum_files))
    regimes = np.asarray([identity["regime"] for identity in sunny_identities])
    for key, selector in (
        ("sunny_by_view", [(str(a), angles == a) for a in SUNNY_VIEW_ANGLES_DEG]),
        ("sunny_by_regime", [(r, regimes == r) for r in ("clear_sky", "cloudy")]),
    ):
        for name, selected in selector:
            summary[key][name] = summarize_classification(
                sunny_features[selected], sunny_labels[selected],
                sunny_probabilities[selected], sunny_scores[selected], threshold,
            )
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(f"Wrote spectral comparison to {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abi-files", nargs="+", type=Path, required=True)
    parser.add_argument("--sunny-dir", type=Path, default=ROOT / "data" / "Sunny")
    parser.add_argument("--filter-dir", type=Path, default=ROOT / "data" / "goes_channels")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "abi_sunny_comparison")
    parser.add_argument("--components", type=int, default=7)
    parser.add_argument("--points-per-file", type=int, default=50000)
    parser.add_argument("--pixel-step", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gmm-jobs", type=int, default=1)
    parser.add_argument("--max-roundtrip-error-K", dest="max_roundtrip_error_K", type=float, default=0.1)
    args = parser.parse_args()
    if not np.isfinite(args.max_roundtrip_error_K) or args.max_roundtrip_error_K <= 0:
        parser.error("--max-roundtrip-error-K must be finite and positive")
    run_comparison(args)


if __name__ == "__main__":
    main()
