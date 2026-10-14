"""Serialization helpers for clean grouped-held-out ECO/Sunny flux products."""

import json
from pathlib import Path

import numpy as np
from product_paths import nominal_product_path


PRODUCT_SCOPE = (
    "Noise-free, simulation/evaluation-only Sunny spectra evaluated with "
    "GroupKFold-held-out ECO scene-library and N2BC fits. These products are "
    "not flight data, geolocated observations, or all-training predictions."
)


def _validate_products(clean_flux, clean_broadband, reference, weights, labels,
                       groups, regimes, fold_ids):
    clean_flux = np.asarray(clean_flux)
    if clean_flux.ndim != 2 or clean_flux.shape[1] != 6:
        raise ValueError("Held-out narrowband flux must have shape (case, 6)")
    n_cases = clean_flux.shape[0]
    clean_broadband = np.asarray(clean_broadband)
    reference = np.asarray(reference)
    weights = np.asarray(weights)
    labels = np.asarray(labels)
    fold_ids = np.asarray(fold_ids)
    vectors = (clean_broadband, reference, weights, labels, groups, regimes, fold_ids)
    if any(np.asarray(vector).shape != (n_cases,) for vector in vectors):
        raise ValueError("Held-out product case metadata must have shape (case,)")
    if (np.any(~np.isfinite(clean_flux)) or np.any(clean_flux <= 0)
            or np.any(~np.isfinite(clean_broadband)) or np.any(clean_broadband <= 0)
            or np.any(~np.isfinite(reference)) or np.any(reference <= 0)
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1)
            or np.any(labels < 0) or np.any(fold_ids < 0)):
        raise ValueError("Held-out flux values or case metadata are invalid")
    return clean_flux, clean_broadband, reference, weights, labels, fold_ids


def write_heldout_products(product_dir, *, clean_flux, clean_broadband, reference,
                           weights, labels, groups, regimes, fold_ids, scenario,
                           provenance, true_band_flux, angles_deg, reference_domain_um,
                           channel_definition):
    """Write paired clean held-out flux arrays and JSON schema/provenance records."""
    (clean_flux, clean_broadband, reference, weights,
     labels, fold_ids) = _validate_products(
        clean_flux, clean_broadband, reference, weights, labels, groups, regimes, fold_ids
    )
    groups = np.asarray(groups).astype(str)
    regimes = np.asarray(regimes).astype(str)
    true_band_flux = np.asarray(true_band_flux)
    angles_deg = np.asarray(angles_deg)
    if (true_band_flux.shape != clean_flux.shape
            or np.any(~np.isfinite(true_band_flux)) or np.any(true_band_flux <= 0)):
        raise ValueError("Reference narrowband flux must be positive with shape (case, 6)")
    if angles_deg.ndim != 1 or not angles_deg.size or np.any(~np.isfinite(angles_deg)):
        raise ValueError("Retrieval angles must be a nonempty finite vector")
    if labels.dtype.kind not in "iu" or fold_ids.dtype.kind not in "iu":
        raise ValueError("Scene/fold IDs must be integers and each group must belong to one fold")
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    first_fold = np.full(len(unique_groups), fold_ids.max(), dtype=fold_ids.dtype)
    last_fold = np.zeros(len(unique_groups), dtype=fold_ids.dtype)
    np.minimum.at(first_fold, group_index, fold_ids)
    np.maximum.at(last_fold, group_index, fold_ids)
    if np.any(first_fold != last_fold):
        raise ValueError("Each group must belong to one fold")
    domain = np.asarray(reference_domain_um, dtype=float)
    if domain.shape != (2,) or not np.all(np.isfinite(domain)) or not 0 < domain[0] < domain[1]:
        raise ValueError("Reference wavelength domain must be an increasing positive pair")
    if len(channel_definition["channel_names"]) != 6:
        raise ValueError("Nominal flux products require six channel names")
    case_index = np.arange(len(reference), dtype=np.int64)
    common_arrays = {
        "case_index": case_index,
        "scene_label": labels,
        "fold_id": fold_ids,
        "groups": groups,
        "regimes": regimes,
        "reference_truth_w_m2": reference,
        "population_weight": weights,
        "retrieval_angles_deg": angles_deg,
    }
    product_dir = Path(product_dir)
    product_specs = (
        ("narrowband_flux",
         {"clean_narrowband_flux_w_m2": clean_flux,
          "reference_narrowband_flux_w_m2": true_band_flux},
         ["case", "eco_channel"], "W m-2"),
        ("broadband_flux",
         {"predicted_clean_broadband_flux_w_m2": clean_broadband},
         ["case"], "W m-2"),
    )
    output_links = {}
    for kind, flux_array, dimensions, units in product_specs:
        path = Path(nominal_product_path(product_dir, scenario, kind))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **common_arrays, **flux_array)
        metadata = {
            "schema_version": 1,
            "product_type": kind,
            "scenario": str(scenario),
            "channel_definition": channel_definition,
            "reference_domain_um": list(reference_domain_um),
            "reference_definition": "Sunny upward hemispheric TOA flux integrated over reference_domain_um",
            "scope": PRODUCT_SCOPE,
            "dimensions": {
                "flux": dimensions,
                **{key: dimensions for key in flux_array},
                "case_index": ["case"],
                "scene_label": ["case"],
                "fold_id": ["case"],
                "groups": ["case"],
                "regimes": ["case"],
                "reference_truth_w_m2": ["case"],
                "population_weight": ["case"],
                "retrieval_angles_deg": ["view"],
            },
            "units": {
                **{key: units for key in flux_array},
                "reference_truth_w_m2": "W m-2",
                "population_weight": "1",
                "case_index": "1",
                "scene_label": "1",
                "fold_id": "1",
                "groups": "identifier",
                "regimes": "label",
                "retrieval_angles_deg": "degree",
            },
            "array_shapes": {
                key: list(value.shape)
                for key, value in {**common_arrays, **flux_array}.items()
            },
            "provenance": provenance,
        }
        with Path(str(path) + ".json").open("w") as target:
            json.dump(metadata, target, indent=2, allow_nan=False)
        output_links[kind] = {
            "npz": str(path.relative_to(product_dir)),
            "metadata": str(Path(str(path) + ".json").relative_to(product_dir)),
        }
    return output_links
