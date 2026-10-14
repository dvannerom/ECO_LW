"""Build a provenance-aware, stage-ordered ECO LW uncertainty registry."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uncertainty import (  # noqa: E402
    build_provenance,
    check_provenance,
    propagate_covariance,
    sha256_file,
)


DEFAULT_METRICS = ROOT / "data" / "uncertainty" / "eco_spectral_reconstruction.json"
DEFAULT_N2BC_METRICS = ROOT / "data" / "uncertainty" / "eco_n2bc_stage_metrics.json"
DEFAULT_ADM_METRICS = ROOT / "data" / "uncertainty" / "eco_adm_stage_metrics.json"
DEFAULT_ABI_SUMMARY = ROOT / "data" / "uncertainty" / "abi_adm_proxy_summary.json"
DEFAULT_RESIDUALS = ROOT / "data" / "uncertainty" / "eco_spectral_residuals.csv"
DEFAULT_SCENARIOS = ROOT / "config" / "eco_channel_scenarios.yaml"
DEFAULT_ERROR_BUDGET = ROOT / "config" / "error_budget.yaml"


def _load_json(path):
    with path.open() as source:
        return json.load(source)


def _artifact_record(path, producer_provenance=None):
    if producer_provenance is None:
        check = {"status": "provenance_missing_or_unsupported", "mismatches": []}
    else:
        check = check_provenance(producer_provenance, ROOT)
    return {
        "path": _display_path(path),
        "sha256": sha256_file(path),
        "producer_provenance_status": check["status"],
        "producer_provenance_mismatches": check["mismatches"],
        "producer_identity": _provenance_identity(producer_provenance),
    }


def _provenance_identity(provenance):
    if not isinstance(provenance, dict):
        return None
    inputs = provenance.get("inputs", {})
    return {
        "code_sha256": provenance.get("code_sha256", {}),
        "configuration_sha256": provenance.get("configuration_sha256", {}),
        "input_identity_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "input_identity_method": provenance.get("input_identity_note"),
    }


def _display_path(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _requirement_reference(error_budget, requirement_id):
    for requirement in error_budget.get("requirements", {}).values():
        if requirement.get("rfmA_id") == requirement_id:
            return {
                "id": requirement_id,
                "source_citation": requirement.get("source"),
                "quantity": requirement.get("quantity"),
                "aggregation": requirement.get("aggregation"),
                "units": requirement.get("units"),
                "wavelength_range": requirement.get("wavelength_range"),
                "wavelength_range": requirement.get("wavelength_range"),
                "goal": requirement.get("goal"),
                "threshold": requirement.get("threshold"),
                "maximum": requirement.get("maximum"),
                "confidence_level": requirement.get("confidence_level"),
            }
    raise KeyError(f"Requirement {requirement_id!r} missing from error_budget.yaml")


def _term_freshness(source_ids, source_artifacts, scenario_current=True):
    statuses = [
        source_artifacts[source_id]["producer_provenance_status"]
        for source_id in source_ids
    ]
    if not scenario_current or "stale" in statuses:
        status = "stale"
    elif any(value != "current" for value in statuses):
        status = "unverified"
    else:
        status = "current"
    return {"status": status, "source_artifacts": list(source_ids)}


def _term_provenance(source_ids, source_artifacts):
    return {
        source_id: {
            "artifact_sha256": source_artifacts[source_id]["sha256"],
            "producer_provenance_status": source_artifacts[source_id][
                "producer_provenance_status"
            ],
            "producer_identity": source_artifacts[source_id]["producer_identity"],
        }
        for source_id in source_ids
    }


def _paired_error_rows(residual_path):
    """Yield one scenario's paired residuals at a time to bound memory use."""
    with residual_path.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "scenario_id",
            "assessment_stage",
            "training_mode",
            "regression_degree",
            "scene_index",
            "regime",
            "population_weight",
            "fold",
            "relative_error_percent",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"Residual CSV {residual_path} lacks fields "
                f"{sorted(required - set(reader.fieldnames or ()))!r}"
            )
        rows = iter(reader)
        pending = next(rows, None)
        seen_scenarios = set()
        while pending is not None:
            scenario_id = pending["scenario_id"]
            if scenario_id in seen_scenarios:
                raise ValueError(
                    f"Residual scenario rows are not contiguous: {scenario_id}"
                )
            seen_scenarios.add(scenario_id)
            if pending["assessment_stage"] != "n2bc_only":
                raise ValueError(
                    f"Expected n2bc_only residuals first for {scenario_id}"
                )

            n2bc_rows = {}
            while (
                pending is not None
                and pending["scenario_id"] == scenario_id
                and pending["assessment_stage"] == "n2bc_only"
            ):
                key = (
                    pending["training_mode"],
                    pending["regression_degree"],
                    pending["scene_index"],
                    pending["regime"],
                )
                if key in n2bc_rows:
                    raise ValueError(
                        f"Duplicate n2bc_only residual for {scenario_id}/{key}"
                    )
                error = float(pending["relative_error_percent"])
                if not np.isfinite(error):
                    raise ValueError(
                        f"Non-finite n2bc_only residual for {scenario_id}/{key}"
                    )
                n2bc_rows[key] = {
                    "error_percent": error,
                    "regime": pending["regime"],
                    "fold": int(pending["fold"]),
                    "population_weight": float(pending["population_weight"]),
                }
                pending = next(rows, None)

            if (
                pending is None
                or pending["scenario_id"] != scenario_id
                or pending["assessment_stage"] != "end_to_end"
            ):
                raise ValueError(
                    f"Missing contiguous end_to_end residuals for {scenario_id}"
                )

            paired = {}
            grouped_folds = {}
            while (
                pending is not None
                and pending["scenario_id"] == scenario_id
                and pending["assessment_stage"] == "end_to_end"
            ):
                key = (
                    pending["training_mode"],
                    pending["regression_degree"],
                    pending["scene_index"],
                    pending["regime"],
                )
                if key not in n2bc_rows:
                    raise ValueError(
                        f"Unmatched end_to_end residual for {scenario_id}/{key}"
                    )
                n2bc = n2bc_rows.pop(key)
                error = float(pending["relative_error_percent"])
                fold = int(pending["fold"])
                if not np.isfinite(error):
                    raise ValueError(
                        f"Non-finite end_to_end residual for {scenario_id}/{key}"
                    )
                weight = float(pending["population_weight"])
                if (
                    n2bc["fold"] != fold
                    or n2bc["regime"] != pending["regime"]
                    or n2bc["population_weight"] != weight
                ):
                    raise ValueError(
                        f"Residual regime/fold/weight differs within paired scene "
                        f"{scenario_id}/{key}"
                    )
                model_key = key[:2]
                paired.setdefault(model_key, []).append(
                    {
                        "n2bc_only": n2bc,
                        "end_to_end": {
                            "error_percent": error,
                            "regime": pending["regime"],
                            "fold": fold,
                        },
                        "scene_index": key[2],
                        "population_weight": weight,
                    }
                )
                shared_group = key[:3]
                previous_fold = grouped_folds.setdefault(shared_group, fold)
                if previous_fold != fold:
                    raise ValueError(
                        "Shared clear/cloud Sunny index crosses validation folds: "
                        f"{scenario_id}/{shared_group}"
                    )
                pending = next(rows, None)

            if n2bc_rows:
                raise ValueError(
                    f"Incomplete spectral/end-to-end residual pairs for "
                    f"{scenario_id}: {len(n2bc_rows)} rows unmatched"
                )
            yield scenario_id, paired


def _paired_statistics(records):
    """ABI-population-weighted paired spectral/angular statistics [%].

    Weights are renormalized within the record group; variances and the
    covariance use unbiased reliability-weight estimators (ddof=1 when equal).
    """
    weights = np.asarray(
        [record["population_weight"] for record in records], dtype=np.float64
    )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not weights.sum() > 0:
        raise ValueError("Paired residual population weights must be non-negative")
    weights = weights / weights.sum()
    spectral = np.asarray(
        [record["n2bc_only"]["error_percent"] for record in records],
        dtype=np.float64,
    )
    end_to_end = np.asarray(
        [record["end_to_end"]["error_percent"] for record in records],
        dtype=np.float64,
    )
    if spectral.size < 2:
        raise ValueError("At least two matched scenes are needed for sample covariance")

    angular_increment = end_to_end - spectral
    covariance = np.cov(np.stack((spectral, angular_increment)), aweights=weights)
    total_variance = float(propagate_covariance(np.ones(2), covariance))
    spectral_bias = float(weights @ spectral)
    angular_bias = float(weights @ angular_increment)
    return {
        "sample_count": int(spectral.size),
        "effective_sample_size": float(1.0 / (weights @ weights)),
        "weighting": "ABI scene population",
        "spectral_only": {
            "signed_bias_percent": spectral_bias,
            "standard_deviation_percent": float(np.sqrt(covariance[0, 0])),
            "rmse_percent": float(np.sqrt(weights @ spectral**2)),
        },
        "angular_plus_interaction_increment": {
            "signed_bias_percent": angular_bias,
            "standard_deviation_percent": float(np.sqrt(covariance[1, 1])),
        },
        "covariance_matrix_percent_squared": covariance.tolist(),
        "covariance_spectral_with_angular_increment_percent_squared": float(
            covariance[0, 1]
        ),
        "total_end_to_end": {
            "signed_bias_percent": spectral_bias + angular_bias,
            "standard_deviation_percent": float(np.sqrt(total_variance)),
            "rmse_percent": float(np.sqrt(weights @ end_to_end**2)),
            "k2_scatter_percent": float(2.0 * np.sqrt(total_variance)),
        },
    }


def _paired_terms(
    scenario_id,
    records_by_model,
    scenario_validity,
    scenario_catalog,
    requirements,
    sources,
    residuals_path,
    simulation_reference_range_um,
    geometry,
):
    terms = []
    for (mode, degree), records in sorted(records_by_model.items()):
        groups = {
            "all_regimes": records,
            **{
                regime: [
                    record
                    for record in records
                    if record["n2bc_only"]["regime"] == regime
                ]
                for regime in sorted(
                    {record["n2bc_only"]["regime"] for record in records}
                )
            },
        }
        statistics = {
            group_name: _paired_statistics(group_records)
            for group_name, group_records in groups.items()
        }
        scenario_current = scenario_validity.get(scenario_id, False)
        source_ids = [
            "eco_n2bc_metrics",
            "eco_adm_metrics",
            "eco_chain_metrics",
            "eco_residuals",
        ]
        freshness = _term_freshness(
            source_ids, sources, scenario_current=scenario_current
        )
        terms.append(
            {
                "term_id": (
                    f"sunny_paired_n2bc_adm:{scenario_id}:{mode}:degree_{degree}"
                ),
                "stage": "N2BC application / broadband flux",
                "quantity": (
                    "Held-out relative broadband OLR error decomposition: "
                    "spectral-only plus paired angular-and-interaction increment"
                ),
                "units": "percent of simulated reference OLR",
                "statistic": (
                    "Grouped out-of-fold ABI-population-weighted signed mean, "
                    "standard deviation, RMSE, and paired covariance"
                ),
                "signed_bias": {
                    group: values["total_end_to_end"]["signed_bias_percent"]
                    for group, values in statistics.items()
                },
                "covariance": {
                    "units": "percent squared",
                    "spectral_with_angular_increment": {
                        group: values[
                            "covariance_spectral_with_angular_increment_percent_squared"
                        ]
                        for group, values in statistics.items()
                    },
                    "matrix_order": [
                        "spectral_only",
                        "angular_plus_interaction_increment",
                    ],
                    "matrices_by_scene_domain": {
                        group: values["covariance_matrix_percent_squared"]
                        for group, values in statistics.items()
                    },
                },
                "statistics_by_scene_domain": statistics,
                "scene_geometry_spectral_domain": {
                    "scene_regimes": sorted(
                        {record["n2bc_only"]["regime"] for record in records}
                    ),
                    "record_count": len(records),
                    "shared_scene_index_count": len(
                        {record["scene_index"] for record in records}
                    ),
                    "regression_training_mode": mode,
                    "regression_degree": int(degree),
                    "wavelength_range_um": simulation_reference_range_um,
                    "view_geometry": {
                        "retrieval_view_angles_deg": geometry.get(
                            "retrieval_view_angles_deg"
                        ),
                        "retrieval_note": geometry.get("retrieval_note"),
                    },
                },
                "eco_scenario": {
                    "id": scenario_id,
                    **scenario_catalog.get(scenario_id, {}),
                },
                "aggregation_scale": (
                    "Instantaneous simulated scene; no spatial or temporal "
                    "aggregation has been applied"
                ),
                "requirement_references": [
                    requirements["ObsReq_16"],
                    requirements["ObsReq_15"],
                    requirements["ObsReq_8"],
                ],
                "source_artifact": _display_path(residuals_path),
                "source_artifact_ids": source_ids,
                "provenance_identity": _term_provenance(source_ids, sources),
                "derivation_method": (
                    "Use the same scene index, regression mode/degree, and CV fold. "
                    "Pair individual clear/cloud records by their shared index and "
                    "regime; shared indices are required to retain one CV fold. "
                    "The spectral-only residual uses true integrated band fluxes; "
                    "the end-to-end residual applies the fold-trained model to "
                    "ADM-retrieved band fluxes. Angular increment = end-to-end "
                    "minus spectral-only residual. Each Sunny file is weighted by "
                    "the ABI population mass of its majority-vote ABI-GMM scene "
                    "(renormalized within each scene domain). Propagate the "
                    "weighted 2x2 covariance with J=[1,1]."
                ),
                "direct_applicability_or_transfer": (
                    "Directly evaluated for the configured ECO channel scenario "
                    "within the Sunny simulation. Conditional on simulation "
                    "representativeness, idealized SRFs, noise-free geometry, "
                    "and grouped validation; not flight-performance evidence."
                ),
                "correlation_and_averaging": (
                    "Paired covariance is retained. No independence assumption "
                    "or 1/sqrt(N) scaling is used; scene-level errors are not "
                    "aggregated to monthly/seasonal products."
                ),
                "assumptions_and_gaps": [
                    "Clear/cloud records sharing a four-digit Sunny index are grouped in the same fold.",
                    "ABI scene weighting retargets scene frequencies only; within-scene Sunny distributions are not reweighted, and sparsely populated ABI scenes reduce the effective sample size.",
                    "The angular increment includes ADM error and its interaction with N2BC; it is not an independently isolated physical source.",
                    (
                        "The integrated Sunny reference is "
                        f"{simulation_reference_range_um} um, while ObsReq 8 "
                        f"defines {requirements['ObsReq_8']['wavelength_range']} um."
                    ),
                    "Requirement confidence is unspecified; 2*standard deviation is scatter only, not accuracy or compliance.",
                ],
                "evidence_status": "conditional_simulation",
                "freshness": freshness,
                "scenario_catalog_match": scenario_current,
                "component_definition_ids": [
                    "sunny_n2bc_spectral_only",
                    "sunny_angular_plus_spectral",
                ],
            }
        )
    return terms


def _adm_terms(
    adm_metrics, requirements, sources, adm_metrics_path, scenario_validity
):
    terms = []
    for scenario_id, channels in adm_metrics.get("scenarios", {}).items():
        for channel_name, channel_data in channels.items():
            errors = channel_data["narrowband_flux_error_percent"]
            for scene_domain, summary in {
                "all_regimes": errors["overall"],
                **errors.get("by_regime", {}),
            }.items():
                source_ids = ["eco_adm_metrics"]
                scenario_current = scenario_validity.get(scenario_id, False)
                freshness = _term_freshness(
                    source_ids, sources, scenario_current=scenario_current
                )
                terms.append(
                    {
                        "term_id": (
                            f"sunny_adm_band_flux:{scenario_id}:{channel_name}:"
                            f"{scene_domain}"
                        ),
                        "stage": "ADM angular conversion / narrowband flux",
                        "quantity": "ADM-retrieved narrowband band-flux relative error",
                        "units": "percent of true simulated band flux",
                        "statistic": (
                            "Signed mean bias, sample standard deviation, and RMSE"
                        ),
                        "signed_bias": summary["bias_percent"],
                        "covariance": None,
                        "statistics": {
                            "sample_count": summary["n_spectra"],
                            "standard_deviation_percent": summary[
                                "standard_deviation_percent"
                            ],
                            "rmse_percent": summary["rmse_percent"],
                            "k2_scatter_percent": summary["k2_scatter_percent"],
                        },
                        "scene_geometry_spectral_domain": {
                            "scene_domain": scene_domain,
                            "scene_index_count": int(adm_metrics["scene_index_count"]),
                            "channel": channel_name,
                            "view_geometry": adm_metrics["geometry"].get(
                                "retrieval_note"
                            ),
                        },
                        "eco_scenario": {"id": scenario_id},
                        "aggregation_scale": (
                            "Instantaneous simulated scene/channel; no spatial or "
                            "temporal aggregation"
                        ),
                        "requirement_references": [
                            requirements["ObsReq_15"],
                            requirements["ObsReq_12"],
                        ],
                        "source_artifact": _display_path(adm_metrics_path),
                        "source_artifact_ids": source_ids,
                        "provenance_identity": _term_provenance(
                            source_ids, sources
                        ),
                        "derivation_method": (
                            "Compare simulated ADM-retrieved ECO channel flux with "
                            "the true hemispheric channel flux; summarize relative "
                            "error by clear/cloud regime."
                        ),
                        "direct_applicability_or_transfer": (
                            "ECO channel-scenario simulation, conditional on Sunny "
                            "representativeness, assumed SRFs, and the configured "
                            "15-view noise-free case. Band-flux diagnostic is not "
                            "itself the instantaneous broadband ObsReq 15 metric."
                        ),
                        "correlation_and_averaging": (
                            "No channel covariance is available in the source summary; "
                            "do not combine channels independently."
                        ),
                        "assumptions_and_gaps": [
                            "The 15-view case does not represent the full ECO N=1..20 geometry/noise distribution.",
                            "No scene-classification or co-registration error is included.",
                        ],
                        "evidence_status": "conditional_simulation",
                        "freshness": freshness,
                        "component_definition_id": "sunny_adm_narrowband_diagnostic",
                        "scenario_catalog_match": scenario_current,
                    }
                )
    return terms


def _abi_proxy_terms(abi_summary, requirements, sources, abi_summary_path):
    terms = []
    for group in abi_summary.get("groups", []):
        summary = group["shape_parameter_b"]
        source_ids = ["abi_adm_proxy_summary"]
        terms.append(
            {
                "term_id": (
                    f"abi_between_day_adm:{group['abi_channel']}:"
                    f"proxy_scene_{group['proxy_scene']}"
                ),
                "stage": "ADM angular conversion",
                "quantity": "Between-day spread of ABI scene-average ADM shape parameter",
                "units": "dimensionless ADM shape parameter b",
                "statistic": "Between-day sample standard deviation of fitted daily values",
                "signed_bias": None,
                "covariance": None,
                "statistics": {
                    "sample_count_days": group["n_days"],
                    "mean": summary["mean"],
                    "standard_deviation": summary["between_day_standard_deviation"],
                    "minimum": summary["minimum"],
                    "maximum": summary["maximum"],
                },
                "scene_geometry_spectral_domain": {
                    "proxy_scene": group["proxy_scene"],
                    "abi_channel": group["abi_channel"],
                    "viewing_angles_deg": abi_summary["configuration"][
                        "viewing_angles_deg"
                    ],
                    "spatial_resolution_km": abi_summary["configuration"][
                        "resolution_km"
                    ],
                    "days": abi_summary["configuration"]["days"],
                },
                "eco_scenario": None,
                "aggregation_scale": (
                    "Between-day variability of scene-average ABI ADM fit; not "
                    "within-scene pixel spread"
                ),
                "requirement_references": [
                    requirements["ObsReq_12"],
                    requirements["ObsReq_15"],
                    requirements["ObsReq_17"],
                ],
                "source_artifact": _display_path(abi_summary_path),
                "source_artifact_ids": source_ids,
                "provenance_identity": _term_provenance(source_ids, sources),
                "derivation_method": (
                    "Reuse the existing stratified ABI proxy summary; no flux-space "
                    "uncertainty is inferred from parameter spread."
                ),
                "direct_applicability_or_transfer": (
                    "Proxy evidence only. GOES channel response, two fixed views, "
                    "and proxy scene classes are not validated transfers to ECO."
                ),
                "correlation_and_averaging": (
                    "No fit covariance is retained. Do not combine with Sunny/ECO "
                    "simulation terms or treat between-day spread as an ECO prior."
                ),
                "assumptions_and_gaps": [
                    "Daily scene-average fits do not measure within-scene pixel-level variability.",
                    "The ABI-to-ECO channel, scene, and geometry mapping is unvalidated.",
                ],
                "evidence_status": "empirical_proxy_not_ECO_uncertainty",
                "freshness": _term_freshness(source_ids, sources),
                "component_definition_id": "abi_adm_proxy_between_day",
            }
        )
    return terms


def _gap_records(error_budget, requirements):
    records = []
    for gap_id, gap in error_budget.get("unquantified_terms", {}).items():
        requirement_ids = gap["requirement_ids"]
        records.append(
            {
                "term_id": f"gap:{gap_id}",
                "stage": gap["stage"],
                "quantity": gap["quantity"],
                "units": gap.get("units"),
                "statistic": "not estimated; missing is not zero",
                "signed_bias": None,
                "covariance": None,
                "source_artifact": None,
                "source_artifact_status": "missing_or_not_generated",
                "provenance_identity": None,
                "derivation_method": "No validated estimate is available in the configured workflow artifacts.",
                "scene_geometry_spectral_domain": None,
                "eco_scenario": "all scenarios unless otherwise stated",
                "aggregation_scale": "not estimated",
                "requirement_references": [
                    requirements[requirement_id] for requirement_id in requirement_ids
                ],
                "direct_applicability_or_transfer": (
                    "Not estimated for ECO. Do not infer a value from missing evidence."
                ),
                "correlation_and_averaging": (
                    "Unknown; no independence or averaging reduction assumed."
                ),
                "assumptions_and_gaps": [gap["next_input"]],
                "evidence_status": "unquantified",
                "missing_is_zero": False,
            }
        )
    return records


def _scenario_checks(
    expected_ids, scenario_catalog, default_edge_slope, *metric_documents
):
    expected = set(expected_ids)
    checks = {}
    fields = (
        "source",
        "description",
        "channel_names",
        "channel_bands_um",
        "response_model",
        "edge_slope_per_um",
    )
    for name, document, has_scenario_metadata in metric_documents:
        observed = set(document.get("scenarios", {}))
        mismatched_definitions = []
        if has_scenario_metadata:
            for scenario_id in sorted(expected & observed):
                expected_config = scenario_catalog[scenario_id]
                actual_config = document["scenarios"][scenario_id]
                expected_values = {
                    field: expected_config.get(field) for field in fields
                }
                expected_values["edge_slope_per_um"] = expected_config.get(
                    "edge_slope_per_um", default_edge_slope
                )
                actual_values = {
                    field: actual_config.get(field) for field in fields
                }
                if actual_values != expected_values:
                    mismatched_definitions.append(scenario_id)
        checks[name] = {
            "matches_catalog": (
                observed == expected and not mismatched_definitions
            ),
            "missing_scenarios": sorted(expected - observed),
            "unexpected_scenarios": sorted(observed - expected),
            "scenario_definition_mismatches": mismatched_definitions,
        }
    return checks


def _stage_assessments(requirements):
    stages = [
        {
            "stage": "Radiances",
            "uncertainty_sources": (
                "Camera noise, absolute calibration, channel response, and input "
                "radiance convention."
            ),
            "existing_evidence": (
                "GOES-16/18 ABI real-scene inputs support proxy algorithm behavior; "
                "Sunny supplies simulated directional radiances and hemispheric spectra."
            ),
            "eco_interpretation": (
                "ABI is term-specific proxy evidence; Sunny is simulation evidence, "
                "not ECO flight radiometry."
            ),
            "downstream_propagation": (
                "No ECO camera noise/calibration covariance is propagated through "
                "scene selection or flux retrieval."
            ),
            "requirement_ids": ["MeasReq_8", "MeasReq_9", "ObsReq_15"],
            "gap_ids": ["radiance_noise_and_calibration"],
        },
        {
            "stage": "Scene identification / co-registration",
            "uncertainty_sources": (
                "Scene misclassification, CTH/parallax, inter-view registration, "
                "and scene evolution between views."
            ),
            "existing_evidence": (
                "ABI GMM scene identification is implemented; ABI ADM summaries are "
                "stratified by proxy scene but do not quantify classification errors."
            ),
            "eco_interpretation": (
                "No validated ECO scene labels or multi-view co-registration error "
                "distribution is available."
            ),
            "downstream_propagation": (
                "Wrong scene/pixel assignment can select an inappropriate ADM and "
                "alter the narrowband inputs; no output-space term is quantified."
            ),
            "requirement_ids": ["ObsReq_12", "ObsReq_15", "ObsReq_17", "MeasReq_3"],
            "gap_ids": [
                "scene_id_cth_parallax_coregistration",
                "temporal_coincidence",
            ],
        },
        {
            "stage": "ADM angular conversion",
            "uncertainty_sources": (
                "ADM residual/model form, view geometry and count, scene population, "
                "measurement noise, and fallback/prior behavior."
            ),
            "existing_evidence": (
                "Sunny gives per-scene/channel band-flux residuals for 15 noise-free "
                "views; ABI contributes between-day scene-average parameter spread."
            ),
            "eco_interpretation": (
                "Sunny results are conditional on the simulated population and ideal "
                "geometry; ABI spread is not an ECO prior or flux uncertainty."
            ),
            "downstream_propagation": (
                "Sunny retrieved band fluxes feed the paired ObsReq 15 end-to-end "
                "residual; paired covariance with spectral error is retained."
            ),
            "requirement_ids": ["ObsReq_12", "ObsReq_15", "ObsReq_17"],
            "gap_ids": ["adm_population_and_model_form"],
        },
        {
            "stage": "Narrowband fluxes",
            "uncertainty_sources": (
                "Angular retrieval error, radiance noise/calibration, spectral "
                "response uncertainty, and channel covariance."
            ),
            "existing_evidence": (
                "Sunny ADM stage metrics report channel/regime bias, sample spread, "
                "and RMSE; ABI processing writes corrected narrowband BT inputs."
            ),
            "eco_interpretation": (
                "Sunny band-flux errors are conditional simulation diagnostics; ABI "
                "narrowband products are proxy workflow evidence."
            ),
            "downstream_propagation": (
                "The ADM-retrieved ECO band fluxes are passed through the same "
                "fold-trained N2BC models in the paired end-to-end score."
            ),
            "requirement_ids": ["MeasReq_8", "ObsReq_15", "ObsReq_16"],
            "gap_ids": ["channel_and_shared_source_covariance"],
        },
        {
            "stage": "N2BC application",
            "uncertainty_sources": (
                "Spectral fit residual, regression/model selection, assumed SRFs, "
                "reference spectral interval, and regime dependence."
            ),
            "existing_evidence": (
                "Grouped held-out predictions from true band fluxes quantify "
                "spectral-only error; applying the same models to ADM fluxes gives "
                "paired end-to-end error."
            ),
            "eco_interpretation": (
                "Spectral-only maps to ObsReq 16; angular-plus-spectral maps to "
                "ObsReq 15. Both remain conditional Sunny/SRF evidence."
            ),
            "downstream_propagation": (
                "Signed component biases add; sample spread is propagated with the "
                "observed spectral/angular-increment covariance, not independence."
            ),
            "requirement_ids": ["ObsReq_8", "ObsReq_15", "ObsReq_16"],
            "gap_ids": ["narrowband_srf_and_reference_domain"],
        },
        {
            "stage": "Broadband flux",
            "uncertainty_sources": (
                "N2BC output, camera-to-WFOV anchoring, absolute validation, and "
                "spectral-domain definition."
            ),
            "existing_evidence": (
                "Sunny supports simulated OLR residuals; `all_goes_proxy` supports "
                "the separate ABI broadband chain."
            ),
            "eco_interpretation": (
                "Sunny broadband values integrate 2.5-500 um, whereas ObsReq 8 "
                "defines 4-100 um; the mismatch is unresolved."
            ),
            "downstream_propagation": (
                "No ECO WFOV anchoring residual or independent CERES absolute "
                "validation is propagated."
            ),
            "requirement_ids": ["ObsReq_8", "ObsReq_14", "ObsReq_15", "MeasReq_9"],
            "gap_ids": [
                "broadband_wfov_anchoring",
                "independent_validation",
            ],
        },
        {
            "stage": "Aggregation",
            "uncertainty_sources": (
                "Spatial/temporal sampling, correlated errors, representativeness, "
                "resolution, and long-term stability."
            ),
            "existing_evidence": (
                "The ABI proxy has a monthly aggregation workflow; this is not an "
                "ECO sampling/stability uncertainty estimate."
            ),
            "eco_interpretation": (
                "No ECO monthly/seasonal covariance or mission sampling model is "
                "available; do not assume automatic 1/sqrt(N) improvement."
            ),
            "downstream_propagation": (
                "Instantaneous Sunny errors are not propagated to ObsReq 10/11 "
                "aggregation scales."
            ),
            "requirement_ids": ["ObsReq_8", "ObsReq_9", "ObsReq_10", "ObsReq_11"],
            "gap_ids": [
                "aggregation_sampling_covariance",
                "olr_stability",
            ],
        },
    ]
    for stage in stages:
        stage["requirement_references"] = [
            requirements[requirement_id] for requirement_id in stage["requirement_ids"]
        ]
    return stages


def _render_report(registry):
    simulation_range = registry["requirement_conventions"][
        "Sunny_simulation_reference_um"
    ]
    requirement_range = registry["requirement_conventions"]["ObsReq_8_reference_um"]
    simulation_domain_text = (
        f"{simulation_range[0]}–{simulation_range[1]} µm"
        if simulation_range
        else "unreported Sunny interval"
    )
    requirement_domain_text = (
        f"{requirement_range[0]}–{requirement_range[1]} µm"
        if requirement_range
        else "unreported ObsReq 8 interval"
    )
    lines = [
        "# ECO LW uncertainty registry and gaps",
        "",
        f"- Registry schema: `{registry['schema_version']}`",
        "- Scope: ECO MO2 LW processing evidence; no total uncertainty or compliance claim.",
        "- Missing evidence is not zero. Components are not combined unless a paired covariance is measured.",
        "",
        "## Evidence freshness and provenance",
        "",
        "| Artifact | SHA-256 | Producer provenance |",
        "|---|---|---|",
    ]
    for name, artifact in registry["source_artifacts"].items():
        lines.append(
            f"| `{name}` (`{artifact['path']}`) | `{artifact['sha256'][:16]}…` "
            f"| {artifact['producer_provenance_status']} |"
        )

    lines.extend(
        [
            "",
            "Scenario catalog comparison:",
            "",
            "| Artifact | Matches current catalog | Missing | Unexpected | Definition mismatch |",
            "|---|---:|---|---|---|",
        ]
    )
    for name, check in registry["scenario_catalog"]["checks"].items():
        lines.append(
            f"| `{name}` | {check['matches_catalog']} | "
            f"{', '.join(check['missing_scenarios']) or '—'} | "
            f"{', '.join(check['unexpected_scenarios']) or '—'} | "
            f"{', '.join(check['scenario_definition_mismatches']) or '—'} |"
        )

    lines.extend(
        [
            "",
            "## Stage-ordered assessment",
            "",
            "| Retrieval stage | Uncertainty sources and evidence | ECO interpretation and downstream propagation | Requirement(s) / unresolved gaps |",
            "|---|---|---|---|",
        ]
    )
    for stage in registry["stage_assessment"]:
        evidence = (
            f"{stage['uncertainty_sources']} Existing evidence: "
            f"{stage['existing_evidence']}"
        ).replace("|", "\\|")
        interpretation = (
            f"{stage['eco_interpretation']} Propagation: "
            f"{stage['downstream_propagation']}"
        ).replace("|", "\\|")
        requirement_ids = ", ".join(stage["requirement_ids"])
        gap_ids = ", ".join(stage["gap_ids"])
        lines.append(
            f"| {stage['stage']} | {evidence} | {interpretation} | "
            f"{requirement_ids}; gaps: {gap_ids} |"
        )

    abi_assessment = registry["abi_production_assessment"]
    abi_products = abi_assessment["product_outputs"]
    abi_sensitivity = abi_assessment["uncertainty_assessments"][0]
    lines.extend(
        [
            "",
            "## ABI production products and linked sensitivity evidence",
            "",
            f"The registry is built with {len(abi_products['narrowband_fluxes'])} "
            "narrowband-flux files, "
            f"{len(abi_products['daily_broadband_fluxes'])} daily broadband-flux "
            "files, and monthly product "
            f"`{abi_products['monthly_flux']}` as Snakemake dependencies. These "
            "are GOES ABI demonstration products, not ECO products.",
            "",
            f"The linked sensitivity assessment `{abi_sensitivity['assessment_id']}` "
            f"contains {len(abi_sensitivity['experiment_ids'])} one-at-a-time "
            "model-choice variants. ABI variants are paired over every native pixel "
            "of the configured days and propagated to monthly 10 km cell means "
            f"(daily: `{abi_sensitivity['daily_broadband_uncertainty_status']}`, "
            f"monthly: `{abi_sensitivity['monthly_uncertainty_status']}`); Sunny "
            "variants are conditional ECO simulation evidence. Sensitivities are not "
            "summed. Missing product-level uncertainties are not zero.",
            "",
            "Requirement links indicate scientific relevance for the experiment "
            "set only; they are not compliance scores or approved error allocations.",
            "",
            "Product-level gaps:",
        ]
    )
    for gap in abi_assessment["product_level_uncertainty_gaps"]:
        lines.append(
            f"- **{gap['product']}** ({gap['status']}): {gap['next_step']}"
        )
    nominal_assessment = registry["conditional_nominal_assessment"]
    if nominal_assessment is not None:
        statistics = nominal_assessment["combined_statistics_w_m2"]
        lines.extend(
            [
                "",
                "## Conditional ECO/Sunny nominal assessment",
                "",
                f"Scenario `{nominal_assessment['scenario']}`, "
                f"reference domain {nominal_assessment['reference_domain_um']} um; "
                f"conditional bias {statistics.get('bias_w_m2')} W m-2, "
                f"scene SD {statistics.get('scene_sd_w_m2')} W m-2, "
                f"random SD {statistics.get('random_sd_w_m2')} W m-2, and "
                f"RMSE {statistics.get('ensemble_rmse_w_m2')} W m-2.",
                "",
                nominal_assessment["interpretation"],
                "",
            ]
        )

    taxonomy = registry.get("scene_taxonomy_assessment")
    if taxonomy is not None:
        lines.extend(
            [
                "",
                "## Scene taxonomy component count",
                "",
                f"Reviewed K = {taxonomy['n_components']} ({taxonomy['feature_set']}; "
                f"diagnostics SHA-256 `{taxonomy['diagnostics_csv_sha256'][:12]}`); "
                f"rationale: {taxonomy['selection']['rationale']}",
                "",
                f"Maximum |K+-1 change|: {json.dumps(taxonomy['max_abs_neighbour_delta'])}"
                + (f"; untested neighbours {taxonomy['missing_neighbours']}"
                   if taxonomy["missing_neighbours"] else ""),
                "",
                taxonomy["interpretation"],
                "",
            ]
        )

    paired_terms = [
        term
        for term in registry["quantified_terms"]
        if term["term_id"].startswith("sunny_paired_n2bc_adm:")
        and "all_regimes" in term["statistics_by_scene_domain"]
    ]
    lines.extend(
        [
            "",
            "## Paired Sunny OLR residual evidence",
            "",
            "The spectral-only term uses true integrated band fluxes (ObsReq 16); "
            "end-to-end applies those fold-trained models to ADM-retrieved fluxes "
            "(ObsReq 15). The incremental error is end-to-end minus spectral-only. "
            "Covariance is retained. These are conditional sample statistics, not "
            "accuracy/compliance; the RfMA confidence convention is unresolved.",
            "",
            "| ECO scenario | Train mode / degree | Spectral bias / SD / RMSE (%) | Angular+interaction bias / SD (%) | Covariance (%²) | End-to-end bias / SD / RMSE (%) | Clear end-to-end bias / SD / RMSE (%) | Cloud end-to-end bias / SD / RMSE (%) | Freshness |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for term in paired_terms:
        overall = term["statistics_by_scene_domain"]["all_regimes"]
        clear = term["statistics_by_scene_domain"].get("clear_sky", {})
        cloud = term["statistics_by_scene_domain"].get("cloudy", {})
        spectral = overall["spectral_only"]
        angular = overall["angular_plus_interaction_increment"]
        total = overall["total_end_to_end"]
        covariance = overall[
            "covariance_spectral_with_angular_increment_percent_squared"
        ]
        lines.append(
            f"| `{term['eco_scenario']['id']}` | "
            f"{term['scene_geometry_spectral_domain']['regression_training_mode']} / "
            f"{term['scene_geometry_spectral_domain']['regression_degree']} | "
            f"{spectral['signed_bias_percent']:.3f} / "
            f"{spectral['standard_deviation_percent']:.3f} / "
            f"{spectral['rmse_percent']:.3f} | "
            f"{angular['signed_bias_percent']:.3f} / "
            f"{angular['standard_deviation_percent']:.3f} | {covariance:.4f} | "
            f"{total['signed_bias_percent']:.3f} / "
            f"{total['standard_deviation_percent']:.3f} / "
            f"{total['rmse_percent']:.3f} | "
            f"{clear.get('total_end_to_end', {}).get('signed_bias_percent', float('nan')):.3f} / "
            f"{clear.get('total_end_to_end', {}).get('standard_deviation_percent', float('nan')):.3f} / "
            f"{clear.get('total_end_to_end', {}).get('rmse_percent', float('nan')):.3f} | "
            f"{cloud.get('total_end_to_end', {}).get('signed_bias_percent', float('nan')):.3f} / "
            f"{cloud.get('total_end_to_end', {}).get('standard_deviation_percent', float('nan')):.3f} / "
            f"{cloud.get('total_end_to_end', {}).get('rmse_percent', float('nan')):.3f} | "
            f"{term['freshness']['status']} |"
        )

    lines.extend(
        [
            "",
            f"The simulated reference spans {simulation_domain_text}; "
            f"ObsReq 8 defines {requirement_domain_text}. "
            "Neither domain is silently substituted for the other. No 1/√N aggregation "
            "or monthly/seasonal propagation is made.",
            "",
            "## ADM narrowband diagnostics and ABI proxy",
            "",
            "ADM band-flux diagnostics and ABI between-day scene-average parameter spreads "
            "are retained in the JSON registry. ABI terms remain proxy-only and are not "
            "transferred to ECO or combined with Sunny evidence.",
            "",
            "## Unquantified terms",
            "",
            "| Retrieval stage | Missing quantity | Requirement(s) | Next scientific input |",
            "|---|---|---|---|",
        ]
    )
    for gap in registry["unquantified_terms"]:
        refs = ", ".join(
            requirement["id"] for requirement in gap["requirement_references"]
        )
        next_input = "; ".join(gap["assumptions_and_gaps"]).replace("|", "\\|")
        lines.append(
            f"| {gap['stage']} | {gap['quantity']} | {refs} | {next_input} |"
        )
    lines.extend(
        [
            "",
            "Quantified terms are algorithm residuals or conditional simulation/proxy "
            "estimates. The registry does not allocate the mission error budget, infer "
            "a total ECO uncertainty, or claim requirement compliance or mission-wide SRL.",
            "",
        ]
    )
    return "\n".join(lines)


def _abi_production_assessment(
    narrowband_paths,
    daily_paths,
    monthly_path,
    sensitivity_path,
    requirements,
):
    """Register ABI product outputs and the production-aligned (schema 2)
    sensitivity evidence without treating either as an ECO product uncertainty.
    """
    product_groups = {
        "narrowband_fluxes": narrowband_paths,
        "daily_broadband_fluxes": daily_paths,
    }
    missing_products = [
        str(path)
        for paths in product_groups.values()
        for path in paths
        if not Path(path).is_file()
    ]
    if monthly_path is not None and not Path(monthly_path).is_file():
        missing_products.append(str(monthly_path))
    if missing_products:
        raise FileNotFoundError(
            "ABI production products are missing: " + ", ".join(missing_products)
        )

    sensitivity = None
    sensitivity_artifact = None
    if sensitivity_path is not None:
        if not Path(sensitivity_path).is_file():
            raise FileNotFoundError(
                f"ABI/Sunny sensitivity report is missing: {sensitivity_path}"
            )
        sensitivity = _load_json(sensitivity_path)
        if sensitivity.get("schema_version") != 2:
            raise ValueError("ABI/Sunny sensitivity report must use schema_version 2")
        sensitivity_artifact = _artifact_record(
            sensitivity_path, sensitivity.get("provenance")
        )
    measured = sensitivity is not None
    abi_variants = sensitivity["abi"]["variants"] if measured else {}
    sunny_variants = sensitivity["sunny"]["variants"] if measured else {}

    def abi_summary(entry):
        return {
            "source_group": entry["source_group"],
            "description": entry["description"],
            "daily_pixel_paired_difference_w_m2": {
                split: {sat: entry["splits"][split][sat]["pixel"]
                        for sat in entry["splits"][split]}
                for split in ("held_in", "held_out")
            },
            "monthly_10km_paired_difference_w_m2": entry["monthly_10km"],
        }

    def sunny_summary(entry):
        if entry["status"] != "quantified":
            return {key: entry[key] for key in ("source_group", "description", "status", "reason")}
        return {
            "source_group": entry["source_group"],
            "description": entry["description"],
            "status": entry["status"],
            "paired_difference_w_m2": {
                key: entry["paired_difference_w_m2"][key]
                for key in ("bias_w_m2", "scene_sd_w_m2", "class_rmse_w_m2")
            },
            "label_change_fraction": entry["label_change_fraction"],
        }

    assessment = {
        "assessment_id": "production_aligned_sensitivity",
        "evidence_status": (
            "paired_model_choice_sensitivity_full_population" if measured else "not_run"
        ),
        "source_artifact_id": (
            "abi_proxy_sensitivity" if sensitivity_artifact else None
        ),
        "scope": sensitivity.get("scope") if measured else None,
        "experiment_ids": sorted(
            [f"abi_{name}" for name in abi_variants]
            + [f"sunny_{name}" for name in sunny_variants]
        ),
        "abi_variants": {name: abi_summary(entry) for name, entry in abi_variants.items()},
        "sunny_variants": {name: sunny_summary(entry) for name, entry in sunny_variants.items()},
        "single_source_terms": sensitivity.get("single_source_terms") if measured else None,
        "estimated_quantity": (
            "One-at-a-time paired broadband-flux changes (variant minus production) "
            "over every native ABI pixel of the configured days, aggregated to daily "
            "pixels and monthly 10 km cell means; and paired changes of the nominal "
            "Sunny/ECO grouped held-out estimator."
        ),
        "requirement_references": [
            requirements[requirement_id]
            for requirement_id in (
                "ObsReq_9",
                "ObsReq_12",
                "ObsReq_15",
                "ObsReq_16",
                "ObsReq_17",
                "MeasReq_8",
            )
            if requirement_id in requirements
        ],
        "requirement_relationship": (
            "Relevance/traceability across the experiment set only; these "
            "sensitivities are not requirement-compliance scores."
        ),
        "transfer_statement": (
            "ABI variants are model-choice sensitivities of the ABI demonstration "
            "products (two-view ABI geometry and channels). They transfer to ECO only "
            "through the matching Sunny/ECO variant. Sensitivities are not "
            "uncertainties of record and are not summed."
        ),
        "daily_broadband_uncertainty_status": (
            "paired_model_choice_sensitivity_on_complete_products" if measured
            else "not_propagated_to_complete_products"
        ),
        "monthly_uncertainty_status": (
            "paired_model_choice_sensitivity_on_monthly_10km_means" if measured
            else "not_estimated"
        ),
    }
    return {
        "product_outputs": {
            "narrowband_fluxes": [_display_path(path) for path in narrowband_paths],
            "daily_broadband_fluxes": [_display_path(path) for path in daily_paths],
            "monthly_flux": _display_path(monthly_path) if monthly_path else None,
            "lineage_note": (
                "These files are dependencies of the registry rule in the "
                "Snakemake DAG; they are registered by path, not re-hashed here."
            ),
        },
        "uncertainty_assessments": [assessment],
        "product_level_uncertainty_gaps": [
            {
                "product": "daily broadband flux",
                "status": assessment["daily_broadband_uncertainty_status"],
                "missing_is_zero": False,
                "next_step": (
                    "Model-choice sensitivities are measured on the complete products; "
                    "statistical terms (radiometric, spatial, assignment) have a single "
                    "source in the nominal Sunny/ECO budget and are not yet mapped per "
                    "ABI pixel. A covariance model is needed before any combination."
                ),
            },
            {
                "product": "monthly flux",
                "status": assessment["monthly_uncertainty_status"],
                "missing_is_zero": False,
                "next_step": (
                    "Temporal sampling (diurnal, missing slots) and inter-day error "
                    "correlation are not estimated; do not assume a 1/sqrt(N) "
                    "reduction of daily statistical terms."
                ),
            },
        ],
        "sensitivity_artifact": sensitivity_artifact,
    }


def _nominal_budget_assessment(report_path, requirements):
    if report_path is None:
        return None
    if not Path(report_path).is_file():
        raise FileNotFoundError(f"Conditional nominal budget is missing: {report_path}")
    report = _load_json(report_path)
    combined_rows = [
        row for row in report.get("rows", []) if row.get("id") == "combined"
    ]
    if len(combined_rows) != 1:
        raise ValueError(
            f"Expected one combined row in nominal budget report {report_path}"
        )
    combined = combined_rows[0]
    artifact = _artifact_record(report_path, report.get("provenance"))
    return {
        "assessment_id": "scene_conditioned_nominal_budget",
        "source_artifact": artifact,
        "scope": report.get("settings", {}).get("scope"),
        "scenario": report.get("settings", {}).get("scenario"),
        "reference_domain_um": report.get("settings", {}).get("reference_domain_um"),
        "combined_statistics_w_m2": {
            key: combined[key]
            for key in (
                "bias_w_m2",
                "scene_sd_w_m2",
                "random_sd_w_m2",
                "ensemble_rmse_w_m2",
            )
            if key in combined
        },
        "requirement_references": [
            requirements[requirement_id]
            for requirement_id in ("ObsReq_9", "ObsReq_15", "ObsReq_16")
            if requirement_id in requirements
        ],
        "evidence_status": "conditional_transfer_and_independence_hypotheses",
        "interpretation": (
            "This is a conditional ECO/Sunny estimate using ABI-derived scene "
            "assignment and spatial evidence under explicit transfer and "
            "independence hypotheses. It is not uncertainty propagated over "
            "the complete ABI daily/monthly production products, a complete "
            "ECO product uncertainty, or a requirement-compliance result."
        ),
    }


def _scene_taxonomy_assessment(gate_path):
    """Reviewed scene-taxonomy K and its K+-1 structural sensitivity (not propagated)."""
    if gate_path is None:
        return None
    gate = _load_json(gate_path)
    structural = gate["structural_taxonomy_choice"]
    return {
        "assessment_id": "scene_taxonomy_component_count",
        "source_artifact": _artifact_record(gate_path),
        "feature_set": gate["feature_set"],
        "n_components": gate["n_components"],
        "selection": gate["selection"],
        "diagnostics_csv": gate["diagnostics_csv"],
        "diagnostics_csv_sha256": gate["diagnostics_csv_sha256"],
        "neighbours": structural["neighbours"],
        "missing_neighbours": structural["missing_neighbours"],
        "neighbour_deltas": structural["deltas"],
        "max_abs_neighbour_delta": structural["max_abs_delta"],
        "evidence_status": structural["status"],
        "interpretation": (
            "K is a reviewed manual choice shared by ABI production and the "
            "nominal budget. K+-1 diagnostic changes bound a structural "
            "classification-choice sensitivity; they are not a statistical "
            "error. Their flux impact is measured by the production-aligned "
            "sensitivity study (components_minus_1/components_plus_1)."
        ),
    }


def build_registry(
    *,
    metrics_path=DEFAULT_METRICS,
    n2bc_metrics_path=DEFAULT_N2BC_METRICS,
    adm_metrics_path=DEFAULT_ADM_METRICS,
    abi_summary_path=DEFAULT_ABI_SUMMARY,
    residuals_path=DEFAULT_RESIDUALS,
    scenarios_path=DEFAULT_SCENARIOS,
    error_budget_path=DEFAULT_ERROR_BUDGET,
    abi_narrowband_paths=(),
    abi_daily_paths=(),
    abi_monthly_path=None,
    abi_sensitivity_path=None,
    nominal_budget_path=None,
    scene_taxonomy_gate_path=None,
):
    metrics = _load_json(metrics_path)
    n2bc_metrics = _load_json(n2bc_metrics_path)
    adm_metrics = _load_json(adm_metrics_path)
    abi_summary = _load_json(abi_summary_path)
    with scenarios_path.open() as source:
        scenario_document = yaml.safe_load(source)
    with error_budget_path.open() as source:
        error_budget = yaml.safe_load(source)

    scenario_catalog = scenario_document["scenarios"]
    scenario_ids = sorted(scenario_catalog)
    requirements = {
        requirement_id: _requirement_reference(error_budget, requirement_id)
        for requirement_id in (
            "ObsReq_8",
            "ObsReq_9",
            "ObsReq_10",
            "ObsReq_11",
            "ObsReq_12",
            "ObsReq_14",
            "ObsReq_15",
            "ObsReq_16",
            "ObsReq_17",
            "MeasReq_3",
            "MeasReq_7",
            "MeasReq_8",
            "MeasReq_9",
        )
    }

    source_artifacts = {
        "eco_chain_metrics": _artifact_record(
            metrics_path, metrics.get("provenance")
        ),
        "eco_n2bc_metrics": _artifact_record(
            n2bc_metrics_path, n2bc_metrics.get("provenance")
        ),
        "eco_adm_metrics": _artifact_record(
            adm_metrics_path, adm_metrics.get("provenance")
        ),
        "abi_adm_proxy_summary": _artifact_record(
            abi_summary_path, abi_summary.get("provenance")
        ),
    }
    residual_record = _artifact_record(residuals_path)
    expected_residual = metrics.get("derived_residuals", {})
    if expected_residual.get("sha256") == residual_record["sha256"]:
        residual_record["producer_provenance_status"] = (
            source_artifacts["eco_chain_metrics"]["producer_provenance_status"]
        )
    elif expected_residual:
        residual_record["producer_provenance_status"] = "stale"
        residual_record["producer_provenance_mismatches"] = ["sha256"]
    else:
        residual_record["producer_provenance_status"] = (
            "provenance_missing_or_unsupported"
        )
    source_artifacts["eco_residuals"] = residual_record
    abi_assessment = _abi_production_assessment(
        abi_narrowband_paths,
        abi_daily_paths,
        abi_monthly_path,
        abi_sensitivity_path,
        requirements,
    )
    nominal_assessment = _nominal_budget_assessment(
        nominal_budget_path, requirements
    )
    if abi_assessment["sensitivity_artifact"] is not None:
        source_artifacts["abi_proxy_sensitivity"] = abi_assessment[
            "sensitivity_artifact"
        ]
    if nominal_assessment is not None:
        source_artifacts["conditional_nominal_budget"] = nominal_assessment[
            "source_artifact"
        ]

    scenario_checks = _scenario_checks(
        scenario_ids,
        scenario_catalog,
        scenario_document["default_edge_slope_per_um"],
        ("eco_chain_metrics", metrics, True),
        ("eco_n2bc_metrics", n2bc_metrics, True),
        ("eco_adm_metrics", adm_metrics, False),
    )
    scenario_validity = {
        scenario_id: all(
            scenario_checks[name]["matches_catalog"]
            for name in ("eco_chain_metrics", "eco_n2bc_metrics")
        )
        for scenario_id in scenario_ids
    }
    paired_terms = []
    observed_residual_scenarios = set()
    for scenario_id, records_by_model in _paired_error_rows(residuals_path):
        observed_residual_scenarios.add(scenario_id)
        paired_terms.extend(
            _paired_terms(
                scenario_id,
                records_by_model,
                scenario_validity,
                scenario_catalog,
                requirements,
                source_artifacts,
                residuals_path,
                metrics.get("reference_data", {}).get("wavelength_range_um"),
                adm_metrics.get("geometry", {}),
            )
        )
    scenario_checks["eco_residuals"] = {
        "matches_catalog": observed_residual_scenarios == set(scenario_ids),
        "missing_scenarios": sorted(set(scenario_ids) - observed_residual_scenarios),
        "unexpected_scenarios": sorted(
            observed_residual_scenarios - set(scenario_ids)
        ),
        "scenario_definition_mismatches": [],
    }

    freshness_provenance = [
        artifact["producer_provenance_status"]
        for artifact in source_artifacts.values()
    ]
    any_scenario_stale = any(
        not check["matches_catalog"] for check in scenario_checks.values()
    )
    if any_scenario_stale or "stale" in freshness_provenance:
        overall_freshness = "stale"
    elif any(status != "current" for status in freshness_provenance):
        overall_freshness = "unverified"
    else:
        overall_freshness = "current"

    adm_terms = _adm_terms(
        adm_metrics,
        requirements,
        source_artifacts,
        adm_metrics_path,
        scenario_validity,
    )
    abi_terms = _abi_proxy_terms(
        abi_summary, requirements, source_artifacts, abi_summary_path
    )
    quantified_terms = paired_terms + adm_terms + abi_terms
    gaps = _gap_records(error_budget, requirements)

    report_provenance = build_provenance(
        root=ROOT,
        code_paths=(
            Path(__file__).resolve(),
            ROOT / "src" / "uncertainty.py",
        ),
        configuration_paths=(
            ROOT / "config.yaml",
            scenarios_path.resolve(),
            error_budget_path.resolve(),
        ),
        input_paths={
            "eco_chain_metrics": metrics_path.resolve(),
            "eco_n2bc_metrics": n2bc_metrics_path.resolve(),
            "eco_adm_metrics": adm_metrics_path.resolve(),
            "abi_adm_proxy_summary": abi_summary_path.resolve(),
            "eco_residuals": residuals_path.resolve(),
            **(
                {"abi_proxy_sensitivity": Path(abi_sensitivity_path).resolve()}
                if abi_sensitivity_path is not None else {}
            ),
            **(
                {"conditional_nominal_budget": Path(nominal_budget_path).resolve()}
                if nominal_budget_path is not None else {}
            ),
        },
    )
    return {
        "schema_version": 1,
        "assessment_scope": (
            "ECO MO2 LW radiance-to-broadband-flux evidence registry; "
            "not a total uncertainty budget or compliance assessment"
        ),
        "stage_order": [
            "Radiances",
            "Scene identification / co-registration",
            "ADM angular conversion",
            "Narrowband fluxes",
            "N2BC application",
            "Broadband flux",
            "Aggregation",
        ],
        "generation_provenance": report_provenance,
        "overall_evidence_freshness": overall_freshness,
        "source_artifacts": source_artifacts,
        "abi_production_assessment": abi_assessment,
        "conditional_nominal_assessment": nominal_assessment,
        "scene_taxonomy_assessment": _scene_taxonomy_assessment(scene_taxonomy_gate_path),
        "scenario_catalog": {
            "path": _display_path(scenarios_path),
            "sha256": sha256_file(scenarios_path),
            "scenario_ids": scenario_ids,
            "checks": scenario_checks,
        },
        "requirement_conventions": {
            "ObsReq_8_reference_um": requirements["ObsReq_8"].get(
                "wavelength_range"
            ),
            "Sunny_simulation_reference_um": metrics.get("reference_data", {}).get(
                "wavelength_range_um"
            ),
            "spectral_domain_reconciliation": (
                "unresolved: preserve the observed Sunny integration interval and "
                "ObsReq 8's configured interval as distinct domains"
            ),
            "confidence_convention": (
                "RfMA confidence for ObsReq 10/15/16 is unspecified in the catalog. "
                "Twice sample standard deviation is scatter, excludes bias, and is "
                "not automatically accuracy."
            ),
            "allocation": "unallocated; no per-source budget allocation inferred",
        },
        "component_definitions": error_budget.get("uncertainty_components", {}),
        "quantified_terms": quantified_terms,
        "unquantified_terms": gaps,
        "stage_assessment": _stage_assessments(requirements),
        "propagation_policy": {
            "paired_decomposition": (
                "For Sunny held-out pairs, end-to-end error equals spectral-only "
                "error plus the observed angular-plus-interaction increment."
            ),
            "covariance": (
                "Use the paired sample covariance matrix for variance propagation; "
                "never assume independence between stages."
            ),
            "other_cross_source_terms": "not combined",
            "missing_values": "not zero; retained as explicit unquantified terms",
            "aggregation": (
                "No monthly/seasonal covariance model is available; no 1/sqrt(N) "
                "reduction is applied."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--n2bc-metrics", type=Path, default=DEFAULT_N2BC_METRICS)
    parser.add_argument("--adm-metrics", type=Path, default=DEFAULT_ADM_METRICS)
    parser.add_argument("--abi-summary", type=Path, default=DEFAULT_ABI_SUMMARY)
    parser.add_argument(
        "--abi-narrowband-products", type=Path, nargs="*", default=[]
    )
    parser.add_argument("--abi-daily-products", type=Path, nargs="*", default=[])
    parser.add_argument("--abi-monthly-product", type=Path)
    parser.add_argument("--abi-sensitivity-report", type=Path)
    parser.add_argument("--nominal-budget-report", type=Path)
    parser.add_argument("--scene-taxonomy-gate", type=Path)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--error-budget", type=Path, default=DEFAULT_ERROR_BUDGET)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    registry = build_registry(
        metrics_path=args.metrics,
        n2bc_metrics_path=args.n2bc_metrics,
        adm_metrics_path=args.adm_metrics,
        abi_summary_path=args.abi_summary,
        residuals_path=args.residuals,
        scenarios_path=args.scenarios,
        error_budget_path=args.error_budget,
        abi_narrowband_paths=args.abi_narrowband_products,
        abi_daily_paths=args.abi_daily_products,
        abi_monthly_path=args.abi_monthly_product,
        abi_sensitivity_path=args.abi_sensitivity_report,
        nominal_budget_path=args.nominal_budget_report,
        scene_taxonomy_gate_path=args.scene_taxonomy_gate,
    )
    args.registry_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.registry_output.open("w") as output:
        json.dump(registry, output, indent=2)
    report = _render_report(registry)
    args.report_output.write_text(report)
    print(f"Wrote uncertainty registry: {args.registry_output}")
    print(f"Wrote uncertainty gap report: {args.report_output}")


if __name__ == "__main__":
    main()
