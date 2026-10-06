"""Build a provenance-aware, stage-ordered ECO LW uncertainty registry."""

import argparse
import csv
import hashlib
import json
import os
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
                if n2bc["fold"] != fold or n2bc["regime"] != pending["regime"]:
                    raise ValueError(
                        f"Residual regime/fold differs within paired scene "
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
    covariance = np.cov(np.stack((spectral, angular_increment)), ddof=1)
    total_variance = float(propagate_covariance(np.ones(2), covariance))
    total_bias = float(np.mean(spectral) + np.mean(angular_increment))
    return {
        "sample_count": int(spectral.size),
        "spectral_only": {
            "signed_bias_percent": float(np.mean(spectral)),
            "standard_deviation_percent": float(np.std(spectral, ddof=1)),
            "rmse_percent": float(np.sqrt(np.mean(spectral**2))),
        },
        "angular_plus_interaction_increment": {
            "signed_bias_percent": float(np.mean(angular_increment)),
            "standard_deviation_percent": float(np.std(angular_increment, ddof=1)),
        },
        "covariance_matrix_percent_squared": covariance.tolist(),
        "covariance_spectral_with_angular_increment_percent_squared": float(
            covariance[0, 1]
        ),
        "total_end_to_end": {
            "signed_bias_percent": total_bias,
            "standard_deviation_percent": float(np.sqrt(total_variance)),
            "rmse_percent": float(np.sqrt(np.mean(end_to_end**2))),
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
                    "Grouped out-of-fold signed mean, sample standard deviation, "
                    "RMSE, and paired sample covariance"
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
                    "minus spectral-only residual. Propagate the observed 2x2 "
                    "sample covariance with J=[1,1]."
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
        "envelope_band_um",
        "envelope_edge_slope_per_um",
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


def build_registry(
    *,
    metrics_path=DEFAULT_METRICS,
    n2bc_metrics_path=DEFAULT_N2BC_METRICS,
    adm_metrics_path=DEFAULT_ADM_METRICS,
    abi_summary_path=DEFAULT_ABI_SUMMARY,
    residuals_path=DEFAULT_RESIDUALS,
    scenarios_path=DEFAULT_SCENARIOS,
    error_budget_path=DEFAULT_ERROR_BUDGET,
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
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--error-budget", type=Path, default=DEFAULT_ERROR_BUDGET)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--figure-paths", type=Path, nargs=3,
        metavar=("EVIDENCE_MAP", "RESIDUAL_SUMMARY", "VARIANCE_DECOMPOSITION"),
        help="Link the three registry-driven workflow figures from the report.",
    )
    parser.add_argument(
        "--numerical-figure-paths", type=Path, nargs=3,
        metavar=("NUMERICAL_BUDGET", "SCENARIO_BUDGETS", "CHANNEL_DIAGNOSTICS"),
        help="Link the LW numerical budget and its scenario/channel companions.",
    )
    args = parser.parse_args()

    registry = build_registry(
        metrics_path=args.metrics,
        n2bc_metrics_path=args.n2bc_metrics,
        adm_metrics_path=args.adm_metrics,
        abi_summary_path=args.abi_summary,
        residuals_path=args.residuals,
        scenarios_path=args.scenarios,
        error_budget_path=args.error_budget,
    )
    args.registry_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.registry_output.open("w") as output:
        json.dump(registry, output, indent=2)
    report = _render_report(registry)
    figures = []
    if args.figure_paths:
        figures.extend(zip((
            "Stage-ordered evidence coverage",
            "Signed bias, sample standard deviation and RMSE",
            "Covariance-aware variance decomposition",
        ), args.figure_paths))
    if args.numerical_figure_paths:
        figures.extend(zip((
            "Numerical LW budget: combined currently quantified contribution",
            "Scenario companions with fixed N2BC model settings",
            "Per-channel ADM diagnostics (not additive broadband contributions)",
        ), args.numerical_figure_paths))
    if figures:
        report += "\n## Graphical interpretation\n\n"
        report += (
            "These figures are generated from the JSON registry by the "
            "`uncertainty_assessment` workflow target. They are conditional "
            "evidence summaries, not a total uncertainty budget or compliance test.\n\n"
        )
        for title, path in figures:
            relative_path = Path(os.path.relpath(
                path.resolve(), args.report_output.parent.resolve()
            )).as_posix()
            report += f"### {title}\n\n![{title}](<{relative_path}>)\n\n"
    args.report_output.write_text(report)
    print(f"Wrote uncertainty registry: {args.registry_output}")
    print(f"Wrote uncertainty gap report: {args.report_output}")


if __name__ == "__main__":
    main()
