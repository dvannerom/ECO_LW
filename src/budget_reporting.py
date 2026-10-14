"""Numerical uncertainty tables and evidence selection, independent of plotting."""

import csv
from pathlib import Path

import numpy as np
import yaml

from uncertainty import check_provenance

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_SPEC = Path(__file__).resolve().parents[1] / "config" / "uncertainty_experiments.yaml"

ECO_BUDGET_COLUMNS = (
    "Contributor", "ECO uncertainty sources / experiment",
    "Impact on final broadband flux",
)

REGIME_LABELS = {
    "all_regimes": "All",
    "clear_sky": "Clear",
    "cloudy": "Cloud",
}

def paired_terms(registry):
    terms = [
        term for term in registry["quantified_terms"]
        if term["term_id"].startswith("sunny_paired_n2bc_adm:")
    ]
    if not terms:
        raise ValueError("Registry contains no paired Sunny residual terms")
    return sorted(
        terms,
        key=lambda term: (
            term["eco_scenario"]["id"],
            term["scene_geometry_spectral_domain"]["regression_training_mode"],
            term["scene_geometry_spectral_domain"]["regression_degree"],
        ),
    )

def component_statistics(statistics, component):
    """Return bias, sample SD and RMSE (%) without treating SD as a new source."""
    values = statistics[component]
    bias = values["signed_bias_percent"]
    sd = values["standard_deviation_percent"]
    if "rmse_percent" in values:
        rmse = values["rmse_percent"]
    else:
        count = statistics["sample_count"]
        if count < 2:
            raise ValueError("At least two samples are required for sample SD")
        rmse = np.sqrt(bias**2 + (count - 1) / count * sd**2)
    result = np.asarray((bias, sd, rmse), dtype=float)
    if not np.all(np.isfinite(result)) or sd < 0 or rmse < 0:
        raise ValueError("Invalid residual statistics in registry")
    return result

def variance_contributions(statistics):
    """Return spectral variance, increment variance, 2*covariance and total (%²)."""
    covariance = np.asarray(
        statistics["covariance_matrix_percent_squared"], dtype=float
    )
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("Expected a finite 2x2 sample covariance matrix")
    if not np.allclose(covariance, covariance.T) or np.any(
        np.linalg.eigvalsh(covariance) < -1e-10
    ):
        raise ValueError("Sample covariance must be symmetric positive semidefinite")
    contributions = np.array(
        [covariance[0, 0], covariance[1, 1], 2 * covariance[0, 1]]
    )
    total = statistics["total_end_to_end"]["standard_deviation_percent"] ** 2
    if not np.isclose(contributions.sum(), total, rtol=1e-8, atol=1e-10):
        raise ValueError("Covariance contributions do not reconstruct end-to-end variance")
    return contributions, total

def selected_models(registry, training_mode, degree):
    """Select fixed N2BC settings, never the best score on the evaluation sample."""
    selected = {}
    groups = {}
    for term in paired_terms(registry):
        groups.setdefault(term["eco_scenario"]["id"], []).append(term)
    for scenario, terms in groups.items():
        matches = [
            term for term in terms
            if term["scene_geometry_spectral_domain"]["regression_training_mode"] == training_mode
            and term["scene_geometry_spectral_domain"]["regression_degree"] == degree
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {training_mode}/degree {degree} result for {scenario}")
        selected[scenario] = matches[0]
    return selected

def lw_requirement(registry, requirement_id):
    references = [
        reference
        for term in registry["quantified_terms"] + registry["unquantified_terms"]
        for reference in term.get("requirement_references", [])
        if reference["id"] == requirement_id
    ]
    if not references or any(reference != references[0] for reference in references):
        raise ValueError(f"Missing or inconsistent LW requirement {requirement_id}")
    reference = references[0]
    if reference["units"] != "percent":
        raise ValueError(f"Expected a percentage LW requirement for {requirement_id}")
    values = [
        f"{label} {reference[field]:g}%"
        for field, label in (("goal", "G"), ("threshold", "T"), ("maximum", "Max"))
        if reference[field] is not None
    ]
    return f"{requirement_id.replace('_', ' ')} OLR\n" + "; ".join(values)

def budget_rows(registry, term):
    """Build output-space contributions and their combined residual, not a mission total."""
    regimes = tuple(REGIME_LABELS)
    missing = ["Unquantified"] * len(regimes)
    rows = [
        ["Radiometric noise / calibration", *missing, "Output impact missing"],
        ["Scene ID / CTH / registration", *missing, "ObsReq 17: qualitative"],
        ["Temporal coincidence / scene evolution", *missing, "ObsReq 12: qualitative"],
        ["ADM population / geometry / model form\nbeyond the simulated case", *missing, "Not quantified beyond Sunny"],
    ]
    for component, label, reference in (
        ("spectral_only", "Spectral reconstruction", lw_requirement(registry, "ObsReq_16")),
        ("angular_plus_interaction_increment", "Angular retrieval + N2BC interaction", "No separate allocation"),
    ):
        cells = []
        for regime in regimes:
            statistics = term["statistics_by_scene_domain"].get(regime)
            if statistics is None:
                cells.append("Not available")
                continue
            bias, sd, rmse = component_statistics(statistics, component)
            cells.append(f"{bias:+.3f} / {sd:.3f} / {rmse:.3f}")
        rows.append([label, *cells, reference])
    covariance_cells = []
    total_cells = []
    for regime in regimes:
        statistics = term["statistics_by_scene_domain"].get(regime)
        if statistics is None:
            covariance_cells.append("Not available")
            total_cells.append("Not available")
            continue
        variance_contributions(statistics)
        spectral_bias = statistics["spectral_only"]["signed_bias_percent"]
        increment_bias = statistics["angular_plus_interaction_increment"]["signed_bias_percent"]
        bias, sd, rmse = component_statistics(statistics, "total_end_to_end")
        if not np.isclose(spectral_bias + increment_bias, bias, atol=1e-10):
            raise ValueError("Component biases do not reconstruct end-to-end bias")
        covariance = statistics["covariance_spectral_with_angular_increment_percent_squared"]
        matrix = np.asarray(statistics["covariance_matrix_percent_squared"])
        if not np.isclose(covariance, matrix[0, 1]):
            raise ValueError("Covariance scalar differs from covariance matrix")
        covariance_cells.append(f"{covariance:+.4f} % squared")
        total_cells.append(f"{bias:+.3f} / {sd:.3f} / {rmse:.3f}")
    rows.append(["Spectral/increment covariance\n(not an additional error source)",
                 *covariance_cells, "Included as 2 Cov in variance"])
    rows.extend([
        ["SRF / reference-domain uncertainty", *missing, "ObsReq 8/16; unresolved"],
        ["Instrument / shared-source covariance", *missing, "Not inferred from paired covariance"],
        ["WFOV anchoring / absolute validation", *missing, "No quantified output contribution"],
        ["Combined currently quantified LW error\n(Sunny angular + spectral only)",
         *total_cells, lw_requirement(registry, "ObsReq_15")],
    ])
    return rows

def load_experiment_spec(path=EXPERIMENT_SPEC):
    with Path(path).open() as source:
        specification = yaml.safe_load(source)
    if specification["schema_version"] != 1:
        raise ValueError("Unsupported uncertainty experiment schema")
    contributors = specification["contributors"]
    ids = [row["id"] for row in contributors]
    if not ids or len(ids) != len(set(ids)) or ids[-1] != "total":
        raise ValueError("Experiment contributors must be unique with total last")
    for row in contributors:
        if row["status"] not in ("pending", "existing_conditional", "incomplete"):
            raise ValueError(f"Unsupported experiment status: {row['status']}")
        if row.get("supporting_component") not in (
            None, "spectral_only", "angular_plus_interaction_increment", "total_end_to_end"
        ):
            raise ValueError(f"Unsupported supporting component for {row['id']}")
    return specification

def experiment_budget_rows(registry, term, specification):
    """Keep proposed source sensitivities separate from existing pooled evidence."""
    # Validate the paired bias/covariance closure before displaying supporting scores.
    budget_rows(registry, term)
    statistics = term["statistics_by_scene_domain"].get("all_regimes")
    if statistics is None:
        raise ValueError("Unified ECO budget requires pooled all-regimes statistics")
    rows = []
    for contributor in specification["contributors"]:
        impact = contributor["impact"]
        component = contributor.get("supporting_component")
        if component:
            bias, sd, rmse = component_statistics(statistics, component)
            impact += f"\nBias {bias:+.3f}%; SD {sd:.3f}%; RMSE {rmse:.3f}%"
            requirement_id = {
                "spectral_only": "ObsReq_16",
                "total_end_to_end": "ObsReq_15",
            }.get(component)
            if requirement_id:
                impact += "\nContext: " + lw_requirement(registry, requirement_id).replace("\n", ": ")
        elif contributor["id"] == "covariance":
            covariance = statistics["covariance_spectral_with_angular_increment_percent_squared"]
            impact += f"\nCov {covariance:+.4f} % squared\nVariance includes 2 Cov."
        if component or contributor["id"] == "covariance":
            impact += f"\nEvidence freshness: {term['freshness']['status']}"
        rows.append([contributor["label"], contributor["sources"], impact])
    return rows

def write_budget_csv(rows, output_path):
    """Export the unified budget rows without visual line wrapping."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(ECO_BUDGET_COLUMNS)
        writer.writerows(rows)

ABI_SENSITIVITY_GROUPS = {
    "scene_identification": ("components_minus_1", "components_plus_1"),
    "angular": ("adm_daily", "adm_quadratic"),
    "spectral": ("n2b_quadratic",),
}

SUNNY_SENSITIVITY_GROUPS = {
    "scene_identification": ("components_minus_1", "components_plus_1"),
    "angular": ("adm_per_file", "adm_quadratic"),
    "spectral": ("n2bc_stratified", "n2bc_robust", "n2bc_degree3"),
}

def _finite(*values):
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Invalid sensitivity impact statistics")
    return values

def sensitivity_budget_rows(rows, report):
    """Add one table row per production-aligned paired sensitivity."""
    if report["schema_version"] != 2:
        raise ValueError("Unsupported sensitivity report schema")
    freshness = check_provenance(report["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale sensitivity report: {freshness['mismatches']}")
    if report["sunny"]["scenario"] != "rfma_goal_6":
        raise ValueError("First draft budget integration requires the RfMA goal scenario")
    abi, sunny = report["abi"]["variants"], report["sunny"]["variants"]

    def abi_line(name):
        entry = abi[name]
        held_out = entry["splits"]["held_out"]
        satellites = sorted(held_out)
        pixel = [held_out[sat]["pixel"] for sat in satellites]
        monthly = [entry["monthly_10km"][sat] for sat in satellites]
        pixel_rmse = _finite(*(row["rmse_w_m2"] for row in pixel))
        pixel_bias = _finite(*(row["bias_w_m2"] for row in pixel))
        monthly_rmse = _finite(*(row["rmse_w_m2"] for row in monthly))
        joined = "/".join
        return (f"ABI {name} ({joined(satellites)}): held-out pixel bias "
                f"{joined(f'{v:+.3f}' for v in pixel_bias)}, RMSE {joined(f'{v:.3f}' for v in pixel_rmse)}; "
                f"monthly 10 km RMSE {joined(f'{v:.3f}' for v in monthly_rmse)} W/m2")

    def sunny_line(name):
        entry = sunny[name]
        if entry["status"] != "quantified":
            return f"ECO/Sunny {name}: {entry['status']} ({entry.get('reason', '')})"
        flux, percent = entry["paired_difference_w_m2"], entry["paired_difference_percent"]
        bias, rmse, rmse_percent = _finite(flux["bias_w_m2"], flux["class_rmse_w_m2"],
                                           percent["class_rmse_w_m2"])
        return f"ECO/Sunny {name}: bias {bias:+.3f}; RMSE {rmse:.3f} W/m2 ({rmse_percent:.3f}%)"

    updated = [list(row) for row in rows]
    nominal_pointer = "Single source: nominal Sunny/ECO budget report ({}); not a model-choice sensitivity."
    updated[0][2] = nominal_pointer.format("ECO NEdT propagated on Sunny")
    updated[1][2] = nominal_pointer.format("ABI 2-to-10 km footprint proxy")
    updated[5][2] = ("Not evaluated by one-at-a-time sensitivities (not additive).\n"
                     "Baseline spectral/angular covariance: nominal budget report.")
    updated[7][2] = ("Single source: nominal Sunny/ECO budget combined row.\n"
                     "Model-choice sensitivities are reported beside it, never added.")

    sensitivity_rows = []
    for group, names in ABI_SENSITIVITY_GROUPS.items():
        for name in names:
            entry = abi.get(name)
            if entry is None or entry["source_group"] != group:
                continue
            description = entry.get("description", "")
            sensitivity_rows.append([
                f"ABI proxy sensitivity: {name}",
                f"Analysis/model-choice sensitivity; ABI proxy evidence. {description}",
                f"{abi_line(name)}\nNot added to the nominal uncertainty subtotal.",
            ])
    for group, names in SUNNY_SENSITIVITY_GROUPS.items():
        for name in names:
            entry = sunny.get(name)
            if entry is None or entry["source_group"] != group:
                continue
            description = entry.get("description", "")
            sensitivity_rows.append([
                f"Sunny/ECO sensitivity: {name}",
                f"Analysis/model-choice sensitivity; conditional Sunny/ECO evidence. {description}",
                f"{sunny_line(name)}\nNot added to the nominal uncertainty subtotal.",
            ])
    updated.extend(sensitivity_rows)
    return updated

def write_registry_budget_csv(
    registry, csv_path, scenario, training_mode, degree, specification, sensitivity_report=None
):
    """Write the unified budget CSV retained as an input to sensitivity assessment."""
    models = selected_models(registry, training_mode, degree)
    if scenario not in models:
        raise ValueError(f"Requested budget scenario is missing: {scenario}")
    term = models[scenario]
    rows = experiment_budget_rows(registry, term, specification)
    if sensitivity_report is not None:
        if scenario != sensitivity_report["sunny"]["scenario"]:
            raise ValueError("Budget channel scenario differs from sensitivity run")
        rows = sensitivity_budget_rows(rows, sensitivity_report)
    write_budget_csv(rows, csv_path)


def nominal_budget_table(report):
    """Return the source/error-class numerical table without rendering."""
    names = {
        "spectral": "N2BC reconstruction (within-library)",
        "angular": "Scene-ADM approximation / within-scene diversity",
        "radiometric": "Radiometric noise (scene label held fixed)",
        "assignment": "Noise-mediated scene assignment (ABI-to-ECO assumption)",
        "total": "Conditional joint subtotal (not full ECO uncertainty)",
        "spatial": "Spatial processing: ABI 2-to-10 km proxy",
        "combined": "Combined budget (spatial independent; not full ECO uncertainty)",
    }
    records = []
    for row in report["rows"]:
        records.append([
            names[row["id"]], f"{row['bias_w_m2']:+.4f}", f"{row['scene_sd_w_m2']:.4f}",
            f"{row['random_sd_w_m2']:.4f}", f"{row['ensemble_rmse_w_m2']:.4f}",
        ])
        if row["negative_corrected_scene_variance"]:
            records[-1][2] = "Below MC resolution"
    records.extend([
        ["Training / library estimation variability", "Unquantified", "Unquantified", "Unquantified", "--"],
        ["ECO spatial PSF / spatial-noise interactions", "Unquantified", "Unquantified", "Unquantified", "--"],
        ["Calibration, SRF, registration, evolution, library discrepancy",
         "Unquantified", "Unquantified", "Unquantified", "--"],
    ])
    headers = ["Error source", "Bias\nW/m2", "Scene-dependent SD\nW/m2", "Random SD\nW/m2",
               "RMSE\nW/m2"]
    return headers, records


def write_nominal_budget_csv(report, path):
    headers, records = nominal_budget_table(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([text.replace("\n", " ") for text in headers])
        writer.writerows(records)
