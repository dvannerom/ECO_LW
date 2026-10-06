"""Plot evidence coverage and conditional residuals from the small JSON registry."""

import argparse
import csv
import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncertainty import check_provenance

EXPERIMENT_SPEC = Path(__file__).resolve().parents[1] / "config" / "uncertainty_experiments.yaml"
ECO_BUDGET_COLUMNS = (
    "Contributor", "ECO uncertainty sources / experiment",
    "Impact on final broadband flux",
)


COMPONENTS = (
    ("spectral_only", "Spectral only (ObsReq 16 evidence)", "#267a8a", "o"),
    ("angular_plus_interaction_increment", "Angular + interaction", "#b07828", "s"),
    ("total_end_to_end", "End-to-end (ObsReq 15 evidence)", "#7156a5", "^"),
)
REGIME_LABELS = {
    "all_regimes": "All",
    "clear_sky": "Clear",
    "cloudy": "Cloud",
}
CAUTION = (
    "Conditional Sunny/SRF/geometry evidence; not total ECO uncertainty or compliance.\n"
    "Bias is signed; SD excludes bias. Missing terms are not zero. ABI evidence is separate."
)


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


def evidence_rows(registry):
    """Summarize coverage by stage; counts describe records, not budget fractions."""
    rows = []
    has_paired_output = any(
        term["term_id"].startswith("sunny_paired_n2bc_adm:")
        for term in registry["quantified_terms"]
    )
    for stage in registry["stage_order"]:
        terms = [
            term for term in registry["quantified_terms"]
            if stage.lower() in term["stage"].lower()
            or (stage == "Narrowband fluxes" and "narrowband flux" in term["stage"].lower())
        ]
        simulation = [
            term for term in terms if term["evidence_status"] == "conditional_simulation"
        ]
        proxy = [
            term for term in terms
            if term["evidence_status"] == "empirical_proxy_not_ECO_uncertainty"
        ]
        gaps = [
            gap for gap in registry["unquantified_terms"]
            if stage.lower() in gap["stage"].lower()
        ]
        statuses = sorted({term["freshness"]["status"] for term in terms})
        if any(term["term_id"].startswith("sunny_paired_n2bc_adm:") for term in terms):
            propagation = "Paired covariance\n(output residual only)"
        elif simulation and has_paired_output and stage in (
            "ADM angular conversion", "Narrowband fluxes"
        ):
            propagation = "Effect in paired output;\nchannel covariance missing"
        else:
            propagation = "No propagated\nECO estimate"
        rows.append({
            "stage": stage,
            "simulation": len(simulation),
            "proxy": len(proxy),
            "gaps": len(gaps),
            "freshness": ", ".join(statuses) if statuses else "No estimate",
            "propagation": propagation,
        })
    return rows


def _save(figure, output_path):
    from figure_layout import figure_path
    output_path = figure_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_evidence_map(registry, output_path):
    rows = evidence_rows(registry)
    figure, axis = plt.subplots(figsize=(14, 7))
    columns = ("Sunny simulation", "ABI proxy only", "ECO propagation", "Open gaps", "Freshness")
    for index, row in enumerate(rows):
        cells = (
            (f"{row['simulation']} conditional records" if row["simulation"] else "Not quantified",
             "#d7e9ec", not row["simulation"]),
            (f"{row['proxy']} proxy records" if row["proxy"] else "No registered estimate",
             "#e5deef", not row["proxy"]),
            (row["propagation"], "#e4edf4", row["propagation"].startswith("No")),
            (f"{row['gaps']} unquantified terms" if row["gaps"] else "No listed gap",
             "#f4e6cc", bool(row["gaps"])),
            (row["freshness"], "#eeeeee", False),
        )
        for column, (text, color, hatch) in enumerate(cells):
            axis.add_patch(Rectangle(
                (column, index), 1, 1, facecolor=color, edgecolor="#bbbbbb",
                hatch="///" if hatch else None, linewidth=0.6,
            ))
            axis.text(column + 0.5, index + 0.5, text, ha="center", va="center", fontsize=9)
    axis.set(xlim=(0, len(columns)), ylim=(len(rows), 0))
    axis.set_xticks(np.arange(len(columns)) + 0.5, columns)
    axis.set_yticks(np.arange(len(rows)) + 0.5, [row["stage"] for row in rows])
    axis.tick_params(length=0)
    axis.xaxis.tick_top()
    axis.set_title("Stage-ordered evidence coverage (not a completeness or compliance score)", pad=35)
    figure.text(
        0.5, 0.015,
        "Hatching denotes absent estimates or unresolved terms, never zero uncertainty.\n"
        "Record counts depend on scenarios/models/channels; they are not independent sources or budget shares.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    _save(figure, output_path)


def _scenario_groups(registry):
    groups = {}
    for term in paired_terms(registry):
        groups.setdefault(term["eco_scenario"]["id"], []).append(term)
    return groups


def _model_label(term):
    domain = term["scene_geometry_spectral_domain"]
    mode = domain["regression_training_mode"]
    mode = {"clear_cloud_stratified": "Stratified", "unstratified": "Unstratified"}.get(mode, mode)
    return f"{mode} d{domain['regression_degree']}"


def plot_residual_summary(registry, output_path):
    groups = _scenario_groups(registry)
    max_rows = max(sum(len(term["statistics_by_scene_domain"]) for term in terms) for terms in groups.values())
    figure, axes = plt.subplots(
        len(groups), 3, figsize=(17, len(groups) * max(4, max_rows * 0.24)),
        squeeze=False, sharex="col",
    )
    for row, (scenario, terms) in enumerate(groups.items()):
        records = [
            (term, regime, statistics)
            for term in terms
            for regime, statistics in term["statistics_by_scene_domain"].items()
        ]
        labels = [
            f"{_model_label(term)} / {REGIME_LABELS.get(regime, regime)} [{term['freshness']['status']}]"
            for term, regime, _ in records
        ]
        for metric_index, title in enumerate(("Signed bias (%)", "Sample SD (%)", "RMSE (%)")):
            axis = axes[row, metric_index]
            for component_index, (component, label, color, marker) in enumerate(COMPONENTS):
                values = [component_statistics(statistics, component)[metric_index] for _, _, statistics in records]
                axis.scatter(
                    values, np.arange(len(records)) + (component_index - 1) * 0.20,
                    color=color, marker=marker, s=22, label=label,
                )
            axis.axvline(0, color="#777777", linewidth=0.8)
            axis.set_yticks(np.arange(len(records)), labels if metric_index == 0 else [])
            axis.set_ylim(len(records) - 0.5, -0.5)
            axis.set_title(f"{scenario}\n{title}")
            axis.grid(axis="x", alpha=0.25)
            if metric_index:
                axis.set_xlim(left=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.text(
        0.5, 0.005,
        CAUTION + "\nIncrement RMSE is derived from bias, sample SD and sample count; it is not an independent angular source.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.965))
    _save(figure, output_path)


def plot_variance_decomposition(registry, output_path):
    groups = _scenario_groups(registry)
    max_models = max(len(terms) for terms in groups.values())
    figure, axes = plt.subplots(
        len(groups), max_models, figsize=(3.5 * max_models, 3.5 * len(groups)),
        squeeze=False, sharey="row",
    )
    colors = ("#267a8a", "#b07828", "#777777", "#7156a5")
    for row, (scenario, terms) in enumerate(groups.items()):
        row_peak = 0.0
        for column in range(max_models):
            axis = axes[row, column]
            if column >= len(terms):
                axis.set_visible(False)
                continue
            term = terms[column]
            contributions, total = variance_contributions(
                term["statistics_by_scene_domain"]["all_regimes"]
            )
            starts = np.concatenate(([0.0], np.cumsum(contributions)[:-1], [0.0]))
            heights = np.append(contributions, total)
            row_peak = max(row_peak, float(np.max(np.cumsum(contributions))), total)
            for index, (start, height, color) in enumerate(zip(starts, heights, colors)):
                axis.bar(
                    index, height, bottom=start, color=color, width=0.65,
                    hatch="//" if height < 0 else None,
                )
                axis.annotate(f"{height:+.3g}", (index, start + height),
                              xytext=(0, 4 if height >= 0 else -12),
                              textcoords="offset points", ha="center", fontsize=8)
                if index < 2:
                    endpoint = start + height
                    axis.plot([index + 0.325, index + 0.675], [endpoint, endpoint],
                              color="#777777", linewidth=0.7)
            axis.axhline(0, color="#777777", linewidth=0.7)
            axis.set_xticks(range(4), ["Spectral", "Increment", "2 Cov", "Total"], rotation=25)
            axis.set_ylabel("Sample variance (% squared)" if column == 0 else "")
            axis.set_title(
                f"{scenario}\n{_model_label(term)} [{term['freshness']['status']}]",
                fontsize=9,
            )
        axes[row, 0].set_ylim(0, row_peak * 1.25 if row_peak > 0 else 1)
    figure.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(
            colors, ("Spectral variance", "Angular + interaction variance", "Signed covariance contribution", "End-to-end variance")
        )],
        loc="upper center", ncol=4,
    )
    figure.text(
        0.5, 0.008,
        CAUTION + "\nAll regimes pooled. Negative 2 Cov is sample cancellation, not zero angular error; bias is not part of this variance sum.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.955))
    _save(figure, output_path)


def selected_models(registry, training_mode, degree):
    """Select fixed N2BC settings, never the best score on the evaluation sample."""
    selected = {}
    for scenario, terms in _scenario_groups(registry).items():
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


def _numeric_table(axis, rows, columns, widths, highlight_total=True):
    axis.axis("off")
    table = axis.table(
        cellText=rows, colLabels=columns, colWidths=widths,
        cellLoc="center", loc="center", bbox=(0, 0, 1, 0.94),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        cell.set_linewidth(0.6)
        text = cell.get_text().get_text()
        if row == 0:
            cell.set_facecolor("#e4edf4")
            cell.get_text().set_weight("bold")
        elif highlight_total and row == len(rows):
            cell.set_facecolor("#d7e9ec")
            cell.get_text().set_weight("bold")
        elif text in ("Unquantified", "Not available") or text.startswith("Pending"):
            cell.set_facecolor("#f4e6cc")
            cell.set_hatch("///")
        if column == 0 and row:
            cell.get_text().set_ha("left")
    return table


def _budget_footer(registry):
    conventions = registry["requirement_conventions"]
    return (
        "Cells: signed bias / sample SD / RMSE (% of simulated reference LW flux), except the covariance row.\n"
        "Combined variance = spectral variance + increment variance + 2 Cov; combined bias is the signed sum.\n"
        "Separate contributions are NOT assumed independent. Channel diagnostics and ABI spread are NOT added again.\n"
        "RfMA OLR values are contextual only: accuracy/confidence convention unresolved; no pass/fail or budget allocation.\n"
        "ObsReq 15 is for instantaneous maps at 10 km (goal) / 50 km (threshold); no such spatial averaging is applied here.\n"
        f"Sunny reference {conventions['Sunny_simulation_reference_um']} um; ObsReq 8 {conventions['ObsReq_8_reference_um']} um: unresolved.\n"
        f"Aggregation (not combined): {lw_requirement(registry, 'ObsReq_10').replace(chr(10), ': ')}; "
        f"{lw_requirement(registry, 'ObsReq_11').replace(chr(10), ': ')} stability. Both unquantified.\n"
        "Conditional idealized Sunny/SRF/geometry case; missing terms are not zero. Not total ECO uncertainty or a lower bound."
        "\nRfMA citations: " + "; ".join(sorted({
            reference["source_citation"]
            for term in registry["quantified_terms"] + registry["unquantified_terms"]
            for reference in term.get("requirement_references", [])
            if reference["id"] in ("ObsReq_10", "ObsReq_11", "ObsReq_15", "ObsReq_16")
        }))
    )


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
    """Export exactly the unified figure's rows, without visual line wrapping."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(ECO_BUDGET_COLUMNS)
        writer.writerows(rows)


def sensitivity_budget_rows(rows, report):
    """Replace pending impacts with measured sensitivities, not mission allocations."""
    if report["schema_version"] != 1:
        raise ValueError("Unsupported sensitivity report schema")
    freshness = check_provenance(report["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale sensitivity report: {freshness['mismatches']}")
    if report["settings"]["sunny"]["scenario"] != "rfma_goal_6":
        raise ValueError("First draft budget integration requires the RfMA goal scenario")

    def metrics(prefix, truth=False):
        records = [value for name, value in report["results"].items()
                   if name.rsplit("_", 1)[0] == prefix]
        if not records:
            raise ValueError(f"Sensitivity report lacks required experiment {prefix}")
        result = [record["truth_error"] if truth else record for record in records]
        lines = []
        for unit, label in (("w_m2", "W/m2"), ("percent", "%")):
            numbers = np.array([
                [record[unit][key] for key in ("bias", "sd", "rmse")] for record in result
            ])
            if not np.all(np.isfinite(numbers)):
                raise ValueError("Invalid sensitivity impact statistics")
            bias, sd, rmse = numbers.mean(axis=0)
            lines.append(f"Bias {bias:+.3f}; SD {sd:.3f}; RMSE {rmse:.3f} {label}")
        return "\n".join(lines)

    updated = [list(row) for row in rows]
    settings = report["settings"]
    radiometric = "ECO/Sunny noise only; no scene-selection effect:\n" + metrics("sunny_noise")
    spatial = "ABI proxy, matched 10 km footprints:\n" + metrics("abi_spatial")
    alternatives = [
        count for count in settings["abi"]["components"]
        if count != settings["abi"]["baseline_components"]
    ]
    model_lines = [
        f"ABI GMM {count} vs {settings['abi']['baseline_components']}:\n"
        + metrics(f"abi_components_{count}") for count in alternatives
    ]
    scene = "\n".join(model_lines) + "\nABI 10% second-choice assignment:\n" + metrics("abi_assignment")
    angular = "ECO/Sunny ADM-form change:\n" + metrics("sunny_adm")
    angular += "\nABI pooled ADM-form change:\n" + metrics("abi_adm")
    spectral = "ECO/Sunny robust vs ordinary N2BC:\n" + metrics("sunny_robust")
    spectral += "\nBaseline spectral-only error against truth:\n" + metrics("sunny_spectral", truth=True)
    spectral += "\nRepresentativity and physical-realism terms remain unquantified."
    covariance = np.asarray(report["covariance_percent_squared"])
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("Invalid sensitivity covariance")
    cov = f"Baseline spectral/angular Cov {covariance[0, 1]:+.4f} % squared.\n"
    interactions = report["joint_interactions"]
    if not interactions:
        raise ValueError("Joint interaction results are missing")
    interaction_rmse = np.mean([record["w_m2"]["rmse"] for record in interactions])
    cov += f"Joint minus sum of individual changes: RMSE {interaction_rmse:.3f} W/m2.\n"
    cov += "Joint scenario measured directly; source sensitivities are not added."
    total = "Conditional ECO/Sunny joint error against simulated truth:\n"
    total += metrics("sunny_joint", truth=True)
    total += "\nNoise + quadratic ADM + robust N2BC only.\nFull ECO total remains unquantified."
    for index, impact in ((0, radiometric), (1, spatial), (2, scene), (3, angular),
                          (4, spectral), (5, cov), (7, total)):
        updated[index][2] = impact
    return updated


def plot_numerical_budget(
    registry, output_path, scenario, training_mode, degree,
    specification=None, csv_path=None, sensitivity_report=None,
):
    models = selected_models(registry, training_mode, degree)
    if scenario not in models:
        raise ValueError(f"Requested budget scenario is missing: {scenario}")
    term = models[scenario]
    if specification is None:
        specification = load_experiment_spec()
    rows = experiment_budget_rows(registry, term, specification)
    if sensitivity_report is not None:
        if scenario != sensitivity_report["settings"]["sunny"]["scenario"]:
            raise ValueError("Budget channel scenario differs from sensitivity run")
        rows = sensitivity_budget_rows(rows, sensitivity_report)
    if csv_path is not None:
        write_budget_csv(rows, csv_path)
    wrapped_rows = [
        ["\n".join(textwrap.fill(line, width=width) for line in cell.splitlines())
         for cell, width in zip(row, (27, 88, 56))]
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(20, 16 if sensitivity_report else 13))
    _numeric_table(
        axis, wrapped_rows, ECO_BUDGET_COLUMNS, [0.18, 0.49, 0.33],
    )
    axis.set_title(
        f"ECO LW uncertainty experiments and current evidence: {scenario} | {_model_label(term)} | "
        f"freshness: {term['freshness']['status']}\n"
        "One ECO assessment; pooled held-out scenes, not minimum clear/cloud scores",
        fontsize=12,
    )
    extra_footer = ""
    if sensitivity_report is not None:
        settings = sensitivity_report["settings"]
        extra_footer = (
            f"Executed draft: ABI training days {settings['training_days']}; held-out days "
            f"{settings['evaluation_days']}; {settings['abi']['tiles_per_day']} "
            f"tiles/day of {settings['abi']['tile_size']}x{settings['abi']['tile_size']} native pixels.\n"
            f"Full Sunny library; {settings['realizations']} noise/assignment/joint realizations. "
            "Displayed metrics are means across realizations; not confidence intervals.\n"
            "ABI percent uses baseline flux; Sunny percent uses simulated true flux. "
            "No scene-selection noise propagation in the Sunny branch.\n"
            "Both ADM forms enforce positive 0-90 deg profiles (0.1 deg grid); "
            "this draft baseline differs from the earlier unconstrained fit.\n"
        )
    footer = _budget_footer(registry)
    if sensitivity_report is not None:
        footer = (
            "Paired sensitivities and truth errors: signed bias / sample SD / RMSE, in W/m2 and relative percent.\n"
            "Spectral/angular covariance refers to baseline residuals; "
            "joint scenario is evaluated directly, not assembled by quadrature.\n"
            + f"{lw_requirement(registry, 'ObsReq_16').replace(chr(10), ': ')}; "
            + f"{lw_requirement(registry, 'ObsReq_15').replace(chr(10), ': ')}. "
            + "Context only: confidence convention unresolved; no compliance claim.\n"
            "Sunny reference 2.5-500 um vs ObsReq 8 4-100 um: unresolved. "
            "ABI output is averaged to 10 km; Sunny has no spatial averaging.\n"
            "Monthly sampling, stability, calibration and other missing terms are not zero. "
            "Not total ECO uncertainty or a lower bound."
        )
    figure.text(
        0.5, 0.02,
        extra_footer
        + "Unresolved terms are not zero. Sensitivity to assumed changes is not a calibrated mission uncertainty.\n"
        + footer, ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.20, 1, 0.98))
    _save(figure, output_path)


def plot_scenario_budgets(registry, output_path, training_mode, degree):
    models = selected_models(registry, training_mode, degree)
    figure, axes = plt.subplots(
        len(models), 1, figsize=(18, 7 * len(models) + 2), squeeze=False,
    )
    for axis, (scenario, term) in zip(axes[:, 0], models.items()):
        _numeric_table(
            axis, budget_rows(registry, term),
            ["LW contribution", "All scenes", "Clear", "Cloud", "RfMA LW reference / scope"],
            [0.30, 0.15, 0.15, 0.15, 0.25],
        )
        axis.set_title(
            f"{scenario} | {_model_label(term)} | freshness: {term['freshness']['status']}",
            fontsize=12,
        )
    figure.text(0.5, 0.008, _budget_footer(registry), ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.075, 1, 0.995))
    _save(figure, output_path)


def channel_rows(registry, scenario):
    terms = [
        term for term in registry["quantified_terms"]
        if term["term_id"].startswith("sunny_adm_band_flux:")
        and term["eco_scenario"]["id"] == scenario
    ]
    if not terms:
        raise ValueError(f"No ADM channel diagnostics for {scenario}")
    channels = {}
    for term in terms:
        domain = term["scene_geometry_spectral_domain"]
        channel = channels.setdefault(domain["channel"], {})
        if domain["scene_domain"] in channel:
            raise ValueError(f"Duplicate channel diagnostic for {scenario}/{domain}")
        channel[domain["scene_domain"]] = term
    rows = []
    for name, domains in channels.items():
        cells = []
        statuses = set()
        for regime in REGIME_LABELS:
            term = domains.get(regime)
            if term is None:
                cells.append("Not available")
                continue
            summary = term["statistics"]
            values = np.array([
                term["signed_bias"], summary["standard_deviation_percent"],
                summary["rmse_percent"],
            ], dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values[1:] < 0):
                raise ValueError(f"Invalid channel statistics for {scenario}/{name}")
            cells.append(f"{values[0]:+.3f} / {values[1]:.3f} / {values[2]:.3f}")
            statuses.add(term["freshness"]["status"])
        rows.append([name.replace("_", " "), *cells, ", ".join(sorted(statuses))])
    return rows


def plot_channel_diagnostics(registry, output_path):
    scenarios = _scenario_groups(registry)
    figure, axes = plt.subplots(
        len(scenarios), 1, figsize=(16, 4 * len(scenarios) + 1), squeeze=False,
    )
    for axis, scenario in zip(axes[:, 0], scenarios):
        _numeric_table(
            axis, channel_rows(registry, scenario),
            ["ECO channel", "All scenes", "Clear", "Cloud", "Freshness"],
            [0.24, 0.20, 0.20, 0.20, 0.16],
            highlight_total=False,
        )
        axis.set_title(f"{scenario}: conditional ADM narrowband flux-error diagnostics", fontsize=12)
    figure.text(
        0.5, 0.012,
        "Cells: signed bias / sample SD / RMSE (% of TRUE CHANNEL flux, not broadband flux).\n"
        "No approved per-channel flux-error allocation. ObsReq 15 concerns broadband OLR, not each channel.\n"
        "Not additive budget rows: their broadband effect is already included in the angular + interaction increment.\n"
        "Conditional Sunny/SRF/idealized geometry case; no instrument noise or scene-selection uncertainty included.",
        ha="center", fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.995))
    _save(figure, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evidence-map", type=Path, required=True)
    parser.add_argument("--residual-summary", type=Path, required=True)
    parser.add_argument("--variance-decomposition", type=Path, required=True)
    parser.add_argument("--numerical-budget", type=Path)
    parser.add_argument("--experiment-spec", type=Path, default=EXPERIMENT_SPEC)
    parser.add_argument("--budget-csv", type=Path)
    parser.add_argument("--sensitivity-report", type=Path)
    parser.add_argument("--scenario-budgets", type=Path)
    parser.add_argument("--channel-diagnostics", type=Path)
    parser.add_argument("--budget-scenario", default="rfma_goal_6")
    parser.add_argument(
        "--training-mode", choices=("clear_cloud_stratified", "unstratified"),
        default="clear_cloud_stratified",
    )
    parser.add_argument("--degree", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    if args.budget_csv and not args.numerical_budget:
        parser.error("--budget-csv requires --numerical-budget")
    with args.registry.open() as source:
        registry = json.load(source)
    if registry["schema_version"] != 1:
        raise ValueError("Unsupported uncertainty registry schema")
    plot_evidence_map(registry, args.evidence_map)
    plot_residual_summary(registry, args.residual_summary)
    plot_variance_decomposition(registry, args.variance_decomposition)
    if args.numerical_budget:
        sensitivity_report = None
        if args.sensitivity_report:
            with args.sensitivity_report.open() as source:
                sensitivity_report = json.load(source)
        plot_numerical_budget(
            registry, args.numerical_budget, args.budget_scenario, args.training_mode, args.degree,
            specification=load_experiment_spec(args.experiment_spec), csv_path=args.budget_csv,
            sensitivity_report=sensitivity_report,
        )
    if args.scenario_budgets:
        plot_scenario_budgets(registry, args.scenario_budgets, args.training_mode, args.degree)
    if args.channel_diagnostics:
        plot_channel_diagnostics(registry, args.channel_diagnostics)


if __name__ == "__main__":
    main()
