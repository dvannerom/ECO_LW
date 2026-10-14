"""Plot saved uncertainty-registry evidence; numerical tables are separate outputs."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from budget_reporting import (
    ABI_SENSITIVITY_GROUPS,
    ECO_BUDGET_COLUMNS,
    EXPERIMENT_SPEC,
    REGIME_LABELS,
    SUNNY_SENSITIVITY_GROUPS,
    _finite,
    budget_rows,
    component_statistics,
    experiment_budget_rows,
    load_experiment_spec,
    lw_requirement,
    paired_terms,
    selected_models,
    sensitivity_budget_rows,
    variance_contributions,
    write_budget_csv,
    write_registry_budget_csv,
)




CAUTION = (
    "Conditional Sunny/SRF/geometry evidence; not total ECO uncertainty or compliance.\n"
    "All regimes pooled; bias is excluded. Negative 2 Cov is sample cancellation, not zero angular error."
)








def _save(figure, output_path):
    from figure_layout import figure_path
    output_path = figure_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _scenario_groups(registry):
    groups = {}
    for term in paired_terms(registry):
        groups.setdefault(term["eco_scenario"]["id"], []).append(term)
    return groups


def _model_label(term):
    domain = term["scene_geometry_spectral_domain"]
    mode = domain["regression_training_mode"]
    mode = {"abi_population_weighted": "ABI-weighted"}.get(mode, mode)
    return f"{mode} d{domain['regression_degree']}"


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
        CAUTION,
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.955))
    _save(figure, output_path)






















def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--variance-decomposition", type=Path)
    args = parser.parse_args()
    with args.registry.open() as source:
        registry = json.load(source)
    if registry["schema_version"] != 1:
        raise ValueError("Unsupported uncertainty registry schema")
    if args.variance_decomposition:
        plot_variance_decomposition(registry, args.variance_decomposition)


if __name__ == "__main__":
    main()
