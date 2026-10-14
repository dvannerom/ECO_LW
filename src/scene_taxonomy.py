"""Shared spectral-only scene taxonomy and its manual component-count gate.

One GMM taxonomy (10 spectral features, one human-selected K) is used by the
ABI production chain, the full-population diagnostics and the nominal budget.
K is chosen by a person from the K-independent full-population diagnostics
table; the choice is pinned to that table by SHA-256 so that a regenerated
table forces a new review instead of silently reusing an old decision.
"""

import csv
import math
from pathlib import Path

import yaml

from scene_features import spectral_feature_names
from uncertainty import sha256_file
from workflow_config import load_settings

DEFAULT_TAXONOMY_FILE = Path("config/scene_taxonomy.yaml")
FEATURE_SET = "spectral_10"
REQUIRED_RECORD_FIELDS = ("rationale", "selected_by", "date")
# Physical/statistical columns compared between K and its neighbours.
NEIGHBOUR_COLUMNS = (
    "full_heldout_loglik", "adm_ratio_rmse", "exact_corrected_flux_std",
    "exact_flux_improvement", "ari_min", "full_minimum_occupancy",
)
NON_PROFILE_SETTINGS_FIELDS = {
    "output_dir", "diagnostic_dir", "figure_dir", "diagnostic_figure_dir",
    "components", "scope", "product_dir",
}


class TaxonomySelectionError(ValueError):
    """The reviewed component-count selection is missing, stale or invalid."""


def load_scene_taxonomy(path=DEFAULT_TAXONOMY_FILE):
    """Load and structurally validate ``config/scene_taxonomy.yaml``.

    ``selection.n_components`` may be ``null`` (diagnostics not yet reviewed);
    use :func:`selected_components` where a K is required.
    """
    path = Path(path)
    with path.open() as handle:
        taxonomy = yaml.safe_load(handle)
    if not isinstance(taxonomy, dict) or taxonomy.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported scene taxonomy schema")
    if taxonomy.get("feature_set") != FEATURE_SET:
        raise ValueError(f"{path}: feature_set must be {FEATURE_SET}")
    if list(taxonomy.get("features", [])) != spectral_feature_names():
        raise ValueError(f"{path}: features must be {spectral_feature_names()}")
    diagnostics = taxonomy.get("diagnostics")
    if not isinstance(diagnostics, dict) or not {"settings", "csv"} <= set(diagnostics):
        raise ValueError(f"{path}: diagnostics.settings and diagnostics.csv are required")
    selection = taxonomy.get("selection")
    if not isinstance(selection, dict) or "n_components" not in selection:
        raise ValueError(f"{path}: selection.n_components is required (may be null)")
    count = selection["n_components"]
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 2):
        raise ValueError(f"{path}: selection.n_components must be null or an integer >= 2")
    return taxonomy


def selected_components(taxonomy, path=DEFAULT_TAXONOMY_FILE):
    count = taxonomy["selection"]["n_components"]
    if count is None:
        raise TaxonomySelectionError(
            f"No GMM component count has been selected in {path}. Run the "
            "'gmm_selection_diagnostics' target, review its table and figures, "
            "then record n_components, rationale, selected_by, date and "
            "diagnostics.csv_sha256 (print it with "
            "'python scripts/check_scene_taxonomy.py --summary')."
        )
    return count


def read_diagnostics(csv_path):
    """Return diagnostics rows keyed by integer K; numeric cells as floats."""
    rows = {}
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows[int(parsed["n_components"])] = parsed
    if not rows:
        raise TaxonomySelectionError(f"No diagnostic rows in {csv_path}")
    return rows


def neighbour_comparison(rows, components):
    """Differences of K-1 and K+1 relative to K for the structural-choice term.

    Returns ``{"neighbours": [...], "deltas": {K': {column: value(K')-value(K)}},
    "max_abs_delta": {column: max |delta|}}``. Missing neighbours (search-range
    edges) are reported, never extrapolated.
    """
    neighbours = [count for count in (components - 1, components + 1) if count in rows]
    deltas, worst = {}, {}
    for count in neighbours:
        deltas[count] = {}
        for column in NEIGHBOUR_COLUMNS:
            if column in rows[components] and column in rows[count]:
                delta = float(rows[count][column]) - float(rows[components][column])
                if math.isfinite(delta):
                    deltas[count][column] = delta
                    worst[column] = max(worst.get(column, 0.), abs(delta))
    return {
        "neighbours": neighbours,
        "missing_neighbours": [count for count in (components - 1, components + 1)
                               if count not in rows],
        "deltas": {str(count): values for count, values in deltas.items()},
        "max_abs_delta": worst,
    }


def verify_selection(taxonomy, root=Path("."), active_settings=None,
                     path=DEFAULT_TAXONOMY_FILE):
    """Check the reviewed selection against the current diagnostics table.

    Raises :class:`TaxonomySelectionError` with an actionable message when K is
    unset, the recorded SHA-256 does not match the table on disk, K was not
    evaluated, the human record is incomplete, or the active nominal settings
    differ from the profile the diagnostics were computed with.
    """
    root = Path(root)
    components = selected_components(taxonomy, path)
    diagnostics, selection = taxonomy["diagnostics"], taxonomy["selection"]
    if active_settings is not None:
        diagnostic_settings = (root / diagnostics["settings"]).resolve()
        production_settings = (root / active_settings).resolve()
        if diagnostic_settings != production_settings:
            diagnostic_profile = load_settings(diagnostic_settings)
            production_profile = load_settings(production_settings)
            diagnostic_profile = {
                key: value for key, value in diagnostic_profile.items()
                if key not in NON_PROFILE_SETTINGS_FIELDS
            }
            production_profile = {
                key: value for key, value in production_profile.items()
                if key not in NON_PROFILE_SETTINGS_FIELDS
            }
            differences = (
                ["no shared scientific settings"]
                if not diagnostic_profile or not production_profile
                else sorted(
                    key for key in diagnostic_profile.keys() | production_profile.keys()
                    if (key not in diagnostic_profile or key not in production_profile
                        or diagnostic_profile[key] != production_profile[key])
                )
            )
            if differences:
                raise TaxonomySelectionError(
                    f"Active nominal settings {active_settings} differ from the "
                    f"taxonomy diagnostics profile {diagnostics['settings']} in "
                    f"{differences}; the shared model must come from the reviewed "
                    "population."
                )
    csv_path = root / diagnostics["csv"]
    if not csv_path.is_file():
        raise TaxonomySelectionError(
            f"Diagnostics table {csv_path} is missing; run 'gmm_selection_diagnostics'."
        )
    digest = sha256_file(csv_path)
    if diagnostics.get("csv_sha256") != digest:
        raise TaxonomySelectionError(
            f"{path}: diagnostics.csv_sha256 does not match {csv_path} ({digest}). "
            "The diagnostics changed since K was reviewed; re-review them and "
            "update the selection record."
        )
    missing = [field for field in REQUIRED_RECORD_FIELDS
               if not str(selection.get(field) or "").strip()]
    if missing:
        raise TaxonomySelectionError(f"{path}: selection record lacks {missing}")
    rows = read_diagnostics(csv_path)
    if components not in rows:
        raise TaxonomySelectionError(
            f"K={components} is not in the diagnostics table ({sorted(rows)})"
        )
    return {
        "schema_version": 1,
        "feature_set": taxonomy["feature_set"],
        "features": list(taxonomy["features"]),
        "n_components": components,
        # str(): YAML parses an unquoted date to datetime.date (not JSON-serializable).
        "selection": {field: str(selection[field]) for field in REQUIRED_RECORD_FIELDS},
        "diagnostics_settings": diagnostics["settings"],
        "diagnostics_csv": diagnostics["csv"],
        "diagnostics_csv_sha256": digest,
        "selected_row": rows[components],
        "structural_taxonomy_choice": {
            "status": "neighbour_sensitivity_not_propagated",
            "definition": (
                "Change of full-population diagnostics when K is replaced by "
                "K-1 or K+1 on the same held-out ABI days; a structural "
                "classification-choice sensitivity, not a statistical error."
            ),
            **neighbour_comparison(rows, components),
        },
    }
