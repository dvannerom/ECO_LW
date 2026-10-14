"""Manual GMM component-count gate for the shared spectral scene taxonomy.

``--summary`` prints the diagnostics-table SHA-256 and a compact per-K table
for review. ``--output`` validates the reviewed selection and writes the gate
record consumed by production classification and the nominal budget; it
exits non-zero with an actionable message when the selection is not valid.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scene_taxonomy import (
    DEFAULT_TAXONOMY_FILE, NEIGHBOUR_COLUMNS, TaxonomySelectionError,
    load_scene_taxonomy, neighbour_comparison, read_diagnostics, verify_selection,
)
from uncertainty import sha256_file


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summary(taxonomy, root):
    csv_path = root / taxonomy["diagnostics"]["csv"]
    if not csv_path.is_file():
        raise TaxonomySelectionError(f"{csv_path} is missing; run 'gmm_selection_diagnostics'.")
    rows = read_diagnostics(csv_path)
    print(f"diagnostics.csv: {taxonomy['diagnostics']['csv']}")
    print(f"diagnostics.csv_sha256: {sha256_file(csv_path)}")
    columns = [column for column in NEIGHBOUR_COLUMNS if column in next(iter(rows.values()))]
    print("K  " + "  ".join(f"{column:>24}" for column in columns))
    for count in sorted(rows):
        print(f"{count:<2} " + "  ".join(f"{float(rows[count][column]):24.6g}"
                                         for column in columns))
    chosen = taxonomy["selection"]["n_components"]
    if chosen is not None and chosen in rows:
        print(json.dumps(neighbour_comparison(rows, chosen), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_FILE)
    parser.add_argument("--active-settings", type=Path,
                        help="Nominal settings file of the running workflow")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Directory that taxonomy paths are relative to")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.summary and args.output is None:
        parser.error("Use --summary and/or --output")
    taxonomy = load_scene_taxonomy(args.taxonomy)
    try:
        if args.summary:
            summary(taxonomy, args.root)
        if args.output is not None:
            record = verify_selection(taxonomy, args.root, args.active_settings, args.taxonomy)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(finite_or_none(record), indent=2,
                                              allow_nan=False) + "\n")
    except TaxonomySelectionError as error:
        print(f"SCENE TAXONOMY GATE: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
