"""K-independent review figures for the manual GMM component-count choice.

No K is marked: these figures are the evidence the reviewer uses to fill
config/scene_taxonomy.yaml, so they must not depend on that choice.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "plotting"))

from plot_n_components import plot_n_components
from plot_nominal_budget import plot_gmm_diagnostics
from uncertainty import check_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path, record_path in ((args.diagnostics, Path(str(args.diagnostics) + ".json")),
                              (args.candidates, args.candidates)):
        freshness = check_provenance(json.loads(record_path.read_text())["provenance"], ROOT)
        if freshness["status"] != "current":
            raise ValueError(f"Stale GMM selection evidence {path}: {freshness}")
    candidates = json.loads(args.candidates.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_n_components(args.diagnostics, args.output_dir / "gmm_selection_overview.png")
    plot_gmm_diagnostics({"candidates": candidates["candidates"], "components": None},
                         args.diagnostics, {}, args.output_dir / "gmm_selection_stability.png")


if __name__ == "__main__":
    main()
