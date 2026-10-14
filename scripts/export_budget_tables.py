"""Export numerical budget tables without importing plotting or Matplotlib."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from budget_reporting import (
    EXPERIMENT_SPEC, load_experiment_spec, write_nominal_budget_csv,
    write_registry_budget_csv,
)
from uncertainty import check_provenance


def current_report(path):
    report = json.loads(Path(path).read_text())
    freshness = check_provenance(report["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale report {path}: {freshness}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("nominal", "registry"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sensitivity-report")
    parser.add_argument("--experiment-spec", default=EXPERIMENT_SPEC)
    parser.add_argument("--scenario", default="rfma_goal_6")
    parser.add_argument("--training-mode", default="abi_population_weighted",
                        choices=("abi_population_weighted",))
    parser.add_argument("--degree", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    if args.kind == "nominal":
        report = current_report(args.input)
        if args.sensitivity_report:
            parser.error("--sensitivity-report is only valid for registry tables")
        write_nominal_budget_csv(report, args.output)
    else:
        report = json.loads(Path(args.input).read_text())
        if report["schema_version"] != 1:
            raise ValueError("Unsupported uncertainty registry schema")
        sensitivity = current_report(args.sensitivity_report) if args.sensitivity_report else None
        write_registry_budget_csv(
            report, args.output, args.scenario, args.training_mode, args.degree,
            load_experiment_spec(args.experiment_spec), sensitivity_report=sensitivity,
        )


if __name__ == "__main__":
    main()
