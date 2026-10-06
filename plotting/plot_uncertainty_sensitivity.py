"""Plot paired broadband sensitivities; proxy and ECO evidence are not combined."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uncertainty import check_provenance


def load_report(path):
    with Path(path).open() as source:
        report = json.load(source)
    if report["schema_version"] != 1:
        raise ValueError("Unsupported sensitivity report schema")
    check = check_provenance(report["provenance"], ROOT)
    if check["status"] != "current":
        raise ValueError(f"Stale sensitivity report: {check['mismatches']}")
    return report


def group_results(report):
    grouped = {}
    for name, result in report["results"].items():
        key = name.rsplit("_", 1)[0]
        grouped.setdefault(key, []).append(result)
    return grouped


def plot(report, output):
    groups = group_results(report)
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    for axis, branch in zip(axes, ("abi", "sunny")):
        selected = [(name, results) for name, results in groups.items()
                    if name.startswith(branch + "_")]
        for row, (name, results) in enumerate(selected):
            for offset, metric, color in zip(
                (-0.15, 0, 0.15), ("bias", "sd", "rmse"),
                ("#267a8a", "#b07828", "#7156a5"),
            ):
                values = [result["w_m2"][metric] for result in results]
                axis.scatter(values, np.full(len(values), row + offset),
                             label=metric.upper() if row == 0 else None, s=28, color=color)
        axis.set_yticks(range(len(selected)),
                        [name.removeprefix(branch + "_").replace("_", " ") for name, _ in selected])
        axis.axvline(0, color="gray", linewidth=0.7)
        axis.set_xlabel("Paired broadband-flux change [W m$^{-2}$]")
        axis.set_title("ABI proxy / temporal holdout" if branch == "abi"
                       else "ECO / full Sunny / grouped holdout")
        axis.legend()
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("One-source sensitivities and conditional joint scenario (not additive)")
    figure.text(0.5, 0.015,
                "Each dot is a realization. Sunny spectral compares true-band N2BC with the angular baseline.\n"
                "Joint = noise + higher-order ADM + robust N2BC; not a full ECO total. "
                "ABI and ECO scores are not summed.", ha="center")
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    from figure_layout import figure_path
    output = figure_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(load_report(args.report), args.output)
