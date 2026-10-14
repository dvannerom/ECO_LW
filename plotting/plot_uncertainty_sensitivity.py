"""Plot production-aligned one-at-a-time sensitivities (schema 2).

Figure 1 (``--output``): paired broadband-flux differences per variant.
  Left:   ABI pixel-level bias and RMSE [W m-2] per satellite, held-out days
          (filled) and held-in days (open).
  Middle: ABI monthly 10 km cell-mean difference RMSE and |bias| [W m-2].
  Right:  Sunny/ECO paired bias and RMSE [W m-2] (ECO goal channels).
Figure 2 (``--maps-output``): monthly 10 km paired-difference maps
  [W m-2], one row per ABI variant and one column per satellite.
Variants are model-choice sensitivities, never summed.
"""

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

GROUP_COLORS = {"scene_identification": "#7156a5", "angular": "#267a8a", "spectral": "#b07828"}
SATELLITE_MARKERS = {"G16": "o", "G18": "s"}


def load_report(path):
    with Path(path).open() as source:
        report = json.load(source)
    if report["schema_version"] != 2:
        raise ValueError("Unsupported sensitivity report schema")
    check = check_provenance(report["provenance"], ROOT)
    if check["status"] != "current":
        raise ValueError(f"Stale sensitivity report: {check['mismatches']}")
    return report


def _labels(axis, names, variants):
    axis.set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
    for tick, name in zip(axis.get_yticklabels(), names):
        tick.set_color(GROUP_COLORS[variants[name]["source_group"]])
    axis.axvline(0, color="gray", linewidth=0.7)
    axis.grid(axis="x", alpha=0.25)
    axis.invert_yaxis()


def plot_summary(report, output):
    abi = report["abi"]["variants"]
    sunny = report["sunny"]["variants"]
    names = list(abi)
    figure, axes = plt.subplots(1, 3, figsize=(20, 7))
    for row, name in enumerate(names):
        for sat_index, (sat, marker) in enumerate(SATELLITE_MARKERS.items()):
            offset = (sat_index - 0.5) * 0.3
            for split, filled in (("held_out", True), ("held_in", False)):
                pixel = abi[name]["splits"][split][sat]["pixel"]
                if not pixel.get("count"):
                    continue
                for metric, color in (("bias_w_m2", "#267a8a"), ("rmse_w_m2", "#7156a5")):
                    axes[0].scatter(pixel[metric], row + offset, marker=marker, s=36,
                                    facecolors=color if filled else "none", edgecolors=color)
            monthly = abi[name]["monthly_10km"][sat]
            if monthly.get("count"):
                axes[1].scatter(abs(monthly["bias_w_m2"]), row + offset, marker=marker, color="#267a8a", s=36)
                axes[1].scatter(monthly["rmse_w_m2"], row + offset, marker=marker, color="#7156a5", s=36)
                axes[1].scatter(monthly["p95_abs_w_m2"], row + offset, marker=marker, color="#b07828", s=36)
    for metric, color in (("bias", "#267a8a"), ("RMSE", "#7156a5")):
        axes[0].scatter([], [], color=color, label=f"{metric} (filled: held-out, open: held-in)")
    for metric, color in (("|bias|", "#267a8a"), ("RMSE", "#7156a5"), ("P95 |diff|", "#b07828")):
        axes[1].scatter([], [], color=color, label=metric)
    for sat, marker in SATELLITE_MARKERS.items():
        axes[0].scatter([], [], marker=marker, color="black", label=sat)
    _labels(axes[0], names, abi)
    _labels(axes[1], names, abi)
    axes[0].set_title("ABI native pixels, paired (variant - production)")
    axes[1].set_title("ABI monthly 10 km cell means, all days")
    axes[0].set_xlabel("Paired broadband-flux difference [W m$^{-2}$]")
    axes[1].set_xlabel("Monthly cell-mean difference [W m$^{-2}$]")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    quantified = [name for name, result in sunny.items() if result["status"] == "quantified"]
    for row, name in enumerate(quantified):
        paired = sunny[name]["paired_difference_w_m2"]
        axes[2].scatter(paired["bias_w_m2"], row - 0.1, color="#267a8a", s=36,
                        label="Paired bias" if row == 0 else None)
        axes[2].scatter(paired["class_rmse_w_m2"], row + 0.1, color="#7156a5", s=36,
                        label="Paired RMSE" if row == 0 else None)
    _labels(axes[2], quantified, sunny)
    baseline = report["sunny"]["baseline_truth_error_w_m2"]["class_rmse_w_m2"]
    axes[2].set_title(f"Sunny / ECO {report['sunny']['scenario']}, grouped held-out\n"
                      f"(baseline RMSE vs truth {baseline:.2f} W m$^{{-2}}$)")
    axes[2].set_xlabel("Population-weighted paired difference [W m$^{-2}$]")
    axes[2].legend(fontsize=8)
    unsupported = [name for name, result in sunny.items() if result["status"] != "quantified"]
    footer = ("Tick colour = source group (purple: scene identification, teal: angular, ochre: spectral). "
              "One-at-a-time model-choice sensitivities; not additive, not uncertainties of record.")
    if unsupported:
        footer += f"\nUnsupported Sunny variants: {', '.join(unsupported)}."
    figure.suptitle(f"Sensitivity around the production configuration "
                    f"(K={report['nominal']['components']}, {report['production']['n2b_form']} N2B, "
                    f"VZA <= {report['production']['max_vza_deg']:g} deg)")
    figure.text(0.5, 0.01, footer, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(figure, output)


def plot_maps(report, maps_path, output):
    names = list(report["abi"]["variants"])
    satellites = list(SATELLITE_MARKERS)
    figure, axes = plt.subplots(len(names), len(satellites), figsize=(6 * len(satellites), 3.6 * len(names)),
                                squeeze=False)
    with np.load(maps_path) as maps:
        for row, name in enumerate(names):
            limit = max(float(np.nanpercentile(np.abs(maps[f"{name}_{sat}"]), 99)) for sat in satellites) or 1.
            for column, sat in enumerate(satellites):
                axis = axes[row, column]
                image = axis.imshow(maps[f"{name}_{sat}"], cmap="RdBu_r", vmin=-limit, vmax=limit,
                                    interpolation="nearest")
                axis.set_title(f"{sat}: {name.replace('_', ' ')}", fontsize=10)
                axis.set_axis_off()
                figure.colorbar(image, ax=axis, shrink=0.8, label="W m$^{-2}$")
    figure.suptitle("Monthly 10 km paired difference (variant - production), all configured days\n"
                    "Colour limits: 99th percentile of |difference| per variant")
    figure.tight_layout()
    save(figure, output)


def save(figure, output):
    from figure_layout import figure_path
    output = figure_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps-output", type=Path, required=True)
    args = parser.parse_args()
    report = load_report(args.report)
    plot_summary(report, args.output)
    plot_maps(report, args.maps, args.maps_output)
