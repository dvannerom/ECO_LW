"""Plot file-level Sunny scene population and measure consistency across views."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scene_population import summarize_files


def plot_population(consistency_csv, report_path, output_dir):
    """Render saved numerical population summaries without exporting data."""
    report = json.loads(Path(report_path).read_text())
    if report["schema_version"] != 1:
        raise ValueError("Unsupported population-summary schema")
    with Path(consistency_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != report["n_files"]:
        raise ValueError("Consistency CSV file count disagrees with population report")
    n_scenes = report["n_scenes"]
    population = np.asarray(report["scene_file_counts"])
    view_population = np.asarray(report["scene_view_counts"])
    consistency = np.array([float(row["consistency_percent"]) for row in rows])
    complete = report["n_fully_consistent_files"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scene_ids = np.arange(n_scenes)
    for counts, xlabel, ylabel, title, filename in (
        (
            population, "Main scene ID (most frequent across 18 views)",
            "Number of Sunny files",
            f"Sunny scene population - {len(rows):,} files, counted once each",
            "sunny_scene_population.png",
        ),
        (
            view_population, "Scene ID", "Number of Sunny file/view assignments",
            f"Sunny scene population - all {int(view_population.sum()):,} view assignments",
            "sunny_scene_population_all_views.png",
        ),
    ):
        fig, axis = plt.subplots(figsize=(8, 4.8))
        bars = axis.bar(scene_ids, counts, width=0.8, color="steelblue")
        axis.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=3)
        axis.set_xticks(scene_ids)
        axis.set_xlim(-0.5, n_scenes - 0.5)
        axis.set_ylim(0, max(1, int(counts.max())) * 1.18)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    # One bin for each attainable modal count (1..18); centers are percentages.
    step = 100.0 / 18
    axis.hist(consistency, bins=(np.arange(0.5, 19.5) * step), color="steelblue", rwidth=0.85)
    axis.set_xlabel("Views matching the file's main scene (%)")
    axis.set_ylabel("Number of Sunny files")
    axis.set_title(
        f"Sunny angular scene consistency - {complete:,}/{len(rows):,} files at 100%"
    )
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output / "sunny_scene_consistency.png", dpi=180)
    plt.close(fig)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Saved sunny_file_scene_consistency.csv")
    parser.add_argument("--summary", type=Path, required=True, help="Saved sunny_scene_population.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/nominal_production/diagnostics/abi_sunny/comparison"),
    )
    args = parser.parse_args(argv)
    report = plot_population(
        args.input, args.summary, args.output_dir
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
