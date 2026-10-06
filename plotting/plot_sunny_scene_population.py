"""Plot file-level Sunny scene population and measure consistency across views."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize_files(classifications, n_scenes):
    """Stream view rows; retain O(n_files * n_scenes) counts/posterior sums.

    Each source file must contain exactly the 18 provided views (0..85 deg).
    Main scene is the mode; count ties use mean posterior over all views,
    then lowest scene ID if posterior means also tie. Consistency is the
    percentage of hard view labels matching the main scene, not confidence.
    """
    if n_scenes < 1:
        raise ValueError("n_scenes must be positive")
    posterior_columns = [f"p_scene_{i}" for i in range(n_scenes)]
    files = {}
    with Path(classifications).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_file", "scene_index", "regime", "view_angle_deg", "scene_id"}
        if not required.union(posterior_columns).issubset(reader.fieldnames or []):
            raise ValueError("Classification CSV lacks identifiers, labels or scene posteriors")
        for row_number, row in enumerate(reader, start=2):
            source = row["source_file"]
            scene = int(row["scene_id"])
            angle = float(row["view_angle_deg"])
            posterior = np.array([float(row[column]) for column in posterior_columns])
            if (
                not source or not 0 <= scene < n_scenes
                or angle not in range(0, 90, 5)
                or not np.all(np.isfinite(posterior))
                or np.any(posterior < 0) or np.any(posterior > 1)
                or not np.isclose(posterior.sum(), 1.0, atol=1e-6)
                or posterior[scene] < posterior.max() - 1e-12
            ):
                raise ValueError(f"Invalid scene/view/posterior in CSV row {row_number}")
            if source not in files:
                files[source] = {
                    "scene_index": row["scene_index"], "regime": row["regime"],
                    "angles": set(), "counts": np.zeros(n_scenes, dtype=np.int64),
                    "posterior_sum": np.zeros(n_scenes, dtype=np.float64),
                }
            record = files[source]
            if (
                record["scene_index"] != row["scene_index"]
                or record["regime"] != row["regime"]
                or angle in record["angles"]
            ):
                raise ValueError(f"Duplicate view or inconsistent identity for {source}")
            record["angles"].add(angle)
            record["counts"][scene] += 1
            record["posterior_sum"] += posterior
    if not files:
        raise ValueError("Classification CSV contains no Sunny files")
    rows = []
    for source, record in sorted(files.items()):
        if record["angles"] != set(range(0, 90, 5)):
            raise ValueError(f"Expected all 18 Sunny views for {source}")
        counts = record["counts"]
        candidates = np.flatnonzero(counts == counts.max())
        means = record["posterior_sum"] / 18
        main = int(candidates[np.argmax(means[candidates])])
        rows.append({
            "source_file": source,
            "scene_index": record["scene_index"], "regime": record["regime"],
            "main_scene_id": main, "n_views": 18,
            "main_scene_view_count": int(counts[main]),
            "consistency_percent": 100.0 * int(counts[main]) / 18,
            "n_distinct_scenes": int(np.count_nonzero(counts)),
            "mode_tied": bool(len(candidates) > 1),
            "tied_scene_ids": ";".join(str(i) for i in candidates) if len(candidates) > 1 else "",
            "main_scene_mean_posterior": float(means[main]),
            **{f"scene_{i}_view_count": int(counts[i]) for i in range(n_scenes)},
        })
    return rows


def plot_population(classifications, summary_path, output_dir):
    """Write modal-file and all-view populations, consistency CSV and aggregate JSON."""
    with Path(summary_path).open() as handle:
        comparison = json.load(handle)
    n_scenes = comparison["settings"]["n_components"]
    rows = summarize_files(classifications, n_scenes)
    if len(rows) != comparison["sunny_spectrum_count"]:
        raise ValueError("CSV file count disagrees with comparison summary")
    population = np.bincount(
        [row["main_scene_id"] for row in rows], minlength=n_scenes
    )
    view_population = np.array([
        sum(row[f"scene_{scene}_view_count"] for row in rows)
        for scene in range(n_scenes)
    ], dtype=np.int64)
    consistency = np.array([row["consistency_percent"] for row in rows])
    complete = sum(row["main_scene_view_count"] == row["n_views"] for row in rows)
    report = {
        "classification_source": str(classifications),
        "comparison_summary": str(summary_path),
        "n_scenes": n_scenes, "n_files": len(rows),
        "scene_file_counts": population.tolist(),
        "scene_view_counts": view_population.tolist(),
        "n_file_view_assignments": int(view_population.sum()),
        "consistency_definition": "100 * modal hard-label view count / 18",
        "tie_break": "Highest mean posterior over all views among count-tied scenes; lowest ID on exact posterior tie",
        "n_mode_ties": sum(row["mode_tied"] for row in rows),
        "n_fully_consistent_files": complete,
        "fully_consistent_percent": 100.0 * complete / len(rows),
        "mean_consistency_percent": float(consistency.mean()),
        "median_consistency_percent": float(np.median(consistency)),
        "interpretation": "Angular label consistency, not physical ground-truth accuracy or proven algorithm bias",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sunny_file_scene_consistency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "sunny_scene_population.json").open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

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
    parser.add_argument("--input", type=Path, required=True, help="sunny_scenes.csv")
    parser.add_argument("--summary", type=Path, help="Defaults to summary.json beside input")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/diagnostics/abi_sunny/comparison"))
    args = parser.parse_args(argv)
    report = plot_population(
        args.input, args.summary or args.input.parent / "summary.json", args.output_dir
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
