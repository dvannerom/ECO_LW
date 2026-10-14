"""Numerical file-level Sunny population and angular-consistency summaries."""

import csv
import json
from pathlib import Path

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

def write_population_summary(classifications, summary_path, output_csv, output_json):
    """Export O(files * scenes) metadata; never load the original spectral cubes."""
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
        "schema_version": 1,
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
    output_csv, output_json = Path(output_csv), Path(output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with output_json.open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    return report
