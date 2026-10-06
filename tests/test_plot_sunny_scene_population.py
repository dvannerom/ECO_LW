"""File-level modal populations, angular consistency, and plot output checks."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from plotting.plot_sunny_scene_population import plot_population, summarize_files


def write_rows(path, files):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_file", "scene_index", "regime", "view_angle_deg", "scene_id",
            "p_scene_0", "p_scene_1", "p_scene_2",
        ])
        for name, labels, probabilities in files:
            for view, (label, posterior) in enumerate(zip(labels, probabilities)):
                writer.writerow([name, "0000", "clear_sky", view * 5, label, *posterior])


class SunnyPopulationTests(unittest.TestCase):
    def test_mode_percentage_and_posterior_tie_break(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.csv"
            write_rows(path, [
                ("consistent", [0] * 18, [[.8, .1, .1]] * 18),
                ("mixed", [0] * 12 + [1] * 6,
                 [[.8, .1, .1]] * 12 + [[.1, .8, .1]] * 6),
                ("tied", [0] * 9 + [1] * 9,
                 [[.6, .3, .1]] * 9 + [[.05, .9, .05]] * 9),
            ])
            rows = {row["source_file"]: row for row in summarize_files(path, 3)}
            self.assertEqual(rows["consistent"]["consistency_percent"], 100)
            self.assertAlmostEqual(rows["mixed"]["consistency_percent"], 100 * 12 / 18)
            self.assertFalse(rows["mixed"]["mode_tied"])
            self.assertEqual(rows["tied"]["main_scene_id"], 1)
            self.assertTrue(rows["tied"]["mode_tied"])
            self.assertEqual(rows["tied"]["consistency_percent"], 50)

    def test_exact_posterior_tie_uses_lowest_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.csv"
            write_rows(path, [("tied", [0] * 9 + [1] * 9,
                               [[.75, .125, .125]] * 9 + [[.125, .75, .125]] * 9)])
            self.assertEqual(summarize_files(path, 3)[0]["main_scene_id"], 0)

    def test_incomplete_duplicate_and_invalid_views_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.csv"
            write_rows(path, [("incomplete", [0] * 17, [[.8, .1, .1]] * 17)])
            with self.assertRaisesRegex(ValueError, "18 Sunny views"):
                summarize_files(path, 3)
            write_rows(path, [
                ("duplicate", [0] * 18, [[.8, .1, .1]] * 18),
                ("duplicate", [0], [[.8, .1, .1]]),
            ])
            with self.assertRaisesRegex(ValueError, "Duplicate view"):
                summarize_files(path, 3)
            write_rows(path, [("bad", [0] * 18, [[.1, .8, .1]] * 18)])
            with self.assertRaisesRegex(ValueError, "Invalid"):
                summarize_files(path, 3)

    def test_output_has_one_count_per_file_and_empty_scene_bin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "views.csv"
            write_rows(path, [
                ("a", [0] * 18, [[.8, .1, .1]] * 18),
                ("b", [1] * 18, [[.1, .8, .1]] * 18),
            ])
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "settings": {"n_components": 3}, "sunny_spectrum_count": 2,
            }))
            report = plot_population(path, summary, root / "figures")
            self.assertEqual(report["scene_file_counts"], [1, 1, 0])
            self.assertEqual(sum(report["scene_file_counts"]), 2)
            self.assertEqual(report["scene_view_counts"], [18, 18, 0])
            self.assertEqual(report["n_file_view_assignments"], 36)
            self.assertEqual(report["fully_consistent_percent"], 100)
            for name in (
                "sunny_scene_population.png", "sunny_scene_population_all_views.png",
                "sunny_scene_consistency.png",
                "sunny_file_scene_consistency.csv", "sunny_scene_population.json",
            ):
                self.assertGreater((root / "figures" / name).stat().st_size, 0)
            with (root / "figures" / "sunny_file_scene_consistency.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            summary.write_text(json.dumps({
                "settings": {"n_components": 3}, "sunny_spectrum_count": 3,
            }))
            with self.assertRaisesRegex(ValueError, "count disagrees"):
                plot_population(path, summary, root / "bad")

    def test_mixed_file_contributes_each_scene_occurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "views.csv"
            write_rows(path, [
                ("mixed", [0] * 12 + [1] * 6,
                 [[.8, .1, .1]] * 12 + [[.1, .8, .1]] * 6),
            ])
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "settings": {"n_components": 3}, "sunny_spectrum_count": 1,
            }))
            report = plot_population(path, summary, root / "figures")
            self.assertEqual(report["scene_file_counts"], [1, 0, 0])
            self.assertEqual(report["scene_view_counts"], [12, 6, 0])
            self.assertEqual(report["n_file_view_assignments"], 18)


if __name__ == "__main__":
    unittest.main()
