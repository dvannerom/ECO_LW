"""Regression checks for registry-driven uncertainty figures."""

import copy
import csv
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import yaml

from build_uncertainty_registry import _paired_statistics, _requirement_reference


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_uncertainty_registry", ROOT / "plotting" / "plot_uncertainty_registry.py"
)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def fixture_registry():
    statistics = _paired_statistics([
        {"n2bc_only": {"error_percent": spectral},
         "end_to_end": {"error_percent": total}, "population_weight": 1.0}
        for spectral, total in ((-2, 1), (0, 2), (2, 3))
    ])
    term = {
        "term_id": "sunny_paired_n2bc_adm:example:abi_population_weighted:degree_2",
        "stage": "N2BC application / broadband flux",
        "eco_scenario": {"id": "example"},
        "scene_geometry_spectral_domain": {
            "regression_training_mode": "abi_population_weighted",
            "regression_degree": 2,
        },
        "statistics_by_scene_domain": {
            group: copy.deepcopy(statistics)
            for group in ("all_regimes", "clear_sky", "cloudy")
        },
        "evidence_status": "conditional_simulation",
        "freshness": {"status": "stale"},
    }
    return {
        "schema_version": 1,
        "stage_order": ["Radiances", "ADM angular conversion", "N2BC application", "Broadband flux", "Aggregation"],
        "quantified_terms": [
            term,
            {"term_id": "abi_between_day_adm:ch08:proxy_scene_0",
             "stage": "ADM angular conversion",
             "evidence_status": "empirical_proxy_not_ECO_uncertainty",
             "freshness": {"status": "unverified"}},
        ],
        "unquantified_terms": [
            {"stage": "Radiances", "term_id": "gap:noise"},
            {"stage": "Broadband flux / aggregation", "term_id": "gap:validation"},
        ],
    }


class RegistryPlotTests(unittest.TestCase):
    def numerical_fixture(self):
        registry = fixture_registry()
        with (ROOT / "config/error_budget.yaml").open() as source:
            budget = yaml.safe_load(source)
        registry["quantified_terms"][0]["requirement_references"] = [
            _requirement_reference(budget, requirement_id)
            for requirement_id in ("ObsReq_10", "ObsReq_11", "ObsReq_15", "ObsReq_16")
        ]
        registry["requirement_conventions"] = {
            "Sunny_simulation_reference_um": [2.5, 500],
            "ObsReq_8_reference_um": [4, 100],
        }
        return registry

    def test_measured_sensitivity_rows_and_missing_results(self):
        registry = self.numerical_fixture()
        rows = plotter.experiment_budget_rows(
            registry, registry["quantified_terms"][0], plotter.load_experiment_spec()
        )
        pixel = {"count": 9, "bias_w_m2": .1, "sd_w_m2": .2, "rmse_w_m2": .3}
        split = {"G16": {"pixel": pixel}, "G18": {"pixel": pixel}}
        classes = {"bias_w_m2": .4, "scene_sd_w_m2": .5, "class_rmse_w_m2": .6}
        abi_groups = {"components_minus_1": "scene_identification", "adm_quadratic": "angular",
                      "n2b_quadratic": "spectral"}
        report = {
            "schema_version": 2, "provenance": {},
            "abi": {"variants": {name: {
                "source_group": group, "description": name,
                "splits": {"held_in": split, "held_out": split, "all": split},
                "monthly_10km": {"G16": {**pixel, "rmse_w_m2": .05}, "G18": {**pixel, "rmse_w_m2": .07}},
            } for name, group in abi_groups.items()}},
            "sunny": {"scenario": "rfma_goal_6", "baseline_truth_error_w_m2": classes, "variants": {
                "n2bc_robust": {"source_group": "spectral", "description": "", "status": "quantified",
                                "paired_difference_w_m2": classes, "paired_difference_percent": classes},
                "adm_per_file": {"source_group": "angular", "description": "", "status": "unsupported",
                                 "reason": "no support"},
            }},
        }
        with patch.object(plotter, "check_provenance", return_value={"status": "current"}):
            measured = plotter.sensitivity_budget_rows(rows, report)
            self.assertIn("nominal Sunny/ECO budget", measured[0][2])
            abi_row = next(row for row in measured if row[0] == "ABI proxy sensitivity: components_minus_1")
            self.assertIn("ABI components_minus_1 (G16/G18)", abi_row[2])
            self.assertIn("monthly 10 km RMSE 0.050/0.070 W/m2", abi_row[2])
            per_file_row = next(row for row in measured if row[0] == "Sunny/ECO sensitivity: adm_per_file")
            self.assertIn("adm_per_file: unsupported", per_file_row[2])
            robust_row = next(row for row in measured if row[0] == "Sunny/ECO sensitivity: n2bc_robust")
            self.assertIn("ECO/Sunny n2bc_robust: bias +0.400; RMSE 0.600 W/m2", robust_row[2])
            self.assertIn("Not added", robust_row[2])
            self.assertEqual(len(measured), len(rows) + 5)
            self.assertIn("never added", measured[7][2])
            self.assertEqual(rows[0][2], "Pending: NEdT is an input, not a broadband-flux error.")
            pixel["rmse_w_m2"] = float("nan")
            with self.assertRaisesRegex(ValueError, "Invalid sensitivity"):
                plotter.sensitivity_budget_rows(rows, report)
            report["schema_version"] = 1
            with self.assertRaisesRegex(ValueError, "schema"):
                plotter.sensitivity_budget_rows(rows, report)
        report["schema_version"] = 2
        with patch.object(plotter, "check_provenance",
                          return_value={"status": "stale", "mismatches": ["code"]}):
            with self.assertRaisesRegex(ValueError, "Stale"):
                plotter.sensitivity_budget_rows(rows, report)

    def test_unified_budget_keeps_proposed_impacts_separate_from_evidence(self):
        registry = self.numerical_fixture()
        specification = plotter.load_experiment_spec()
        rows = plotter.experiment_budget_rows(
            registry, registry["quantified_terms"][0], specification
        )
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(len(row) == 3 for row in rows))
        self.assertEqual(
            [row[0] for row in rows[:5]],
            ["Radiometric noise", "Spatial resolution", "Scene identification",
             "Radiance-to-flux conversion", "Narrowband-to-broadband conversion"],
        )
        self.assertTrue(all(row[2].startswith("Pending") for row in rows[:5]))
        self.assertIn("0.4 K at 255 K", rows[0][1])
        self.assertIn("5x5", rows[1][1])
        self.assertIn("10%", rows[2][1])
        self.assertIn("Pending model comparison", rows[3][2])
        self.assertIn("Bias +2.000%; SD 1.000%; RMSE 2.160%", rows[-1][2])
        self.assertIn("Cov -2.0000 % squared", rows[5][2])
        self.assertIn("Full ECO total pending", rows[-1][2])
        self.assertNotIn("Clear", str(plotter.ECO_BUDGET_COLUMNS))
        self.assertNotIn("Cloud", str(plotter.ECO_BUDGET_COLUMNS))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "budget.csv"
            plotter.write_budget_csv(rows, output)
            with output.open(newline="") as source:
                exported = list(csv.reader(source))
            self.assertEqual(exported[0], list(plotter.ECO_BUDGET_COLUMNS))
            self.assertEqual(exported[1:], rows)
        del registry["quantified_terms"][0]["statistics_by_scene_domain"]["all_regimes"]
        with self.assertRaisesRegex(ValueError, "pooled"):
            plotter.experiment_budget_rows(
                registry, registry["quantified_terms"][0], specification
            )

    def test_experiment_inputs_match_rfma_and_preserve_sensitivity_caveats(self):
        specification = plotter.load_experiment_spec()
        rows = {row["id"]: row for row in specification["contributors"]}
        with (ROOT / "config/error_budget.yaml").open() as source:
            budget = yaml.safe_load(source)
        channels = budget["requirements"]["tir_radiometric_sensitivity"]["channels"]
        noise = rows["radiometric_noise"]["experiment"]
        self.assertEqual(
            noise["goal_nedt_k"], [channel["goal"] for channel in channels.values()]
        )
        self.assertEqual(
            noise["threshold_nedt_k"], [channel["threshold"] for channel in channels.values()]
        )
        self.assertEqual(noise["reference_temperature_k"], 255)
        self.assertEqual(rows["scene_identification"]["experiment"]["n_components"], [6, 7])
        self.assertEqual(rows["scene_identification"]["experiment"]["second_choice_fraction"], 0.1)
        self.assertIn("two-view", rows["radiance_to_flux"]["experiment"]["caveat"])
        self.assertIn("not absolute truth", rows["spatial_resolution"]["sources"])

    def test_numerical_budget_is_covariance_aware_and_lw_only(self):
        registry = self.numerical_fixture()
        term = registry["quantified_terms"][0]
        rows = plotter.budget_rows(registry, term)
        self.assertEqual(rows[-1][1], "+2.000 / 1.000 / 2.160")
        self.assertIn("Combined currently quantified", rows[-1][0])
        self.assertEqual(rows[0][1], "Unquantified")
        self.assertEqual(rows[6][1], "-2.0000 % squared")
        self.assertEqual(rows[4][-1], "ObsReq 16 OLR\nMax 0.84%")
        self.assertEqual(rows[-1][-1], "ObsReq 15 OLR\nG 1.42%; T 5.83%")
        self.assertNotIn("RSR", str(rows))
        self.assertNotIn("ABI", str(rows))
        self.assertEqual(
            plotter.lw_requirement(registry, "ObsReq_10"),
            "ObsReq 10 OLR\nG 0.3%; T 1.25%",
        )
        term["statistics_by_scene_domain"]["all_regimes"]["total_end_to_end"]["signed_bias_percent"] = 0
        with self.assertRaisesRegex(ValueError, "biases"):
            plotter.budget_rows(registry, term)

    def test_fixed_model_selection_and_missing_data(self):
        registry = self.numerical_fixture()
        self.assertEqual(
            plotter.selected_models(registry, "abi_population_weighted", 2)["example"]["freshness"]["status"],
            "stale",
        )
        with self.assertRaisesRegex(ValueError, "Expected one"):
            plotter.selected_models(registry, "legacy_mode", 3)
        term = registry["quantified_terms"][0]
        del term["statistics_by_scene_domain"]["cloudy"]
        self.assertEqual(plotter.budget_rows(registry, term)[-1][3], "Not available")
        with self.assertRaisesRegex(ValueError, "Missing or inconsistent"):
            plotter.lw_requirement(registry, "ObsReq_1")

    def test_variance_figure_and_required_budget_csv_render(self):
        registry = self.numerical_fixture()
        with tempfile.TemporaryDirectory() as directory:
            variance_figure = Path(directory) / "variance.png"
            budget_csv = Path(directory) / "budget.csv"
            plotter.plot_variance_decomposition(registry, variance_figure)
            plotter.write_registry_budget_csv(
                registry,
                budget_csv,
                "example",
                "abi_population_weighted",
                2,
                plotter.load_experiment_spec(),
            )
            self.assertGreater(variance_figure.stat().st_size, 1000)
            self.assertEqual(variance_figure.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue(budget_csv.is_file())

    def test_increment_rmse_uses_sample_count_and_bias(self):
        statistics = fixture_registry()["quantified_terms"][0]["statistics_by_scene_domain"]["all_regimes"]
        result = plotter.component_statistics(statistics, "angular_plus_interaction_increment")
        np.testing.assert_allclose(result, [2, 1, np.sqrt(14 / 3)])
        # SD is about the mean error, not the zero-error reference.
        np.testing.assert_allclose(
            plotter.component_statistics(statistics, "total_end_to_end"),
            [2, 1, np.sqrt(14 / 3)],
        )

    def test_negative_covariance_is_not_clipped(self):
        statistics = fixture_registry()["quantified_terms"][0]["statistics_by_scene_domain"]["all_regimes"]
        contributions, total = plotter.variance_contributions(statistics)
        np.testing.assert_allclose(contributions, [4, 1, -4])
        self.assertAlmostEqual(total, 1)
        inconsistent = copy.deepcopy(statistics)
        inconsistent["total_end_to_end"]["standard_deviation_percent"] = 10
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            plotter.variance_contributions(inconsistent)

    def test_constant_bias_has_zero_sd_but_nonzero_rmse(self):
        statistics = _paired_statistics([
            {"n2bc_only": {"error_percent": 0}, "end_to_end": {"error_percent": 5},
             "population_weight": 1.0}
            for _ in range(3)
        ])
        np.testing.assert_allclose(
            plotter.component_statistics(statistics, "angular_plus_interaction_increment"),
            [5, 0, 5],
        )
        contributions, total = plotter.variance_contributions(statistics)
        np.testing.assert_array_equal(contributions, [0, 0, 0])
        self.assertEqual(total, 0)

    def test_variance_decomposition_figure_renders(self):
        registry = fixture_registry()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "variance.png"
            plotter.plot_variance_decomposition(registry, output)
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_missing_pairs_and_invalid_covariance_are_explicit(self):
        registry = fixture_registry()
        registry["quantified_terms"] = []
        with self.assertRaisesRegex(ValueError, "no paired"):
            plotter.paired_terms(registry)
        statistics = fixture_registry()["quantified_terms"][0]["statistics_by_scene_domain"]["all_regimes"]
        statistics["covariance_matrix_percent_squared"] = [[1, 2], [2, 1]]
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            plotter.variance_contributions(statistics)


if __name__ == "__main__":
    unittest.main()
