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
         "end_to_end": {"error_percent": total}}
        for spectral, total in ((-2, 1), (0, 2), (2, 3))
    ])
    term = {
        "term_id": "sunny_paired_n2bc_adm:example:stratified:degree_2",
        "stage": "N2BC application / broadband flux",
        "eco_scenario": {"id": "example"},
        "scene_geometry_spectral_domain": {
            "regression_training_mode": "clear_cloud_stratified",
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
        for regime in plotter.REGIME_LABELS:
            registry["quantified_terms"].append({
                "term_id": f"sunny_adm_band_flux:example:water_vapour:{regime}",
                "eco_scenario": {"id": "example"},
                "scene_geometry_spectral_domain": {
                    "scene_domain": regime, "channel": "water_vapour",
                },
                "signed_bias": 0.25,
                "statistics": {"standard_deviation_percent": 0.5, "rmse_percent": 0.559},
                "freshness": {"status": "unverified"},
            })
        return registry

    def test_measured_sensitivity_rows_and_missing_results(self):
        registry = self.numerical_fixture()
        rows = plotter.experiment_budget_rows(
            registry, registry["quantified_terms"][0], plotter.load_experiment_spec()
        )
        metrics = {
            "w_m2": {"bias": .1, "sd": .2, "rmse": .3},
            "percent": {"bias": .4, "sd": .5, "rmse": .6},
        }
        report = {
            "schema_version": 1, "provenance": {},
            "settings": {"sunny": {"scenario": "rfma_goal_6"},
                         "abi": {"components": [6, 7], "baseline_components": 7}},
            "results": {
                f"{name}_0": {**metrics, "truth_error": metrics}
                for name in ("sunny_noise", "abi_spatial", "abi_components_6",
                             "abi_assignment", "sunny_adm", "abi_adm",
                             "sunny_robust", "sunny_spectral", "sunny_joint")
            },
            "covariance_percent_squared": [[1, -.1], [-.1, .2]],
            "joint_interactions": [{"w_m2": {"rmse": .01}}],
        }
        with patch.object(plotter, "check_provenance", return_value={"status": "current"}):
            measured = plotter.sensitivity_budget_rows(rows, report)
            self.assertIn("0.300 W/m2", measured[0][2])
            self.assertIn("ABI GMM 6 vs 7", measured[2][2])
            self.assertIn("spectral-only error against truth", measured[4][2])
            self.assertIn("Full ECO total remains unquantified", measured[-1][2])
            self.assertIn("RMSE 0.010 W/m2", measured[5][2])
            self.assertEqual(rows[0][2], "Pending: NEdT is an input, not a broadband-flux error.")
            del report["results"]["sunny_noise_0"]
            with self.assertRaisesRegex(ValueError, "required experiment"):
                plotter.sensitivity_budget_rows(rows, report)
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

    def test_fixed_model_selection_missing_data_and_channel_units(self):
        registry = self.numerical_fixture()
        self.assertEqual(
            plotter.selected_models(registry, "clear_cloud_stratified", 2)["example"]["freshness"]["status"],
            "stale",
        )
        with self.assertRaisesRegex(ValueError, "Expected one"):
            plotter.selected_models(registry, "unstratified", 3)
        term = registry["quantified_terms"][0]
        del term["statistics_by_scene_domain"]["cloudy"]
        self.assertEqual(plotter.budget_rows(registry, term)[-1][3], "Not available")
        rows = plotter.channel_rows(registry, "example")
        self.assertEqual(rows[0][1], "+0.250 / 0.500 / 0.559")
        self.assertEqual(rows[0][-1], "unverified")
        with self.assertRaisesRegex(ValueError, "No ADM"):
            plotter.channel_rows(registry, "missing")
        with self.assertRaisesRegex(ValueError, "Missing or inconsistent"):
            plotter.lw_requirement(registry, "ObsReq_1")

    def test_three_numerical_companions_render(self):
        registry = self.numerical_fixture()
        with tempfile.TemporaryDirectory() as directory:
            outputs = [Path(directory) / f"{name}.png" for name in ("budget", "scenarios", "channels")]
            plotter.plot_numerical_budget(
                registry, outputs[0], "example", "clear_cloud_stratified", 2,
                csv_path=Path(directory) / "budget.csv",
            )
            self.assertTrue((Path(directory) / "budget.csv").is_file())
            plotter.plot_scenario_budgets(
                registry, outputs[1], "clear_cloud_stratified", 2
            )
            plotter.plot_channel_diagnostics(registry, outputs[2])
            for output in outputs:
                self.assertGreater(output.stat().st_size, 1000)
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

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
        self.assertEqual(total, 1)
        inconsistent = copy.deepcopy(statistics)
        inconsistent["total_end_to_end"]["standard_deviation_percent"] = 10
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            plotter.variance_contributions(inconsistent)

    def test_map_keeps_missing_proxy_and_stale_separate(self):
        rows = {row["stage"]: row for row in plotter.evidence_rows(fixture_registry())}
        self.assertEqual(rows["Radiances"]["simulation"], 0)
        self.assertEqual(rows["Radiances"]["gaps"], 1)
        self.assertEqual(rows["ADM angular conversion"]["proxy"], 1)
        self.assertEqual(rows["ADM angular conversion"]["simulation"], 0)
        self.assertEqual(rows["ADM angular conversion"]["freshness"], "unverified")
        self.assertEqual(rows["N2BC application"]["freshness"], "stale")
        self.assertEqual(rows["Aggregation"]["gaps"], 1)
        self.assertEqual(rows["Aggregation"]["propagation"], "No propagated\nECO estimate")

    def test_narrowband_diagnostics_cover_both_stages(self):
        registry = fixture_registry()
        registry["stage_order"].append("Narrowband fluxes")
        registry["quantified_terms"].append({
            "term_id": "sunny_adm_band_flux:example:channel:all_regimes",
            "stage": "ADM angular conversion / narrowband flux",
            "evidence_status": "conditional_simulation",
            "freshness": {"status": "current"},
        })
        rows = {row["stage"]: row for row in plotter.evidence_rows(registry)}
        self.assertEqual(rows["ADM angular conversion"]["simulation"], 1)
        self.assertEqual(rows["Narrowband fluxes"]["simulation"], 1)
        self.assertIn("Effect in paired output", rows["Narrowband fluxes"]["propagation"])

    def test_constant_bias_has_zero_sd_but_nonzero_rmse(self):
        statistics = _paired_statistics([
            {"n2bc_only": {"error_percent": 0}, "end_to_end": {"error_percent": 5}}
            for _ in range(3)
        ])
        np.testing.assert_allclose(
            plotter.component_statistics(statistics, "angular_plus_interaction_increment"),
            [5, 0, 5],
        )
        contributions, total = plotter.variance_contributions(statistics)
        np.testing.assert_array_equal(contributions, [0, 0, 0])
        self.assertEqual(total, 0)

    def test_three_figures_render_with_single_scenario(self):
        registry = fixture_registry()
        with tempfile.TemporaryDirectory() as directory:
            for function, name in (
                (plotter.plot_evidence_map, "coverage.png"),
                (plotter.plot_residual_summary, "residuals.png"),
                (plotter.plot_variance_decomposition, "variance.png"),
            ):
                output = Path(directory) / name
                function(registry, output)
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
