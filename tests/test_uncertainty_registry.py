"""Focused tests for traceable, paired uncertainty registration."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from build_uncertainty_registry import (
    _abi_production_assessment,
    _gap_records,
    _nominal_budget_assessment,
    _paired_error_rows,
    _paired_statistics,
    _paired_terms,
    _requirement_reference,
    _scenario_checks,
    _scene_taxonomy_assessment,
)
from evaluate_eco_spectral_reconstruction import requirement_summary
from uncertainty import build_provenance, check_provenance


ROOT = Path(__file__).resolve().parents[1]


class UncertaintyRegistryTests(unittest.TestCase):
    def test_abi_products_and_sensitivities_are_registered_without_overclaim(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            narrowband = root / "narrowband.nc"
            daily = root / "daily.nc"
            monthly = root / "monthly.nc"
            sensitivity = root / "sensitivity.json"
            for path in (narrowband, daily, monthly):
                path.write_bytes(b"test product")
            pixel = {"count": 10, "bias_w_m2": .1, "sd_w_m2": .2, "rmse_w_m2": .22}
            sensitivity.write_text(json.dumps({
                "schema_version": 2, "scope": "test scope",
                "single_source_terms": {"spatial": "nominal budget report"},
                "abi": {"variants": {"n2b_quadratic": {
                    "source_group": "spectral", "description": "quadratic N2B",
                    "splits": {split: {"G16": {"pixel": pixel}} for split in ("held_in", "held_out", "all")},
                    "monthly_10km": {"G16": {**pixel, "p95_abs_w_m2": .4}},
                }}},
                "sunny": {"variants": {"adm_per_file": {
                    "source_group": "angular", "description": "per-file ADM",
                    "status": "unsupported", "reason": "no support",
                }}},
            }))

            assessment = _abi_production_assessment(
                [narrowband], [daily], monthly, sensitivity, {}
            )
            unmeasured = _abi_production_assessment([narrowband], [daily], monthly, None, {})

        self.assertEqual(len(assessment["product_outputs"]["narrowband_fluxes"]), 1)
        measured = assessment["uncertainty_assessments"][0]
        self.assertEqual(measured["experiment_ids"], ["abi_n2b_quadratic", "sunny_adm_per_file"])
        self.assertEqual(measured["monthly_uncertainty_status"],
                         "paired_model_choice_sensitivity_on_monthly_10km_means")
        self.assertEqual(
            measured["abi_variants"]["n2b_quadratic"]["daily_pixel_paired_difference_w_m2"]["held_out"]["G16"],
            pixel)
        self.assertEqual(measured["sunny_variants"]["adm_per_file"]["status"], "unsupported")
        self.assertEqual(unmeasured["uncertainty_assessments"][0]["monthly_uncertainty_status"],
                         "not_estimated")
        self.assertTrue(
            all(
                gap["missing_is_zero"] is False
                for gap in assessment["product_level_uncertainty_gaps"]
            )
        )

    def test_abi_product_registration_fails_for_missing_products(self):
        with self.assertRaisesRegex(FileNotFoundError, "ABI production products"):
            _abi_production_assessment(
                ["missing.nc"], [], None, None, {}
            )

    def test_nominal_budget_is_registered_as_conditional_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "budget_report.json"
            report_path.write_text(
                """
                {
                  "settings": {
                    "scope": "sampled Sunny evaluation",
                    "scenario": "rfma_goal_6",
                    "reference_domain_um": [4, 100]
                  },
                  "rows": [
                    {
                      "id": "combined",
                      "bias_w_m2": 0.6,
                      "scene_sd_w_m2": 1.8,
                      "random_sd_w_m2": 0.4,
                      "ensemble_rmse_w_m2": 2.0
                    }
                  ]
                }
                """
            )
            assessment = _nominal_budget_assessment(
                report_path,
                {
                    requirement_id: {"id": requirement_id}
                    for requirement_id in ("ObsReq_9", "ObsReq_15", "ObsReq_16")
                },
            )

        self.assertEqual(
            assessment["evidence_status"],
            "conditional_transfer_and_independence_hypotheses",
        )
        self.assertEqual(
            assessment["combined_statistics_w_m2"]["ensemble_rmse_w_m2"], 2.0
        )
        self.assertEqual(
            [item["id"] for item in assessment["requirement_references"]],
            ["ObsReq_9", "ObsReq_15", "ObsReq_16"],
        )
        self.assertIn("not uncertainty propagated", assessment["interpretation"])

    def test_scene_taxonomy_gate_is_registered_as_unpropagated_structural_term(self):
        self.assertIsNone(_scene_taxonomy_assessment(None))
        with tempfile.TemporaryDirectory() as temporary_directory:
            gate_path = Path(temporary_directory) / "gate.json"
            gate_path.write_text(json.dumps({
                "feature_set": "spectral_10", "n_components": 7,
                "selection": {"rationale": "plateau", "selected_by": "x", "date": "2026-10-09"},
                "diagnostics_csv": "diag.csv", "diagnostics_csv_sha256": "a" * 64,
                "structural_taxonomy_choice": {
                    "status": "neighbour_sensitivity_not_propagated",
                    "neighbours": [6, 8], "missing_neighbours": [],
                    "deltas": {"6": {"adm_ratio_rmse": .01}, "8": {"adm_ratio_rmse": -.02}},
                    "max_abs_delta": {"adm_ratio_rmse": .02},
                },
            }))
            assessment = _scene_taxonomy_assessment(gate_path)
        self.assertEqual(assessment["n_components"], 7)
        self.assertEqual(assessment["evidence_status"], "neighbour_sensitivity_not_propagated")
        self.assertEqual(assessment["max_abs_neighbour_delta"]["adm_ratio_rmse"], .02)
        self.assertIn("production-aligned sensitivity study", assessment["interpretation"])

    def test_provenance_detects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            code_path = root / "producer.py"
            config_path = root / "settings.yaml"
            input_path = root / "inputs"
            input_path.mkdir()
            source_file = input_path / "sample.dat"
            code_path.write_text("producer = 1\n")
            config_path.write_text("setting: 1\n")
            source_file.write_text("sample\n")

            provenance = build_provenance(
                root=root,
                code_paths=(code_path,),
                configuration_paths=(config_path,),
                input_paths={"sample_inputs": input_path},
            )
            self.assertEqual(check_provenance(provenance, root)["status"], "current")

            config_path.write_text("setting: 2\n")
            check = check_provenance(provenance, root)
            self.assertEqual(check["status"], "stale")
            self.assertIn("settings.yaml", check["mismatches"])

    def test_covariance_propagation_retains_signed_bias(self):
        records = [
            {
                "n2bc_only": {"error_percent": 1.0},
                "end_to_end": {"error_percent": 2.0},
                "population_weight": 1.0,
            },
            {
                "n2bc_only": {"error_percent": 3.0},
                "end_to_end": {"error_percent": 6.0},
                "population_weight": 1.0,
            },
            {
                "n2bc_only": {"error_percent": 5.0},
                "end_to_end": {"error_percent": 10.0},
                "population_weight": 1.0,
            },
        ]
        result = _paired_statistics(records)
        self.assertEqual(
            result["spectral_only"]["signed_bias_percent"], 3.0
        )
        self.assertEqual(
            result["angular_plus_interaction_increment"]["signed_bias_percent"],
            3.0,
        )
        self.assertAlmostEqual(result["total_end_to_end"]["signed_bias_percent"], 6.0)
        self.assertAlmostEqual(
            result["covariance_spectral_with_angular_increment_percent_squared"],
            4.0,
        )
        self.assertAlmostEqual(result["total_end_to_end"]["standard_deviation_percent"], 4.0)
        self.assertAlmostEqual(result["effective_sample_size"], 3.0)

    def test_paired_statistics_apply_population_weights(self):
        records = [
            {
                "n2bc_only": {"error_percent": spectral},
                "end_to_end": {"error_percent": 2.0 * spectral},
                "population_weight": weight,
            }
            for spectral, weight in ((1.0, 0.6), (3.0, 0.2), (5.0, 0.2))
        ]
        result = _paired_statistics(records)
        self.assertAlmostEqual(result["spectral_only"]["signed_bias_percent"], 2.2)
        self.assertAlmostEqual(result["total_end_to_end"]["signed_bias_percent"], 4.4)
        self.assertAlmostEqual(result["effective_sample_size"], 1.0 / 0.44)
        weights = np.asarray([0.6, 0.2, 0.2])
        spectral = np.asarray([1.0, 3.0, 5.0])
        expected_variance = (
            weights @ (spectral - 2.2) ** 2 / (1.0 - weights @ weights)
        )
        self.assertAlmostEqual(
            result["spectral_only"]["standard_deviation_percent"] ** 2,
            expected_variance,
        )

    def test_requirement_mapping_and_regime_paired_terms(self):
        with (ROOT / "config" / "error_budget.yaml").open() as source:
            error_budget = yaml.safe_load(source)
        requirement_ids = ("ObsReq_8", "ObsReq_15", "ObsReq_16")
        requirements = {
            requirement_id: _requirement_reference(error_budget, requirement_id)
            for requirement_id in requirement_ids
        }
        self.assertEqual(requirement_summary(error_budget, "ObsReq_16")["maximum"], 0.84)
        self.assertEqual(
            requirement_summary(error_budget, "ObsReq_15")["goal"], 1.42
        )
        paired_records = []
        for index, (spectral, end_to_end, regime) in enumerate(
            (
                (1.0, 2.0, "clear_sky"),
                (2.0, 4.0, "cloudy"),
                (3.0, 6.0, "clear_sky"),
                (4.0, 8.0, "cloudy"),
            )
        ):
            paired_records.append(
                {
                    "scene_index": f"{index // 2:04d}",
                    "n2bc_only": {
                        "error_percent": spectral,
                        "regime": regime,
                        "fold": index // 2 + 1,
                    },
                    "end_to_end": {
                        "error_percent": end_to_end,
                        "regime": regime,
                        "fold": index // 2 + 1,
                    },
                    "population_weight": 0.25,
                }
            )
        sources = {
            source_id: {
                "producer_provenance_status": "current",
                "sha256": "test",
                "producer_identity": {},
            }
            for source_id in (
                "eco_n2bc_metrics",
                "eco_adm_metrics",
                "eco_chain_metrics",
                "eco_residuals",
            )
        }
        terms = _paired_terms(
            "scenario_a",
            {("abi_population_weighted", "1"): paired_records},
            {"scenario_a": True},
            {"scenario_a": {"channel_names": ["test_channel"]}},
            requirements,
            sources,
            ROOT / "residuals.csv",
            [2.5, 500.0],
            {"retrieval_view_angles_deg": [0.0, 55.0, 70.0]},
        )
        self.assertEqual(len(terms), 1)
        term = terms[0]
        self.assertEqual(
            [ref["id"] for ref in term["requirement_references"]],
            ["ObsReq_16", "ObsReq_15", "ObsReq_8"],
        )
        self.assertIn("clear_sky", term["statistics_by_scene_domain"])
        self.assertIn("cloudy", term["statistics_by_scene_domain"])
        self.assertEqual(term["freshness"]["status"], "current")

    def test_scenario_mismatch_and_missing_terms_are_explicit(self):
        checks = _scenario_checks(
            ["expected_a", "expected_b"],
            {
                "expected_a": {
                    "source": "catalog",
                    "description": "current",
                    "channel_names": ["a"],
                    "channel_bands_um": [[1.0, 2.0]],
                    "response_model": "smooth_bands",
                },
                "expected_b": {},
            },
            10.0,
            (
                "metrics",
                {
                    "scenarios": {
                        "expected_a": {
                            "source": "catalog",
                            "description": "stale",
                            "channel_names": ["a"],
                            "channel_bands_um": [[1.0, 2.0]],
                            "response_model": "smooth_bands",
                            "edge_slope_per_um": 10.0,
                        }
                    }
                },
                True,
            ),
        )
        self.assertFalse(checks["metrics"]["matches_catalog"])
        self.assertEqual(checks["metrics"]["missing_scenarios"], ["expected_b"])
        self.assertEqual(
            checks["metrics"]["scenario_definition_mismatches"], ["expected_a"]
        )

        with (ROOT / "config" / "error_budget.yaml").open() as source:
            error_budget = yaml.safe_load(source)
        requirements = {
            requirement_id: _requirement_reference(error_budget, requirement_id)
            for requirement in error_budget["unquantified_terms"].values()
            for requirement_id in requirement["requirement_ids"]
        }
        gaps = _gap_records(error_budget, requirements)
        self.assertTrue(gaps)
        self.assertTrue(all(gap["missing_is_zero"] is False for gap in gaps))
        self.assertIn(
            "gap:scene_id_cth_parallax_coregistration",
            {gap["term_id"] for gap in gaps},
        )
        self.assertIn(
            "gap:channel_and_shared_source_covariance",
            {gap["term_id"] for gap in gaps},
        )

    def test_shared_clear_cloud_indices_remain_in_one_validation_fold(self):
        header = (
            "scenario_id,assessment_stage,training_mode,regression_degree,"
            "scene_index,regime,population_weight,fold,relative_error_percent\n"
        )
        rows = (
            "scenario_a,n2bc_only,abi_population_weighted,1,0001,clear_sky,0.5,2,1.0\n"
            "scenario_a,n2bc_only,abi_population_weighted,1,0001,cloudy,0.5,2,3.0\n"
            "scenario_a,end_to_end,abi_population_weighted,1,0001,clear_sky,0.5,2,2.0\n"
            "scenario_a,end_to_end,abi_population_weighted,1,0001,cloudy,0.5,2,4.0\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            residual_path = Path(temporary_directory) / "residuals.csv"
            residual_path.write_text(header + rows)
            paired_scenarios = list(_paired_error_rows(residual_path))
            self.assertEqual(len(paired_scenarios), 1)
            self.assertEqual(
                len(paired_scenarios[0][1][("abi_population_weighted", "1")]), 2
            )

            residual_path.write_text(
                header
                + rows.replace(
                    "n2bc_only,abi_population_weighted,1,0001,cloudy,0.5,2,3.0",
                    "n2bc_only,abi_population_weighted,1,0001,cloudy,0.5,3,3.0",
                ).replace(
                    "end_to_end,abi_population_weighted,1,0001,cloudy,0.5,2,4.0",
                    "end_to_end,abi_population_weighted,1,0001,cloudy,0.5,3,4.0",
                )
            )
            with self.assertRaisesRegex(ValueError, "crosses validation folds"):
                list(_paired_error_rows(residual_path))


if __name__ == "__main__":
    unittest.main()
