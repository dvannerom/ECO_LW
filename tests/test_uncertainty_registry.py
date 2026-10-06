"""Focused tests for traceable, paired uncertainty registration."""

import tempfile
import unittest
from pathlib import Path

import yaml

from build_uncertainty_registry import (
    _gap_records,
    _paired_error_rows,
    _paired_statistics,
    _paired_terms,
    _requirement_reference,
    _scenario_checks,
)
from evaluate_eco_spectral_reconstruction import requirement_summary
from uncertainty import build_provenance, check_provenance


ROOT = Path(__file__).resolve().parents[1]


class UncertaintyRegistryTests(unittest.TestCase):
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
            },
            {
                "n2bc_only": {"error_percent": 3.0},
                "end_to_end": {"error_percent": 6.0},
            },
            {
                "n2bc_only": {"error_percent": 5.0},
                "end_to_end": {"error_percent": 10.0},
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
        self.assertEqual(result["total_end_to_end"]["signed_bias_percent"], 6.0)
        self.assertEqual(
            result["covariance_spectral_with_angular_increment_percent_squared"],
            4.0,
        )
        self.assertEqual(result["total_end_to_end"]["standard_deviation_percent"], 4.0)

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
            {("unstratified", "1"): paired_records},
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
                            "envelope_band_um": None,
                            "envelope_edge_slope_per_um": None,
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
            "scene_index,regime,fold,relative_error_percent\n"
        )
        rows = (
            "scenario_a,n2bc_only,unstratified,1,0001,clear_sky,2,1.0\n"
            "scenario_a,n2bc_only,unstratified,1,0001,cloudy,2,3.0\n"
            "scenario_a,end_to_end,unstratified,1,0001,clear_sky,2,2.0\n"
            "scenario_a,end_to_end,unstratified,1,0001,cloudy,2,4.0\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            residual_path = Path(temporary_directory) / "residuals.csv"
            residual_path.write_text(header + rows)
            paired_scenarios = list(_paired_error_rows(residual_path))
            self.assertEqual(len(paired_scenarios), 1)
            self.assertEqual(
                len(paired_scenarios[0][1][("unstratified", "1")]), 2
            )

            residual_path.write_text(
                header
                + rows.replace(
                    "n2bc_only,unstratified,1,0001,cloudy,2,3.0",
                    "n2bc_only,unstratified,1,0001,cloudy,3,3.0",
                ).replace(
                    "end_to_end,unstratified,1,0001,cloudy,2,4.0",
                    "end_to_end,unstratified,1,0001,cloudy,3,4.0",
                )
            )
            with self.assertRaisesRegex(ValueError, "crosses validation folds"):
                list(_paired_error_rows(residual_path))


if __name__ == "__main__":
    unittest.main()
