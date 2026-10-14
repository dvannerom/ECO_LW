"""Schema and validation tests for nominal grouped-held-out flux products."""

import json

import numpy as np
import pytest

from nominal_products import PRODUCT_SCOPE, write_heldout_products
from plotting.plot_nominal_flux import plot_product


def test_writes_separate_clean_flux_products_with_case_provenance(tmp_path, monkeypatch):
    n_cases = 4
    clean_flux = np.arange(1, n_cases*6+1, dtype=float).reshape(n_cases, 6)
    clean_broadband = np.array([10., 11., 12., 13.])
    reference = np.array([8., 9., 10., 11.])
    weights = np.full(n_cases, 1/n_cases)
    groups = np.array(["g0", "g0", "g1", "g1"])
    regimes = np.array(["clear", "cloudy", "clear", "cloudy"])
    links = write_heldout_products(
        tmp_path,
        clean_flux=clean_flux,
        clean_broadband=clean_broadband,
        reference=reference,
        weights=weights,
        labels=np.array([0, 1, 0, 1]),
        groups=groups,
        regimes=regimes,
        fold_ids=np.array([0, 0, 1, 1]),
        scenario="rfma_goal_6",
        provenance={"runner": "nominal-budget"},
        true_band_flux=clean_flux + 1,
        angles_deg=[0, 35, 70],
        reference_domain_um=[4, 100],
        channel_definition={"channel_names": [f"c{i}" for i in range(6)]},
    )

    narrowband_path = tmp_path/links["narrowband_flux"]["npz"]
    broadband_path = tmp_path/links["broadband_flux"]["npz"]
    assert narrowband_path.parent.name == "narrowband_flux"
    assert broadband_path.parent.name == "broadband_flux"
    assert "rfma_goal_6_grouped_heldout_clean" in narrowband_path.name
    assert "rfma_goal_6_grouped_heldout_clean" in broadband_path.name
    with np.load(narrowband_path) as narrowband:
        np.testing.assert_array_equal(narrowband["clean_narrowband_flux_w_m2"], clean_flux)
        np.testing.assert_array_equal(narrowband["reference_narrowband_flux_w_m2"], clean_flux + 1)
        np.testing.assert_array_equal(narrowband["case_index"], np.arange(n_cases))
        np.testing.assert_array_equal(narrowband["groups"], groups)
        np.testing.assert_array_equal(narrowband["regimes"], regimes)
        np.testing.assert_array_equal(narrowband["reference_truth_w_m2"], reference)
        np.testing.assert_array_equal(narrowband["population_weight"], weights)
    with np.load(broadband_path) as broadband:
        np.testing.assert_array_equal(
            broadband["predicted_clean_broadband_flux_w_m2"], clean_broadband
        )

    for kind, link, flux_key, flux_dimensions in (
        ("narrowband_flux", links["narrowband_flux"], "clean_narrowband_flux_w_m2",
         ["case", "eco_channel"]),
        ("broadband_flux", links["broadband_flux"], "predicted_clean_broadband_flux_w_m2",
         ["case"]),
    ):
        metadata = json.loads((tmp_path/link["metadata"]).read_text())
        assert metadata["schema_version"] == 1
        assert metadata["product_type"] == kind
        assert metadata["scenario"] == "rfma_goal_6"
        assert metadata["scope"] == PRODUCT_SCOPE
        assert metadata["dimensions"]["flux"] == flux_dimensions
        assert metadata["dimensions"]["groups"] == ["case"]
        assert metadata["units"][flux_key] == "W m-2"
        assert metadata["units"]["reference_truth_w_m2"] == "W m-2"
        assert metadata["provenance"] == {"runner": "nominal-budget"}
        assert flux_key in metadata["array_shapes"]
        assert metadata["reference_domain_um"] == [4, 100]
        assert metadata["units"]["retrieval_angles_deg"] == "degree"
        monkeypatch.setattr("plotting.plot_nominal_flux.check_provenance",
                            lambda *_: {"status": "current"})
        output = tmp_path / "figures" / f"{kind}.png"
        plot_product(tmp_path / link["npz"], output)
        assert output.stat().st_size > 0
    monkeypatch.setattr("plotting.plot_nominal_flux.check_provenance",
                        lambda *_: {"status": "stale"})
    with pytest.raises(ValueError, match="Stale nominal flux product"):
        plot_product(broadband_path, tmp_path / "stale.png")
    assert not (tmp_path / "stale.png").exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"clean_flux": np.ones((3, 5))},
        {"clean_broadband": np.array([1., 2.])},
        {"weights": np.array([.2, .2, .2])},
        {"clean_broadband": np.array([1., 0., 3.])},
        {"groups": np.array(["a", "a", "b"]), "fold_ids": np.array([0, 1, 0])},
        {"labels": np.array([0., 1., 0.])},
        {"reference_domain_um": [100, 4]},
        {"channel_definition": {"channel_names": ["one"]}},
    ],
)
def test_rejects_invalid_heldout_product_inputs(tmp_path, updates):
    values = {
        "clean_flux": np.ones((3, 6)),
        "clean_broadband": np.array([1., 2., 3.]),
        "reference": np.array([1., 2., 3.]),
        "weights": np.full(3, 1/3),
        "labels": np.array([0, 1, 0]),
        "groups": np.array(["a", "b", "c"]),
        "regimes": np.array(["clear", "cloudy", "clear"]),
        "fold_ids": np.array([0, 1, 0]),
        "scenario": "rfma_goal_6",
        "provenance": {},
        "true_band_flux": np.ones((3, 6)),
        "angles_deg": [0, 35, 70],
        "reference_domain_um": [4, 100],
        "channel_definition": {"channel_names": [f"c{i}" for i in range(6)]},
    }
    values.update(updates)
    with pytest.raises(ValueError):
        write_heldout_products(tmp_path, **values)
