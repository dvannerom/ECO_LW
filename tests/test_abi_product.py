"""Production ABI operators: N2B polynomial, ADM chain and ADM library fit."""

import json

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures

import abi_product
import nominal_streaming
from broadband import polynomial_regression, polynomial_term_count
from sensitivity import abi_broadband, angular_basis, angular_integral

PLANCK = np.broadcast_to([200000., 1300., 0.1, 0.999], (6, 4)).copy()


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_regression_matches_sklearn_term_order(degree):
    rng = np.random.default_rng(degree)
    bt = rng.uniform(200, 310, (500, 6))
    terms = polynomial_term_count(degree)
    coefficients = rng.normal(scale=10.0**(-2 * degree), size=terms)
    design = PolynomialFeatures(degree, include_bias=False).fit_transform(bt)
    assert design.shape[1] == terms
    expected = design @ coefficients + 3.
    np.testing.assert_allclose(polynomial_regression(bt, coefficients, 3., degree),
                               expected, rtol=1e-12, atol=1e-9)


def test_satellite_chain_matches_sensitivity_operator_and_masks():
    rng = np.random.default_rng(1)
    n, k = 200, 3
    library = rng.uniform(-.25, .1, (k, 6, 1))
    labels = rng.integers(0, k, n)
    angles = rng.uniform(0, 69, n)
    bt = rng.uniform(220, 300, (n, 6))
    radiance = 200000. / np.expm1(1300. / bt)
    coefficients = np.zeros(83)
    coefficients[:6] = 1 / 6
    integral = angular_integral(library, "regularized")
    regression = (coefficients, 0., 3)
    _, flux = abi_product.satellite_chain(
        radiance, angles, labels, PLANCK, library, "regularized", integral, regression, 70.)
    reference = abi_broadband(
        np.stack([radiance] * 2, axis=1), np.stack([angles] * 2, axis=1),
        np.stack([PLANCK] * 2), labels, library, "regularized", (coefficients, 0.))
    np.testing.assert_allclose(flux, reference[:, 0], rtol=2e-5)

    spectral_irradiance = np.empty((n, 6), dtype=np.float32)
    corrected_expected = abi_product.adm_correct(
        radiance, angles, labels, library, "regularized", integral
    )
    _, _ = abi_product.satellite_chain(
        radiance, angles, labels, PLANCK, library, "regularized", integral,
        None, 70., spectral_irradiance_out=spectral_irradiance,
    )
    np.testing.assert_allclose(
        spectral_irradiance,
        corrected_expected * (np.pi / 1000.),
        rtol=1e-6,
    )

    labels[0], angles[1], radiance[2, 3] = -1, 75., 0.
    bt_out, flux = abi_product.satellite_chain(
        radiance, angles, labels, PLANCK, library, "regularized", integral, regression, 70.)
    assert np.isnan(flux[:3]).all() and np.isnan(bt_out[:3]).all()
    assert np.isfinite(flux[3:]).all()


class FixedScenes:
    def __init__(self, components):
        self.n_components = components

    def __getitem__(self, index):
        return self

    def predict(self, features):
        return features[:, 0].astype(np.int64)


@pytest.mark.parametrize("form", ["regularized", "quadratic"])
def test_fit_library_recovers_paired_ratio_truth(tmp_path, monkeypatch, form):
    rng = np.random.default_rng(7)
    k, n = 2, 4000
    width = 1 if form == "regularized" else 2
    truth = rng.uniform(-.2, .05, (k, 6, width))
    labels = np.arange(n) % k
    angles = np.column_stack([rng.uniform(5, 70, n), rng.uniform(5, 70, n)])
    shape = 1 + np.einsum("nsp,ncp->nsc", angular_basis(angles, form), truth[labels])
    corrected = 50. * shape
    features = labels[:, None].astype(float)
    monkeypatch.setattr(nominal_streaming, "open_features",
                        lambda settings, day: ({"pixels": n}, None))
    monkeypatch.setattr(nominal_streaming, "paired_records", lambda settings, days: iter(
        [(None, None, corrected, angles, None, features)]))
    settings = {"training_days": [245], "batch_points": 1000}
    library = nominal_streaming.fit_library(settings, FixedScenes(k), tmp_path, form=form)
    assert library.shape == (k, 6, width)
    np.testing.assert_allclose(library, truth, atol=2e-4)
    assert not (tmp_path / "adm_fit_records.npy").exists()


def test_load_adm_library_reads_form_sidecar(tmp_path):
    library = np.full((2, 6, 2), .01)
    path = tmp_path / "adm_k2_quadratic.joblib"
    joblib.dump(library, path)
    (tmp_path / "adm_k2_quadratic.joblib.json").write_text(json.dumps({"form": "quadratic"}))
    loaded, form, _ = abi_product.load_adm_library(path, 2)
    assert form == "quadratic"
    np.testing.assert_array_equal(loaded, library)
    with pytest.raises(ValueError, match="expected 3"):
        abi_product.load_adm_library(path, 3)


def test_create_product_assigns_spectral_irradiance_units(tmp_path):
    target = abi_product.create_product(
        tmp_path / "narrowband.nc",
        2,
        3,
        ("spectral_irradiance_G16", "BT_G16", "flux_G16"),
        {},
        1,
    )
    try:
        assert target.variables["spectral_irradiance_G16"].units == "W m-2 (cm-1)-1"
        assert target.variables["BT_G16"].units == "K"
        assert target.variables["flux_G16"].units == "W m-2"
    finally:
        target.close()
