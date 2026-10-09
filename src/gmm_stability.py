"""Shared candidate fitting and bootstrap assignment-stability metrics."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def fit_candidate(features, n_components, use_pca, pca_var, random_state, report_n_pc=False):
    scaler = StandardScaler()
    # Float64 avoids spurious non-positive-definite covariances in GaussianMixture.
    scaled = scaler.fit_transform(features).astype(np.float64, copy=False)
    pca = None
    transformed = scaled
    if use_pca:
        full_pca = PCA(svd_solver="full")
        full_pca.fit(scaled)
        n_pc = int(np.searchsorted(np.cumsum(full_pca.explained_variance_ratio_), pca_var) + 1)
        if report_n_pc:
            print("n_pc = " + str(n_pc))
        pca = PCA(n_components=n_pc, svd_solver="full", whiten=False)
        transformed = pca.fit_transform(scaled)

    model = GaussianMixture(
        n_components=n_components,
        n_init=5,
        covariance_type="full",
        random_state=random_state,
        reg_covar=1e-6,
    )
    model.fit(transformed)
    return scaler, pca, model, transformed


def transform_predict(features, scaler, pca, model):
    transformed = scaler.transform(features).astype(np.float64, copy=False)
    if pca is not None:
        transformed = pca.transform(transformed)
    return transformed, model.predict(transformed)


def cluster_stability_metrics(
    train_features,
    test_features,
    baseline_labels,
    n_components,
    use_pca,
    pca_var,
    random_state,
    repeats,
):
    """Measure bootstrap-refit assignment stability on fixed held-out points."""
    rng = np.random.default_rng(random_state)
    ari_scores = []
    for repeat in range(repeats):
        bootstrap_indices = rng.integers(0, train_features.shape[0], train_features.shape[0])
        scaler, pca, model, _ = fit_candidate(
            train_features[bootstrap_indices],
            n_components,
            use_pca,
            pca_var,
            random_state + repeat + 1,
        )
        _, labels = transform_predict(test_features, scaler, pca, model)
        ari_scores.append(adjusted_rand_score(baseline_labels, labels))

    return (
        float(np.min(ari_scores)) if ari_scores else np.nan,
        float(np.std(ari_scores)) if ari_scores else np.nan,
    )
