"""Label permutation and missing-scene handling in bootstrap detail metrics."""

import numpy as np
import pytest

import gmm_stability as stability


def test_bootstrap_details_match_labels_and_ignore_absent_reference_scene(monkeypatch):
    baseline = np.array([0, 0, 1, 1])
    permuted = np.array([1, 1, 0, 0])
    monkeypatch.setattr(stability, "fit_candidate", lambda *args: (None, None, None, None))
    monkeypatch.setattr(stability, "transform_predict", lambda *args: (None, permuted))
    args = (np.ones((4, 10)), np.ones((4, 10)), baseline, 3, True, .98, 42, 2)
    details = stability.cluster_stability_details(*args)
    assert len(details) == 2
    for repeat in details:
        assert repeat["ari"] == pytest.approx(1)
        assert repeat["scene_jaccard"] == [1., 1., None]
        assert repeat["minimum_scene_jaccard"] == pytest.approx(1)
    assert stability.cluster_stability_metrics(*args) == (1., 0.)
