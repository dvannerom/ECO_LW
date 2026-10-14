"""Exact-only and two-stage component diagnostic plotting."""

import csv
import json
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pytest

from plot_n_components import plot_n_components


@pytest.mark.parametrize("mode", ["all_exact", "different_exact", "partial", "missing"])
def test_adm_curves_follow_exact_evaluation_coverage(tmp_path, monkeypatch, mode):
    rows = []
    for k in (2, 3):
        row = {
            "n_components": k, "heldout_loglik": -1 / k, "icl": 100 / k,
            "ari_min": .9, "ari_std": .01, "minimum_angular_coverage": 7 / 9,
            "exact_flux_improvement": .5, "exact_corrected_flux_std": 8.,
            "shortlisted": not (mode == "partial" and k == 3),
            "all_criteria": True,
        }
        for name in ("adm_ratio_rmse", "worst_channel_adm_rmse", "worst_scene_adm_rmse"):
            row[name] = .1 / k
            row["exact_" + name] = .2 / k if mode == "different_exact" else .1 / k
        if mode == "missing":
            row["exact_worst_scene_adm_rmse"] = np.nan
        for name in (
            "adm_plateau", "occupancy", "stability", "angular_coverage",
            "adm_coverage", "exact_adm_coverage", "exact_improvement",
        ):
            row["criterion_" + name] = True
        rows.append(row)
    table = tmp_path / "diagnostics.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    captured = []
    original_savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        if len(figure.axes) > 2 and figure.axes[2].get_title() == "Physical ADM quality":
            axis = figure.axes[2]
            captured.append((
                [line.get_label() for line in axis.lines],
                [line.get_ydata().copy() for line in axis.lines],
                len(axis.collections),
                [text.get_text() for text in axis.get_legend().get_texts()],
            ))
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    paths = plot_n_components(table, tmp_path / "diagnostics.png")
    assert all(path.stat().st_size > 1000 for path in paths)
    labels, values, scatters, legend = captured[0]
    assert len(labels) == 3
    if mode in ("all_exact", "different_exact"):
        assert all(label.startswith("Exact ") for label in labels)
        assert scatters == 0
        assert legend == labels
        expected = .2 if mode == "different_exact" else .1
        np.testing.assert_allclose(values[0], [expected / 2, expected / 3])
    else:
        assert all(not label.startswith("Exact ") for label in labels)
        assert scatters == 3
        assert len(legend) == 6


def test_full_overview_uses_full_statistics_and_paired_day_changes(tmp_path, monkeypatch):
    rows, details = [], {}
    for k in (2, 3):
        row = {
            "n_components": k, "heldout_loglik": -99, "icl": 9999,
            "full_heldout_loglik": -1 / k, "full_training_icl": 100 / k,
            "ari_min": .8, "heldout_normalized_entropy": .2,
            "heldout_ambiguous_fraction": .1, "full_minimum_occupancy": .1,
            "minimum_day_occupancy": .08, "minimum_effective_angular_bins": 4,
            "minimum_ratio_sensitivity_rms": .02,
            "shortlisted": True, "all_criteria": True,
            "exact_adm_ratio_rmse": .1, "exact_worst_channel_adm_rmse": .2,
            "exact_worst_scene_adm_rmse": .3,
        }
        for name in (
            "adm_plateau", "occupancy", "stability", "angular_coverage",
            "adm_coverage", "exact_adm_coverage", "exact_improvement",
        ):
            row["criterion_" + name] = True
        rows.append(row)
        details[str(k)] = {
            "heldout_day_log_likelihood": {"246": -1 / k, "247": -1 / k},
            "bootstrap": [{"ari": .8, "minimum_scene_jaccard": .6},
                          {"ari": .9, "minimum_scene_jaccard": .7}],
            "scene_ratio_sensitivity_rms": [[.02, .03, .02, .03, .02, .03]] * k,
            "by_day": {
                "246": {"sd": 10 - k, "bias": k, "rmse": 12 - k},
                "247": {"sd": 11 - k, "bias": k + 1, "rmse": 13 - k},
            },
        }
    table = tmp_path / "diagnostics.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"diagnostic_schema_version": 2, "component_details": details}
    Path(str(table) + ".json").write_text(json.dumps(metadata))
    captured = []
    original_savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        if len(figure.axes) > 4:
            lines = {line.get_label(): line.get_ydata().copy()
                     for axis in figure.axes for line in axis.lines
                     if not line.get_label().startswith("_")}
            captured.append((lines, figure))
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    paths = plot_n_components(table, tmp_path / "updated.png")
    assert all(path.stat().st_size > 1000 for path in paths)
    lines, figure = captured[0]
    np.testing.assert_allclose(lines["Full held-out mean log likelihood"], [-.5, -1 / 3])
    np.testing.assert_allclose(lines["Full-training ICL"], [50, 100 / 3])
    np.testing.assert_allclose(lines["SD: equal-day mean"], [0, -1])
    np.testing.assert_allclose(lines["Signed bias: equal-day mean"], [0, 1])
    np.testing.assert_allclose(lines["RMSE: equal-day mean"], [0, -1])
    assert len(figure.axes) == 6  # Five subplots plus the ICL axis.
    fit, robust, adm, flux, occupancy, icl_axis = figure.axes
    assert len(fit.lines) == len(icl_axis.lines) == 1
    assert len(robust.lines) == 1
    assert len(robust.collections) == 4
    assert [text.get_text() for text in robust.get_legend().get_texts()] == [
        "Bootstrap ARI", "Worst matched-scene Jaccard", "Minimum bootstrap ARI",
    ]
    assert len(adm.lines) == 3
    assert len(flux.lines) == 4
    assert not flux.collections
    for line in flux.lines[:3]:
        np.testing.assert_array_equal(line.get_xdata(), [2, 3])
    assert occupancy.get_title() == "Scene occupancy"
    assert not any("angular" in axis.get_ylabel().lower() for axis in figure.axes)
    assert not any("entropy" in label.lower() or "posterior" in label.lower() for label in lines)
    assert any("spatial dependence uncorrected" in text.get_text() for text in figure.texts)
    details["3"]["by_day"].pop("247")
    Path(str(table) + ".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="same nonempty held-out days"):
        plot_n_components(table, tmp_path / "invalid.png")
