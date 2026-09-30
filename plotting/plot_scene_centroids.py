#!/usr/bin/env python3
"""Plot scene BT, BTD, and spatial stddev centroids from a saved GMM model."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from parallel_gmm import load_pipeline
from scene_features import (
    FEATURE_GROUPS,
    build_scene_features,
    feature_names,
    scene_label_mapping,
)

BTD_LIMITS = {
    "BTD14-11": (-10.0, 10.0),
    "BTD14-15": (-5.0, 10.0),
    "BTD14-08": (-10.0, 80.0),
    "BTD14-16": (-10.0, 40.0),
}


def heatmap(data, row_labels, col_labels, ax, cbarlabel, cmap, value_format):
    image = ax.imshow(data, cmap=cmap)
    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(range(data.shape[1]), labels=col_labels, rotation=30, ha="right")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            ax.text(column, row, value_format.format(data[row, column]), ha="center", va="center")
    return image


def plot_feature_distributions(
    histograms,
    bin_edges,
    scene_counts,
    feature_means,
    underflow,
    overflow,
    names,
    output_dir,
    suffix,
):
    plot_groups = (
        ("averages", "BT (K)"),
        ("differences", "BTD (K)"),
        ("local_std_5x5", "BT standard deviation (K)"),
        ("local_std_9x9", "BT standard deviation (K)"),
    )
    cmap = mpl.cm.turbo
    bounds = [scene_id - 0.5 for scene_id in range(histograms.shape[0] + 1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    colors = cmap(norm(np.arange(histograms.shape[0])))
    paths = []

    for group, value_label in plot_groups:
        columns = np.arange(histograms.shape[1])[FEATURE_GROUPS[group]]
        n_columns = 2
        n_rows = int(np.ceil(len(columns) / n_columns))
        figure, axes = plt.subplots(n_rows, n_columns, figsize=(12, 3.2 * n_rows), squeeze=False)
        for axis, column in zip(axes.flat, columns):
            edges = bin_edges[column]
            bin_width = np.diff(edges)
            for scene_id in range(histograms.shape[0]):
                total = scene_counts[scene_id]
                if total == 0:
                    continue
                density = histograms[scene_id, column] / (total * bin_width)
                axis.plot(
                    edges[:-1] + bin_width / 2,
                    density,
                    color=colors[scene_id],
                    drawstyle="steps-mid",
                    linewidth=1.5,
                )
            outside = int(np.sum(underflow[:, column] + overflow[:, column]))
            title = names[column]
            if outside:
                title += f" ({outside:,} outside plotted range)"
            axis.set_title(title)
            axis.set_xlabel(value_label)
            axis.set_ylabel("Fraction per K")
            logarithmic = group in ("local_std_5x5", "local_std_9x9")
            if logarithmic:
                axis.set_yscale("log")
            axis.grid(alpha=0.2)
            handles = [
                mpl.lines.Line2D(
                    [0],
                    [0],
                    color=colors[scene_id],
                    label=f"Scene {scene_id} (mean={feature_means[scene_id, column]:.1f} K)",
                )
                for scene_id in range(histograms.shape[0])
            ]
            axis.legend(
                handles=handles,
                loc="best",
                ncol=2,
                fontsize=7,
                framealpha=0.85,
                borderpad=0.3,
                labelspacing=0.2,
                handlelength=1.2,
            )

        for axis in axes.flat[len(columns):]:
            axis.remove()
        figure.suptitle(f"All-pixel feature distributions: {group.replace('_', ' ')}", y=0.99)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        path = os.path.join(output_dir, f"feature_distributions_{group}_{suffix}.png")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def plot_scene_centroids(model_file, input_files, output_dir, label_order, chunk_rows=128):
    pipeline = load_pipeline(model_file)
    gmm = pipeline.named_steps["gmm"]
    if "pca" in pipeline.named_steps:
        means_scaled = pipeline.named_steps["pca"].inverse_transform(gmm.means_)
    else:
        means_scaled = gmm.means_
    label_mapping = scene_label_mapping(means_scaled, label_order)

    names = feature_names()
    n_scenes = gmm.n_components
    n_features = len(names)
    histogram_bins = 300
    bin_edges = []
    for column in range(n_features):
        if column < FEATURE_GROUPS["averages"].stop:
            lower, upper = 180.0, 320.0
        elif column < FEATURE_GROUPS["differences"].stop:
            lower, upper = BTD_LIMITS[names[column]]
        else:
            lower, upper = 0.0, 10.0
        bin_edges.append(np.linspace(lower, upper, histogram_bins + 1))

    histograms = np.zeros((n_scenes, n_features, histogram_bins), dtype=np.int64)
    underflow = np.zeros((n_scenes, n_features), dtype=np.int64)
    overflow = np.zeros((n_scenes, n_features), dtype=np.int64)
    scene_counts = np.zeros(n_scenes, dtype=np.int64)
    feature_sums = np.zeros((n_scenes, n_features), dtype=np.float64)
    component_to_scene = np.array([label_mapping[i] for i in range(n_scenes)])

    def accumulate_chunk(features, _flat_indices):
        component_labels = pipeline.predict(features)
        scene_labels = component_to_scene[component_labels]
        for scene_id in range(n_scenes):
            scene_features = features[scene_labels == scene_id]
            if not len(scene_features):
                continue
            scene_counts[scene_id] += len(scene_features)
            feature_sums[scene_id] += np.sum(scene_features, axis=0, dtype=np.float64)
            for column in range(n_features):
                values = scene_features[:, column]
                edges = bin_edges[column]
                underflow[scene_id, column] += np.count_nonzero(values < edges[0])
                overflow[scene_id, column] += np.count_nonzero(values > edges[-1])
                histograms[scene_id, column] += np.histogram(values, bins=edges)[0]

    for input_file in input_files:
        build_scene_features(
            input_file,
            chunk_rows=chunk_rows,
            pixel_step=1,
            chunk_callback=accumulate_chunk,
        )

    if not np.all(scene_counts):
        raise ValueError(f"No valid pixels assigned to scenes: {np.flatnonzero(scene_counts == 0).tolist()}")
    centroids = (feature_sums / scene_counts[:, None]).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    model_stem = Path(model_file).stem.replace("gmm_pipeline_", "")
    suffix = f"{model_stem}_{label_order}"

    plot_groups = (
        ("averages", "centroids", "BT (K)", "rainbow", (10, 6)),
        ("differences", "centroids_diff", "BTD (K)", mpl.cm.bwr, (4, 6)),
        ("local_std_5x5", "centroids_std5x5", "BT stddev (K)", "rainbow", (10, 6)),
        ("local_std_9x9", "centroids_std9x9", "BT stddev (K)", "rainbow", (10, 6)),
    )
    paths = []
    for group, prefix, colorbar_label, cmap, figure_size in plot_groups:
        columns = FEATURE_GROUPS[group]
        figure, axis = plt.subplots(figsize=figure_size)
        heatmap(
            centroids[:, columns],
            [str(index) for index in range(gmm.n_components)],
            names[columns],
            axis,
            colorbar_label,
            cmap,
            "{:.1f}",
        )
        axis.set_xlabel("Feature")
        axis.set_ylabel("Scene ID")
        figure.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_{suffix}.png")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    distribution_paths = plot_feature_distributions(
        histograms,
        bin_edges,
        scene_counts,
        centroids,
        underflow,
        overflow,
        names,
        output_dir,
        suffix,
    )
    return tuple(paths) + distribution_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Saved GMM pipeline")
    parser.add_argument("--input-file", nargs="+", required=True, help="Training preprocessed files")
    parser.add_argument("--output-dir", default="figures/scene_id", help="Directory for PNG output")
    parser.add_argument("--chunk-rows", type=int, default=128, help="Rows processed per feature chunk")
    parser.add_argument(
        "--label-order",
        choices=("c14", "c14_btd14_08"),
        default="c14_btd14_08",
        help="Ordering from cold/icy to hot/clear",
    )
    args = parser.parse_args()
    plot_scene_centroids(args.model, args.input_file, args.output_dir, args.label_order, args.chunk_rows)
