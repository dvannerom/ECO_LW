#!/usr/bin/env python3
"""Plot scene BT and BTD centroids from a saved GMM model."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scene_features import (
    CHANNEL_LABELS,
    DIFFERENCE_LABELS,
    FEATURE_GROUPS,
    build_scene_features,
    scene_label_mapping,
)


FEATURE_NAMES = [f"{label}_av" for label in CHANNEL_LABELS]
DIFFERENCE_NAMES = list(DIFFERENCE_LABELS)


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


def plot_scene_centroids(model_file, input_files, output_dir, label_order):
    pipeline = joblib.load(model_file)
    gmm = pipeline.named_steps["gmm"]
    feature_arrays = [build_scene_features(path, pixel_step=3)[0] for path in input_files]
    features = np.concatenate(feature_arrays, axis=0)
    component_labels = pipeline.predict(features)

    if "pca" in pipeline.named_steps:
        means_scaled = pipeline.named_steps["pca"].inverse_transform(gmm.means_)
    else:
        means_scaled = gmm.means_
    label_mapping = scene_label_mapping(means_scaled, label_order)

    centroids = np.empty((gmm.n_components, 6), dtype=np.float32)
    centroids_diff = np.empty((gmm.n_components, 4), dtype=np.float32)
    for component in range(gmm.n_components):
        scene_id = label_mapping[component]
        component_features = features[component_labels == component]
        centroids[scene_id] = np.mean(component_features[:, FEATURE_GROUPS["averages"]], axis=0)
        centroids_diff[scene_id] = np.mean(component_features[:, FEATURE_GROUPS["differences"]], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    model_stem = Path(model_file).stem.replace("gmm_pipeline_", "")
    suffix = f"{model_stem}_{label_order}"

    figure, axis = plt.subplots(figsize=(10, 6))
    heatmap(
        centroids,
        [str(index) for index in range(gmm.n_components)],
        FEATURE_NAMES,
        axis,
        "BT (K)",
        "rainbow",
        "{:.1f}",
    )
    axis.set_xlabel("Feature")
    axis.set_ylabel("Scene ID")
    figure.tight_layout()
    centroid_path = os.path.join(output_dir, f"centroids_{suffix}.png")
    figure.savefig(centroid_path, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4, 6))
    heatmap(
        centroids_diff,
        [str(index) for index in range(gmm.n_components)],
        DIFFERENCE_NAMES,
        axis,
        "BTD",
        mpl.cm.bwr,
        "{:.1f}",
    )
    axis.set_xlabel("Feature")
    axis.set_ylabel("Scene ID")
    figure.tight_layout()
    difference_path = os.path.join(output_dir, f"centroids_diff_{suffix}.png")
    figure.savefig(difference_path, dpi=150)
    plt.close(figure)
    return centroid_path, difference_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Saved GMM pipeline")
    parser.add_argument("--input-file", nargs="+", required=True, help="Training preprocessed files")
    parser.add_argument("--output-dir", default="figures/scene_id", help="Directory for PNG output")
    parser.add_argument(
        "--label-order",
        choices=("c14", "c14_btd14_08"),
        default="c14_btd14_08",
        help="Ordering from cold/icy to hot/clear",
    )
    args = parser.parse_args()
    plot_scene_centroids(args.model, args.input_file, args.output_dir, args.label_order)
