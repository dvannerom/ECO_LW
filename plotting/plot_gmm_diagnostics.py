#!/usr/bin/env python3
"""Plot PCA diagnostics for a saved GMM scene model."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parallel_gmm import load_pipeline
from scene_features import build_scene_features, feature_names


def plot_gmm_diagnostics(
    input_files,
    model_file,
    output_dir=ROOT / "figures" / "scene_id",
    pca_var=0.98,
):
    """Recompute full-PCA diagnostics and plot the saved model's selected PCs."""
    pipeline = load_pipeline(model_file)
    if "pca" not in pipeline.named_steps:
        raise ValueError("The saved model does not contain a PCA step")

    features = np.concatenate(
        [build_scene_features(input_file, pixel_step=3)[0] for input_file in input_files],
        axis=0,
    )
    scaled = StandardScaler().fit_transform(features)
    full_pca = PCA(svd_solver="full").fit(scaled)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    selected_pca = pipeline.named_steps["pca"]
    n_components = selected_pca.n_components_

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        np.arange(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o",
    )
    axis.axhline(pca_var, color="r", linestyle="--", label=f"Target var={pca_var:.2f}")
    axis.axvline(n_components, color="g", linestyle=":", label=f"Selected PCs={n_components}")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Cumulative explained variance")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_scree_merged_2km.png", dpi=150)
    plt.close(fig)

    loadings = selected_pca.components_.T
    pc_to_show = loadings.shape[1]
    names = feature_names()[: loadings.shape[0]]
    fig, axes = plt.subplots(pc_to_show, 1, figsize=(8, 2.2 * pc_to_show), sharex=True)
    axes = np.atleast_1d(axes)
    for index in range(pc_to_show):
        axes[index].bar(np.arange(len(names)), loadings[:, index])
        axes[index].set_title(f"PC{index + 1} loadings")
        axes[index].grid(True, alpha=0.3)
    axes[-1].set_xticks(np.arange(len(names)))
    axes[-1].set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_loadings_merged_2km.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", nargs="+", required=True, help="Training preprocessed NetCDF files")
    parser.add_argument("--model", type=Path, required=True, help="Saved GMM pipeline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "scene_id")
    parser.add_argument("--pca-var", type=float, default=0.98)
    args = parser.parse_args()
    plot_gmm_diagnostics(args.input_file, args.model, args.output_dir, args.pca_var)
