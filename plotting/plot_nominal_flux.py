"""Plot clean held-out ECO/Sunny flux products, without rerunning retrieval."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncertainty import check_provenance


def plot_product(path, output):
    """Render O(cases * channels) simulation comparisons, not geographical maps."""
    path, output = Path(path), Path(output)
    metadata = json.loads(Path(str(path) + ".json").read_text())
    if metadata["schema_version"] != 1:
        raise ValueError("Unsupported nominal flux product schema")
    freshness = check_provenance(metadata["provenance"], ROOT)
    if freshness["status"] != "current":
        raise ValueError(f"Stale nominal flux product {path}: {freshness}")
    kind = metadata["product_type"]
    if kind not in ("narrowband_flux", "broadband_flux"):
        raise ValueError(f"Unknown nominal product type: {kind}")
    with np.load(path) as data:
        if kind == "narrowband_flux":
            predicted = data["clean_narrowband_flux_w_m2"]
            reference = data["reference_narrowband_flux_w_m2"]
            names = metadata["channel_definition"]["channel_names"]
            figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        else:
            predicted = data["predicted_clean_broadband_flux_w_m2"][:, None]
            reference = data["reference_truth_w_m2"][:, None]
            names = ["Broadband OLR"]
            figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
            axes = np.array([axis])
        for channel, axis in enumerate(axes.flat):
            truth, estimate = reference[:, channel], predicted[:, channel]
            lower, upper = min(truth.min(), estimate.min()), max(truth.max(), estimate.max())
            axis.scatter(truth, estimate, s=8, alpha=.5, rasterized=True)
            axis.plot([lower, upper], [lower, upper], color="black", linewidth=.8)
            axis.set(title=names[channel], xlabel="Sunny reference flux (W m-2)",
                     ylabel="Retrieved flux (W m-2)")
        figure.suptitle(f"{metadata['scenario']}: clean grouped-held-out simulation")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_product(args.input, args.output)


if __name__ == "__main__":
    main()
