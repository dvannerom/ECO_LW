"""Canonical ABI filenames; workflow callers may override their parent roots."""

from pathlib import Path


def preprocessed_path(day, resolution=2, step=1, root="data/preprocessing/abi"):
    return str(Path(root) / f"abi_{day}_res{resolution}km_step{step}.nc")


def scene_id_path(day, resolution, n_components, root="data/abi_proxy/products/scene_id"):
    return str(Path(root) / f"scene_id_{day}_res{resolution}km_{n_components}comp.nc")


def narrowband_path(day, resolution, channel, root="data/abi_proxy/products/narrowband_flux"):
    return str(Path(root) / f"narrowband_flux_{day}_res{resolution}km_C{channel}.nc")


def broadband_path(day, resolution, root="data/abi_proxy/products/broadband_flux"):
    return str(Path(root) / f"broadband_flux_{day}_res{resolution}km.nc")


def candidate_directory(settings):
    return Path(settings.get("candidate_dir", settings["output_dir"]))


def preprocessing_directory(settings):
    return Path(settings.get("training_dir", candidate_directory(settings) / "training_pca"))


def nominal_model_directory(settings):
    return Path(settings.get("nominal_model_dir", settings["output_dir"]))


def nominal_product_path(root, scenario, kind):
    """Name a clean grouped-held-out ECO/Sunny simulation product."""
    if kind not in ("narrowband_flux", "broadband_flux"):
        raise ValueError(f"Unsupported nominal flux product kind: {kind}")
    slug = "".join(char.lower() if char.isalnum() else "_" for char in str(scenario)).strip("_")
    if not slug:
        raise ValueError("Nominal product scenario must contain letters or digits")
    return str(Path(root) / kind / f"eco_sunny_{slug}_grouped_heldout_clean_{kind}.npz")
