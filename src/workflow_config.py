"""Explicit, shallow composition of workflow study settings."""

from pathlib import Path

import yaml


def load_settings(path):
    """Load a YAML mapping, optionally extending one file relative to this file.

    Overrides replace whole top-level values; nested mappings are not merged.
    Cycles, missing files and malformed settings are errors.
    """
    return _load(Path(path), ())


def _load(path, parents):
    path = path.resolve()
    if path in parents:
        raise ValueError(f"Cyclic settings inheritance: {path}")
    with path.open() as source:
        settings = yaml.safe_load(source)
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: settings must be a mapping")
    base = settings.pop("extends", None)
    if base is None:
        return settings
    if not isinstance(base, str) or not base.strip():
        raise ValueError(f"{path}: extends must name one settings file")
    return {**_load(path.parent / base, (*parents, path)), **settings}


def settings_paths(path):
    """Return every configuration file in a composition for provenance tracking."""
    paths = []
    while path is not None:
        path = Path(path).resolve()
        if path in paths:
            raise ValueError(f"Cyclic settings inheritance: {path}")
        paths.append(path)
        with path.open() as source:
            settings = yaml.safe_load(source)
        if not isinstance(settings, dict):
            raise ValueError(f"{path}: settings must be a mapping")
        base = settings.get("extends")
        if base is not None and (not isinstance(base, str) or not base.strip()):
            raise ValueError(f"{path}: extends must name one settings file")
        path = path.parent / base if base is not None else None
    return paths


def resolve_components(settings, taxonomy, sensitivity):
    """Resolve an explicit candidate list or the reviewed K and enabled neighbours."""
    request = settings["components"]
    if isinstance(request, list):
        if (not request or len(set(request)) != len(request)
                or any(isinstance(k, bool) or not isinstance(k, int) or k < 2 for k in request)):
            raise ValueError("components must contain unique integers >= 2")
        return settings
    if request != "reviewed_and_adjacent":
        raise ValueError("components must be a list or reviewed_and_adjacent")
    count = taxonomy["selection"]["n_components"] or 2
    candidates = set(load_settings(taxonomy["diagnostics"]["settings"])["components"])
    variants = set(sensitivity["abi_variants"]) | set(sensitivity["sunny_variants"])
    components = {count}
    for name, shift in (("components_minus_1", -1), ("components_plus_1", 1)):
        if name in variants and count + shift in candidates:
            components.add(count + shift)
    return {**settings, "components": sorted(components)}
