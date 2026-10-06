"""Small NumPy helpers for traceable, decomposed uncertainty propagation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


AVERAGING_CLASSES = {"random", "scene_dependent", "systematic"}
_HASH_BLOCK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _path_identity(path: Path, root: Path) -> dict[str, object]:
    """Fingerprint a source by path, size, and mtime without reading large inputs."""
    path = path.resolve()
    if not path.exists():
        return {"path": _relative_path(path, root), "exists": False}
    if path.is_file():
        stat = path.stat()
        return {
            "path": _relative_path(path, root),
            "exists": True,
            "kind": "file",
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    entries = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        entries.append(
            (
                child.relative_to(path).as_posix(),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    serialized = json.dumps(entries, separators=(",", ":"), ensure_ascii=True)
    return {
        "path": _relative_path(path, root),
        "exists": True,
        "kind": "directory_inventory",
        "file_count": len(entries),
        "size_bytes": sum(entry[1] for entry in entries),
        "inventory_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "identity_method": "relative_path_size_mtime_ns",
    }


def _runtime_identity() -> dict[str, str]:
    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "scikit-learn",
        "PyYAML": "PyYAML",
    }
    versions = {}
    for label, distribution in distributions.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "not-installed"
    return {"python": platform.python_version(), **versions}


def build_provenance(
    *,
    root: Path,
    code_paths: Sequence[Path],
    configuration_paths: Sequence[Path],
    input_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Capture exact code/config hashes and bounded-cost input inventory identities."""
    root = root.resolve()

    def hash_files(paths: Sequence[Path]) -> dict[str, str]:
        return {
            _relative_path(path, root): sha256_file(path.resolve())
            for path in paths
        }

    return {
        "schema_version": 1,
        "code_sha256": hash_files(code_paths),
        "configuration_sha256": hash_files(configuration_paths),
        "inputs": {
            name: _path_identity(path, root) for name, path in input_paths.items()
        },
        "runtime": _runtime_identity(),
        "input_identity_note": (
            "Code and configuration use SHA-256. Input files/directories use "
            "path, size, and mtime metadata to avoid reading large source data."
        ),
    }


def check_provenance(provenance: object, root: Path) -> dict[str, object]:
    """Check recorded code, configuration, and input identities against disk."""
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
        return {"status": "provenance_missing_or_unsupported", "mismatches": []}
    root = root.resolve()
    mismatches = []

    for section in ("code_sha256", "configuration_sha256"):
        identities = provenance.get(section)
        if not isinstance(identities, dict):
            mismatches.append(section)
            continue
        for relative_path, expected_hash in identities.items():
            path = Path(relative_path)
            if not path.is_absolute():
                path = root / path
            if not path.is_file() or sha256_file(path) != expected_hash:
                mismatches.append(str(relative_path))

    inputs = provenance.get("inputs")
    if not isinstance(inputs, dict):
        mismatches.append("inputs")
    else:
        for name, expected_identity in inputs.items():
            if not isinstance(expected_identity, dict):
                mismatches.append(str(name))
                continue
            relative_path = Path(str(expected_identity.get("path", "")))
            path = relative_path if relative_path.is_absolute() else root / relative_path
            if _path_identity(path, root) != expected_identity:
                mismatches.append(str(name))

    runtime = provenance.get("runtime")
    if runtime != _runtime_identity():
        mismatches.append("runtime")
    return {
        "status": "stale" if mismatches else "current",
        "mismatches": mismatches,
    }


@dataclass(frozen=True)
class UncertaintyComponent:
    """One output-space variance component with provenance metadata."""

    variance: np.ndarray
    averaging_class: str
    requirement_ids: tuple[str, ...] = ()
    source: Optional[str] = None


@dataclass(frozen=True)
class BiasComponent:
    """One signed output-space bias; biases combine algebraically, not in quadrature."""

    bias: np.ndarray
    requirement_ids: tuple[str, ...] = ()
    source: Optional[str] = None


def _numeric_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinite values")
    return array


def combine_variances(
    variances: Sequence[object], out: Optional[np.ndarray] = None
) -> np.ndarray:
    """Sum independent variances, broadcasting scalars and compatible arrays.

    NaNs are preserved so invalid pixels remain invalid. If ``out`` is supplied,
    it is cleared and reused; it must not overlap any input array.
    """
    arrays = [_numeric_array(value, "variance") for value in variances]
    for array in arrays:
        if np.any(array < 0):
            raise ValueError("variances must be non-negative")

    shape = np.broadcast_shapes(*(array.shape for array in arrays)) if arrays else ()
    if out is None:
        result = np.zeros(shape, dtype=np.float64)
    else:
        result = np.asarray(out)
        if result.shape != shape or not np.issubdtype(result.dtype, np.floating):
            raise ValueError(f"out must be a writable floating array with shape {shape}")
        if not result.flags.writeable:
            raise ValueError("out must be writable")
        if any(np.may_share_memory(result, array) for array in arrays):
            raise ValueError("out must not overlap any input variance")
        result.fill(0.0)

    for array in arrays:
        np.add(result, array, out=result, casting="unsafe")
    return result


def propagate_covariance(jacobian: object, covariance: object) -> np.ndarray:
    """Return ``J C J.T`` as output variance for a linearized scalar output.

    The final Jacobian dimension and covariance's final two dimensions represent
    the input variables. Leading dimensions are broadcast using NumPy rules.
    Covariance matrices must be positive semidefinite; materially negative
    propagated variances raise ``ValueError``.
    """
    derivative = _numeric_array(jacobian, "jacobian")
    matrix = _numeric_array(covariance, "covariance")
    if derivative.ndim < 1 or matrix.ndim < 2:
        raise ValueError("jacobian and covariance need at least 1 and 2 dimensions")
    n_variables = derivative.shape[-1]
    if matrix.shape[-2:] != (n_variables, n_variables):
        raise ValueError("covariance's last dimensions must match the Jacobian")

    variance = np.einsum(
        "...i,...ij,...j->...", derivative, matrix, derivative, optimize=True
    )
    if np.any(np.isinf(variance)):
        raise ValueError("propagated variance is infinite")
    if np.any(variance < 0):
        scale = np.einsum(
            "...i,...ij,...j->...",
            np.abs(derivative),
            np.abs(matrix),
            np.abs(derivative),
            optimize=True,
        )
        tolerance = 32 * np.finfo(np.float64).eps * np.maximum(scale, 1.0)
        if np.any(variance < -tolerance):
            raise ValueError("covariance produced a negative variance")
        np.maximum(variance, 0.0, out=variance)
    return variance


def propagate_covariance_components(
    jacobian: object, covariance_components: Mapping[str, object]
) -> dict[str, np.ndarray]:
    """Propagate each independent source covariance separately through ``J``."""
    return {
        name: propagate_covariance(jacobian, covariance)
        for name, covariance in covariance_components.items()
    }


class UncertaintyBudget:
    """Collect output-space components linked to mission requirement IDs.

    Variance components are summed in quadrature and therefore must be
    independent of one another. Put cross-source/channel covariance into the
    covariance passed to ``propagate_covariance`` before registering a component.
    Signed biases are stored separately and summed algebraically.
    """

    def __init__(self) -> None:
        self.variances: dict[str, UncertaintyComponent] = {}
        self.biases: dict[str, BiasComponent] = {}

    def add_variance(
        self,
        name: str,
        variance: object,
        averaging_class: str,
        requirement_ids: Sequence[str] = (),
        source: Optional[str] = None,
    ) -> None:
        """Register an independent non-negative variance component."""
        if name in self.variances or name in self.biases:
            raise KeyError(f"uncertainty component {name!r} already exists")
        if averaging_class not in AVERAGING_CLASSES:
            raise ValueError(
                f"averaging_class must be one of {sorted(AVERAGING_CLASSES)}"
            )
        values = _numeric_array(variance, "variance")
        if np.any(values < 0):
            raise ValueError("variances must be non-negative")
        self.variances[name] = UncertaintyComponent(
            values, averaging_class, tuple(requirement_ids), source
        )

    def add_bias(
        self,
        name: str,
        bias: object,
        requirement_ids: Sequence[str] = (),
        source: Optional[str] = None,
    ) -> None:
        """Register a signed bias, which is not an uncertainty standard deviation."""
        if name in self.variances or name in self.biases:
            raise KeyError(f"uncertainty component {name!r} already exists")
        values = _numeric_array(bias, "bias")
        self.biases[name] = BiasComponent(values, tuple(requirement_ids), source)

    def variance_components(
        self,
        requirement_id: Optional[str] = None,
        averaging_class: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """Select named variance components by requirement and/or averaging class."""
        if averaging_class is not None and averaging_class not in AVERAGING_CLASSES:
            raise ValueError(
                f"averaging_class must be one of {sorted(AVERAGING_CLASSES)}"
            )
        return {
            name: component.variance
            for name, component in self.variances.items()
            if (requirement_id is None or requirement_id in component.requirement_ids)
            and (
                averaging_class is None
                or averaging_class == component.averaging_class
            )
        }

    def total_variance(
        self,
        requirement_id: Optional[str] = None,
        averaging_class: Optional[str] = None,
    ) -> np.ndarray:
        """Return the quadrature sum for the selected independent components."""
        return combine_variances(
            list(self.variance_components(requirement_id, averaging_class).values())
        )

    def total_uncertainty(
        self,
        requirement_id: Optional[str] = None,
        averaging_class: Optional[str] = None,
    ) -> np.ndarray:
        """Return the standard uncertainty for selected variance components."""
        return np.sqrt(self.total_variance(requirement_id, averaging_class))

    def bias_components(
        self, requirement_id: Optional[str] = None
    ) -> dict[str, np.ndarray]:
        """Select signed biases by requirement ID."""
        return {
            name: component.bias
            for name, component in self.biases.items()
            if requirement_id is None or requirement_id in component.requirement_ids
        }

    def total_bias(self, requirement_id: Optional[str] = None) -> np.ndarray:
        """Return the algebraic sum of selected signed biases."""
        components = list(self.bias_components(requirement_id).values())
        if not components:
            return np.asarray(0.0)
        shape = np.broadcast_shapes(*(component.shape for component in components))
        total = np.zeros(shape, dtype=np.float64)
        for component in components:
            np.add(total, component, out=total, casting="unsafe")
        return total
