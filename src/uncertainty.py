"""Small NumPy helpers for traceable, decomposed uncertainty propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


AVERAGING_CLASSES = {"random", "scene_dependent", "systematic"}


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
