"""Direct equal-weight ADM fitting without an observed normalization view."""

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import LinearConstraint, minimize

from sensitivity import angular_basis, angular_integral


def retrieve_geometry(radiance, angles, form):
    """Fit L=A+sum(q_j*basis_j), returning flux[scene,channel] in W/m2.

    Radiance has shape (scene,view,channel); angles are degrees. A is the
    fitted 55-degree amplitude, not an observed reference. Memory is bounded
    by scenes*channels*views plus one constrained fit at a time.
    """
    radiance = np.asarray(radiance, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if (radiance.ndim != 3 or radiance.shape[1] != len(angles)
            or not np.all(np.isfinite(radiance)) or np.any(radiance <= 0)
            or np.any(~np.isfinite(angles)) or np.any((angles < 0) | (angles > 70))
            or len(np.unique(angles)) != len(angles)):
        raise ValueError("Need positive radiances and distinct valid 0-70 degree views")
    design = np.column_stack((np.ones(len(angles)), angular_basis(angles, form)))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Geometry does not identify amplitude and shape")
    norms = np.linalg.norm(design, axis=0)
    normalized_condition = float(np.linalg.cond(design / norms))
    if normalized_condition > 1e8:
        raise ValueError("Geometry design is ill-conditioned")
    targets = radiance.transpose(0, 2, 1).reshape(-1, len(angles))
    coefficients = targets @ np.linalg.pinv(design).T
    grid = np.column_stack((np.ones(901), angular_basis(np.linspace(0, 90, 901), form)))
    constrained = []
    for start in range(0, len(coefficients), 512):
        local = coefficients[start:start+512]
        # Relative profile floor also enforces positive amplitude.
        constraints = grid.copy()
        constraints[:, 0] -= 1e-6
        bad = np.any(local @ constraints.T < 0, axis=1) | (local[:, 0] <= 0)
        constrained.extend((np.flatnonzero(bad) + start).tolist())
    for index in constrained:
        target = targets[index]
        scale = target.mean()
        scaled_target = target / scale
        constraint = grid.copy()
        constraint[:, 0] -= 1e-6
        result = minimize(
            lambda value: .5 * np.sum((design @ value - scaled_target)**2),
            np.r_[1., np.zeros(design.shape[1]-1)],
            jac=lambda value: design.T @ (design @ value - scaled_target),
            constraints=[
                LinearConstraint(constraint, 0, np.inf),
                LinearConstraint(np.eye(design.shape[1])[0:1], 1e-8, np.inf),
            ], method="SLSQP", options={"ftol": 1e-12, "maxiter": 300},
        )
        if not result.success:
            raise RuntimeError(f"Constrained geometry fit failed: {result.message}")
        coefficients[index] = result.x * scale
    amplitude = coefficients[:, 0]
    parameters = coefficients[:, 1:] / amplitude[:, None]
    integral = angular_integral(parameters, form)
    flux = (np.pi * amplitude * integral).reshape(radiance.shape[0], radiance.shape[2])
    residual = targets - coefficients @ design.T
    theta = np.deg2rad(np.linspace(0, 90, 901))
    integrated_design = simpson(
        grid * (2*np.cos(theta)*np.sin(theta))[:, None], x=theta, axis=0
    )
    flux_derivative = np.pi * integrated_design @ np.linalg.pinv(design)
    return flux, {
        "views": len(angles), "parameters": design.shape[1],
        "normalized_design_condition": normalized_condition,
        "design_condition": float(np.linalg.cond(design)),
        "unconstrained_flux_noise_gain": float(np.linalg.norm(flux_derivative)),
        "noise_gain_note": "Flux SD per unit independent equal radiance SD; local unconstrained linear estimator",
        "constrained_scene_channels": len(constrained),
        "scene_channels": len(coefficients),
        "radiance_fit_rmse": float(np.sqrt(np.mean(residual**2))),
        "degrees_of_freedom": len(angles) - design.shape[1],
    }


def nested_switch_order(probabilities, seed):
    """Stable top-two labels and one permutation for nested fraction prefixes."""
    probabilities = np.asarray(probabilities)
    if (probabilities.ndim != 2 or probabilities.shape[1] < 2
            or np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Invalid GMM posteriors")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    return order[:, 0], order[:, 1], np.random.default_rng(seed).permutation(len(order))
