"""Stage-scoped identities for the nominal ABI feature and GMM pipeline."""

from __future__ import annotations

import hashlib
import ast
import copy
import json
from pathlib import Path
from typing import Mapping


STAGE_CODE_PATHS = {
    "features": (
        "scripts/run_nominal_budget.py",
        "src/nominal_streaming.py",
        "src/nominal_provenance.py",
        "src/scene_features.py",
        "src/uncertainty.py",
    ),
    "pca": (
        "scripts/run_nominal_budget.py",
        "src/nominal_streaming.py",
        "src/streaming_gmm.py",
        "src/nominal_provenance.py",
        "src/scene_features.py",
        "src/uncertainty.py",
    ),
    "gmm": (
        "scripts/run_nominal_budget.py",
        "src/nominal_streaming.py",
        "src/streaming_gmm.py",
        "src/nominal_provenance.py",
        "src/uncertainty.py",
    ),
    "sweep": (
        "scripts/run_nominal_budget.py",
        "src/nominal_streaming.py",
        "src/streaming_gmm.py",
        "src/nominal_provenance.py",
        "src/uncertainty.py",
    ),
    "candidate": (
        "scripts/run_nominal_budget.py",
        "src/nominal_streaming.py",
        "src/streaming_gmm.py",
        "src/nominal_provenance.py",
        "src/uncertainty.py",
    ),
}

_GMM_MATH = (
    "_positive_integer", "_chunk_rows", "_chunks", "_merge_moments",
    "_stream_moments", "_preprocessing", "prepare_streaming_preprocessing",
    "_validate_preprocessor", "_transformed_chunks", "_training_chunks",
    "_initial_centers", "_set_parameters", "_chunk_statistics",
    "_bounded_statistics", "_expectation_statistics", "_validate_masses",
    "_maximization", "fit_streaming_gmm", "_fit_streaming_gmm",
)
_GMM_MAPPED = (
    "MappedChunk", "_mapped_file", "MappedTrainingData",
    "_mapped_chunk_statistics",
)

STAGE_SCOPED_CODE = {
    "features": {
        "scripts/run_nominal_budget.py": (
            "ROOT", "SETTINGS", "CONFIG_PATH", "TAXONOMY_PATH",
            "options", "write", "provenance", "provenance_identity",
            "check_stage_provenance", "verify",
        ),
        "src/nominal_streaming.py": (
            "validate_settings", "source_path", "cache_path",
            "configure_chunk_cache", "observations", "prepare_cache",
        ),
        "src/uncertainty.py": (
            "sha256_file", "_relative_path", "_path_identity",
            "_runtime_identity", "build_provenance", "check_provenance",
        ),
    },
    "pca": {
        "scripts/run_nominal_budget.py": (
            "ROOT", "SETTINGS", "CONFIG_PATH", "TAXONOMY_PATH",
            "options", "write", "provenance", "provenance_identity",
            "check_stage_provenance", "verify",
        ),
        "src/nominal_streaming.py": (
            "cache_path", "open_features", "feature_batches",
            "prepare_training",
        ),
        "src/uncertainty.py": (
            "sha256_file", "_relative_path", "_path_identity",
            "_runtime_identity", "build_provenance", "check_provenance",
        ),
        "src/streaming_gmm.py": (
            "_FEATURES", "_CHUNK_ROWS", "_REG_COVAR",
            "_positive_integer", "_chunk_rows", "_chunks", "_merge_moments",
            "_stream_moments", "_preprocessing", "prepare_streaming_preprocessing",
        ),
    },
    "gmm": {
        "scripts/run_nominal_budget.py": (
            "ROOT", "SETTINGS", "CONFIG_PATH", "TAXONOMY_PATH",
            "options", "write", "provenance", "provenance_identity",
            "check_stage_provenance", "verify",
        ),
        "src/nominal_streaming.py": (
            "validate_settings", "cache_path", "open_features",
            "training_options", "candidate_initialization_path",
            "_initialization_inputs", "_atomic_output",
            "fit_candidate_initialization", "combine_candidate_initializations",
        ),
        "src/uncertainty.py": (
            "sha256_file", "_relative_path", "_path_identity",
            "_runtime_identity", "build_provenance", "check_provenance",
        ),
        "src/streaming_gmm.py": (
            *_GMM_MAPPED, "_FEATURES", "_CHUNK_ROWS", "_REG_COVAR",
            *_GMM_MATH, "_worker_initialize", "_emit_progress",
        ),
    },
    "sweep": {
        "scripts/run_nominal_budget.py": (
            "ROOT", "SETTINGS", "CONFIG_PATH", "TAXONOMY_PATH",
            "options", "write", "provenance", "provenance_identity",
            "check_stage_provenance", "verify",
        ),
        "src/nominal_streaming.py": (
            "candidate_initialization_path", "_initialization_inputs",
            "_atomic_output", "fit_candidate_initialization",
            "fit_initialization_sweep",
        ),
        "src/streaming_gmm.py": (
            *_GMM_MAPPED, "_FEATURES", "_CHUNK_ROWS", "_REG_COVAR",
            *_GMM_MATH, "_worker_initialize", "_emit_progress",
        ),
        "src/uncertainty.py": (
            "sha256_file", "_relative_path", "_path_identity",
            "_runtime_identity", "build_provenance", "check_provenance",
        ),
    },
    "candidate": {
        "scripts/run_nominal_budget.py": (
            "ROOT", "SETTINGS", "CONFIG_PATH", "TAXONOMY_PATH",
            "options", "write", "provenance", "provenance_identity",
            "check_stage_provenance", "verify",
        ),
        "src/nominal_streaming.py": (
            "candidate_initialization_path", "_initialization_inputs",
            "combine_candidate_initializations",
        ),
        "src/streaming_gmm.py": (
            *_GMM_MAPPED, "_FEATURES", "_CHUNK_ROWS", "_REG_COVAR",
            *_GMM_MATH, "_worker_initialize", "_emit_progress",
        ),
        "src/uncertainty.py": (
            "sha256_file", "_relative_path", "_path_identity",
            "_runtime_identity", "build_provenance", "check_provenance",
        ),
    },
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def configuration_identity(
    stage: str,
    settings: Mapping[str, object],
    identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the scientifically relevant settings for one stage.

    Deliberately do not include output paths, downstream analysis settings, or
    the full K list. A GMM start is keyed by its individual K and seed
    through ``identity``, so adding another candidate leaves compatible starts
    reusable.
    """
    if stage == "features":
        selected = {
            "maximum_vza_deg": settings.get("maximum_vza_deg"),
        }
    elif stage == "pca":
        selected = {
            "training_days": settings.get("training_days"),
            "gmm_chunk_points": settings.get("gmm_chunk_points", 65536),
        }
    elif stage == "gmm":
        selected = {
            "training_days": settings.get("training_days"),
            "gmm_chunk_points": settings.get("gmm_chunk_points", 65536),
            "batch_points": settings.get("batch_points"),
            "gmm_max_iter": settings.get("gmm_max_iter"),
            "gmm_tol": settings.get("gmm_tol"),
        }
    elif stage == "sweep":
        selected = {
            "training_days": settings.get("training_days"),
            "gmm_chunk_points": settings.get("gmm_chunk_points", 65536),
            "batch_points": settings.get("batch_points"),
            "gmm_max_iter": settings.get("gmm_max_iter"),
            "gmm_tol": settings.get("gmm_tol"),
        }
    elif stage == "candidate":
        selected = {
            "training_days": settings.get("training_days"),
            "gmm_chunk_points": settings.get("gmm_chunk_points", 65536),
            "batch_points": settings.get("batch_points"),
            "gmm_max_iter": settings.get("gmm_max_iter"),
            "gmm_tol": settings.get("gmm_tol"),
        }
    else:
        raise ValueError(f"Unsupported nominal provenance stage: {stage}")

    result: dict[str, object] = {"identity_schema": 1, "settings": selected}
    if identity:
        result["request"] = dict(identity)
    return result


def initialization_identity(components: int, seed: int, initialization: int) -> dict[str, int]:
    return {
        "components": int(components),
        "seed": int(seed),
        "initialization": int(initialization),
    }


def candidate_identity(
    settings: Mapping[str, object], components: int, seed: int
) -> dict[str, object]:
    """Identity for a combined candidate, independent of other K/seed requests."""
    return {
        "components": int(components),
        "seed": int(seed),
        "initializations": list(range(int(settings["gmm_initializations"]))),
    }


def sweep_identity(settings: Mapping[str, object]) -> dict[str, object]:
    """Identity for the sweep manifest's exact requested queue."""
    return {
        "components": list(settings["components"]),
        "seed": int(settings["gmm_seed"]),
        "initializations": list(range(int(settings["gmm_initializations"]))),
    }


def stage_signature(
    stage: str,
    settings: Mapping[str, object],
    identity: Mapping[str, object] | None = None,
    root: Path | None = None,
) -> str:
    """Stable signature for stage config and stage-relevant source code."""
    root = root or Path(__file__).resolve().parents[1]
    signature_payload = {
        "configuration_identity": configuration_identity(stage, settings, identity),
        "code_sha256": stage_code_identity(stage, root),
    }
    serialized = _canonical_json(signature_payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def stage_code_paths(stage: str) -> tuple[str, ...]:
    """Return safe file inputs; AST-scoped files are tracked by signatures."""
    try:
        root = Path(__file__).resolve().parents[1]
        return tuple(
            path.relative_to(root).as_posix()
            for path in provenance_code_paths(stage, root)
        )
    except KeyError as error:
        raise ValueError(f"Unsupported nominal provenance stage: {stage}") from error


def scoped_code_paths(stage: str, root: Path) -> dict[Path, tuple[str, ...]]:
    try:
        scopes = STAGE_SCOPED_CODE[stage]
    except KeyError as error:
        raise ValueError(f"Unsupported nominal provenance stage: {stage}") from error
    return {root / relative: symbols for relative, symbols in scopes.items()}


def provenance_code_paths(stage: str, root: Path) -> tuple[Path, ...]:
    """Files to hash byte-for-byte; scoped Python modules use AST hashes."""
    try:
        paths = STAGE_CODE_PATHS[stage]
    except KeyError as error:
        raise ValueError(f"Unsupported nominal provenance stage: {stage}") from error
    scoped = {path.relative_to(root).as_posix() for path in scoped_code_paths(stage, root)}
    return tuple(root / relative for relative in paths if relative not in scoped)


def scoped_code_sha256(path: Path, symbols: tuple[str, ...] | list[str]) -> str:
    """Hash selected top-level Python definitions, ignoring unrelated functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    requested = set(symbols)
    selected = []
    found = set()
    remaining = []
    for node in tree.body:
        names = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        matched = names & requested
        if matched:
            selected.append(node)
            found.update(matched)
        else:
            remaining.append(node)
    missing = requested - found
    if missing:
        raise ValueError(f"Missing scoped code symbols in {path}: {sorted(missing)}")

    def loaded_names(node):
        return {
            child.id for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }

    def bound_import_names(node):
        if isinstance(node, ast.Import):
            return {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
        if isinstance(node, ast.ImportFrom):
            return {alias.asname or alias.name for alias in node.names}
        return set()

    def bound_assignment_names(node):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {target.id for target in targets if isinstance(target, ast.Name)}

    needed = set().union(*(loaded_names(node) for node in selected))
    # Include only imports and module constants referenced by the selected
    # functions. This prevents downstream-only imports/constants in a shared
    # module from invalidating feature/PCA/GMM artifacts.
    changed = True
    while changed:
        changed = False
        for node in remaining:
            bound = bound_import_names(node)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                bound |= bound_assignment_names(node)
            is_future_import = (
                isinstance(node, ast.ImportFrom) and node.module == "__future__"
            )
            if is_future_import or bound & needed:
                selected.append(node)
                remaining.remove(node)
                needed.update(loaded_names(node))
                changed = True
                break
    class ExcludeFitTelemetry(ast.NodeTransformer):
        """Keep the fit call/math but exclude progress/scheduler-only wrappers."""

        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            if node.name != "fit_candidate_initialization":
                return node
            node.body = [
                statement for statement in node.body
                if not (
                    isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and statement.name == "report_progress"
                )
                and not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id == "print"
                )
            ]
            if node.args.kwonlyargs:
                kept = [
                    (argument, default)
                    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if argument.arg not in {"statistics_scheduler", "event_callback"}
                ]
                node.args.kwonlyargs = [argument for argument, _ in kept]
                node.args.kw_defaults = [default for _, default in kept]
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "fit_streaming_gmm":
                node.keywords = [
                    keyword for keyword in node.keywords
                    if keyword.arg not in {
                        "statistics_scheduler", "event_callback", "progress_callback",
                    }
                ]
            return node

    selected = ExcludeFitTelemetry().visit(copy.deepcopy(ast.Module(
        body=selected, type_ignores=[]
    ))).body
    serialized = ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def stage_code_identity(stage: str, root: Path) -> dict[str, object]:
    scopes = scoped_code_paths(stage, root)
    full = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in provenance_code_paths(stage, root)
    }
    scoped = {
        path.relative_to(root).as_posix(): {
            "symbols": list(symbols),
            "sha256": scoped_code_sha256(path, symbols),
        }
        for path, symbols in scopes.items()
    }
    return {"files": full, "scoped": scoped}


def identity_sha256(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def check_stage_provenance(
    provenance: object,
    stage: str,
    settings: Mapping[str, object],
    identity: Mapping[str, object] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Validate stage metadata without importing a workflow runner module.

    ``provenance`` is the metadata's provenance value itself, not the enclosing
    JSON record. This helper is safe for ``nominal_streaming.open_features``.
    """
    from uncertainty import check_provenance

    root = root or Path(__file__).resolve().parents[1]
    expected = configuration_identity(stage, settings, identity)
    return check_provenance(
        provenance, root, expected_stage=stage,
        expected_configuration_identity=expected,
    )
