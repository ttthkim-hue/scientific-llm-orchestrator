"""Path, protected-literal, and validation-ID checks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import PurePosixPath
from typing import Iterable, List, Sequence

ALLOWED_VALIDATION_IDS = frozenset(
    {
        "schema",
        "path_scope",
        "protected_literals",
        "fixture_expected",
        "compile",
        "unit",
        "integration",
        "svg_labels",
        "cad_geometry",
    }
)


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    validation_id: str
    errors: List[str]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "validation_id": self.validation_id, "errors": list(self.errors)}


def _normalise_public_path(value: str) -> str:
    return value.replace("\\", "/")


def check_path_scope(paths: Iterable[str], root: str = "workspace") -> ValidationReport:
    """Reject absolute paths, traversal, and empty paths.

    ``root`` is descriptive only; the public contract carries relative paths
    and does not expose a local absolute workspace path.
    """

    errors: List[str] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw:
            errors.append("path must be a non-empty string")
            continue
        path = _normalise_public_path(raw)
        if os.path.isabs(raw) or path.startswith("/") or (len(path) > 1 and path[1] == ":"):
            errors.append("absolute path is not allowed: " + raw)
            continue
        parts = PurePosixPath(path).parts
        if ".." in parts:
            errors.append("path traversal is not allowed: " + raw)
    return ValidationReport(not errors, "path_scope", errors)


def check_protected_literals(required: Sequence[str], candidate: str) -> ValidationReport:
    """Require each exact protected literal to survive in a candidate."""

    errors: List[str] = []
    if not isinstance(candidate, str):
        errors.append("candidate must be text")
    else:
        for literal in required:
            if not isinstance(literal, str) or not literal:
                errors.append("protected literals must be non-empty strings")
            elif literal not in candidate:
                errors.append("missing protected literal: " + literal)
    return ValidationReport(not errors, "protected_literals", errors)


def check_validation_allowlist(validation_ids: Iterable[str]) -> ValidationReport:
    ids = list(validation_ids)
    errors = ["validation ID is not allowlisted: " + str(item) for item in ids if item not in ALLOWED_VALIDATION_IDS]
    return ValidationReport(not errors, "schema", errors)
