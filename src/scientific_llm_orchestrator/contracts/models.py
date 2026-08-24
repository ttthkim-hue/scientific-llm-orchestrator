"""Small JSON-safe contracts used by the public prototype.

The contracts intentionally describe proposals and validation receipts. They
never represent permission to apply code or to make a scientific decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Mapping, Optional

CONTRACT_VERSION = "1.0"
ARCHITECTURES = (
    "FRONTIER_ONLY",
    "FRONTIER_HOSTED_WORKER",
    "FRONTIER_LOCAL_WORKER",
    "FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL",
)
EVIDENCE_LABELS = (
    "observed-real-model",
    "observed-deterministic",
    "observed-mock-runtime",
    "vendor-reported",
    "inferred",
    "proposed",
    "not-run",
)
TASK_CLASSES = (
    "coding",
    "data_scientific_mechanics",
    "visual_figure_qa",
    "parametric_cad",
)
ROUTES = ("frontier", "hosted_worker", "local_worker", "blocked")


def _copy_json(value: Any) -> Any:
    """Return a JSON-compatible deep copy without non-standard dependencies."""

    return json.loads(json.dumps(value, sort_keys=True))


@dataclass(frozen=True)
class AuthorityFlags:
    """Authorities retained by the frontier reviewer."""

    scientific: bool = True
    citation: bool = True
    security: bool = True
    canonical: bool = True
    release: bool = True

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "AuthorityFlags":
        if value is None:
            return cls()
        expected = set(asdict(cls()).keys())
        if set(value) != expected or not all(isinstance(item, bool) for item in value.values()):
            raise ValueError("authority_flags must contain exactly five boolean authorities")
        flags = cls(**dict(value))
        if not all(flags.to_dict().values()):
            raise ValueError("all authority flags must remain frontier-owned")
        return flags


@dataclass(frozen=True)
class WorkOrder:
    """A bounded, non-applying request for a worker proposal."""

    work_order_id: str
    task_class: str
    operation: str
    objective: str
    architecture: str
    allowed_paths: List[str]
    protected_literals: List[str]
    validation_ids: List[str]
    authority_flags: AuthorityFlags = field(default_factory=AuthorityFlags)
    privacy_class: str = "public-synthetic"
    automatic_apply: bool = False
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported work-order contract version")
        if not self.work_order_id or not self.work_order_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("work_order_id must be a simple public identifier")
        if self.task_class not in TASK_CLASSES:
            raise ValueError("unsupported task_class")
        if self.architecture not in ARCHITECTURES:
            raise ValueError("unsupported architecture")
        if not self.operation or not self.objective:
            raise ValueError("operation and objective are required")
        if not self.allowed_paths:
            raise ValueError("at least one allowed path is required")
        if self.automatic_apply is not False:
            raise ValueError("automatic_apply is permanently disabled in the public MVP")
        AuthorityFlags.from_dict(self.authority_flags.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["authority_flags"] = self.authority_flags.to_dict()
        return _copy_json(value)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkOrder":
        data = dict(value)
        data["authority_flags"] = AuthorityFlags.from_dict(data.get("authority_flags"))
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "WorkOrder":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class Result:
    """Provider-neutral, sanitized result receipt."""

    work_order_id: str
    route: str
    status: str
    evidence_label: str
    provider: str
    dry_run: bool
    automatic_apply: bool = False
    protected_literals_preserved: Optional[bool] = None
    validation_results: Dict[str, bool] = field(default_factory=dict)
    proposal_summary: str = ""
    route_reason: str = ""
    authority_flags: AuthorityFlags = field(default_factory=AuthorityFlags)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported result contract version")
        if self.route not in ROUTES:
            raise ValueError("unsupported route")
        if self.evidence_label not in EVIDENCE_LABELS:
            raise ValueError("unsupported evidence label")
        if self.automatic_apply is not False:
            raise ValueError("automatic_apply is permanently disabled in the public MVP")
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be boolean")
        if self.protected_literals_preserved is not None and not isinstance(
            self.protected_literals_preserved, bool
        ):
            raise ValueError("protected_literals_preserved must be boolean or null")
        if not all(isinstance(value, bool) for value in self.validation_results.values()):
            raise ValueError("validation_results values must be boolean")
        AuthorityFlags.from_dict(self.authority_flags.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["authority_flags"] = self.authority_flags.to_dict()
        return _copy_json(value)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Result":
        data = dict(value)
        data["authority_flags"] = AuthorityFlags.from_dict(data.get("authority_flags"))
        return cls(**data)


@dataclass(frozen=True)
class ReviewBundle:
    """A deterministic, sanitized collection of receipts."""

    bundle_id: str
    results: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    evidence_label: str = "observed-deterministic"
    semantic_review_status: str = "pending-human-review"
    automatic_apply: bool = False
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported review-bundle contract version")
        if self.evidence_label not in EVIDENCE_LABELS:
            raise ValueError("unsupported evidence label")
        if self.automatic_apply is not False:
            raise ValueError("review bundles cannot authorize application")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return _copy_json(value)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewBundle":
        return cls(**dict(value))
