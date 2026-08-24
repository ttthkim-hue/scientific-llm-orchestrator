"""Provider-neutral, evidence-gated scientific LLM orchestration prototype."""

__version__ = "0.1.0"

from .contracts.models import (
    ARCHITECTURES,
    EVIDENCE_LABELS,
    AuthorityFlags,
    Result,
    ReviewBundle,
    WorkOrder,
)

__all__ = [
    "ARCHITECTURES",
    "EVIDENCE_LABELS",
    "AuthorityFlags",
    "Result",
    "ReviewBundle",
    "WorkOrder",
    "__version__",
]
