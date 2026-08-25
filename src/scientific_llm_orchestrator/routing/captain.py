"""Provider-neutral Captain portfolio routing for public review.

The module selects a role, not a product or model. It performs only preflight
routing and never retries across hosted providers, applies a patch, or grants
scientific, security, canonical, or release authority to a worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


PRIVACY_CLASSES = frozenset({"public", "public-synthetic", "local-private"})
TASK_SHAPES = frozenset({"authority", "deterministic", "mechanical", "routine", "complex"})


@dataclass(frozen=True)
class CaptainRequest:
    """Sanitized facts needed for one zero-inference route decision."""

    privacy_class: str
    task_shape: str
    exact_tool_available: bool = False
    deterministic_evaluator_available: bool = True


@dataclass(frozen=True)
class CaptainAvailability:
    """Role availability without provider identity, credentials, or quotas."""

    frontier: bool = True
    local_mechanical_worker: bool = False
    private_implementation_worker: bool = False
    routine_hosted_worker: bool = False
    complex_hosted_worker: bool = False
    hosted_route_active: bool = False
    hosted_usage_headroom: bool = False


@dataclass(frozen=True)
class CaptainDecision:
    role: str
    allowed: bool
    reason: str
    requires_frontier_acceptance: bool = True
    automatic_apply: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "allowed": self.allowed,
            "reason": self.reason,
            "requires_frontier_acceptance": self.requires_frontier_acceptance,
            "automatic_apply": self.automatic_apply,
        }


def _frontier_or_blocked(availability: CaptainAvailability, reason: str) -> CaptainDecision:
    if availability.frontier:
        return CaptainDecision("frontier", True, reason)
    return CaptainDecision("blocked", False, "frontier_unavailable")


def route_captain_task(
    request: CaptainRequest, availability: CaptainAvailability
) -> CaptainDecision:
    """Choose one provider-neutral role with fail-closed authority boundaries."""

    if request.privacy_class not in PRIVACY_CLASSES:
        return CaptainDecision("blocked", False, "privacy_class_not_allowlisted")
    if request.task_shape not in TASK_SHAPES:
        return CaptainDecision("blocked", False, "task_shape_not_allowlisted")

    if request.task_shape == "authority":
        return _frontier_or_blocked(availability, "authority_reserved_for_frontier")

    if request.exact_tool_available:
        return CaptainDecision("deterministic_tool", True, "exact_tool_preferred")

    if not request.deterministic_evaluator_available:
        return _frontier_or_blocked(availability, "worker_evaluator_unavailable")

    if request.task_shape == "mechanical" and availability.local_mechanical_worker:
        return CaptainDecision("local_mechanical_worker", True, "bounded_local_mechanics")

    if request.privacy_class == "local-private":
        if availability.private_implementation_worker:
            return CaptainDecision(
                "private_implementation_worker", True, "private_content_stays_local"
            )
        return _frontier_or_blocked(availability, "private_worker_unavailable")

    hosted_ready = availability.hosted_route_active and availability.hosted_usage_headroom
    if request.task_shape == "routine" and hosted_ready and availability.routine_hosted_worker:
        return CaptainDecision("routine_hosted_worker", True, "routine_hosted_preflight_pass")
    if request.task_shape == "complex" and hosted_ready and availability.complex_hosted_worker:
        return CaptainDecision("complex_hosted_worker", True, "complex_hosted_preflight_pass")

    if availability.private_implementation_worker:
        return CaptainDecision(
            "private_implementation_worker", True, "hosted_preflight_unavailable_private_route"
        )
    return _frontier_or_blocked(availability, "worker_preflight_unavailable")
