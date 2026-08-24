"""Architecture-to-route mapping for the public dry-run MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..contracts.models import ARCHITECTURES, ROUTES, WorkOrder
from ..validators import check_path_scope, check_validation_allowlist


@dataclass(frozen=True)
class Availability:
    """Public capability flags; no credentials or provider details are stored."""

    frontier: bool = True
    hosted_worker: bool = False
    local_worker: bool = False


@dataclass(frozen=True)
class RouteDecision:
    route: str
    allowed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"route": self.route, "allowed": self.allowed, "reason": self.reason}


def route_work_order(order: WorkOrder, availability: Availability) -> RouteDecision:
    """Return the same route for the same order and availability.

    Worker routes are proposal lanes only. The order's five authority flags are
    required to stay true, and every route remains non-applying.
    """

    if order.architecture not in ARCHITECTURES:
        return RouteDecision("blocked", False, "architecture_not_allowlisted")
    if not check_path_scope(order.allowed_paths).passed:
        return RouteDecision("blocked", False, "path_scope_failed")
    if not check_validation_allowlist(order.validation_ids).passed:
        return RouteDecision("blocked", False, "validation_allowlist_failed")
    if order.automatic_apply is not False:
        return RouteDecision("blocked", False, "automatic_apply_disabled")
    if not all(order.authority_flags.to_dict().values()):
        return RouteDecision("blocked", False, "frontier_authority_required")

    if order.architecture == "FRONTIER_ONLY":
        return (
            RouteDecision("frontier", True, "frontier_only_baseline")
            if availability.frontier
            else RouteDecision("blocked", False, "frontier_unavailable")
        )
    if order.architecture == "FRONTIER_HOSTED_WORKER":
        if availability.hosted_worker:
            return RouteDecision("hosted_worker", True, "hosted_worker_available")
        if availability.frontier:
            return RouteDecision("frontier", True, "hosted_worker_unavailable_frontier_fallback")
        return RouteDecision("blocked", False, "hosted_and_frontier_unavailable")
    if order.architecture == "FRONTIER_LOCAL_WORKER":
        if availability.local_worker:
            return RouteDecision("local_worker", True, "local_worker_available")
        if availability.frontier:
            return RouteDecision("frontier", True, "local_worker_unavailable_frontier_fallback")
        return RouteDecision("blocked", False, "local_and_frontier_unavailable")

    if availability.local_worker:
        return RouteDecision("local_worker", True, "local_worker_available_primary")
    if availability.hosted_worker:
        return RouteDecision("hosted_worker", True, "hosted_worker_residual")
    if availability.frontier:
        return RouteDecision("frontier", True, "residual_frontier_fallback")
    return RouteDecision("blocked", False, "all_routes_unavailable")
