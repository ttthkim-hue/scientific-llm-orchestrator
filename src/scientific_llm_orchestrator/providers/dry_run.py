"""Credential-free adapter that never calls a model or applies a patch."""

from __future__ import annotations

from ..contracts.models import Result, WorkOrder
from ..routing.router import RouteDecision


class DryRunAdapter:
    """Emit a sanitized route receipt without contacting any provider."""

    provider_name = "dry-run-no-provider"

    def execute(self, order: WorkOrder, decision: RouteDecision) -> Result:
        if not decision.allowed:
            return Result(
                work_order_id=order.work_order_id,
                route="blocked",
                status="blocked",
                evidence_label="not-run",
                provider=self.provider_name,
                dry_run=True,
                protected_literals_preserved=None,
                proposal_summary="No provider call was made because the route was blocked.",
                route_reason=decision.reason,
            )
        return Result(
            work_order_id=order.work_order_id,
            route=decision.route,
            status="dry-run",
            evidence_label="observed-mock-runtime",
            provider=self.provider_name,
            dry_run=True,
            protected_literals_preserved=None,
            validation_results={},
            proposal_summary="Dry-run route receipt only; no model output or code application.",
            route_reason=decision.reason,
        )
