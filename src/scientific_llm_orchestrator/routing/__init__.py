"""Deterministic routing with fail-closed authority boundaries."""

from .router import Availability, RouteDecision, route_work_order

__all__ = ["Availability", "RouteDecision", "route_work_order"]
