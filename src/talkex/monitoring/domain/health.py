"""Health + engagement value objects for M8 (blueprint D1/D2/D4).

`HealthReport` is a readiness snapshot (session state + queue depth + DB reachability); `EngagementMetric`
is the alert-engagement proxy (`acted_on_rate` = labels ÷ alerts in a window). Frozen so the serialized
snapshot is immutable (livekit-style metric records).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthReport(BaseModel):
    """A readiness snapshot of the monitor itself."""

    model_config = ConfigDict(frozen=True)

    ready: bool
    session_state: str
    queue_depth: int
    queue_maxsize: int
    db_reachable: bool


class EngagementMetric(BaseModel):
    """Alert-engagement proxy — the supervisor-acted-on-alert rate over a window.

    NOTE: a north-star PROXY. The plumbing (label ÷ alert) is real and tested; the RATE itself is a
    production signal only under a real pilot with supervisors acting on alerts (M8 honesty caveat).
    """

    model_config = ConfigDict(frozen=True)

    alerts: int
    labels: int
    acted_on_rate: float
