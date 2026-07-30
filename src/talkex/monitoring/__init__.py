"""Real-time attendance monitoring (M0 walking skeleton).

Consumes streaming transcripts, runs the cascade (segmentation → context window →
DSL rule), persists to a TimescaleDB hypertable, and pushes evidence-backed alerts
to a supervisor. Package-by-layer: interface → application → domain ← infrastructure
(DIP at the borders, see .claude/rules/architecture.md and ADR-005).
"""

from talkex.monitoring.config import MonitoringConfig

__all__ = ["MonitoringConfig"]
