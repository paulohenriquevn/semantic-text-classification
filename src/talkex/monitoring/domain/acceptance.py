"""V1-acceptance value objects (M8 D3).

A `Criterion` is one V1 ship gate with a measured value (from a live check or a fresh metrics artifact),
a threshold, and a direction. `ReadinessReport` aggregates them into a PASS/FAIL verdict — PASS only when
EVERY criterion passes. A missing measurement (`measured is None`) is a FAIL, never a silent PASS
(no hard-coded PASS — acceptance theatre is the failure this guards against).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Criterion(BaseModel):
    """One V1 ship criterion. `direction`: 'min' → measured must be ≥ threshold; 'max' → measured < threshold."""

    model_config = ConfigDict(frozen=True)

    name: str
    measured: float | None
    threshold: float
    direction: str  # "min" | "max"

    @property
    def passed(self) -> bool:
        if self.measured is None:
            return False  # missing evidence is a FAIL, not a silent PASS
        if self.direction == "min":
            return self.measured >= self.threshold
        if self.direction == "max":
            return self.measured < self.threshold
        raise ValueError(f"unknown direction {self.direction!r} (expected 'min' or 'max')")


_SYNTHETIC_CAVEAT = (
    "Validated on synthetic load; real 8 kHz call-center drift is NOT yet validated — "
    "re-run this harness against real pilot data before an unqualified V1 claim."
)


class ReadinessReport(BaseModel):
    """The V1 readiness verdict over all criteria, with the mandatory honesty caveat."""

    model_config = ConfigDict(frozen=True)

    criteria: list[Criterion]
    verdict: str  # "PASS" | "FAIL"
    caveat: str


def build_readiness_report(criteria: list[Criterion]) -> ReadinessReport:
    """PASS iff every criterion passes; the synthetic-load caveat is always attached (M8 D6)."""
    verdict = "PASS" if all(c.passed for c in criteria) else "FAIL"
    return ReadinessReport(criteria=criteria, verdict=verdict, caveat=_SYNTHETIC_CAVEAT)
