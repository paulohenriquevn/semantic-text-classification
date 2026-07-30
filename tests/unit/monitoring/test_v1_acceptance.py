"""Unit tests for the M8 V1-acceptance report builder (T2.1/T3.1) — no hard-coded PASS."""

from talkex.monitoring.domain.acceptance import Criterion, build_readiness_report


def _ok(name: str) -> Criterion:
    return Criterion(name=name, measured=1.0, threshold=1.0, direction="min")


class TestV1Acceptance:
    def test_report_passes_when_all_meet(self) -> None:
        report = build_readiness_report([_ok("a"), _ok("b")])
        assert report.verdict == "PASS"
        assert "synthetic load" in report.caveat.lower()  # the honesty caveat is always present

    def test_report_fails_when_a_criterion_fails(self) -> None:
        failing = Criterion(name="retrieval_p95", measured=250.0, threshold=200.0, direction="max")
        report = build_readiness_report([_ok("a"), failing])
        assert report.verdict == "FAIL"  # a single unmet criterion fails the whole verdict

    def test_missing_artifact_is_fail(self) -> None:
        missing = Criterion(name="no_evidence", measured=None, threshold=1.0, direction="min")
        assert missing.passed is False  # missing measurement is never a silent PASS
        assert build_readiness_report([missing]).verdict == "FAIL"

    def test_max_direction_boundary(self) -> None:
        assert Criterion(name="p95", measured=160.95, threshold=200.0, direction="max").passed is True
        assert Criterion(name="p95", measured=200.0, threshold=200.0, direction="max").passed is False
