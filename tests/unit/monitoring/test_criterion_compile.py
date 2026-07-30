"""Unit tests for M5 criterion → bound-SQL compilation (T2.1, injection defense)."""

import pytest

from talkex.monitoring.domain.search import Criterion
from talkex.monitoring.infrastructure.read_repo import TimescaleReadRepository

_compile = TimescaleReadRepository._criteria_clause


class TestCriterionCompile:
    def test_empty_criteria_is_noop(self) -> None:
        clause, params = _compile(())
        assert clause == ""
        assert params == []

    def test_whitelisted_column_compiles_to_bound_param(self) -> None:
        clause, params = _compile((Criterion(field="speaker", value="customer"),))
        assert clause == " AND speaker = %s"
        assert params == ["customer"]

    def test_metadata_criterion_binds_key_and_value(self) -> None:
        clause, params = _compile((Criterion(field="metadata.queue", value="retention"),))
        assert clause == " AND metadata->>%s = %s"
        assert params == ["queue", "retention"]

    def test_unknown_field_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="unknown criterion field"):
            _compile((Criterion(field="raw_text; DROP TABLE turns", value="x"),))

    def test_injection_value_is_bound_never_interpolated(self) -> None:
        malicious = "customer'; DROP TABLE turns; --"
        clause, params = _compile((Criterion(field="speaker", value=malicious),))
        # The dangerous value lives ONLY in the bound params tuple, never in the SQL string.
        assert malicious in params
        assert malicious not in clause
        assert clause.count("%s") == 1
