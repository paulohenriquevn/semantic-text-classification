"""Unit tests for domain exception hierarchy."""

import pytest

from semantic_conversation_engine.exceptions import (
    ConfigurationError,
    EngineError,
    EngineValidationError,
    ModelError,
    PipelineError,
    RuleError,
)


class TestEngineError:
    def test_stores_message(self) -> None:
        err = EngineError("something failed")
        assert err.message == "something failed"

    def test_context_defaults_to_empty_dict(self) -> None:
        err = EngineError("fail")
        assert err.context == {}

    def test_stores_context(self) -> None:
        ctx = {"conversation_id": "conv_123", "stage": "ingestion"}
        err = EngineError("fail", context=ctx)
        assert err.context == ctx

    def test_str_without_context(self) -> None:
        err = EngineError("something failed")
        assert str(err) == "something failed"

    def test_str_with_context(self) -> None:
        err = EngineError("processing failed", context={"turn_id": "turn_42", "step": "normalize"})
        result = str(err)
        assert "processing failed" in result
        assert "turn_id='turn_42'" in result
        assert "step='normalize'" in result

    def test_is_exception_subclass(self) -> None:
        assert issubclass(EngineError, Exception)

    def test_catchable_as_exception(self) -> None:
        with pytest.raises(Exception, match="test"):
            raise EngineError("test")


class TestExceptionHierarchy:
    """All domain exceptions must inherit from EngineError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            EngineValidationError,
            PipelineError,
            ModelError,
            RuleError,
            ConfigurationError,
        ],
    )
    def test_is_subclass_of_engine_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, EngineError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            EngineValidationError,
            PipelineError,
            ModelError,
            RuleError,
            ConfigurationError,
        ],
    )
    def test_is_subclass_of_exception(self, exc_class: type) -> None:
        assert issubclass(exc_class, Exception)

    @pytest.mark.parametrize(
        "exc_class",
        [
            EngineValidationError,
            PipelineError,
            ModelError,
            RuleError,
            ConfigurationError,
        ],
    )
    def test_inherits_message_and_context(self, exc_class: type) -> None:
        ctx = {"key": "value"}
        err = exc_class("test message", context=ctx)
        assert err.message == "test message"
        assert err.context == ctx

    @pytest.mark.parametrize(
        "exc_class",
        [
            EngineValidationError,
            PipelineError,
            ModelError,
            RuleError,
            ConfigurationError,
        ],
    )
    def test_str_representation(self, exc_class: type) -> None:
        err = exc_class("failure", context={"id": "abc"})
        assert "failure" in str(err)
        assert "id='abc'" in str(err)

    @pytest.mark.parametrize(
        "exc_class",
        [
            EngineValidationError,
            PipelineError,
            ModelError,
            RuleError,
            ConfigurationError,
        ],
    )
    def test_catchable_as_engine_error(self, exc_class: type) -> None:
        with pytest.raises(EngineError):
            raise exc_class("test")


class TestEngineValidationErrorNaming:
    """EngineValidationError must NOT shadow pydantic.ValidationError."""

    def test_no_collision_with_pydantic(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        # Hierarchy isolation: neither is a subclass of the other
        assert not issubclass(EngineValidationError, PydanticValidationError)
        assert not issubclass(PydanticValidationError, EngineValidationError)

    def test_different_class_names(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        assert EngineValidationError.__name__ != PydanticValidationError.__name__


class TestExceptionsImportFromPackageRoot:
    """Verify exceptions are properly re-exported from package root."""

    def test_all_exceptions_importable(self) -> None:
        from semantic_conversation_engine import (  # noqa: F401
            ConfigurationError,
            EngineError,
            EngineValidationError,
            ModelError,
            PipelineError,
            RuleError,
        )
