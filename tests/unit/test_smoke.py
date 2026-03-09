"""Smoke tests verifying project setup and importability."""

import semantic_conversation_engine


class TestProjectSetup:
    def test_package_is_importable(self) -> None:
        assert hasattr(semantic_conversation_engine, "__version__")

    def test_version_is_string(self) -> None:
        assert isinstance(semantic_conversation_engine.__version__, str)

    def test_subpackages_are_importable(self) -> None:
        from semantic_conversation_engine import (  # noqa: F401
            analytics,
            classification,
            context,
            embeddings,
            ingestion,
            models,
            retrieval,
            rules,
            segmentation,
        )
