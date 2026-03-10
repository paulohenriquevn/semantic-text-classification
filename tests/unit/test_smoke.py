"""Smoke tests verifying project setup and importability."""

import talkex


class TestProjectSetup:
    def test_package_is_importable(self) -> None:
        assert hasattr(talkex, "__version__")

    def test_version_is_string(self) -> None:
        assert isinstance(talkex.__version__, str)

    def test_subpackages_are_importable(self) -> None:
        from talkex import (  # noqa: F401
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
