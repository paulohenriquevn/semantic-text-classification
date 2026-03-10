"""Unit tests for pipeline stage protocols.

Tests verify that the Protocol classes are importable and define
the expected method signatures. Structural protocol compliance
is verified via a placeholder implementation.
"""

from datetime import UTC, datetime

from talkex.context.config import ContextWindowConfig
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.context_window import ContextWindow
from talkex.models.conversation import Conversation
from talkex.models.enums import Channel
from talkex.models.turn import Turn
from talkex.models.types import ConversationId
from talkex.pipeline.protocols import ContextBuilder, Segmenter
from talkex.segmentation.config import SegmentationConfig

# ---------------------------------------------------------------------------
# Placeholder implementations for structural protocol check
# ---------------------------------------------------------------------------


class _StubSegmenter:
    """Minimal implementation satisfying the Segmenter protocol."""

    def segment(self, transcript: TranscriptInput, config: SegmentationConfig) -> list[Turn]:
        return []


class _StubContextBuilder:
    """Minimal implementation satisfying the ContextBuilder protocol."""

    def build(
        self,
        conversation: Conversation,
        turns: list[Turn],
        config: ContextWindowConfig,
    ) -> list[ContextWindow]:
        return []


# ---------------------------------------------------------------------------
# Protocol importability
# ---------------------------------------------------------------------------


class TestProtocolImportability:
    def test_segmenter_importable(self) -> None:
        assert Segmenter is not None

    def test_context_builder_importable(self) -> None:
        assert ContextBuilder is not None


# ---------------------------------------------------------------------------
# Structural protocol compliance
# ---------------------------------------------------------------------------


class TestSegmenterProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        """A class with the right method signature satisfies the Protocol."""
        stub: Segmenter = _StubSegmenter()
        transcript = TranscriptInput(
            conversation_id=ConversationId("conv_001"),
            channel=Channel.CHAT,
            raw_text="CUSTOMER: hello",
            source_format=SourceFormat.LABELED,
        )
        config = SegmentationConfig()
        result = stub.segment(transcript, config)
        assert isinstance(result, list)

    def test_segment_returns_list_of_turns(self) -> None:
        stub: Segmenter = _StubSegmenter()
        transcript = TranscriptInput(
            conversation_id=ConversationId("conv_001"),
            channel=Channel.CHAT,
            raw_text="CUSTOMER: hello",
            source_format=SourceFormat.LABELED,
        )
        result = stub.segment(transcript, SegmentationConfig())
        assert isinstance(result, list)


class TestContextBuilderProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        """A class with the right method signature satisfies the Protocol."""
        stub: ContextBuilder = _StubContextBuilder()
        conversation = Conversation(
            conversation_id=ConversationId("conv_001"),
            channel=Channel.CHAT,
            start_time=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
        )
        config = ContextWindowConfig()
        result = stub.build(conversation, [], config)
        assert isinstance(result, list)

    def test_build_returns_list_of_context_windows(self) -> None:
        stub: ContextBuilder = _StubContextBuilder()
        conversation = Conversation(
            conversation_id=ConversationId("conv_001"),
            channel=Channel.VOICE,
            start_time=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
        )
        result = stub.build(conversation, [], ContextWindowConfig())
        assert isinstance(result, list)
