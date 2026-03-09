"""SlidingWindowBuilder — orchestrator for the context window pipeline.

Implements the ContextBuilder protocol from pipeline.protocols.
Composes pure functions from windowing, rendering, and metrics modules
into a fully deterministic pipeline:

    generate slices → render text → extract role views → compute metrics → build ContextWindows

Window IDs are deterministic: ``{conversation_id}_win_{index}`` where
index is the zero-based position in the output list. Same input and
config always produce byte-identical output.
"""

from semantic_conversation_engine.context.config import ContextWindowConfig
from semantic_conversation_engine.context.metrics import compute_window_metrics
from semantic_conversation_engine.context.rendering import (
    extract_role_text,
    render_window_text,
)
from semantic_conversation_engine.context.windowing import generate_window_slices
from semantic_conversation_engine.models.context_window import ContextWindow
from semantic_conversation_engine.models.conversation import Conversation
from semantic_conversation_engine.models.enums import SpeakerRole
from semantic_conversation_engine.models.turn import Turn
from semantic_conversation_engine.models.types import WindowId


class SlidingWindowBuilder:
    """Builds context windows from conversation turns using a sliding window.

    Implements the ContextBuilder protocol. Orchestrates three pure-function
    modules: windowing, rendering, and metrics.

    The pipeline is fully deterministic: same input + same config produces
    byte-identical output, including window IDs.
    """

    def build(
        self,
        conversation: Conversation,
        turns: list[Turn],
        config: ContextWindowConfig,
    ) -> list[ContextWindow]:
        """Build context windows from a conversation's turns.

        Args:
            conversation: The parent conversation (provides conversation_id).
            turns: Ordered list of turns to window over. Empty list
                produces empty result.
            config: Window configuration controlling size, stride,
                rendering, and tail behavior.

        Returns:
            Ordered list of ContextWindow objects. Each window carries
            rendered text, role-aware views, and operational metrics
            in its metadata. Empty list if no valid windows can be
            generated.
        """
        slices = generate_window_slices(turns, config)

        windows: list[ContextWindow] = []
        for i, ws in enumerate(slices):
            # 1. Render window text
            window_text = render_window_text(ws.turns, config)

            # 2. Extract role-aware views
            customer_text = extract_role_text(ws.turns, SpeakerRole.CUSTOMER, config.render_turn_delimiter)
            agent_text = extract_role_text(ws.turns, SpeakerRole.AGENT, config.render_turn_delimiter)

            # 3. Compute metrics and assemble metadata
            metadata = compute_window_metrics(ws.turns, window_text, customer_text, agent_text)

            # 4. Build ContextWindow
            window = ContextWindow(
                window_id=WindowId(f"{conversation.conversation_id}_win_{i}"),
                conversation_id=conversation.conversation_id,
                turn_ids=[t.turn_id for t in ws.turns],
                window_text=window_text,
                start_index=ws.start_index,
                end_index=ws.end_index,
                window_size=len(ws.turns),
                stride=config.stride,
                metadata=metadata,
            )
            windows.append(window)

        return windows
