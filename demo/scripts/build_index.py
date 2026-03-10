"""Offline precompute pipeline: conversations → segments → embeddings → indexes.

Reads ingested conversations from JSONL, runs the TalkEx pipeline
(segmentation → context windows → embeddings), and builds both
lexical (BM25) and vector (Qdrant) indexes for search.

Output: Persisted indexes + metadata ready for the demo API server.

Usage:
    python demo/scripts/build_index.py --input demo/data/conversations.jsonl --output demo/data/index/
    python demo/scripts/build_index.py --input demo/data/conversations.jsonl --embedding-model all-MiniLM-L6-v2
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import time
from pathlib import Path

import click

from talkex.context.builder import SlidingWindowBuilder
from talkex.context.config import ContextWindowConfig
from talkex.embeddings.config import EmbeddingModelConfig
from talkex.embeddings.generator import NullEmbeddingGenerator
from talkex.embeddings.inputs import EmbeddingBatch, EmbeddingInput
from talkex.ingestion.enums import SourceFormat
from talkex.ingestion.inputs import TranscriptInput
from talkex.models.enums import Channel, ObjectType, PoolingStrategy
from talkex.models.types import ConversationId, EmbeddingId
from talkex.retrieval.bm25 import InMemoryBM25Index
from talkex.retrieval.builders import context_windows_to_lexical_docs
from talkex.retrieval.config import DistanceMetric, LexicalIndexConfig, VectorIndexConfig
from talkex.retrieval.qdrant import QdrantVectorIndex
from talkex.segmentation.config import SegmentationConfig
from talkex.segmentation.segmenter import TurnSegmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_conversations(input_path: str) -> list[dict]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    return conversations


def _create_embedding_generator(model_name: str, dimensions: int) -> tuple[object, int]:
    """Create an embedding generator.

    Returns the generator and actual dimensions.
    For sentence-transformers models, dimensions are auto-detected.
    """
    if model_name == "null":
        config = EmbeddingModelConfig(
            model_name="null",
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
        )
        return NullEmbeddingGenerator(model_config=config, dimensions=dimensions), dimensions

    try:
        from talkex.embeddings.generator import SentenceTransformerGenerator

        config = EmbeddingModelConfig(
            model_name=model_name,
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
        )
        gen = SentenceTransformerGenerator(model_config=config)
        # Detect dimensions from model
        test_batch = EmbeddingBatch(
            items=[
                EmbeddingInput(
                    embedding_id=EmbeddingId("emb_test"),
                    object_type=ObjectType.CONTEXT_WINDOW,
                    object_id="test",
                    text="test",
                )
            ]
        )
        test_result = gen.generate(test_batch)
        actual_dims = test_result[0].dimensions
        logger.info("Model %s detected dimensions: %d", model_name, actual_dims)
        return gen, actual_dims
    except ImportError:
        logger.warning("sentence-transformers not installed, falling back to NullEmbeddingGenerator")
        config = EmbeddingModelConfig(
            model_name="null",
            model_version="1.0",
            pooling_strategy=PoolingStrategy.MEAN,
        )
        return NullEmbeddingGenerator(model_config=config, dimensions=dimensions), dimensions


@click.command()
@click.option("--input", "input_path", default="demo/data/conversations.jsonl", help="Input JSONL file.")
@click.option("--output", "output_dir", default="demo/data/index", help="Output directory for indexes.")
@click.option("--embedding-model", default="null", help="Embedding model name (or 'null' for deterministic fake).")
@click.option("--dimensions", default=384, type=int, help="Embedding dimensions (for null model).")
@click.option("--window-size", default=5, type=int, help="Context window size (turns).")
@click.option("--stride", default=3, type=int, help="Context window stride.")
@click.option("--batch-size", default=64, type=int, help="Embedding batch size.")
def main(
    input_path: str,
    output_dir: str,
    embedding_model: str,
    dimensions: int,
    window_size: int,
    stride: int,
    batch_size: int,
) -> None:
    """Build search indexes from ingested conversations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load conversations
    logger.info("Loading conversations from %s...", input_path)
    raw_conversations = _load_conversations(input_path)
    logger.info("Loaded %d conversations", len(raw_conversations))

    # Initialize components
    segmenter = TurnSegmenter()
    seg_config = SegmentationConfig(min_turn_chars=5, merge_consecutive_same_speaker=False)
    ctx_builder = SlidingWindowBuilder()
    ctx_config = ContextWindowConfig(window_size=window_size, stride=stride)

    emb_gen, actual_dims = _create_embedding_generator(embedding_model, dimensions)

    # Initialize indexes
    lex_config = LexicalIndexConfig()
    bm25_index = InMemoryBM25Index(config=lex_config)

    vec_config = VectorIndexConfig(dimensions=actual_dims, metric=DistanceMetric.COSINE)
    qdrant_index = QdrantVectorIndex(
        config=vec_config,
        collection_name="talkex_demo",
        path=str(output_path / "qdrant_storage"),
    )

    # Process conversations
    all_conversations_meta: list[dict] = []
    all_windows_meta: list[dict] = []
    total_turns = 0
    total_windows = 0
    total_embeddings = 0

    pipeline_start = time.monotonic()

    for i, conv_data in enumerate(raw_conversations):
        conv_id = conv_data["conversation_id"]

        # Preprocess: split continuous ASR text into sentence-based turns.
        # The dataset has no newlines, so we split by sentence-ending punctuation
        # to create meaningful turn boundaries for context windows.
        raw_text = conv_data["text"]
        sentences = re.split(r"(?<=[.!?])\s+", raw_text)
        multiline_text = "\n".join(s for s in (s.strip() for s in sentences) if s)

        # Create TranscriptInput
        transcript = TranscriptInput(
            conversation_id=ConversationId(conv_id),
            source_format=SourceFormat.MULTILINE,
            raw_text=multiline_text,
            channel=Channel.VOICE,
            metadata={
                "domain": conv_data.get("domain", "unknown"),
                "topic": conv_data.get("topic", "unknown"),
                "asr_confidence": conv_data.get("asr_confidence", 0.0),
                "audio_duration_seconds": conv_data.get("audio_duration_seconds", 0),
            },
        )

        # Stage 1: Segmentation
        turns = segmenter.segment(transcript, seg_config)
        if not turns:
            continue

        total_turns += len(turns)

        # Stage 2: Build context windows
        from datetime import UTC, datetime

        from talkex.models.conversation import Conversation

        conversation = Conversation(
            conversation_id=ConversationId(conv_id),
            channel=Channel.VOICE,
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            metadata={
                "domain": conv_data.get("domain", "unknown"),
                "topic": conv_data.get("topic", "unknown"),
            },
        )

        windows = ctx_builder.build(conversation, turns, ctx_config)
        total_windows += len(windows)

        # Store conversation metadata
        all_conversations_meta.append(
            {
                "conversation_id": conv_id,
                "domain": conv_data.get("domain", "unknown"),
                "topic": conv_data.get("topic", "unknown"),
                "asr_confidence": conv_data.get("asr_confidence", 0.0),
                "audio_duration_seconds": conv_data.get("audio_duration_seconds", 0),
                "turn_count": len(turns),
                "window_count": len(windows),
                "text": conv_data["text"][:500],
                "turns": [
                    {
                        "turn_id": str(t.turn_id),
                        "speaker": t.speaker.value,
                        "raw_text": t.raw_text,
                        "normalized_text": t.normalized_text,
                    }
                    for t in turns
                ],
            }
        )

        # Store window metadata
        for w in windows:
            all_windows_meta.append(
                {
                    "window_id": str(w.window_id),
                    "conversation_id": conv_id,
                    "window_text": w.window_text,
                    "start_index": w.start_index,
                    "end_index": w.end_index,
                    "window_size": w.window_size,
                    "domain": conv_data.get("domain", "unknown"),
                    "topic": conv_data.get("topic", "unknown"),
                }
            )

        # Stage 3: Build lexical index
        lex_docs = context_windows_to_lexical_docs(windows)
        bm25_index.index(lex_docs)

        # Stage 4: Generate embeddings and build vector index
        emb_inputs = [
            EmbeddingInput(
                embedding_id=EmbeddingId(f"emb_{w.window_id}"),
                object_type=ObjectType.CONTEXT_WINDOW,
                object_id=str(w.window_id),
                text=w.window_text,
            )
            for w in windows
        ]

        # Process in batches
        for batch_start in range(0, len(emb_inputs), batch_size):
            batch_items = emb_inputs[batch_start : batch_start + batch_size]
            batch = EmbeddingBatch(items=batch_items)
            records = emb_gen.generate(batch)
            qdrant_index.upsert(records)
            total_embeddings += len(records)

        if (i + 1) % 500 == 0:
            logger.info(
                "  Processed %d/%d conversations (%d turns, %d windows, %d embeddings)",
                i + 1,
                len(raw_conversations),
                total_turns,
                total_windows,
                total_embeddings,
            )

    pipeline_elapsed = (time.monotonic() - pipeline_start) * 1000

    # Persist BM25 index
    bm25_path = output_path / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)
    logger.info("BM25 index saved to %s", bm25_path)

    # Persist metadata
    conversations_path = output_path / "conversations.json"
    with open(conversations_path, "w") as f:
        json.dump(all_conversations_meta, f, ensure_ascii=False)
    logger.info("Conversation metadata saved to %s (%d entries)", conversations_path, len(all_conversations_meta))

    windows_path = output_path / "windows.json"
    with open(windows_path, "w") as f:
        json.dump(all_windows_meta, f, ensure_ascii=False)
    logger.info("Window metadata saved to %s (%d entries)", windows_path, len(all_windows_meta))

    # Save build manifest
    manifest = {
        "conversations": len(all_conversations_meta),
        "turns": total_turns,
        "windows": total_windows,
        "embeddings": total_embeddings,
        "embedding_model": embedding_model,
        "dimensions": actual_dims,
        "window_size": window_size,
        "stride": stride,
        "pipeline_ms": round(pipeline_elapsed, 2),
        "qdrant_path": str(output_path / "qdrant_storage"),
        "bm25_path": str(bm25_path),
    }
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("─" * 60)
    logger.info("Index build complete")
    logger.info("  Conversations: %d", len(all_conversations_meta))
    logger.info("  Turns:         %d", total_turns)
    logger.info("  Windows:       %d", total_windows)
    logger.info("  Embeddings:    %d", total_embeddings)
    logger.info("  Pipeline time: %.2f ms", pipeline_elapsed)
    logger.info("  Output:        %s", output_path)


if __name__ == "__main__":
    main()
