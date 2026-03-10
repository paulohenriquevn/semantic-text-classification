"""Ingest HuggingFace call center dataset into TalkEx domain models.

Downloads zip files from AIxBlock/92k-real-world-call-center-scripts-english,
extracts JSON transcripts, and converts them to TalkEx Conversation/Turn models.

Output: JSON Lines file with ingested conversations ready for pipeline processing.

Usage:
    python demo/scripts/ingest_dataset.py --limit 5000 --output demo/data/conversations.jsonl
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import click
from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "AIxBlock/92k-real-world-call-center-scripts-english"

# Map zip filenames to domain categories
_DOMAIN_MAP: dict[str, dict[str, str]] = {
    "auto_insurance_customer_service_inbound.zip": {"domain": "auto_insurance", "topic": "inbound"},
    "automotive_and_healthcare_insurance_inbound.zip": {"domain": "automotive_healthcare", "topic": "inbound"},
    "automotive_inbound.zip": {"domain": "automotive", "topic": "inbound"},
    "customer_service_general_inbound.zip": {"domain": "customer_service", "topic": "inbound"},
    "home_service_inbound.zip": {"domain": "home_service", "topic": "inbound"},
    "insurance_outbound.zip": {"domain": "insurance", "topic": "outbound"},
    "medical_equipment_outbound.zip": {"domain": "medical_equipment", "topic": "outbound"},
    "medicare_inbound.zip": {"domain": "medicare", "topic": "inbound"},
    "home_ervice_inbound&telecom _outbound.zip": {"domain": "home_telecom", "topic": "mixed"},
    "(re-uploaded)PII_Redacted_Transcripts_aixblock-automotive-stereo-inbound-104h.zip": {
        "domain": "automotive_stereo",
        "topic": "inbound",
    },
    "(reupload)PII_redacted_auto_insurance_script.zip": {"domain": "auto_insurance_v2", "topic": "inbound"},
}


@dataclass
class IngestedConversation:
    """Lightweight conversation record for JSONL serialization."""

    conversation_id: str
    text: str
    domain: str
    topic: str
    asr_confidence: float
    audio_duration_seconds: int
    word_count: int
    source_file: str

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "text": self.text,
            "domain": self.domain,
            "topic": self.topic,
            "asr_confidence": self.asr_confidence,
            "audio_duration_seconds": self.audio_duration_seconds,
            "word_count": self.word_count,
            "source_file": self.source_file,
        }


@dataclass
class IngestionStats:
    """Track ingestion progress."""

    total_files: int = 0
    ingested: int = 0
    skipped_empty: int = 0
    skipped_short: int = 0
    errors: int = 0
    domains: dict[str, int] = field(default_factory=dict)


def _extract_conversations(
    zip_path: str,
    domain_info: dict[str, str],
    *,
    min_words: int = 20,
    limit: int | None = None,
    current_count: int = 0,
) -> tuple[list[IngestedConversation], IngestionStats]:
    """Extract conversations from a single zip file.

    Args:
        zip_path: Path to the downloaded zip file.
        domain_info: Domain and topic metadata.
        min_words: Minimum word count to include a conversation.
        limit: Global limit on total conversations.
        current_count: How many conversations already ingested.

    Returns:
        Tuple of (conversations, stats).
    """
    conversations: list[IngestedConversation] = []
    stats = IngestionStats()

    with zipfile.ZipFile(zip_path) as zf:
        json_files = [name for name in zf.namelist() if name.endswith(".json") and "__MACOSX" not in name]
        stats.total_files = len(json_files)

        for name in json_files:
            if limit is not None and (current_count + len(conversations)) >= limit:
                break

            try:
                with zf.open(name) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                stats.errors += 1
                continue

            text = data.get("text", "").strip()
            if not text:
                stats.skipped_empty += 1
                continue

            word_count = len(text.split())
            if word_count < min_words:
                stats.skipped_short += 1
                continue

            file_id = Path(name).stem.replace("_transcript", "")
            conv_id = f"conv_{file_id}"

            conv = IngestedConversation(
                conversation_id=conv_id,
                text=text,
                domain=domain_info.get("domain", "unknown"),
                topic=domain_info.get("topic", "unknown"),
                asr_confidence=float(data.get("confidence", 0.0)),
                audio_duration_seconds=int(data.get("audio_duration", 0)),
                word_count=word_count,
                source_file=name,
            )
            conversations.append(conv)
            stats.ingested += 1

    return conversations, stats


@click.command()
@click.option("--limit", default=5000, type=int, help="Max conversations to ingest.")
@click.option("--output", default="demo/data/conversations.jsonl", help="Output JSONL path.")
@click.option("--min-words", default=20, type=int, help="Min words per conversation.")
@click.option(
    "--zips",
    default=None,
    type=str,
    help="Comma-separated zip names to process (default: all).",
)
def main(limit: int, output: str, min_words: int, zips: str | None) -> None:
    """Ingest call center transcripts from HuggingFace into JSONL."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # List available zips
    all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
    available_zips = [f for f in all_files if f.endswith(".zip")]

    if zips:
        selected = [z.strip() for z in zips.split(",")]
        available_zips = [z for z in available_zips if z in selected]

    logger.info("Found %d zip files, ingesting up to %d conversations", len(available_zips), limit)

    total_stats = IngestionStats()
    all_conversations: list[IngestedConversation] = []

    for zip_name in available_zips:
        if len(all_conversations) >= limit:
            break

        domain_info = _DOMAIN_MAP.get(zip_name, {"domain": "unknown", "topic": "unknown"})
        logger.info("Downloading %s (domain: %s)...", zip_name, domain_info["domain"])

        try:
            zip_path = hf_hub_download(REPO_ID, zip_name, repo_type="dataset")
        except Exception as e:
            logger.error("Failed to download %s: %s", zip_name, e)
            total_stats.errors += 1
            continue

        conversations, stats = _extract_conversations(
            zip_path,
            domain_info,
            min_words=min_words,
            limit=limit,
            current_count=len(all_conversations),
        )

        all_conversations.extend(conversations)
        total_stats.total_files += stats.total_files
        total_stats.ingested += stats.ingested
        total_stats.skipped_empty += stats.skipped_empty
        total_stats.skipped_short += stats.skipped_short
        total_stats.errors += stats.errors

        domain = domain_info["domain"]
        total_stats.domains[domain] = total_stats.domains.get(domain, 0) + stats.ingested

        logger.info(
            "  %s: %d ingested, %d skipped, %d errors (total: %d/%d)",
            domain,
            stats.ingested,
            stats.skipped_empty + stats.skipped_short,
            stats.errors,
            len(all_conversations),
            limit,
        )

    # Write output
    with open(output_path, "w") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv.to_dict(), ensure_ascii=False) + "\n")

    logger.info("─" * 60)
    logger.info("Ingestion complete")
    logger.info("  Total conversations: %d", len(all_conversations))
    logger.info("  Total files scanned: %d", total_stats.total_files)
    logger.info("  Skipped (empty):     %d", total_stats.skipped_empty)
    logger.info("  Skipped (short):     %d", total_stats.skipped_short)
    logger.info("  Errors:              %d", total_stats.errors)
    logger.info("  Domains: %s", json.dumps(total_stats.domains, indent=4))
    logger.info("  Output: %s", output_path)


if __name__ == "__main__":
    main()
