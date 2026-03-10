"""Evaluation dataset schema for retrieval benchmarking.

Defines the data structures for evaluation queries, relevance
judgments, and evaluation datasets. Datasets are versionable and
serializable to/from JSON for reproducibility.

An EvaluationExample pairs a query with its known relevant documents.
An EvaluationDataset is a named, versioned collection of examples.

Relevance can be binary (relevant or not) or graded (relevance score
from 0 to N). Graded relevance is used by nDCG; binary relevance
is sufficient for Recall@K, Precision@K, and MRR.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RelevanceJudgment:
    """A relevance judgment for a single document.

    Args:
        document_id: ID of the relevant document.
        relevance: Relevance grade. 1 = binary relevant. Higher values
            indicate stronger relevance (for graded metrics like nDCG).
    """

    document_id: str
    relevance: int = 1


@dataclass(frozen=True)
class EvaluationExample:
    """A single evaluation query with its relevant documents.

    Args:
        query_id: Unique identifier for this example.
        query_text: The query text to evaluate.
        relevant_docs: List of relevance judgments for this query.
        metadata: Additional context (e.g., category, difficulty).
    """

    query_id: str
    query_text: str
    relevant_docs: list[RelevanceJudgment]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def relevant_doc_ids(self) -> set[str]:
        """Set of relevant document IDs for this example."""
        return {j.document_id for j in self.relevant_docs}

    @property
    def relevance_map(self) -> dict[str, int]:
        """Mapping from document_id to relevance grade."""
        return {j.document_id: j.relevance for j in self.relevant_docs}


@dataclass(frozen=True)
class EvaluationDataset:
    """A named, versioned collection of evaluation examples.

    Args:
        name: Human-readable dataset name.
        version: Dataset version string for reproducibility.
        examples: List of evaluation examples.
        description: Optional dataset description.
    """

    name: str
    version: str
    examples: list[EvaluationExample]
    description: str = ""

    def save(self, path: str | Path) -> None:
        """Serialize dataset to JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "examples": [
                {
                    "query_id": ex.query_id,
                    "query_text": ex.query_text,
                    "relevant_docs": [
                        {"document_id": j.document_id, "relevance": j.relevance} for j in ex.relevant_docs
                    ],
                    "metadata": ex.metadata,
                }
                for ex in self.examples
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: str | Path) -> EvaluationDataset:
        """Deserialize dataset from JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded EvaluationDataset.
        """
        data = json.loads(Path(path).read_text())
        examples = [
            EvaluationExample(
                query_id=ex["query_id"],
                query_text=ex["query_text"],
                relevant_docs=[
                    RelevanceJudgment(
                        document_id=j["document_id"],
                        relevance=j["relevance"],
                    )
                    for j in ex["relevant_docs"]
                ],
                metadata=ex.get("metadata", {}),
            )
            for ex in data["examples"]
        ]
        return EvaluationDataset(
            name=data["name"],
            version=data["version"],
            examples=examples,
            description=data.get("description", ""),
        )
