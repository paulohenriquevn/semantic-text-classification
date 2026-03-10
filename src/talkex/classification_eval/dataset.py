"""Classification evaluation dataset schema.

Defines the data structures for classification evaluation examples, ground
truth labels, and evaluation datasets. Datasets are versionable and
serializable to/from JSON for reproducibility.

A ClassificationExample pairs an input (source_id + text) with its known
ground truth labels. Labels carry relevance grade for future weighted
metrics. A ClassificationDataset is a named, versioned collection of examples.

Supports single-label and multi-label ground truth, and multi-level
classification (turn, context_window, conversation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GroundTruthLabel:
    """A ground truth label for a classification example.

    Args:
        label: The correct label name.
        relevance: Relevance grade. 1 = binary relevant. Higher values
            indicate stronger relevance (for future weighted metrics).
    """

    label: str
    relevance: int = 1


@dataclass(frozen=True)
class ClassificationExample:
    """A single classification evaluation example with ground truth.

    Args:
        example_id: Unique identifier for this example.
        text: Text content to classify.
        ground_truth: List of correct labels for this example.
        source_type: Classification level (turn, context_window, conversation).
        embedding: Optional pre-computed embedding vector for this example.
        features: Optional pre-computed features for this example.
        metadata: Additional context (e.g., conversation_id, difficulty).
    """

    example_id: str
    text: str
    ground_truth: list[GroundTruthLabel]
    source_type: str = "context_window"
    embedding: list[float] | None = None
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label_set(self) -> set[str]:
        """Set of ground truth label names for this example."""
        return {gt.label for gt in self.ground_truth}

    @property
    def label_relevance_map(self) -> dict[str, int]:
        """Mapping from label name to relevance grade."""
        return {gt.label: gt.relevance for gt in self.ground_truth}


@dataclass(frozen=True)
class ClassificationDataset:
    """A named, versioned collection of classification evaluation examples.

    Args:
        name: Human-readable dataset name.
        version: Dataset version string for reproducibility.
        examples: List of classification examples with ground truth.
        label_names: Ordered list of all possible labels in this dataset.
        description: Optional dataset description.
    """

    name: str
    version: str
    examples: list[ClassificationExample]
    label_names: list[str]
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
            "label_names": self.label_names,
            "examples": [
                {
                    "example_id": ex.example_id,
                    "text": ex.text,
                    "ground_truth": [{"label": gt.label, "relevance": gt.relevance} for gt in ex.ground_truth],
                    "source_type": ex.source_type,
                    "embedding": ex.embedding,
                    "features": ex.features,
                    "metadata": ex.metadata,
                }
                for ex in self.examples
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: str | Path) -> ClassificationDataset:
        """Deserialize dataset from JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded ClassificationDataset.
        """
        data = json.loads(Path(path).read_text())
        examples = [
            ClassificationExample(
                example_id=ex["example_id"],
                text=ex["text"],
                ground_truth=[
                    GroundTruthLabel(
                        label=gt["label"],
                        relevance=gt["relevance"],
                    )
                    for gt in ex["ground_truth"]
                ],
                source_type=ex.get("source_type", "context_window"),
                embedding=ex.get("embedding"),
                features=ex.get("features", {}),
                metadata=ex.get("metadata", {}),
            )
            for ex in data["examples"]
        ]
        return ClassificationDataset(
            name=data["name"],
            version=data["version"],
            examples=examples,
            label_names=data["label_names"],
            description=data.get("description", ""),
        )
