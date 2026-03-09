"""Label space definition for classification.

Defines the taxonomy of labels a classifier can produce. Centralizes
label names, per-label thresholds, and optional label hierarchy.

A LabelSpace is the single source of truth for what labels exist,
their thresholds, and their ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabelSpace:
    """Ordered set of classification labels with per-label thresholds.

    The label order determines the position in model output vectors.
    Each label can have its own decision threshold (overriding the
    classifier's default).

    Args:
        labels: Ordered list of label names. Must be non-empty, unique.
        thresholds: Per-label thresholds. Keys must be a subset of labels.
            Labels without explicit thresholds use default_threshold.
        default_threshold: Fallback threshold for labels not in thresholds.
    """

    labels: list[str]
    thresholds: dict[str, float] = field(default_factory=dict)
    default_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate label space invariants."""
        if not self.labels:
            raise ValueError("LabelSpace requires at least one label")
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("LabelSpace labels must be unique")
        unknown = set(self.thresholds) - set(self.labels)
        if unknown:
            raise ValueError(f"Thresholds reference unknown labels: {sorted(unknown)}")
        for name, value in self.thresholds.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Threshold for '{name}' must be in [0.0, 1.0], got {value}")
        if self.default_threshold < 0.0 or self.default_threshold > 1.0:
            raise ValueError(f"default_threshold must be in [0.0, 1.0], got {self.default_threshold}")

    def threshold_for(self, label: str) -> float:
        """Get the effective threshold for a label.

        Args:
            label: Label name.

        Returns:
            Per-label threshold if set, otherwise default_threshold.

        Raises:
            ValueError: If label is not in this label space.
        """
        if label not in set(self.labels):
            raise ValueError(f"Unknown label: '{label}'")
        return self.thresholds.get(label, self.default_threshold)

    def label_index(self, label: str) -> int:
        """Get the positional index of a label.

        Args:
            label: Label name.

        Returns:
            Zero-based index in the labels list.

        Raises:
            ValueError: If label is not in this label space.
        """
        try:
            return self.labels.index(label)
        except ValueError:
            raise ValueError(f"Unknown label: '{label}'") from None

    @property
    def size(self) -> int:
        """Number of labels in this space."""
        return len(self.labels)
