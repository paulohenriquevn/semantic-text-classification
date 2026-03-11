"""Statistical tests for hypothesis validation.

Provides non-parametric tests for comparing experiment results,
following the experimental design in desenho-experimental.md.

All functions accept arrays of paired measurements and return
structured results with test statistic, p-value, and effect size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class TestResult:
    """Result of a statistical significance test.

    Attributes:
        test_name: Name of the statistical test applied.
        statistic: Test statistic value.
        p_value: p-value for the test.
        significant: Whether p_value < alpha.
        alpha: Significance level used.
        effect_size: Cohen's d or rank-biserial correlation.
        confidence_interval: 95% CI for the mean difference (if applicable).
        summary: Human-readable summary of the result.
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float | None
    confidence_interval: tuple[float, float] | None
    summary: str


def wilcoxon_signed_rank(
    scores_a: list[float],
    scores_b: list[float],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    label_a: str = "A",
    label_b: str = "B",
) -> TestResult:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test. Appropriate for
    comparing two systems on the same set of queries/examples.

    Args:
        scores_a: Metric scores for system A (one per query/example).
        scores_b: Metric scores for system B (one per query/example).
        alpha: Significance level (default 0.05).
        alternative: 'two-sided', 'greater', or 'less'.
        label_a: Label for system A in summary.
        label_b: Label for system B in summary.

    Returns:
        TestResult with statistic, p-value, and effect size.

    Raises:
        ValueError: If input arrays have different lengths or are empty.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Arrays must have same length: {len(scores_a)} vs {len(scores_b)}")
    if len(scores_a) == 0:
        raise ValueError("Arrays must be non-empty")

    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    diff = a - b

    # Remove zero differences (tied pairs)
    nonzero_mask = diff != 0
    if not np.any(nonzero_mask):
        return TestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            summary=f"No difference between {label_a} and {label_b} (all pairs tied)",
        )

    result = scipy_stats.wilcoxon(a, b, alternative=alternative)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    # Rank-biserial correlation as effect size
    n = int(np.sum(nonzero_mask))
    effect_size = 1 - (2 * statistic) / (n * (n + 1) / 2)

    mean_diff = float(np.mean(diff))
    significant = p_value < alpha

    direction = ">" if mean_diff > 0 else "<" if mean_diff < 0 else "="
    sig_text = "significant" if significant else "not significant"

    return TestResult(
        test_name="Wilcoxon signed-rank",
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        confidence_interval=None,
        summary=(
            f"{label_a} {direction} {label_b} (mean diff={mean_diff:.4f}, p={p_value:.4f}, {sig_text} at alpha={alpha})"
        ),
    )


@dataclass(frozen=True)
class FriedmanResult:
    """Result of Friedman test with optional post-hoc Nemenyi."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    n_systems: int
    n_samples: int
    mean_ranks: dict[str, float]
    post_hoc: dict[tuple[str, str], float] | None
    summary: str


def friedman_nemenyi(
    scores: dict[str, list[float]],
    *,
    alpha: float = 0.05,
) -> FriedmanResult:
    """Friedman test with Nemenyi post-hoc for multiple systems.

    Non-parametric alternative to repeated-measures ANOVA.
    Appropriate for comparing 3+ systems on the same set of queries.

    Args:
        scores: Dict mapping system name to list of scores.
            All lists must have the same length.
        alpha: Significance level (default 0.05).

    Returns:
        FriedmanResult with test statistic, p-value, mean ranks,
        and post-hoc pairwise comparisons if significant.

    Raises:
        ValueError: If fewer than 3 systems or arrays have different lengths.
    """
    names = list(scores.keys())
    if len(names) < 3:
        raise ValueError(f"Friedman test requires at least 3 systems, got {len(names)}")

    arrays = [np.array(scores[name], dtype=np.float64) for name in names]
    n_samples = len(arrays[0])
    if any(len(a) != n_samples for a in arrays):
        raise ValueError("All score arrays must have the same length")
    if n_samples == 0:
        raise ValueError("Score arrays must be non-empty")

    # Friedman test
    result = scipy_stats.friedmanchisquare(*arrays)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    significant = p_value < alpha

    # Compute mean ranks
    data_matrix = np.column_stack(arrays)
    ranks = np.zeros_like(data_matrix, dtype=np.float64)
    for i in range(n_samples):
        ranks[i] = scipy_stats.rankdata(-data_matrix[i])  # Higher score = lower rank (better)
    mean_ranks = {name: float(ranks[:, j].mean()) for j, name in enumerate(names)}

    # Post-hoc Nemenyi if significant
    post_hoc: dict[tuple[str, str], float] | None = None
    if significant:
        post_hoc = _nemenyi_post_hoc(ranks, names, n_samples)

    sig_text = "significant" if significant else "not significant"
    best = min(mean_ranks, key=mean_ranks.get)  # type: ignore[arg-type]
    return FriedmanResult(
        test_name="Friedman + Nemenyi",
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        n_systems=len(names),
        n_samples=n_samples,
        mean_ranks=mean_ranks,
        post_hoc=post_hoc,
        summary=(
            f"Friedman χ²={statistic:.2f}, p={p_value:.4f} ({sig_text}). "
            f"Best mean rank: {best} ({mean_ranks[best]:.2f})"
        ),
    )


def _nemenyi_post_hoc(
    ranks: np.ndarray,
    names: list[str],
    n_samples: int,
) -> dict[tuple[str, str], float]:
    """Compute Nemenyi post-hoc pairwise comparisons.

    Uses the Studentized range distribution approximation.

    Returns:
        Dict mapping (system_a, system_b) to p-value.
    """
    k = len(names)
    mean_ranks = ranks.mean(axis=0)
    pairwise: dict[tuple[str, str], float] = {}

    # Critical difference approach using Tukey's HSD on ranks
    se = np.sqrt(k * (k + 1) / (6 * n_samples))

    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            # Approximate p-value using normal distribution for large samples
            z = diff / (se * np.sqrt(2))
            p = float(2 * (1 - scipy_stats.norm.cdf(z)))
            pairwise[(names[i], names[j])] = min(p * k * (k - 1) / 2, 1.0)  # Bonferroni correction

    return pairwise


@dataclass(frozen=True)
class BootstrapCIResult:
    """Result of bootstrap confidence interval estimation."""

    metric_name: str
    observed: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    summary: str


def bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    *,
    metric_name: str = "difference",
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> BootstrapCIResult:
    """Bootstrap confidence interval for the difference between two systems.

    Estimates the CI for mean(scores_a) - mean(scores_b) via
    bootstrap resampling with replacement.

    Args:
        scores_a: Metric scores for system A.
        scores_b: Metric scores for system B.
        metric_name: Name of the metric being compared.
        confidence_level: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap iterations (default 10000).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapCIResult with observed difference and CI bounds.

    Raises:
        ValueError: If input arrays have different lengths or are empty.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Arrays must have same length: {len(scores_a)} vs {len(scores_b)}")
    if len(scores_a) == 0:
        raise ValueError("Arrays must be non-empty")

    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    observed_diff = float(np.mean(a) - np.mean(b))

    rng = np.random.default_rng(seed)
    n = len(a)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_diffs[i] = np.mean(a[indices]) - np.mean(b[indices])

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))

    contains_zero = ci_lower <= 0 <= ci_upper
    sig_text = "contains zero (not significant)" if contains_zero else "excludes zero (significant)"

    return BootstrapCIResult(
        metric_name=metric_name,
        observed=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        summary=(
            f"{metric_name}: observed diff={observed_diff:.4f}, "
            f"{confidence_level * 100:.0f}% CI=[{ci_lower:.4f}, {ci_upper:.4f}] ({sig_text})"
        ),
    )
