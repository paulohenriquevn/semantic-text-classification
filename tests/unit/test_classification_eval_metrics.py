"""Unit tests for classification evaluation metrics.

Tests cover: precision, recall, f1_score, per_label_metrics, micro_f1, macro_f1.
Each metric is tested for happy path, edge cases, and determinism.
"""

from talkex.classification_eval.metrics import (
    f1_score,
    macro_f1,
    micro_f1,
    per_label_metrics,
    precision,
    recall,
)

# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_perfect_precision(self) -> None:
        assert precision({"billing"}, {"billing"}) == 1.0

    def test_zero_precision(self) -> None:
        assert precision({"cancel"}, {"billing"}) == 0.0

    def test_partial_precision(self) -> None:
        assert precision({"billing", "cancel"}, {"billing"}) == 0.5

    def test_empty_predicted(self) -> None:
        assert precision(set(), {"billing"}) == 0.0

    def test_empty_ground_truth(self) -> None:
        # All predictions are "wrong" since none match ground truth
        assert precision({"billing"}, set()) == 0.0

    def test_both_empty(self) -> None:
        assert precision(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecall:
    def test_perfect_recall(self) -> None:
        assert recall({"billing"}, {"billing"}) == 1.0

    def test_zero_recall(self) -> None:
        assert recall({"cancel"}, {"billing"}) == 0.0

    def test_partial_recall(self) -> None:
        assert recall({"billing"}, {"billing", "cancel"}) == 0.5

    def test_empty_predicted(self) -> None:
        assert recall(set(), {"billing"}) == 0.0

    def test_empty_ground_truth(self) -> None:
        assert recall({"billing"}, set()) == 0.0

    def test_both_empty(self) -> None:
        assert recall(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# F1 Score
# ---------------------------------------------------------------------------


class TestF1Score:
    def test_perfect_f1(self) -> None:
        assert f1_score({"billing"}, {"billing"}) == 1.0

    def test_zero_f1(self) -> None:
        assert f1_score({"cancel"}, {"billing"}) == 0.0

    def test_harmonic_mean(self) -> None:
        # precision=0.5, recall=1.0 → F1 = 2*0.5*1.0/(0.5+1.0) = 2/3
        result = f1_score({"billing", "cancel"}, {"billing"})
        assert abs(result - 2 / 3) < 1e-10

    def test_symmetric_partial(self) -> None:
        # pred={a,b}, gt={b,c} → P=1/2, R=1/2, F1=1/2
        result = f1_score({"a", "b"}, {"b", "c"})
        assert abs(result - 0.5) < 1e-10

    def test_both_empty(self) -> None:
        assert f1_score(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# Per-Label Metrics
# ---------------------------------------------------------------------------


class TestPerLabelMetrics:
    def test_single_label_perfect(self) -> None:
        preds = [{"billing"}]
        gts = [{"billing"}]
        result = per_label_metrics(preds, gts, ["billing"])
        assert result["billing"]["precision"] == 1.0
        assert result["billing"]["recall"] == 1.0
        assert result["billing"]["f1"] == 1.0
        assert result["billing"]["support"] == 1.0

    def test_single_label_missed(self) -> None:
        preds = [set()]
        gts = [{"billing"}]
        result = per_label_metrics(preds, gts, ["billing"])
        assert result["billing"]["precision"] == 0.0
        assert result["billing"]["recall"] == 0.0
        assert result["billing"]["f1"] == 0.0
        assert result["billing"]["support"] == 1.0

    def test_single_label_false_positive(self) -> None:
        preds = [{"billing"}]
        gts = [set()]
        result = per_label_metrics(preds, gts, ["billing"])
        assert result["billing"]["precision"] == 0.0
        assert result["billing"]["recall"] == 0.0
        assert result["billing"]["support"] == 0.0

    def test_multi_label_mixed(self) -> None:
        # Example 1: pred={billing}, gt={billing, cancel}
        # Example 2: pred={cancel}, gt={cancel}
        preds = [{"billing"}, {"cancel"}]
        gts = [{"billing", "cancel"}, {"cancel"}]
        result = per_label_metrics(preds, gts, ["billing", "cancel"])

        # billing: TP=1, FP=0, FN=0 → P=1, R=1, F1=1, support=1
        assert result["billing"]["precision"] == 1.0
        assert result["billing"]["recall"] == 1.0
        assert result["billing"]["support"] == 1.0

        # cancel: TP=1, FP=0, FN=1 → P=1, R=0.5, support=2
        assert result["cancel"]["precision"] == 1.0
        assert result["cancel"]["recall"] == 0.5
        assert result["cancel"]["support"] == 2.0

    def test_label_not_in_any_example(self) -> None:
        preds = [{"billing"}]
        gts = [{"billing"}]
        result = per_label_metrics(preds, gts, ["billing", "refund"])
        assert result["refund"]["precision"] == 0.0
        assert result["refund"]["recall"] == 0.0
        assert result["refund"]["support"] == 0.0


# ---------------------------------------------------------------------------
# Micro F1
# ---------------------------------------------------------------------------


class TestMicroF1:
    def test_perfect_micro_f1(self) -> None:
        preds = [{"billing"}, {"cancel"}]
        gts = [{"billing"}, {"cancel"}]
        assert micro_f1(preds, gts, ["billing", "cancel"]) == 1.0

    def test_zero_micro_f1(self) -> None:
        preds = [{"billing"}]
        gts = [{"cancel"}]
        assert micro_f1(preds, gts, ["billing", "cancel"]) == 0.0

    def test_partial_micro_f1(self) -> None:
        # 2 examples, 2 labels
        # Ex1: pred={billing}, gt={billing, cancel} → TP(billing)=1, FN(cancel)=1
        # Ex2: pred={cancel}, gt={cancel} → TP(cancel)=1
        preds = [{"billing"}, {"cancel"}]
        gts = [{"billing", "cancel"}, {"cancel"}]
        # Global: TP=2, FP=0, FN=1
        # micro_P = 2/2 = 1.0, micro_R = 2/3 ≈ 0.667
        # micro_F1 = 2*1.0*(2/3) / (1.0 + 2/3) = (4/3) / (5/3) = 4/5 = 0.8
        result = micro_f1(preds, gts, ["billing", "cancel"])
        assert abs(result - 0.8) < 1e-10

    def test_empty_predictions(self) -> None:
        preds = [set()]
        gts = [{"billing"}]
        assert micro_f1(preds, gts, ["billing"]) == 0.0

    def test_empty_list(self) -> None:
        assert micro_f1([], [], ["billing"]) == 0.0


# ---------------------------------------------------------------------------
# Macro F1
# ---------------------------------------------------------------------------


class TestMacroF1:
    def test_perfect_macro_f1(self) -> None:
        preds = [{"billing"}, {"cancel"}]
        gts = [{"billing"}, {"cancel"}]
        assert macro_f1(preds, gts, ["billing", "cancel"]) == 1.0

    def test_zero_macro_f1(self) -> None:
        preds = [{"billing"}]
        gts = [{"cancel"}]
        assert macro_f1(preds, gts, ["billing", "cancel"]) == 0.0

    def test_unbalanced_macro_f1(self) -> None:
        # billing: F1=1.0, cancel: F1=0.0 → macro = 0.5
        preds = [{"billing"}]
        gts = [{"billing"}]
        result = macro_f1(preds, gts, ["billing", "cancel"])
        assert abs(result - 0.5) < 1e-10

    def test_empty_label_names(self) -> None:
        assert macro_f1([{"billing"}], [{"billing"}], []) == 0.0

    def test_empty_list(self) -> None:
        assert macro_f1([], [], ["billing"]) == 0.0


# ---------------------------------------------------------------------------
# Reexport
# ---------------------------------------------------------------------------


class TestMetricsReexport:
    def test_importable_from_package(self) -> None:
        from talkex.classification_eval import (
            f1_score as f1,
        )
        from talkex.classification_eval import (
            macro_f1 as maf1,
        )
        from talkex.classification_eval import (
            micro_f1 as mif1,
        )
        from talkex.classification_eval import (
            precision as p,
        )
        from talkex.classification_eval import (
            recall as r,
        )

        assert f1 is f1_score
        assert p is precision
        assert r is recall
        assert mif1 is micro_f1
        assert maf1 is macro_f1
