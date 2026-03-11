/**
 * Shared evidence badge — displays predicate match info with family-coded colors.
 *
 * Used by SearchBuilderPanel, CategoriesPanel, and DSLGuidePanel.
 */

import type { PredicateEvidence } from "@/types/api";

export const PREDICATE_BADGE: Record<string, { label: string; color: string }> = {
  lexical: { label: "Lexical", color: "bg-blue-100 text-blue-700" },
  semantic: { label: "Semantic", color: "bg-purple-100 text-purple-700" },
  structural: { label: "Structural", color: "bg-teal-100 text-teal-700" },
  contextual: { label: "Contextual", color: "bg-amber-100 text-amber-700" },
};

export function EvidenceBadge({ evidence: ev }: { evidence: PredicateEvidence }) {
  const badge = PREDICATE_BADGE[ev.predicate_type] ?? {
    label: ev.predicate_type,
    color: "bg-gray-100 text-gray-600",
  };

  return (
    <span
      className={`inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded ${badge.color}`}
      title={`${ev.field_name} ${ev.operator} — score: ${ev.score.toFixed(2)}, threshold: ${ev.threshold.toFixed(2)}`}
    >
      {badge.label}: {ev.field_name}
      {ev.matched_text && (
        <span className="font-normal opacity-75">
          &quot;{ev.matched_text}&quot;
        </span>
      )}
    </span>
  );
}
