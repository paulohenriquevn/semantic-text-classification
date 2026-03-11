import type { PredicateEvidence } from "@/types/api";

/**
 * Extract individual highlight fragments from evidence matched_text values.
 *
 * The rule evaluator returns composite matched_text for multi-term predicates:
 *   contains_all  → "cancelar, conta"
 *   contains_any  → "cancelar, encerrar"
 *   near          → "cancelar ~ conta (dist=3)"
 *   occurs_after  → "problema → cancelar"
 *   repeated      → "problema (x3)"
 *
 * We split these into individual words so HighlightedText can mark each one.
 */
export function extractHighlightFragments(
  evidence: PredicateEvidence[],
): string[] {
  const fragments: string[] = [];

  for (const ev of evidence) {
    const raw = ev.matched_text;
    if (!raw || raw.length === 0) continue;

    // near: "cancelar ~ conta (dist=3)" → ["cancelar", "conta"]
    if (raw.includes(" ~ ")) {
      const cleaned = raw.replace(/\s*\(dist=\d+\)/, "");
      for (const part of cleaned.split(" ~ ")) {
        const trimmed = part.trim();
        if (trimmed) fragments.push(trimmed);
      }
      continue;
    }

    // occurs_after: "problema → cancelar" → ["problema", "cancelar"]
    if (raw.includes(" → ") || raw.includes(" -> ")) {
      for (const part of raw.split(/\s*(?:→|->)\s*/)) {
        const trimmed = part.trim();
        if (trimmed) fragments.push(trimmed);
      }
      continue;
    }

    // repeated: "problema (x3)" → ["problema"]
    if (/\(x\d+\)/.test(raw)) {
      const cleaned = raw.replace(/\s*\(x\d+\)/, "").trim();
      if (cleaned) fragments.push(cleaned);
      continue;
    }

    // comma-separated lists: "cancelar, conta" → ["cancelar", "conta"]
    if (raw.includes(", ")) {
      for (const part of raw.split(", ")) {
        const trimmed = part.trim();
        if (trimmed) fragments.push(trimmed);
      }
      continue;
    }

    // Single term
    fragments.push(raw);
  }

  // Deduplicate preserving order
  return [...new Set(fragments)];
}
