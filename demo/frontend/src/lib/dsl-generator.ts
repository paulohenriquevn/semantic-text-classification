/**
 * Shared DSL generation from visual builder conditions.
 *
 * Used by SearchBuilderPanel (search queries) and CategoriesPanel (category rules).
 */

import type { BuilderCondition } from "@/types/dsl";

/**
 * Convert a list of builder conditions into a TalkEx DSL rule string.
 *
 * @param conditions - Builder conditions from the visual builder.
 * @param options.ruleName - Rule name (default: "search_query").
 * @param options.priority - Priority level for THEN clause (omitted if "none" or empty).
 */
export function generateDSL(
  conditions: BuilderCondition[],
  options: { ruleName?: string; priority?: string } = {},
): string {
  if (conditions.length === 0) return "";

  const ruleName = options.ruleName || "search_query";
  const priority = options.priority;

  const predicates = conditions.map((c) => {
    const val = c.value.trim();
    const op = c.operator || ">";
    const th = c.threshold || "0.80";

    switch (c.type) {
      case "keyword":
        return val ? `keyword("${val}")` : "";
      case "word":
        return val ? `lexical.word("${val}")` : "";
      case "stem":
        return val ? `lexical.stem("${val}")` : "";
      case "keywords_any": {
        if (!val) return "";
        const words = val
          .split(",")
          .map((w) => `"${w.trim()}"`)
          .filter((w) => w !== '""')
          .join(", ");
        return `lexical.contains_any([${words}])`;
      }
      case "contains_all": {
        if (!val) return "";
        const words = val
          .split(",")
          .map((w) => `"${w.trim()}"`)
          .filter((w) => w !== '""')
          .join(", ");
        return `lexical.contains_all([${words}])`;
      }
      case "not_contains":
        return val ? `lexical.not_contains("${val}")` : "";
      case "excludes_any": {
        if (!val) return "";
        const words = val
          .split(",")
          .map((w) => `"${w.trim()}"`)
          .filter((w) => w !== '""')
          .join(", ");
        return `lexical.excludes_any([${words}])`;
      }
      case "near": {
        if (!val) return "";
        const w2 = c.nearWord?.trim() || "";
        const dist = c.nearDistance || "3";
        return w2 ? `lexical.near("${val}", "${w2}", ${dist})` : "";
      }
      case "starts_with":
        return val ? `lexical.starts_with("${val}")` : "";
      case "ends_with":
        return val ? `lexical.ends_with("${val}")` : "";
      case "regex":
        return val ? `regex("${val}")` : "";
      case "speaker":
        return val ? `speaker == "${val}"` : "";
      case "channel":
        return val ? `channel == "${val}"` : "";
      case "intent_score":
        return val ? `semantic.intent("${val}") ${op} ${th}` : "";
      case "similarity":
        return val ? `semantic.similarity("${val}") ${op} ${th}` : "";
      case "window_count": {
        if (!val) return "";
        const ws = c.windowSize || "5";
        const cv = c.countValue || "2";
        return `context.turn_window(${ws}).count(intent="${val}") ${op} ${cv}`;
      }
    }
  });

  const validPredicates = predicates.filter(Boolean);
  if (validPredicates.length === 0) return "";

  let whenClause = `    ${validPredicates[0]}`;
  for (let i = 1; i < validPredicates.length; i++) {
    whenClause += `\n    ${conditions[i].connector} ${validPredicates[i]}`;
  }

  const thenParts = [`tag("${ruleName}")`];
  if (priority && priority !== "none") thenParts.push(`priority("${priority}")`);

  return `RULE ${ruleName}\nWHEN\n${whenClause}\nTHEN\n    ${thenParts.join(" ")}`;
}

/**
 * Sanitize a category name into a valid DSL rule name.
 */
export function toRuleName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
}
