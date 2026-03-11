/**
 * Shared condition builder configuration — groups, placeholders, and defaults.
 *
 * Used by SearchBuilderPanel and CategoriesPanel.
 */

import type { BuilderCondition, ConditionType } from "@/types/dsl";

export const CONDITION_GROUPS: {
  label: string;
  options: { value: ConditionType; label: string }[];
}[] = [
  {
    label: "Semantic",
    options: [
      { value: "intent_score", label: "Intent score" },
      { value: "similarity", label: "Semantic similarity" },
    ],
  },
  {
    label: "Lexical — Basic",
    options: [
      { value: "keyword", label: "Contains (substring)" },
      { value: "word", label: "Exact word (boundary)" },
      { value: "stem", label: "Word prefix (stem)" },
      { value: "regex", label: "Regex pattern" },
    ],
  },
  {
    label: "Lexical — Lists",
    options: [
      { value: "keywords_any", label: "Contains any (list)" },
      { value: "contains_all", label: "Contains all (list)" },
      { value: "excludes_any", label: "Excludes any (list)" },
    ],
  },
  {
    label: "Lexical — Advanced",
    options: [
      { value: "not_contains", label: "Not contains" },
      { value: "near", label: "Words near each other" },
      { value: "starts_with", label: "Starts with" },
      { value: "ends_with", label: "Ends with" },
    ],
  },
  {
    label: "Structural",
    options: [
      { value: "speaker", label: "Speaker role" },
      { value: "channel", label: "Channel" },
    ],
  },
  {
    label: "Contextual",
    options: [{ value: "window_count", label: "Turn window count" }],
  },
];

export const CONDITION_PLACEHOLDERS: Record<ConditionType, string> = {
  keyword: "e.g. billing",
  word: "e.g. cancelar (whole word only)",
  stem: "e.g. cancel (matches cancelar, cancelamento...)",
  keywords_any: "e.g. cancel, terminate, close",
  contains_all: "e.g. cancelar, conta (all must match)",
  not_contains: "e.g. teste (must NOT be present)",
  excludes_any: "e.g. teste, debug (none must match)",
  near: "First word, e.g. cancelar",
  starts_with: "e.g. FAT- (text starts with this)",
  ends_with: "e.g. .pdf (text ends with this)",
  regex: "e.g. cancel(ar|amento)",
  speaker: "",
  channel: "",
  intent_score: "e.g. cancellation",
  similarity: "e.g. I want to cancel my service",
  window_count: "e.g. dissatisfaction",
};

export function defaultsForType(type: ConditionType): Partial<BuilderCondition> {
  switch (type) {
    case "speaker":
      return { value: "customer" };
    case "channel":
      return { value: "voice" };
    case "intent_score":
      return { value: "", threshold: "0.50", operator: ">" };
    case "similarity":
      return { value: "", threshold: "0.50", operator: ">" };
    case "window_count":
      return { value: "", windowSize: "5", countValue: "2", operator: ">=" };
    case "near":
      return { value: "", nearWord: "", nearDistance: "3" };
    default:
      return { value: "" };
  }
}
