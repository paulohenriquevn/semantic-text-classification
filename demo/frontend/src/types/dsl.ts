/**
 * Shared DSL builder types — used by SearchBuilderPanel and CategoriesPanel.
 */

export type ConditionType =
  | "keyword"
  | "keywords_any"
  | "contains_all"
  | "word"
  | "stem"
  | "not_contains"
  | "excludes_any"
  | "near"
  | "starts_with"
  | "ends_with"
  | "regex"
  | "speaker"
  | "channel"
  | "intent_score"
  | "similarity"
  | "window_count";

export interface BuilderCondition {
  id: string;
  type: ConditionType;
  value: string;
  connector: "AND" | "OR";
  /** Threshold for semantic conditions (e.g. 0.82) */
  threshold?: string;
  /** Comparison operator for semantic/contextual (>, >=, <, <=) */
  operator?: string;
  /** Window size for context.turn_window(N) */
  windowSize?: string;
  /** Count threshold for window_count (e.g. >= 2) */
  countValue?: string;
  /** Second word for near() proximity search */
  nearWord?: string;
  /** Max distance between words for near() */
  nearDistance?: string;
}
