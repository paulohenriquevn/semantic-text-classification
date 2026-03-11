/**
 * Shared condition builder configuration — groups, placeholders, defaults,
 * descriptions, and family colors.
 *
 * Used by SearchBuilderPanel and CategoriesPanel.
 */

import type { BuilderCondition, ConditionType } from "@/types/dsl";

// ---------------------------------------------------------------------------
// Family color system — visual indicator per predicate family
// ---------------------------------------------------------------------------

export type ConditionFamily = "semantic" | "lexical" | "structural" | "contextual";

export const FAMILY_COLORS: Record<ConditionFamily, { dot: string; bg: string; label: string }> = {
  semantic: { dot: "bg-purple-400", bg: "bg-purple-50 border-purple-200", label: "Semântico" },
  lexical: { dot: "bg-blue-400", bg: "bg-blue-50 border-blue-200", label: "Lexical" },
  structural: { dot: "bg-teal-400", bg: "bg-teal-50 border-teal-200", label: "Estrutural" },
  contextual: { dot: "bg-amber-400", bg: "bg-amber-50 border-amber-200", label: "Contextual" },
};

export const CONDITION_FAMILY: Record<ConditionType, ConditionFamily> = {
  intent_score: "semantic",
  similarity: "semantic",
  keyword: "lexical",
  word: "lexical",
  stem: "lexical",
  regex: "lexical",
  keywords_any: "lexical",
  contains_all: "lexical",
  not_contains: "lexical",
  excludes_any: "lexical",
  near: "lexical",
  starts_with: "lexical",
  ends_with: "lexical",
  speaker: "structural",
  channel: "structural",
  window_count: "contextual",
};

// ---------------------------------------------------------------------------
// Condition groups — organized for the dropdown selector (PT-BR)
// ---------------------------------------------------------------------------

export const CONDITION_GROUPS: {
  label: string;
  options: { value: ConditionType; label: string }[];
}[] = [
  {
    label: "Semântico",
    options: [
      { value: "similarity", label: "Similaridade semântica" },
      { value: "intent_score", label: "Intenção (score)" },
    ],
  },
  {
    label: "Lexical — Básico",
    options: [
      { value: "keyword", label: "Contém texto" },
      { value: "word", label: "Palavra exata" },
      { value: "stem", label: "Prefixo de palavra" },
    ],
  },
  {
    label: "Lexical — Listas",
    options: [
      { value: "keywords_any", label: "Contém qualquer (lista)" },
      { value: "contains_all", label: "Contém todos (lista)" },
      { value: "excludes_any", label: "Exclui qualquer (lista)" },
    ],
  },
  {
    label: "Lexical — Avançado",
    options: [
      { value: "not_contains", label: "Não contém" },
      { value: "near", label: "Palavras próximas" },
      { value: "regex", label: "Expressão regular" },
      { value: "starts_with", label: "Começa com" },
      { value: "ends_with", label: "Termina com" },
    ],
  },
  {
    label: "Estrutural",
    options: [
      { value: "speaker", label: "Falante" },
      { value: "channel", label: "Canal" },
    ],
  },
  {
    label: "Contextual",
    options: [{ value: "window_count", label: "Contagem na janela" }],
  },
];

// ---------------------------------------------------------------------------
// Placeholders — PT-BR with realistic examples
// ---------------------------------------------------------------------------

export const CONDITION_PLACEHOLDERS: Record<ConditionType, string> = {
  keyword: "ex: cancelamento",
  word: "ex: conta (palavra exata, não 'contato')",
  stem: "ex: cancel (encontra cancelar, cancelamento...)",
  keywords_any: "ex: cancelar, encerrar, desistir",
  contains_all: "ex: cancelar, conta (todos devem aparecer)",
  not_contains: "ex: teste (não pode estar presente)",
  excludes_any: "ex: teste, debug (nenhum pode aparecer)",
  near: "Primeira palavra, ex: cancelar",
  starts_with: "ex: FAT- (texto começa com isso)",
  ends_with: "ex: .pdf (texto termina com isso)",
  regex: "ex: cancel(ar|amento)",
  speaker: "",
  channel: "",
  intent_score: "ex: cancelamento",
  similarity: "ex: quero cancelar meu serviço",
  window_count: "ex: cancelamento",
};

// ---------------------------------------------------------------------------
// Inline help descriptions — shown below each condition (PT-BR)
// ---------------------------------------------------------------------------

export const CONDITION_DESCRIPTIONS: Record<ConditionType, string> = {
  similarity: "Busca por significado semelhante, mesmo com palavras diferentes",
  intent_score: "Detecta a intenção do cliente por similaridade semântica",
  keyword: "Busca por substring no texto (inclui variações parciais)",
  word: "Busca por palavra inteira — 'conta' não encontra 'contato'",
  stem: "Busca por prefixo — 'cancel' encontra cancelar, cancelamento, cancelada",
  keywords_any: "Pelo menos UMA das palavras da lista deve aparecer",
  contains_all: "TODAS as palavras da lista devem aparecer no texto",
  not_contains: "O texto NÃO pode conter esta palavra",
  excludes_any: "NENHUMA das palavras da lista pode aparecer",
  near: "Duas palavras devem aparecer próximas uma da outra",
  starts_with: "O texto deve começar com este trecho",
  ends_with: "O texto deve terminar com este trecho",
  regex: "Expressão regular para padrões complexos",
  speaker: "Filtra por quem está falando na conversa",
  channel: "Filtra pelo canal de atendimento",
  window_count: "Conta ocorrências de uma intenção em uma janela de turns",
};

// ---------------------------------------------------------------------------
// Quick-add presets — one-click condition templates (PT-BR)
// ---------------------------------------------------------------------------

export interface QuickAddPreset {
  label: string;
  icon: string;
  family: ConditionFamily;
  condition: Partial<BuilderCondition> & { type: ConditionType };
}

export const QUICK_ADD_PRESETS: QuickAddPreset[] = [
  {
    label: "Similaridade",
    icon: "🔮",
    family: "semantic",
    condition: { type: "similarity", value: "", threshold: "0.50", operator: ">" },
  },
  {
    label: "Contém palavra",
    icon: "🔤",
    family: "lexical",
    condition: { type: "keyword", value: "" },
  },
  {
    label: "Lista de palavras",
    icon: "📋",
    family: "lexical",
    condition: { type: "keywords_any", value: "" },
  },
  {
    label: "Prefixo (stem)",
    icon: "🌱",
    family: "lexical",
    condition: { type: "stem", value: "" },
  },
  {
    label: "Não contém",
    icon: "🚫",
    family: "lexical",
    condition: { type: "not_contains", value: "" },
  },
  {
    label: "Falante",
    icon: "👤",
    family: "structural",
    condition: { type: "speaker", value: "customer" },
  },
];

// ---------------------------------------------------------------------------
// Default values per condition type
// ---------------------------------------------------------------------------

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
