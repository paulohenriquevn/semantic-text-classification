import { useMutation } from "@tanstack/react-query";
import {
  BookOpen,
  Code,
  Layers,
  Loader2,
  Plus,
  Search,
  Trash2,
} from "lucide-react";
import { useMemo, useState } from "react";
import { previewDSL } from "@/lib/api";
import { extractHighlightFragments } from "@/lib/evidence";
import type { PreviewDSLResponse, PreviewMatch } from "@/types/api";
import { HighlightedText } from "@/components/HighlightedText";
import { EvidenceBadge } from "@/components/EvidenceBadge";
import { ModeTab } from "@/components/ModeTab";
import { QueryEvaluationPanel } from "@/components/QueryEvaluationPanel";
import { SearchLoadingCompact } from "@/components/SearchLoading";
import { CONDITION_GROUPS, CONDITION_PLACEHOLDERS, CONDITION_DESCRIPTIONS, CONDITION_FAMILY, FAMILY_COLORS, QUICK_ADD_PRESETS, defaultsForType } from "@/lib/condition-config";
import { generateDSL } from "@/lib/dsl-generator";
import type { BuilderCondition, ConditionType } from "@/types/dsl";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SearchMode = "examples" | "builder" | "dsl";

// ---------------------------------------------------------------------------
// Example search templates
// ---------------------------------------------------------------------------

interface SearchExample {
  id: string;
  name: string;
  color: string;
  description: string;
  dsl: string;
  type: "semantic" | "lexical" | "combined";
}

const SEARCH_EXAMPLES: SearchExample[] = [
  {
    id: "cancelamento",
    name: "Cancelamento",
    color: "border-red-300 bg-red-50 hover:bg-red-100",
    description: "Detecta intenção de cancelamento via similaridade semântica",
    dsl: `RULE search_query\nWHEN\n    semantic.similarity("quero cancelar meu serviço") > 0.50\nTHEN\n    tag("search_result")`,
    type: "semantic",
  },
  {
    id: "reclamacao",
    name: "Reclamacao / Insatisfacao",
    color: "border-orange-300 bg-orange-50 hover:bg-orange-100",
    description: "Combina semantica + palavras-chave de reclamacao",
    dsl: `RULE search_query\nWHEN\n    semantic.similarity("reclamação sobre serviço de telefonia") > 0.45\n    AND lexical.contains_any(["internet", "telefone", "plano", "sinal"])\nTHEN\n    tag("search_result")`,
    type: "combined",
  },
  {
    id: "cobranca",
    name: "Cobranca Indevida",
    color: "border-yellow-300 bg-yellow-50 hover:bg-yellow-100",
    description: "Busca por questoes de cobranca e faturamento",
    dsl: `RULE search_query\nWHEN\n    semantic.similarity("recebi uma cobrança indevida na minha fatura") > 0.50\nTHEN\n    tag("search_result")`,
    type: "semantic",
  },
  {
    id: "suporte",
    name: "Suporte Tecnico",
    color: "border-blue-300 bg-blue-50 hover:bg-blue-100",
    description: "Problemas tecnicos como internet, sistema, erro",
    dsl: `RULE search_query\nWHEN\n    semantic.similarity("minha internet não está funcionando") > 0.50\nTHEN\n    tag("search_result")`,
    type: "semantic",
  },
  {
    id: "elogio",
    name: "Elogios",
    color: "border-green-300 bg-green-50 hover:bg-green-100",
    description: "Combina semantica de satisfacao + palavras de agradecimento",
    dsl: `RULE search_query\nWHEN\n    semantic.intent("elogio satisfação") > 0.45\n    AND lexical.contains_any(["obrigado", "obrigada", "parabéns", "excelente"])\nTHEN\n    tag("search_result")`,
    type: "combined",
  },
  {
    id: "entrega",
    name: "Prazo de Entrega",
    color: "border-indigo-300 bg-indigo-50 hover:bg-indigo-100",
    description: "Duvidas e reclamacoes sobre prazo de entrega",
    dsl: `RULE search_query\nWHEN\n    semantic.similarity("qual o prazo de entrega do meu pedido") > 0.50\nTHEN\n    tag("search_result")`,
    type: "semantic",
  },
  {
    id: "cancel_lexical",
    name: "Cancelamento (Lexical)",
    color: "border-gray-300 bg-gray-50 hover:bg-gray-100",
    description: "Busca por palavras-chave de cancelamento — baseline lexical",
    dsl: `RULE search_query\nWHEN\n    lexical.contains_any(["cancelar", "cancelamento", "desistir", "encerrar"])\nTHEN\n    tag("search_result")`,
    type: "lexical",
  },
  {
    id: "cancel_combo",
    name: "Cancelamento (Hibrido)",
    color: "border-purple-300 bg-purple-50 hover:bg-purple-100",
    description: "Semantica + lexical: alta precisao combinando ambos sinais",
    dsl: `RULE search_query\nWHEN\n    semantic.intent("cancelamento") > 0.50\n    AND lexical.contains_any(["cancelar", "desistir", "encerrar"])\nTHEN\n    tag("search_result")`,
    type: "combined",
  },
  {
    id: "stem_cancel",
    name: "Stem: cancel* (prefixo)",
    color: "border-teal-300 bg-teal-50 hover:bg-teal-100",
    description: "Word prefix: encontra cancelar, cancelamento, cancelada, cancelei...",
    dsl: `RULE search_query\nWHEN\n    lexical.stem("cancel")\nTHEN\n    tag("search_result")`,
    type: "lexical",
  },
  {
    id: "word_boundary",
    name: "Palavra exata: conta",
    color: "border-cyan-300 bg-cyan-50 hover:bg-cyan-100",
    description: "Word boundary: 'conta' match exato, nao 'contato' ou 'desconta'",
    dsl: `RULE search_query\nWHEN\n    lexical.word("conta")\nTHEN\n    tag("search_result")`,
    type: "lexical",
  },
  {
    id: "near_proximity",
    name: "Proximidade: cancelar + conta",
    color: "border-rose-300 bg-rose-50 hover:bg-rose-100",
    description: "Proximity: 'cancelar' e 'conta' dentro de 5 palavras de distancia",
    dsl: `RULE search_query\nWHEN\n    lexical.near("cancelar", "conta", 5)\nTHEN\n    tag("search_result")`,
    type: "lexical",
  },
  {
    id: "negation",
    name: "Negacao: reclamacao sem teste",
    color: "border-slate-300 bg-slate-50 hover:bg-slate-100",
    description: "Reclamacoes reais: contem palavras de insatisfacao MAS exclui termos de teste/debug",
    dsl: `RULE search_query\nWHEN\n    lexical.contains_any(["reclamação", "insatisfeito", "frustrado"])\n    AND lexical.excludes_any(["teste", "debug", "exemplo"])\nTHEN\n    tag("search_result")`,
    type: "lexical",
  },
];

function nextId(): string {
  return Math.random().toString(36).slice(2, 8);
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

interface SearchBuilderPanelProps {
  onViewConversation: (conversationId: string, fragments?: string[]) => void;
}

export function SearchBuilderPanel({ onViewConversation }: SearchBuilderPanelProps) {
  const [mode, setMode] = useState<SearchMode>("builder");
  const [dsl, setDsl] = useState("");
  const [searchResult, setSearchResult] = useState<PreviewDSLResponse | null>(null);

  // Builder state — start with a similarity condition
  const [conditions, setConditions] = useState<BuilderCondition[]>([
    { id: nextId(), type: "similarity", value: "", connector: "AND", threshold: "0.50", operator: ">" },
  ]);

  const generatedDsl = useMemo(
    () => generateDSL(conditions),
    [conditions],
  );

  const activeDsl = mode === "builder" ? generatedDsl : dsl;

  const searchMutation = useMutation({
    mutationFn: () => previewDSL(activeDsl),
    onSuccess: (res) => setSearchResult(res),
  });

  return (
    <div className="space-y-4">
      {/* Builder card */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {/* Mode tabs */}
        <div className="flex items-center border-b border-gray-100 px-4 py-2">
          <div className="flex gap-1">
            <ModeTab
              active={mode === "examples"}
              icon={<BookOpen className="h-3.5 w-3.5" />}
              label="Examples"
              onClick={() => setMode("examples")}
            />
            <ModeTab
              active={mode === "builder"}
              icon={<Layers className="h-3.5 w-3.5" />}
              label="Visual Builder"
              onClick={() => setMode("builder")}
            />
            <ModeTab
              active={mode === "dsl"}
              icon={<Code className="h-3.5 w-3.5" />}
              label="DSL Editor"
              onClick={() => setMode("dsl")}
            />
          </div>
        </div>

        <div className="p-4 space-y-4">
          {/* Example templates */}
          {mode === "examples" && (
            <ExamplesGrid
              onSelect={(example) => {
                setDsl(example.dsl);
                setMode("dsl");
                setSearchResult(null);
              }}
            />
          )}

          {/* Visual builder */}
          {mode === "builder" && (
            <ConditionBuilder
              conditions={conditions}
              onChange={setConditions}
            />
          )}

          {/* DSL editor */}
          {mode === "dsl" && (
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Search DSL
              </label>
              <textarea
                value={dsl}
                onChange={(e) => {
                  setDsl(e.target.value);
                  setSearchResult(null);
                }}
                placeholder={`RULE search_query\nWHEN\n    semantic.similarity("quero cancelar meu serviço") > 0.85\n    AND speaker == "customer"\nTHEN\n    tag("search_result")`}
                rows={5}
                className={`w-full px-3 py-2 rounded-lg border text-sm font-mono leading-relaxed
                           focus:outline-none focus:ring-2 focus:ring-blue-500
                           ${searchResult && !searchResult.valid ? "border-red-300 bg-red-50" : "border-gray-200 bg-gray-50"}`}
              />
            </div>
          )}

          {/* Generated DSL preview (builder mode) */}
          {mode === "builder" && generatedDsl && (
            <div>
              <label className="text-xs font-medium text-gray-500 mb-1 block">
                Generated DSL
              </label>
              <pre className="bg-gray-900 text-gray-100 rounded-lg px-4 py-3 text-xs font-mono leading-relaxed overflow-x-auto">
                {generatedDsl}
              </pre>
            </div>
          )}

          {/* Search button */}
          <div className="flex items-center justify-end pt-2 border-t border-gray-100">
            <button
              onClick={() => searchMutation.mutate()}
              disabled={!activeDsl.trim() || searchMutation.isPending}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 text-white
                         text-sm font-medium hover:bg-blue-700 disabled:opacity-40 transition-colors shadow-sm"
            >
              {searchMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              {searchMutation.isPending ? "Searching..." : "Search"}
            </button>
          </div>
        </div>
      </div>

      {/* Loading */}
      <SearchLoadingCompact isLoading={searchMutation.isPending} label="Executando busca DSL..." />

      {/* Error */}
      {searchMutation.isError && !searchMutation.isPending && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
          Busca falhou: {(searchMutation.error as Error).message}
        </div>
      )}

      {/* Results */}
      {searchResult && !searchMutation.isPending && (
        <SearchResults result={searchResult} onViewConversation={onViewConversation} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Search results panel
// ---------------------------------------------------------------------------

function SearchResults({
  result,
  onViewConversation,
}: {
  result: PreviewDSLResponse;
  onViewConversation: (conversationId: string, fragments?: string[]) => void;
}) {
  if (!result.valid) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3">
        <p className="text-sm font-medium text-red-700">Invalid DSL</p>
        <p className="text-xs text-red-600 mt-0.5">{result.error}</p>
      </div>
    );
  }

  const hasMatches = result.match_count > 0;

  return (
    <div className="space-y-3">
      {/* Summary banner */}
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>
          {hasMatches ? (
            <>
              <span className="font-semibold text-gray-900">{result.match_count}</span> window
              {result.match_count !== 1 ? "s" : ""} across{" "}
              <span className="font-semibold text-gray-900">{result.conversation_count}</span>{" "}
              conversation{result.conversation_count !== 1 ? "s" : ""}
            </>
          ) : (
            "No matches found. Try adjusting your conditions."
          )}
        </span>
        <span className="font-mono text-xs bg-gray-100 px-2 py-0.5 rounded">
          {result.latency_ms.toFixed(0)} ms
        </span>
      </div>

      {/* Query evaluation */}
      {result.evaluation && (
        <QueryEvaluationPanel evaluation={result.evaluation} />
      )}

      {/* Match cards */}
      {result.sample_matches.map((m, i) => (
        <SearchMatchCard
          key={`${m.window_id}-${i}`}
          match={m}
          rank={i + 1}
          onViewConversation={onViewConversation}
        />
      ))}

      {result.match_count > result.sample_matches.length && (
        <p className="text-xs text-gray-400 text-center">
          Showing {result.sample_matches.length} of {result.match_count} matches
        </p>
      )}

      {!hasMatches && (
        <div className="text-center py-10 text-gray-400">
          No results found. Try different conditions or lower thresholds.
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Search match card — shows window text with evidence
// ---------------------------------------------------------------------------

function SearchMatchCard({
  match,
  rank,
  onViewConversation,
}: {
  match: PreviewMatch;
  rank: number;
  onViewConversation: (conversationId: string, fragments?: string[]) => void;
}) {
  const fragments = extractHighlightFragments(match.evidence);

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md hover:border-blue-300 transition-all">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
            #{rank}
          </span>
          <button
            onClick={() => onViewConversation(match.conversation_id, fragments)}
            className="text-xs text-blue-600 hover:underline font-mono"
          >
            {match.conversation_id.length > 20
              ? `${match.conversation_id.slice(0, 20)}...`
              : match.conversation_id}
          </button>
        </div>
        <span className="text-xs font-mono bg-purple-50 text-purple-700 px-2 py-0.5 rounded">
          score {match.score.toFixed(3)}
        </span>
      </div>

      {/* Window text with highlights */}
      <div className="px-4 py-3">
        <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-line">
          <HighlightedText text={match.window_text} fragments={fragments} />
        </p>
      </div>

      {/* Evidence badges */}
      {match.evidence.length > 0 && (
        <div className="px-4 py-2 border-t border-gray-100 flex flex-wrap gap-1.5">
          {match.evidence.map((ev, i) => (
            <EvidenceBadge key={i} evidence={ev} />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Condition builder
// ---------------------------------------------------------------------------

function ConditionBuilder({
  conditions,
  onChange,
}: {
  conditions: BuilderCondition[];
  onChange: (c: BuilderCondition[]) => void;
}) {
  const addCondition = (preset?: Partial<BuilderCondition> & { type: ConditionType }) => {
    const base = preset || { type: "keywords_any" as ConditionType };
    const defaults = defaultsForType(base.type);
    onChange([
      ...conditions,
      { id: nextId(), connector: "AND", value: "", ...defaults, ...base },
    ]);
  };

  const removeCondition = (id: string) => {
    if (conditions.length <= 1) return;
    onChange(conditions.filter((c) => c.id !== id));
  };

  const updateCondition = (id: string, patch: Partial<BuilderCondition>) => {
    onChange(conditions.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  };

  const handleTypeChange = (id: string, newType: ConditionType) => {
    const defaults = defaultsForType(newType);
    updateCondition(id, { type: newType, ...defaults });
  };

  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-2">
        Condições de busca
      </label>
      <div className="space-y-2">
        {conditions.map((cond, idx) => {
          const family = CONDITION_FAMILY[cond.type];
          const familyColor = FAMILY_COLORS[family];
          return (
          <div key={cond.id}>
            {/* Connector between conditions */}
            {idx > 0 && (
              <div className="flex items-center gap-2 py-1 pl-2">
                <div className="h-px flex-1 bg-gray-200" />
                <select
                  value={cond.connector}
                  onChange={(e) =>
                    updateCondition(cond.id, {
                      connector: e.target.value as "AND" | "OR",
                    })
                  }
                  className="text-xs font-semibold text-blue-600 bg-blue-50 border border-blue-200
                             rounded px-2 py-0.5 focus:outline-none cursor-pointer"
                >
                  <option value="AND">E (AND)</option>
                  <option value="OR">OU (OR)</option>
                </select>
                <div className="h-px flex-1 bg-gray-200" />
              </div>
            )}

            {/* Condition row — color-coded by family */}
            <div className={`rounded-lg px-3 py-2 space-y-2 border ${familyColor.bg}`}>
              <div className="flex items-center gap-2">
                {/* Family color dot */}
                <span className={`w-2 h-2 rounded-full shrink-0 ${familyColor.dot}`} title={familyColor.label} />

                {/* Type selector with optgroups */}
                <select
                  value={cond.type}
                  onChange={(e) =>
                    handleTypeChange(cond.id, e.target.value as ConditionType)
                  }
                  className="text-xs font-medium text-gray-700 bg-white border border-gray-200
                             rounded-md px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500
                             cursor-pointer min-w-[180px]"
                >
                  {CONDITION_GROUPS.map((group) => (
                    <optgroup key={group.label} label={group.label}>
                      {group.options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>

                {/* Value input */}
                <ConditionValueInput
                  condition={cond}
                  onUpdate={(patch) => updateCondition(cond.id, patch)}
                />

                <button
                  onClick={() => removeCondition(cond.id)}
                  disabled={conditions.length <= 1}
                  className="p-1 rounded text-gray-400 hover:text-red-500 hover:bg-red-50
                             disabled:opacity-30 disabled:hover:text-gray-400 disabled:hover:bg-transparent
                             shrink-0"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>

              {/* Inline description */}
              <p className="text-[11px] text-gray-400 pl-4 leading-tight">
                {CONDITION_DESCRIPTIONS[cond.type]}
              </p>

              {/* Threshold for semantic conditions */}
              {(cond.type === "intent_score" || cond.type === "similarity") && (
                <div className="flex items-center gap-2 pl-4">
                  <span className="text-xs text-gray-500">Limiar:</span>
                  <select
                    value={cond.operator || ">"}
                    onChange={(e) =>
                      updateCondition(cond.id, { operator: e.target.value })
                    }
                    className="text-xs font-mono text-gray-600 bg-white border border-gray-200
                               rounded px-1.5 py-1 focus:outline-none w-14"
                  >
                    <option value=">">{">"}</option>
                    <option value=">=">{">="}</option>
                    <option value="<">{"<"}</option>
                    <option value="<=">{"<="}</option>
                  </select>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={cond.threshold || "0.80"}
                    onChange={(e) =>
                      updateCondition(cond.id, { threshold: e.target.value })
                    }
                    className="w-20 px-2 py-1 rounded border border-gray-200 text-xs font-mono bg-white
                               focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              )}

              {/* Near proximity extras */}
              {cond.type === "near" && (
                <div className="flex items-center gap-2 pl-4 flex-wrap">
                  <span className="text-xs text-gray-500">próxima de</span>
                  <input
                    type="text"
                    value={cond.nearWord || ""}
                    onChange={(e) =>
                      updateCondition(cond.id, { nearWord: e.target.value })
                    }
                    placeholder="segunda palavra, ex: conta"
                    className="flex-1 min-w-[120px] px-2 py-1 rounded border border-gray-200 text-xs bg-white
                               focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-xs text-gray-500">dentro de</span>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={cond.nearDistance || "3"}
                    onChange={(e) =>
                      updateCondition(cond.id, { nearDistance: e.target.value })
                    }
                    className="w-14 px-2 py-1 rounded border border-gray-200 text-xs font-mono bg-white
                               focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-xs text-gray-400">palavras</span>
                </div>
              )}

              {/* Window count extras */}
              {cond.type === "window_count" && (
                <div className="flex items-center gap-2 pl-4 flex-wrap">
                  <span className="text-xs text-gray-500">janela de</span>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={cond.windowSize || "5"}
                    onChange={(e) =>
                      updateCondition(cond.id, { windowSize: e.target.value })
                    }
                    className="w-14 px-2 py-1 rounded border border-gray-200 text-xs font-mono bg-white
                               focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-xs text-gray-500">turns</span>
                  <select
                    value={cond.operator || ">="}
                    onChange={(e) =>
                      updateCondition(cond.id, { operator: e.target.value })
                    }
                    className="text-xs font-mono text-gray-600 bg-white border border-gray-200
                               rounded px-1.5 py-1 focus:outline-none w-14"
                  >
                    <option value=">=">{">="}</option>
                    <option value=">">{">"}</option>
                    <option value="<=">{"<="}</option>
                    <option value="<">{"<"}</option>
                  </select>
                  <input
                    type="number"
                    min="1"
                    value={cond.countValue || "2"}
                    onChange={(e) =>
                      updateCondition(cond.id, { countValue: e.target.value })
                    }
                    className="w-14 px-2 py-1 rounded border border-gray-200 text-xs font-mono bg-white
                               focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-xs text-gray-400">ocorrências</span>
                </div>
              )}
            </div>
          </div>
          );
        })}
      </div>

      {/* Quick-add chips */}
      <div className="flex flex-wrap items-center gap-1.5 mt-3">
        <span className="text-[11px] text-gray-400 mr-1">Adicionar:</span>
        {QUICK_ADD_PRESETS.map((preset) => {
          const familyColor = FAMILY_COLORS[preset.family];
          return (
            <button
              key={preset.label}
              onClick={() => addCondition(preset.condition)}
              className={`flex items-center gap-1 px-2 py-1 rounded-full text-[11px] font-medium
                         border transition-all hover:shadow-sm ${familyColor.bg} hover:opacity-80`}
            >
              <span>{preset.icon}</span>
              <span>{preset.label}</span>
            </button>
          );
        })}
        <button
          onClick={() => addCondition()}
          className="flex items-center gap-1 px-2 py-1 rounded-full text-[11px] font-medium
                     text-gray-500 bg-gray-100 border border-gray-200 hover:bg-gray-200 transition-all"
        >
          <Plus className="h-3 w-3" /> Mais...
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Condition value input — polymorphic by type
// ---------------------------------------------------------------------------

function ConditionValueInput({
  condition,
  onUpdate,
}: {
  condition: BuilderCondition;
  onUpdate: (patch: Partial<BuilderCondition>) => void;
}) {
  const { type, value } = condition;

  if (type === "speaker") {
    return (
      <select
        value={value}
        onChange={(e) => onUpdate({ value: e.target.value })}
        className="flex-1 px-2 py-1.5 rounded-md border border-gray-200 text-sm bg-white
                   focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="customer">Cliente</option>
        <option value="agent">Atendente</option>
      </select>
    );
  }

  if (type === "channel") {
    return (
      <select
        value={value}
        onChange={(e) => onUpdate({ value: e.target.value })}
        className="flex-1 px-2 py-1.5 rounded-md border border-gray-200 text-sm bg-white
                   focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="voice">Voz</option>
        <option value="chat">Chat</option>
        <option value="email">Email</option>
      </select>
    );
  }

  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onUpdate({ value: e.target.value })}
      placeholder={CONDITION_PLACEHOLDERS[type]}
      className="flex-1 px-2 py-1.5 rounded-md border border-gray-200 text-sm bg-white
                 focus:outline-none focus:ring-2 focus:ring-blue-500"
    />
  );
}

// ---------------------------------------------------------------------------
// Examples grid
// ---------------------------------------------------------------------------

const TYPE_BADGE: Record<string, { label: string; color: string }> = {
  semantic: { label: "Semantic", color: "bg-purple-100 text-purple-700" },
  lexical: { label: "Lexical", color: "bg-blue-100 text-blue-700" },
  combined: { label: "Hybrid", color: "bg-indigo-100 text-indigo-700" },
};

function ExamplesGrid({ onSelect }: { onSelect: (example: SearchExample) => void }) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-2">
        Click an example to load it in the DSL Editor
      </label>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {SEARCH_EXAMPLES.map((ex) => {
          const badge = TYPE_BADGE[ex.type];
          return (
            <button
              key={ex.id}
              onClick={() => onSelect(ex)}
              className={`text-left rounded-lg border px-3 py-2.5 transition-all ${ex.color}`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-semibold text-gray-800">
                  {ex.name}
                </span>
                <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${badge.color}`}>
                  {badge.label}
                </span>
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">{ex.description}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
