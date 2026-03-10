import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Code,
  Copy,
  Layers,
  Loader2,
  Play,
  Plus,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import {
  applyCategory,
  createCategory,
  deleteCategory,
  getCategory,
  listCategories,
  previewDSL,
} from "@/lib/api";
import type {
  CategoryDetailResponse,
  CategoryResponse,
  PreviewDSLResponse,
  PreviewMatch,
} from "@/types/api";
import { QueryEvaluationPanel } from "@/components/QueryEvaluationPanel";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CategoriesPanelProps {
  onViewConversation: (conversationId: string) => void;
}

type CreatorMode = "templates" | "builder" | "dsl";

type ConditionType =
  | "keyword"
  | "keywords_any"
  | "regex"
  | "speaker"
  | "channel"
  | "intent_score"
  | "similarity"
  | "window_count";

interface BuilderCondition {
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
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

interface RuleTemplate {
  id: string;
  name: string;
  icon: string;
  color: string;
  description: string;
  dsl: string;
}

const TEMPLATES: RuleTemplate[] = [
  {
    id: "cancelamento",
    name: "Risco de Cancelamento",
    icon: "🔴",
    color: "border-red-400 bg-red-50 hover:bg-red-100",
    description:
      "Detecta intenção de cancelamento por palavras-chave (cancelar, cancelamento, desistir, encerrar)",
    dsl: `RULE risco_cancelamento
WHEN
    contains_any("cancelar", "cancelamento", "desistir", "encerrar")
THEN
    tag("risco_cancelamento") priority("high")`,
  },
  {
    id: "reclamacao",
    name: "Reclamação / Insatisfação",
    icon: "😠",
    color: "border-orange-400 bg-orange-50 hover:bg-orange-100",
    description:
      "Reclamações, frustração e insatisfação do cliente",
    dsl: `RULE reclamacao
WHEN
    contains_any("reclamação", "insatisfeito", "frustrado", "absurdo", "ruim", "péssimo", "lento")
THEN
    tag("reclamacao") priority("high")`,
  },
  {
    id: "suporte_tecnico",
    name: "Suporte Técnico",
    icon: "🔧",
    color: "border-blue-400 bg-blue-50 hover:bg-blue-100",
    description:
      "Problemas técnicos: erros, bugs, defeitos e falhas",
    dsl: `RULE suporte_tecnico
WHEN
    contains_any("erro", "bug", "defeito", "não funciona", "problema", "suporte")
THEN
    tag("suporte_tecnico") priority("medium")`,
  },
  {
    id: "compra",
    name: "Intenção de Compra",
    icon: "🛒",
    color: "border-emerald-400 bg-emerald-50 hover:bg-emerald-100",
    description:
      "Interesse em compra, contratação ou assinatura de produto/serviço",
    dsl: `RULE intencao_compra
WHEN
    contains_any("comprar", "contratar", "assinar", "plano", "preço", "valor")
THEN
    tag("intencao_compra") priority("medium")`,
  },
  {
    id: "elogio",
    name: "Elogio / Feedback Positivo",
    icon: "⭐",
    color: "border-green-400 bg-green-50 hover:bg-green-100",
    description:
      "Experiências positivas: agradecimentos, elogios e satisfação",
    dsl: `RULE elogio
WHEN
    contains_any("obrigado", "excelente", "ótimo", "satisfeito", "parabéns", "adorei")
THEN
    tag("elogio") priority("low")`,
  },
  {
    id: "reembolso",
    name: "Reembolso / Cobrança",
    icon: "💰",
    color: "border-amber-400 bg-amber-50 hover:bg-amber-100",
    description:
      "Pedidos de reembolso, estorno, devolução e disputas de cobrança",
    dsl: `RULE reembolso_cobranca
WHEN
    contains_any("reembolso", "estorno", "devolução", "cobrança", "cobrado", "caro")
THEN
    tag("reembolso_cobranca") priority("high")`,
  },
];

// ---------------------------------------------------------------------------
// Condition type labels
// ---------------------------------------------------------------------------

/** Grouped condition options for <optgroup> rendering */
const CONDITION_GROUPS: { label: string; options: { value: ConditionType; label: string }[] }[] = [
  {
    label: "Lexical",
    options: [
      { value: "keyword", label: "Contains word" },
      { value: "keywords_any", label: "Contains any of" },
      { value: "regex", label: "Matches pattern" },
    ],
  },
  {
    label: "Semantic",
    options: [
      { value: "intent_score", label: "Intent score" },
      { value: "similarity", label: "Semantic similarity" },
    ],
  },
  {
    label: "Structural",
    options: [
      { value: "speaker", label: "Speaker is" },
      { value: "channel", label: "Channel is" },
    ],
  },
  {
    label: "Contextual",
    options: [
      { value: "window_count", label: "Turn window count" },
    ],
  },
];

const CONDITION_PLACEHOLDERS: Record<ConditionType, string> = {
  keyword: "e.g. billing",
  keywords_any: "e.g. cancel, terminate, close",
  regex: "e.g. cancel|terminate",
  speaker: "",
  channel: "",
  intent_score: "e.g. cancellation",
  similarity: "e.g. I want to cancel my service",
  window_count: "e.g. dissatisfaction",
};

/** Default values for new conditions by type */
function defaultsForType(type: ConditionType): Partial<BuilderCondition> {
  switch (type) {
    case "speaker":
      return { value: "customer" };
    case "channel":
      return { value: "voice" };
    case "intent_score":
      return { value: "", threshold: "0.80", operator: ">" };
    case "similarity":
      return { value: "", threshold: "0.85", operator: ">" };
    case "window_count":
      return { value: "", windowSize: "5", countValue: "2", operator: ">=" };
    default:
      return { value: "" };
  }
}

// ---------------------------------------------------------------------------
// DSL generator
// ---------------------------------------------------------------------------

function generateDSL(
  name: string,
  conditions: BuilderCondition[],
  priority: string,
): string {
  if (!name.trim() || conditions.length === 0) return "";

  const ruleName = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");

  const predicates = conditions.map((c) => {
    const val = c.value.trim();
    const op = c.operator || ">";
    const th = c.threshold || "0.80";

    switch (c.type) {
      case "keyword":
        return val ? `keyword("${val}")` : "";
      case "keywords_any": {
        if (!val) return "";
        const words = val
          .split(",")
          .map((w) => `"${w.trim()}"`)
          .filter((w) => w !== '""')
          .join(", ");
        return `lexical.contains_any([${words}])`;
      }
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

function nextId(): string {
  return Math.random().toString(36).slice(2, 8);
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function CategoriesPanel({ onViewConversation }: CategoriesPanelProps) {
  const queryClient = useQueryClient();
  const [showCreator, setShowCreator] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["categories"],
    queryFn: listCategories,
  });

  const handleCreated = useCallback(() => {
    setShowCreator(false);
    queryClient.invalidateQueries({ queryKey: ["categories"] });
  }, [queryClient]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Categories</h2>
          <p className="text-sm text-gray-500">
            Create rules to categorize conversations across the entire corpus
          </p>
        </div>
        {!showCreator && (
          <button
            onClick={() => setShowCreator(true)}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 text-white
                       text-sm font-medium hover:bg-blue-700 transition-colors shadow-sm"
          >
            <Plus className="h-4 w-4" /> New Category
          </button>
        )}
      </div>

      {/* Creator */}
      {showCreator && (
        <CreateCategoryPanel
          onCreated={handleCreated}
          onCancel={() => setShowCreator(false)}
        />
      )}

      {/* Categories list */}
      <div className="space-y-2">
        {data?.categories.map((cat) => (
          <CategoryCard
            key={cat.category_id}
            category={cat}
            expanded={expandedId === cat.category_id}
            onToggle={() =>
              setExpandedId(expandedId === cat.category_id ? null : cat.category_id)
            }
            onViewConversation={onViewConversation}
            onRefresh={() =>
              queryClient.invalidateQueries({ queryKey: ["categories"] })
            }
          />
        ))}
        {data?.categories.length === 0 && !showCreator && (
          <EmptyState onCreateClick={() => setShowCreator(true)} />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

function EmptyState({ onCreateClick }: { onCreateClick: () => void }) {
  return (
    <div className="text-center py-16 border-2 border-dashed border-gray-200 rounded-xl">
      <Layers className="h-10 w-10 text-gray-300 mx-auto mb-3" />
      <p className="text-gray-500 mb-1">No categories yet</p>
      <p className="text-sm text-gray-400 mb-4">
        Use templates to categorize conversations in seconds
      </p>
      <button
        onClick={onCreateClick}
        className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 text-white
                   text-sm font-medium hover:bg-blue-700 transition-colors"
      >
        <Sparkles className="h-4 w-4" /> Get Started
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Create panel — orchestrator
// ---------------------------------------------------------------------------

function CreateCategoryPanel({
  onCreated,
  onCancel,
}: {
  onCreated: () => void;
  onCancel: () => void;
}) {
  const [mode, setMode] = useState<CreatorMode>("templates");
  const [name, setName] = useState("");
  const [dsl, setDsl] = useState("");
  const [copied, setCopied] = useState(false);
  const [previewResult, setPreviewResult] = useState<PreviewDSLResponse | null>(null);

  // Builder state
  const [conditions, setConditions] = useState<BuilderCondition[]>([
    { id: nextId(), type: "keywords_any", value: "", connector: "AND" },
  ]);
  const [priority, setPriority] = useState("medium");

  // Auto-generate DSL from builder
  const generatedDsl = useMemo(
    () => generateDSL(name, conditions, priority),
    [name, conditions, priority],
  );

  // Use generated DSL when in builder mode
  const activeDsl = mode === "builder" ? generatedDsl : dsl;

  const previewMutation = useMutation({
    mutationFn: () => previewDSL(activeDsl),
    onSuccess: (res) => setPreviewResult(res),
  });

  const createMutation = useMutation({
    mutationFn: () => createCategory({ name, dsl_expression: activeDsl }),
    onSuccess: onCreated,
  });

  const handleTemplateSelect = (template: RuleTemplate) => {
    setName(template.name);
    setDsl(template.dsl);
    setPreviewResult(null);
    setMode("dsl");
  };

  const handleCopyDsl = () => {
    navigator.clipboard.writeText(activeDsl);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const canCreate = name.trim().length > 0 && activeDsl.trim().length > 0;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      {/* Tabs */}
      <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2">
        <div className="flex gap-1">
          <ModeTab
            active={mode === "templates"}
            icon={<Sparkles className="h-3.5 w-3.5" />}
            label="Templates"
            onClick={() => setMode("templates")}
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
        <button
          onClick={onCancel}
          className="p-1 rounded text-gray-400 hover:text-gray-600 hover:bg-gray-100"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="p-4 space-y-4">
        {/* Templates grid */}
        {mode === "templates" && (
          <TemplateGrid onSelect={handleTemplateSelect} />
        )}

        {/* Visual builder */}
        {mode === "builder" && (
          <>
            <NameInput value={name} onChange={setName} />
            <ConditionBuilder
              conditions={conditions}
              onChange={setConditions}
            />
            <PrioritySelector value={priority} onChange={setPriority} />
          </>
        )}

        {/* DSL editor */}
        {mode === "dsl" && (
          <>
            <NameInput value={name} onChange={setName} />
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Rule DSL
              </label>
              <textarea
                value={dsl}
                onChange={(e) => {
                  setDsl(e.target.value);
                  setPreviewResult(null);
                }}
                placeholder={`RULE minha_regra\nWHEN\n    speaker == "customer"\n    AND semantic.intent("cancelamento") > 0.82\n    AND lexical.contains_any(["cancelar", "encerrar"])\nTHEN\n    tag("minha_regra") score(0.95) priority("high")`}
                rows={6}
                className={`w-full px-3 py-2 rounded-lg border text-sm font-mono leading-relaxed
                           focus:outline-none focus:ring-2 focus:ring-blue-500
                           ${previewResult && !previewResult.valid ? "border-red-300 bg-red-50" : "border-gray-200 bg-gray-50"}`}
              />
            </div>
          </>
        )}

        {/* DSL preview (for builder mode) */}
        {mode === "builder" && generatedDsl && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs font-medium text-gray-500">
                Generated DSL
              </label>
              <button
                onClick={handleCopyDsl}
                className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600"
              >
                {copied ? (
                  <Check className="h-3 w-3 text-green-500" />
                ) : (
                  <Copy className="h-3 w-3" />
                )}
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <pre className="bg-gray-900 text-gray-100 rounded-lg px-4 py-3 text-xs font-mono leading-relaxed overflow-x-auto">
              {generatedDsl}
            </pre>
          </div>
        )}

        {/* Preview results */}
        {previewResult && (
          <PreviewResultsPanel result={previewResult} />
        )}

        {/* Error */}
        {createMutation.isError && (
          <p className="text-xs text-red-600 bg-red-50 px-3 py-2 rounded-lg">
            {(createMutation.error as Error).message}
          </p>
        )}

        {/* Actions */}
        {(mode !== "templates" || activeDsl) && (
          <div className="flex items-center gap-2 justify-end pt-2 border-t border-gray-100">
            <button
              onClick={() => previewMutation.mutate()}
              disabled={!activeDsl.trim() || previewMutation.isPending}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-amber-300
                         bg-amber-50 text-sm font-medium text-amber-700
                         hover:bg-amber-100 disabled:opacity-40 transition-colors"
            >
              {previewMutation.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Search className="h-3.5 w-3.5" />
              )}
              {previewMutation.isPending ? "Running..." : "Run"}
            </button>
            <button
              onClick={() => createMutation.mutate()}
              disabled={!canCreate || createMutation.isPending}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-medium
                         hover:bg-blue-700 disabled:opacity-40 transition-colors"
            >
              {createMutation.isPending ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> Creating...
                </>
              ) : (
                "Create Category"
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Preview results panel
// ---------------------------------------------------------------------------

function PreviewResultsPanel({ result }: { result: PreviewDSLResponse }) {
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
    <div
      className={`rounded-lg border px-4 py-3 ${
        hasMatches
          ? "border-green-200 bg-green-50"
          : "border-gray-200 bg-gray-50"
      }`}
    >
      {/* Summary header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {hasMatches ? (
            <div className="flex items-center gap-1.5 text-green-700">
              <Check className="h-4 w-4" />
              <span className="text-sm font-semibold">
                {result.match_count} window{result.match_count !== 1 ? "s" : ""}
              </span>
            </div>
          ) : (
            <span className="text-sm font-medium text-gray-500">
              No matches found
            </span>
          )}
          {hasMatches && (
            <span className="text-xs text-green-600">
              across {result.conversation_count} conversation
              {result.conversation_count !== 1 ? "s" : ""}
            </span>
          )}
        </div>
        <span className="text-xs text-gray-400 font-mono">
          {result.latency_ms.toFixed(0)} ms
        </span>
      </div>

      {/* Query evaluation */}
      {result.evaluation && (
        <div className="mt-3">
          <QueryEvaluationPanel evaluation={result.evaluation} />
        </div>
      )}

      {/* Sample matches with highlighted text and evidence */}
      {result.sample_matches.length > 0 && (
        <div className="mt-3 space-y-2 max-h-[400px] overflow-y-auto">
          {result.sample_matches.map((m, i) => (
            <PreviewMatchCard key={`${m.window_id}-${i}`} match={m} />
          ))}
          {result.match_count > result.sample_matches.length && (
            <p className="text-xs text-gray-400 text-center pt-1">
              showing {result.sample_matches.length} of {result.match_count}{" "}
              matches
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Preview match card — shows window text with highlighted matches + evidence
// ---------------------------------------------------------------------------

/** Predicate type label + color mapping */
const PREDICATE_BADGE: Record<string, { label: string; color: string }> = {
  lexical: { label: "Lexical", color: "bg-blue-100 text-blue-700" },
  semantic: { label: "Semantic", color: "bg-purple-100 text-purple-700" },
  structural: { label: "Structural", color: "bg-teal-100 text-teal-700" },
  contextual: { label: "Contextual", color: "bg-amber-100 text-amber-700" },
};

function PreviewMatchCard({ match }: { match: PreviewMatch }) {
  // Collect all matched text fragments for highlighting
  const fragments = match.evidence
    .map((ev) => ev.matched_text)
    .filter((t): t is string => t !== null && t.length > 0);

  return (
    <div className="bg-white rounded-lg border border-green-100 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-gray-50 border-b border-gray-100">
        <span className="text-xs text-gray-500 font-mono">
          {match.conversation_id.slice(0, 16)}...
        </span>
        <span className="text-xs text-gray-400 font-mono">
          score {match.score.toFixed(2)}
        </span>
      </div>

      {/* Window text with highlights */}
      <div className="px-3 py-2">
        <p className="text-sm text-gray-700 leading-relaxed">
          <HighlightedText text={match.window_text} fragments={fragments} />
        </p>
      </div>

      {/* Per-predicate evidence badges */}
      {match.evidence.length > 0 && (
        <div className="px-3 py-2 border-t border-gray-100 flex flex-wrap gap-1.5">
          {match.evidence.map((ev, i) => {
            const badge = PREDICATE_BADGE[ev.predicate_type] ?? {
              label: ev.predicate_type,
              color: "bg-gray-100 text-gray-600",
            };
            return (
              <span
                key={i}
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
          })}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Text highlighter — marks matched fragments within the full text
// ---------------------------------------------------------------------------

function HighlightedText({
  text,
  fragments,
}: {
  text: string;
  fragments: string[];
}) {
  if (fragments.length === 0) return <>{text}</>;

  // Build a regex matching any fragment (case-insensitive, longest first to
  // avoid partial overlap)
  const sorted = [...fragments].sort((a, b) => b.length - a.length);
  const escaped = sorted.map((f) => f.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");

  const parts = text.split(pattern);

  return (
    <>
      {parts.map((part, i) => {
        const isMatch = fragments.some(
          (f) => f.toLowerCase() === part.toLowerCase(),
        );
        return isMatch ? (
          <mark
            key={i}
            className="bg-yellow-200 text-yellow-900 rounded-sm px-0.5"
          >
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        );
      })}
    </>
  );
}

// ---------------------------------------------------------------------------
// Mode tab button
// ---------------------------------------------------------------------------

function ModeTab({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
        active
          ? "bg-blue-50 text-blue-700"
          : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Name input
// ---------------------------------------------------------------------------

function NameInput({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-1">
        Category Name
      </label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="e.g. Cancellation Risk"
        className="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm
                   focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Template grid
// ---------------------------------------------------------------------------

function TemplateGrid({
  onSelect,
}: {
  onSelect: (t: RuleTemplate) => void;
}) {
  return (
    <div>
      <p className="text-sm text-gray-500 mb-3">
        Pick a template to get started instantly
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {TEMPLATES.map((t) => (
          <button
            key={t.id}
            onClick={() => onSelect(t)}
            className={`text-left p-3 rounded-lg border-l-4 transition-all ${t.color}
                       cursor-pointer group`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-lg">{t.icon}</span>
              <span className="text-sm font-semibold text-gray-800 group-hover:text-gray-900">
                {t.name}
              </span>
            </div>
            <p className="text-xs text-gray-500 leading-relaxed">
              {t.description}
            </p>
          </button>
        ))}
      </div>
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
  const addCondition = () => {
    onChange([
      ...conditions,
      { id: nextId(), type: "keywords_any", value: "", connector: "AND" },
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
        Conditions
      </label>
      <div className="space-y-2">
        {conditions.map((cond, idx) => (
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
                  <option value="AND">AND</option>
                  <option value="OR">OR</option>
                </select>
                <div className="h-px flex-1 bg-gray-200" />
              </div>
            )}

            {/* Condition row */}
            <div className="bg-gray-50 rounded-lg px-3 py-2 space-y-2">
              <div className="flex items-center gap-2">
                {/* Type selector with optgroups */}
                <select
                  value={cond.type}
                  onChange={(e) =>
                    handleTypeChange(cond.id, e.target.value as ConditionType)
                  }
                  className="text-xs font-medium text-gray-700 bg-white border border-gray-200
                             rounded-md px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500
                             cursor-pointer min-w-[160px]"
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

                {/* Value input — varies by type */}
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

              {/* Extra fields for semantic/contextual types */}
              {(cond.type === "intent_score" || cond.type === "similarity") && (
                <div className="flex items-center gap-2 pl-[168px]">
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
                  <span className="text-xs text-gray-400">threshold</span>
                </div>
              )}

              {cond.type === "window_count" && (
                <div className="flex items-center gap-2 pl-[168px] flex-wrap">
                  <span className="text-xs text-gray-500">window</span>
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
                  <span className="text-xs text-gray-400">occurrences</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={addCondition}
        className="flex items-center gap-1 mt-2 text-xs font-medium text-blue-600
                   hover:text-blue-800 transition-colors"
      >
        <Plus className="h-3.5 w-3.5" /> Add condition
      </button>
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

  // Speaker/Channel: dropdown
  if (type === "speaker") {
    return (
      <select
        value={value}
        onChange={(e) => onUpdate({ value: e.target.value })}
        className="flex-1 px-2 py-1.5 rounded-md border border-gray-200 text-sm bg-white
                   focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="customer">Customer</option>
        <option value="agent">Agent</option>
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
        <option value="voice">Voice</option>
        <option value="chat">Chat</option>
        <option value="email">Email</option>
      </select>
    );
  }

  // Semantic / Contextual / Lexical: text input
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
// Priority selector
// ---------------------------------------------------------------------------

function PrioritySelector({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const options = [
    { value: "high", label: "High", color: "bg-red-100 text-red-700 border-red-200" },
    { value: "medium", label: "Medium", color: "bg-amber-100 text-amber-700 border-amber-200" },
    { value: "low", label: "Low", color: "bg-green-100 text-green-700 border-green-200" },
  ];

  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-2">
        Priority
      </label>
      <div className="flex gap-2">
        {options.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={`px-3 py-1 rounded-full text-xs font-medium border transition-all ${
              value === opt.value
                ? `${opt.color} ring-2 ring-offset-1 ring-blue-400`
                : "bg-gray-50 text-gray-400 border-gray-200 hover:bg-gray-100"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Category card
// ---------------------------------------------------------------------------

function CategoryCard({
  category,
  expanded,
  onToggle,
  onViewConversation,
  onRefresh,
}: {
  category: CategoryResponse;
  expanded: boolean;
  onToggle: () => void;
  onViewConversation: (id: string) => void;
  onRefresh: () => void;
}) {
  const applyMutation = useMutation({
    mutationFn: () => applyCategory(category.category_id),
    onSuccess: onRefresh,
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteCategory(category.category_id),
    onSuccess: onRefresh,
  });

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center gap-3 min-w-0">
          {expanded ? (
            <ChevronDown className="h-4 w-4 text-gray-400 shrink-0" />
          ) : (
            <ChevronRight className="h-4 w-4 text-gray-400 shrink-0" />
          )}
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-medium text-gray-900">{category.name}</span>
              {category.applied && (
                <span className="inline-flex items-center gap-0.5 text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                  <Check className="h-3 w-3" /> Applied
                </span>
              )}
            </div>
            <p className="text-xs text-gray-400 font-mono mt-0.5 truncate">
              {category.dsl_expression.split("\n")[0]}
            </p>
          </div>
        </div>

        <div
          className="flex items-center gap-3 shrink-0"
          onClick={(e) => e.stopPropagation()}
        >
          {category.applied && (
            <div className="text-right text-xs text-gray-500 hidden sm:block">
              <div>
                <span className="font-semibold text-gray-900">
                  {category.match_count}
                </span>{" "}
                windows
              </div>
              <div>
                <span className="font-semibold text-gray-900">
                  {category.conversation_count}
                </span>{" "}
                conversations
              </div>
            </div>
          )}
          <button
            onClick={() => applyMutation.mutate()}
            disabled={applyMutation.isPending}
            className="flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium
                       bg-green-50 text-green-700 hover:bg-green-100 disabled:opacity-50 transition-colors"
            title="Apply rule to all conversations"
          >
            {applyMutation.isPending ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Play className="h-3 w-3" />
            )}
            {category.applied ? "Re-apply" : "Apply"}
          </button>
          <button
            onClick={() => {
              if (confirm(`Delete category "${category.name}"?`)) {
                deleteMutation.mutate();
              }
            }}
            className="p-1 rounded text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors"
            title="Delete category"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {expanded && category.applied && (
        <CategoryMatchList
          categoryId={category.category_id}
          onViewConversation={onViewConversation}
        />
      )}

      {expanded && !category.applied && (
        <div className="px-4 pb-4 text-sm text-gray-400">
          Click &quot;Apply&quot; to evaluate this rule against all conversations.
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Match list (loaded on expand)
// ---------------------------------------------------------------------------

function CategoryMatchList({
  categoryId,
  onViewConversation,
}: {
  categoryId: string;
  onViewConversation: (id: string) => void;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ["category", categoryId],
    queryFn: () => getCategory(categoryId),
  });

  if (isLoading) {
    return (
      <div className="px-4 pb-4">
        <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
      </div>
    );
  }

  const detail = data as CategoryDetailResponse;
  if (!detail?.matches?.length) {
    return (
      <div className="px-4 pb-4 text-sm text-gray-400">No matches found.</div>
    );
  }

  // Group matches by conversation
  const byConv = new Map<string, typeof detail.matches>();
  for (const m of detail.matches) {
    const list = byConv.get(m.conversation_id) ?? [];
    list.push(m);
    byConv.set(m.conversation_id, list);
  }

  return (
    <div className="border-t border-gray-100">
      <div className="px-4 py-2 text-xs text-gray-500 bg-gray-50 flex justify-between">
        <span>
          {detail.match_count} window matches across {detail.conversation_count}{" "}
          conversations
        </span>
        <span className="font-mono">Applied in {detail.apply_time_ms} ms</span>
      </div>
      <div className="max-h-80 overflow-y-auto divide-y divide-gray-100">
        {[...byConv.entries()].slice(0, 50).map(([convId, matches]) => (
          <div key={convId} className="px-4 py-2 hover:bg-gray-50">
            <div className="flex items-center justify-between">
              <button
                onClick={() => onViewConversation(convId)}
                className="text-xs text-blue-600 hover:underline font-medium"
              >
                {convId}
              </button>
              <span className="text-xs text-gray-400">
                {matches.length} window{matches.length > 1 ? "s" : ""}
              </span>
            </div>
            {matches.slice(0, 3).map((m) => (
              <div key={m.window_id} className="mt-1 text-xs text-gray-500">
                {m.matched_text && (
                  <span className="bg-yellow-100 text-yellow-800 px-1 rounded">
                    {m.matched_text}
                  </span>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
