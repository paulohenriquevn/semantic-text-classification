import {
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Loader2,
  Play,
  Zap,
} from "lucide-react";
import { useState } from "react";
import { previewDSL } from "@/lib/api";
import { EvidenceBadge } from "@/components/EvidenceBadge";
import { HighlightedText } from "@/components/HighlightedText";
import { extractHighlightFragments } from "@/lib/evidence";
import type { PreviewDSLResponse } from "@/types/api";

// ---------------------------------------------------------------------------
// Collapsible section
// ---------------------------------------------------------------------------

function Section({
  title,
  color,
  defaultOpen = false,
  children,
}: {
  title: string;
  color: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={`rounded-xl border ${color} overflow-hidden`}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left font-semibold text-sm hover:opacity-80 transition-opacity"
      >
        {open ? (
          <ChevronDown className="h-4 w-4 shrink-0" />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0" />
        )}
        <span>{title}</span>
      </button>
      {open && <div className="px-4 pb-4 space-y-3">{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Code block with copy
// ---------------------------------------------------------------------------

function CodeBlock({ code, label }: { code: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="relative group">
      {label && (
        <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">
          {label}
        </span>
      )}
      <pre className="bg-gray-900 text-gray-100 rounded-lg px-4 py-3 text-xs font-mono leading-relaxed overflow-x-auto">
        {code}
      </pre>
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 p-1.5 rounded-md bg-gray-700 text-gray-300 opacity-0 group-hover:opacity-100
                   hover:bg-gray-600 transition-all"
        title="Copiar DSL"
      >
        {copied ? (
          <Check className="h-3.5 w-3.5 text-green-400" />
        ) : (
          <Copy className="h-3.5 w-3.5" />
        )}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Predicate row
// ---------------------------------------------------------------------------

function Predicate({
  name,
  syntax,
  description,
  example,
}: {
  name: string;
  syntax: string;
  description: string;
  example?: string;
}) {
  return (
    <div className="border border-gray-100 rounded-lg px-3 py-2 space-y-1">
      <div className="flex items-baseline gap-2 flex-wrap">
        <code className="text-xs font-bold text-blue-700 bg-blue-50 px-1.5 py-0.5 rounded">
          {name}
        </code>
        <code className="text-[11px] text-gray-500 font-mono">{syntax}</code>
      </div>
      <p className="text-xs text-gray-600 leading-relaxed">{description}</p>
      {example && <CodeBlock code={example} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Runnable example — code block + Run button + inline results
// ---------------------------------------------------------------------------

function RunnableExample({
  code,
  title,
  titleColor,
}: {
  code: string;
  title: string;
  titleColor?: string;
}) {
  const [result, setResult] = useState<PreviewDSLResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await previewDSL(code);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro ao executar");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-50 border-b border-gray-200">
        <h4 className="text-xs font-semibold text-gray-700 flex items-center gap-1.5">
          <Zap className={`h-3 w-3 ${titleColor || "text-gray-400"}`} />
          {title}
        </h4>
        <div className="flex items-center gap-1">
          <button
            onClick={handleCopy}
            className="p-1 rounded text-gray-400 hover:text-gray-600 hover:bg-gray-200 transition-colors"
            title="Copiar DSL"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-green-500" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
          </button>
          <button
            onClick={handleRun}
            disabled={loading}
            className="flex items-center gap-1 px-2.5 py-1 rounded bg-blue-600 text-white text-[11px]
                       font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Play className="h-3 w-3" />
            )}
            {loading ? "Executando..." : "Run"}
          </button>
        </div>
      </div>

      {/* Code */}
      <pre className="bg-gray-900 text-gray-100 px-4 py-3 text-xs font-mono leading-relaxed overflow-x-auto">
        {code}
      </pre>

      {/* Error */}
      {error && (
        <div className="px-3 py-2 bg-red-50 border-t border-red-200 text-xs text-red-700">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="border-t border-gray-200">
          {/* Summary */}
          <div className="flex items-center justify-between px-3 py-2 bg-gray-50">
            <span className="text-xs text-gray-600">
              {result.valid ? (
                result.match_count > 0 ? (
                  <>
                    <span className="font-semibold text-gray-900">{result.match_count}</span>
                    {" "}match{result.match_count !== 1 ? "es" : ""} em{" "}
                    <span className="font-semibold text-gray-900">{result.conversation_count}</span>
                    {" "}conversa{result.conversation_count !== 1 ? "s" : ""}
                  </>
                ) : (
                  <span className="text-gray-400">Nenhum match encontrado</span>
                )
              ) : (
                <span className="text-red-600">DSL invalido: {result.error}</span>
              )}
            </span>
            {result.valid && (
              <span className="text-[10px] font-mono text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">
                {result.latency_ms.toFixed(0)} ms
              </span>
            )}
          </div>

          {/* Sample matches */}
          {result.valid && result.sample_matches.length > 0 && (
            <div className="divide-y divide-gray-100">
              {result.sample_matches.slice(0, 3).map((m, i) => {
                const fragments = extractHighlightFragments(m.evidence);

                return (
                  <div key={`${m.window_id}-${i}`} className="px-3 py-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] font-mono text-gray-400">
                        {m.conversation_id.length > 24
                          ? `${m.conversation_id.slice(0, 24)}...`
                          : m.conversation_id}
                      </span>
                      <span className="text-[10px] font-mono text-purple-600 bg-purple-50 px-1 rounded">
                        score {m.score.toFixed(3)}
                      </span>
                    </div>
                    <p className="text-xs text-gray-700 leading-relaxed line-clamp-4 whitespace-pre-line">
                      <HighlightedText text={m.window_text} fragments={fragments} />
                    </p>
                    {m.evidence.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {m.evidence.map((ev, j) => (
                          <EvidenceBadge key={j} evidence={ev} />
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
              {result.match_count > 3 && (
                <div className="px-3 py-1.5 text-center">
                  <span className="text-[10px] text-gray-400">
                    + {result.match_count - 3} matches adicionais
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main guide panel
// ---------------------------------------------------------------------------

export function DSLGuidePanel() {
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm px-6 py-5">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-lg bg-blue-50">
            <BookOpen className="h-5 w-5 text-blue-600" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-gray-900">
              Guia Completo -- TalkEx DSL
            </h2>
            <p className="text-sm text-gray-500">
              Referencia de todas as funcionalidades do sistema de regras
            </p>
          </div>
        </div>
        <p className="text-sm text-gray-600 leading-relaxed">
          O TalkEx usa uma <strong>DSL (Domain-Specific Language)</strong> para
          definir regras de classificacao de conversas. As regras combinam
          predicados de 4 familias -- <strong>lexical</strong>,{" "}
          <strong>semantica</strong>, <strong>estrutural</strong> e{" "}
          <strong>contextual</strong> -- com operadores logicos{" "}
          <code className="text-xs bg-gray-100 px-1 rounded">AND</code>,{" "}
          <code className="text-xs bg-gray-100 px-1 rounded">OR</code> e{" "}
          <code className="text-xs bg-gray-100 px-1 rounded">NOT</code>.
          Todos os predicados lexicais sao{" "}
          <strong>insensitivos a acentos</strong> -- &quot;nao&quot; encontra
          &quot;nao&quot;, &quot;cancelar&quot; encontra &quot;cancelar&quot;.
          Clique <strong>Run</strong> em qualquer exemplo para ver matches reais no dataset.
        </p>
      </div>

      {/* Structure */}
      <Section
        title="Estrutura de uma Regra"
        color="border-gray-200 bg-white"
        defaultOpen={true}
      >
        <p className="text-xs text-gray-600 leading-relaxed">
          Uma regra segue a estrutura{" "}
          <code className="bg-gray-100 px-1 rounded">
            RULE ... WHEN ... THEN
          </code>
          . O bloco <strong>WHEN</strong> define as condicoes (predicados
          combinados com AND/OR/NOT). O bloco <strong>THEN</strong> define as
          acoes a executar quando a regra e satisfeita.
        </p>
        <CodeBlock
          label="Estrutura basica"
          code={`RULE nome_da_regra
WHEN
    <predicado_1>
    AND <predicado_2>
    OR <predicado_3>
    AND NOT <predicado_4>
THEN
    tag("etiqueta") score(0.9) priority("high")`}
        />
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          <div className="border border-gray-100 rounded-lg px-3 py-2">
            <code className="text-xs font-bold text-emerald-700">tag(&quot;nome&quot;)</code>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Etiqueta aplicada a conversas que satisfazem a regra
            </p>
          </div>
          <div className="border border-gray-100 rounded-lg px-3 py-2">
            <code className="text-xs font-bold text-emerald-700">score(0.9)</code>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Score de confianca da regra (0.0 a 1.0)
            </p>
          </div>
          <div className="border border-gray-100 rounded-lg px-3 py-2">
            <code className="text-xs font-bold text-emerald-700">priority(&quot;high&quot;)</code>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Prioridade: &quot;low&quot;, &quot;medium&quot; ou &quot;high&quot;
            </p>
          </div>
        </div>

        <p className="text-xs text-gray-600 leading-relaxed">
          <strong>Operadores logicos:</strong>{" "}
          <code className="bg-gray-100 px-1 rounded">AND</code> (ambos
          devem ser verdadeiros),{" "}
          <code className="bg-gray-100 px-1 rounded">OR</code> (pelo menos
          um verdadeiro),{" "}
          <code className="bg-gray-100 px-1 rounded">NOT</code> (inverte o
          resultado). Use parenteses{" "}
          <code className="bg-gray-100 px-1 rounded">()</code> para
          agrupar.
        </p>
        <RunnableExample
          title="Exemplo com agrupamento"
          titleColor="text-gray-500"
          code={`RULE regra_complexa
WHEN
    speaker == "customer"
    AND (lexical.contains_any(["cancelar", "encerrar"])
         OR semantic.intent("cancelamento") > 0.6)
    AND NOT lexical.contains("teste")
THEN
    tag("risco_cancelamento") priority("high")`}
        />
      </Section>

      {/* ================================================================== */}
      {/* LEXICAL */}
      {/* ================================================================== */}
      <Section
        title="Predicados Lexicais (11 operadores)"
        color="border-blue-200 bg-blue-50/30"
        defaultOpen={true}
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          Predicados lexicais verificam a <strong>presenca de palavras e
          padroes no texto</strong>. Sao rapidos (custo 1) e ideais como
          primeiro filtro. Todos normalizam acentos automaticamente.
        </p>

        <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider mt-3">
          Busca basica
        </h4>
        <div className="space-y-2">
          <Predicate
            name="contains"
            syntax='lexical.contains("texto") / keyword("texto")'
            description="Busca por substring -- verifica se o texto contem a palavra em qualquer posicao. 'net' encontra 'internet', 'network', 'net'."
          />
          <RunnableExample
            title="contains: busca por substring"
            titleColor="text-blue-500"
            code={`RULE exemplo_contains
WHEN
    lexical.contains("fatura")
THEN
    tag("contem_fatura")`}
          />
          <Predicate
            name="word"
            syntax='lexical.word("texto") / word("texto")'
            description="Busca por palavra inteira (word boundary) -- 'net' NAO encontra 'internet' ou 'network', apenas 'net' isolado. Use quando precisar de match exato."
          />
          <RunnableExample
            title="word: match por palavra inteira"
            titleColor="text-blue-500"
            code={`RULE exemplo_word
WHEN
    lexical.word("conta")
THEN
    tag("palavra_conta")`}
          />
          <Predicate
            name="stem"
            syntax='lexical.stem("prefixo") / stem("prefixo")'
            description="Busca por prefixo de palavra -- encontra todas as variantes que comecam com o prefixo. Ideal para verbos em portugues que tem muitas conjugacoes."
          />
          <RunnableExample
            title="stem: prefixo cancel* (cancelar, cancelamento, cancelei...)"
            titleColor="text-blue-500"
            code={`RULE exemplo_stem
WHEN
    lexical.stem("cancel")
THEN
    tag("prefixo_cancel")`}
          />
          <Predicate
            name="regex"
            syntax='lexical.regex("padrao") / regex("padrao")'
            description="Match por expressao regular (case-insensitive, acentos normalizados). Suporta toda a sintaxe regex do Python."
          />
          <RunnableExample
            title="regex: expressao regular"
            titleColor="text-blue-500"
            code={`RULE exemplo_regex
WHEN
    lexical.regex("cancel(ar|amento|ei)")
THEN
    tag("regex_cancel")`}
          />
        </div>

        <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider mt-4">
          Busca em listas
        </h4>
        <div className="space-y-2">
          <Predicate
            name="contains_any"
            syntax='lexical.contains_any(["a", "b", "c"])'
            description="Pelo menos UMA das palavras deve estar presente. O score e proporcional: se 2 de 4 palavras sao encontradas, score = 0.5. Evidencia mostra quais palavras foram encontradas."
          />
          <RunnableExample
            title="contains_any: pelo menos uma da lista"
            titleColor="text-blue-500"
            code={`RULE exemplo_contains_any
WHEN
    lexical.contains_any(["cancelar", "encerrar", "desistir"])
THEN
    tag("qualquer_cancel")`}
          />
          <Predicate
            name="contains_all"
            syntax='lexical.contains_all(["a", "b", "c"])'
            description="TODAS as palavras devem estar presentes. Mais restritivo que contains_any -- ideal para combinar termos que juntos indicam uma intencao especifica."
          />
          <RunnableExample
            title="contains_all: todas devem estar presentes"
            titleColor="text-blue-500"
            code={`RULE exemplo_contains_all
WHEN
    lexical.contains_all(["cancelar", "conta"])
THEN
    tag("cancelar_e_conta")`}
          />
          <Predicate
            name="excludes_any"
            syntax='lexical.excludes_any(["a", "b", "c"])'
            description="NENHUMA das palavras pode estar presente. Use para excluir falsos positivos -- conversas de teste, debug, ou termos que indicam contexto irrelevante."
          />
        </div>

        <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider mt-4">
          Negacao
        </h4>
        <div className="space-y-2">
          <Predicate
            name="not_contains"
            syntax='lexical.not_contains("texto") / not_contains("texto")'
            description='Verifica que o texto NAO contem a palavra. Equivalente a NOT keyword("texto") mas mais legivel.'
          />
        </div>

        <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider mt-4">
          Busca avancada
        </h4>
        <div className="space-y-2">
          <Predicate
            name="near"
            syntax='lexical.near("palavra1", "palavra2", distancia)'
            description="Verifica se duas palavras aparecem proximas no texto (dentro de N palavras de distancia). Usa word boundary para cada palavra. Ideal para detectar co-ocorrencia significativa."
          />
          <RunnableExample
            title="near: proximidade entre palavras"
            titleColor="text-blue-500"
            code={`RULE exemplo_near
WHEN
    lexical.near("cancelar", "conta", 5)
THEN
    tag("cancelar_perto_conta")`}
          />
          <Predicate
            name="starts_with"
            syntax='lexical.starts_with("prefixo") / starts_with("prefixo")'
            description="Verifica se o texto comeca com o prefixo especificado. Util para codigos, protocolos ou prefixos padronizados."
          />
          <Predicate
            name="ends_with"
            syntax='lexical.ends_with("sufixo") / ends_with("sufixo")'
            description="Verifica se o texto termina com o sufixo especificado. Util para extensoes de arquivo ou codigos com sufixo padrao."
          />
        </div>
      </Section>

      {/* ================================================================== */}
      {/* SEMANTIC */}
      {/* ================================================================== */}
      <Section
        title="Predicados Semanticos (2 operadores)"
        color="border-purple-200 bg-purple-50/30"
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          Predicados semanticos usam <strong>embeddings (vetores de
          significado)</strong> para entender a intencao por tras do texto.
          Encontram parafases e variacoes linguisticas que predicados
          lexicais nao conseguem. Sao mais caros (custo 4) -- combine com
          filtros lexicais para melhor performance.
        </p>

        <div className="space-y-2">
          <Predicate
            name="intent"
            syntax='semantic.intent("label") > threshold'
            description="Compara o embedding do texto com o embedding do label de intencao. Retorna um score de similaridade (0.0 a 1.0). Use o operador de comparacao para definir o threshold minimo."
          />
          <RunnableExample
            title="intent: intencao de cancelamento"
            titleColor="text-purple-500"
            code={`RULE exemplo_intent
WHEN
    semantic.intent("cancelamento") > 0.50
THEN
    tag("intencao_cancelamento")`}
          />
          <Predicate
            name="similarity"
            syntax='semantic.similarity("frase de referencia") > threshold'
            description="Compara o embedding do texto com o embedding de uma frase de referencia. Similar ao intent, mas aceita frases completas como referencia em vez de labels curtos."
          />
          <RunnableExample
            title="similarity: similaridade com frase"
            titleColor="text-purple-500"
            code={`RULE exemplo_similarity
WHEN
    semantic.similarity("quero cancelar meu servico") > 0.50
THEN
    tag("similar_cancelamento")`}
          />
        </div>

        <div className="bg-purple-50 border border-purple-200 rounded-lg px-3 py-2 mt-2">
          <p className="text-xs text-purple-800 leading-relaxed">
            <strong>Dica:</strong> Thresholds recomendados: <code className="bg-purple-100 px-1 rounded">&gt; 0.50</code> para buscas
            amplas, <code className="bg-purple-100 px-1 rounded">&gt; 0.75</code> para match moderado,{" "}
            <code className="bg-purple-100 px-1 rounded">&gt; 0.85</code> para alta precisao.
            Operadores disponiveis: <code className="bg-purple-100 px-1 rounded">&gt;</code>{" "}
            <code className="bg-purple-100 px-1 rounded">&gt;=</code>{" "}
            <code className="bg-purple-100 px-1 rounded">&lt;</code>{" "}
            <code className="bg-purple-100 px-1 rounded">&lt;=</code>
          </p>
        </div>
      </Section>

      {/* ================================================================== */}
      {/* STRUCTURAL */}
      {/* ================================================================== */}
      <Section
        title="Predicados Estruturais (5 operadores)"
        color="border-teal-200 bg-teal-50/30"
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          Predicados estruturais verificam <strong>metadados da
          conversa</strong> -- quem falou, qual canal, campos numericos.
          Custo moderado (2). Ideais para segmentar regras por contexto.
        </p>

        <div className="space-y-2">
          <Predicate
            name="speaker"
            syntax='speaker == "role"'
            description='Filtra por quem falou: "customer" (cliente) ou "agent" (atendente). Fundamental para distinguir quem expressou a intencao.'
          />
          <Predicate
            name="channel"
            syntax='channel == "valor"'
            description='Filtra por canal de atendimento: "voice" (telefone), "chat", "email".'
          />
          <RunnableExample
            title="speaker + channel: filtrando por contexto"
            titleColor="text-teal-500"
            code={`RULE exemplo_structural
WHEN
    speaker == "customer"
    AND lexical.contains_any(["problema", "erro", "nao funciona"])
THEN
    tag("cliente_com_problema")`}
          />
          <Predicate
            name="field_eq"
            syntax='field_eq("campo", "valor")'
            description="Compara qualquer campo de metadados por igualdade. Para campos customizados."
            example='field_eq("region", "sudeste")'
          />
          <Predicate
            name="field_gte / field_lte"
            syntax='field_gte("campo", valor) / field_lte("campo", valor)'
            description="Comparacao numerica -- maior/igual ou menor/igual. Util para scores, duracoes, contadores."
            example='field_gte("duration_seconds", 300)'
          />
        </div>

        <div className="bg-teal-50 border border-teal-200 rounded-lg px-3 py-2 mt-2">
          <p className="text-xs text-teal-800 leading-relaxed">
            <strong>Operadores de comparacao:</strong>{" "}
            <code className="bg-teal-100 px-1 rounded">==</code>{" "}
            <code className="bg-teal-100 px-1 rounded">!=</code>{" "}
            <code className="bg-teal-100 px-1 rounded">&gt;</code>{" "}
            <code className="bg-teal-100 px-1 rounded">&gt;=</code>{" "}
            <code className="bg-teal-100 px-1 rounded">&lt;</code>{" "}
            <code className="bg-teal-100 px-1 rounded">&lt;=</code>
          </p>
        </div>
      </Section>

      {/* ================================================================== */}
      {/* CONTEXTUAL */}
      {/* ================================================================== */}
      <Section
        title="Predicados Contextuais (2 operadores)"
        color="border-amber-200 bg-amber-50/30"
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          Predicados contextuais analisam <strong>padroes ao longo de
          multiplos turnos</strong> da conversa -- repeticoes, sequencias.
          Custo moderado (3). Capturam dinamicas conversacionais que
          predicados isolados nao detectam.
        </p>

        <div className="space-y-2">
          <Predicate
            name="turn_window.count"
            syntax='context.turn_window(N).count(intent="label") >= K'
            description="Conta quantas vezes uma intencao/termo aparece dentro de uma janela de N turnos. Detecta insistencia, frustracao repetida, ou escalacao."
          />
          <Predicate
            name="repeated"
            syntax='repeated("campo", "valor", contagem)'
            description="Verifica se um valor aparece repetidamente dentro da janela de contexto. Similar ao turn_window.count mas com sintaxe inline."
          />
          <Predicate
            name="occurs_after"
            syntax='occurs_after("campo", "primeiro", "segundo")'
            description="Verifica se um termo aparece DEPOIS de outro na sequencia da conversa. Detecta padroes causa-efeito."
          />
          <RunnableExample
            title="occurs_after: problema seguido de cancelar"
            titleColor="text-amber-500"
            code={`RULE exemplo_contextual
WHEN
    speaker == "customer"
    AND occurs_after("text", "problema", "cancelar")
THEN
    tag("escalacao_problema")`}
          />
        </div>
      </Section>

      {/* ================================================================== */}
      {/* FEATURES — Search modes */}
      {/* ================================================================== */}
      <Section
        title="Modos de Busca"
        color="border-indigo-200 bg-indigo-50/30"
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          O TalkEx oferece 2 modos de busca na aba Search:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="border border-indigo-200 rounded-lg px-4 py-3 bg-white">
            <h4 className="text-sm font-semibold text-gray-800 mb-1">
              Natural Language
            </h4>
            <p className="text-xs text-gray-600 leading-relaxed">
              Digite uma frase em linguagem natural e o sistema usa busca
              hibrida (BM25 lexical + similaridade semantica com embeddings)
              para encontrar conversas relevantes. Ideal para exploracoes
              rapidas.
            </p>
            <CodeBlock
              label="Exemplo"
              code="cliente quer cancelar o plano ou servico"
            />
          </div>
          <div className="border border-indigo-200 rounded-lg px-4 py-3 bg-white">
            <h4 className="text-sm font-semibold text-gray-800 mb-1">
              DSL Builder
            </h4>
            <p className="text-xs text-gray-600 leading-relaxed">
              Construa regras precisas usando o Visual Builder (interface
              grafica), Examples (templates prontos), ou DSL Editor (codigo
              direto). Controle total sobre cada predicado, threshold e
              combinacao logica.
            </p>
            <CodeBlock
              label="Exemplo"
              code={`RULE search_query
WHEN
    semantic.intent("cancelamento") > 0.5
    AND lexical.stem("cancel")
THEN
    tag("search_result")`}
            />
          </div>
        </div>
      </Section>

      {/* ================================================================== */}
      {/* FEATURES — Categories */}
      {/* ================================================================== */}
      <Section
        title="Sistema de Categorias"
        color="border-rose-200 bg-rose-50/30"
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          Na aba <strong>Categories</strong>, voce pode criar regras
          permanentes que classificam TODAS as conversas do corpus
          automaticamente. Fluxo:
        </p>
        <ol className="text-xs text-gray-600 leading-relaxed space-y-1 list-decimal list-inside">
          <li>
            <strong>Crie a categoria</strong> -- use um template, o visual
            builder, ou escreva DSL direto
          </li>
          <li>
            <strong>Preveja com &quot;Run&quot;</strong> -- teste a regra sem
            salvar, veja quantas conversas casam e a qualidade do match
          </li>
          <li>
            <strong>Salve com &quot;Create&quot;</strong> -- persiste a
            categoria com nome e descricao
          </li>
          <li>
            <strong>Aplique com &quot;Apply&quot;</strong> -- executa a regra
            contra o corpus inteiro e conta matches
          </li>
        </ol>
        <RunnableExample
          title="Exemplo de categoria completa"
          titleColor="text-rose-500"
          code={`RULE risco_cancelamento
WHEN
    speaker == "customer"
    AND lexical.stem("cancel")
    AND lexical.excludes_any(["teste", "debug"])
THEN
    tag("risco_cancelamento") priority("high")`}
        />
      </Section>

      {/* ================================================================== */}
      {/* FEATURES — Normalization */}
      {/* ================================================================== */}
      <Section
        title="Normalizacao e Acentos"
        color="border-gray-200 bg-white"
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          Todos os predicados lexicais passam por normalizacao automatica de
          acentos e diacriticos (NFD decomposition). Isso e essencial para
          PT-BR:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <div className="border border-gray-100 rounded-lg px-3 py-2">
            <code className="text-xs font-mono text-green-700">
              contains(&quot;nao&quot;)
            </code>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Encontra &quot;nao&quot; e &quot;nao&quot; no texto
            </p>
          </div>
          <div className="border border-gray-100 rounded-lg px-3 py-2">
            <code className="text-xs font-mono text-green-700">
              word(&quot;acao&quot;)
            </code>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Match exato com ou sem acento
            </p>
          </div>
          <div className="border border-gray-100 rounded-lg px-3 py-2">
            <code className="text-xs font-mono text-green-700">
              regex(&quot;cancela(r|mento)&quot;)
            </code>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Regex aplicado sobre texto normalizado
            </p>
          </div>
          <div className="border border-gray-100 rounded-lg px-3 py-2">
            <code className="text-xs font-mono text-green-700">
              near(&quot;nao&quot;, &quot;funciona&quot;, 3)
            </code>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Proximidade com acentos normalizados
            </p>
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          A busca BM25 (Natural Language) tambem usa normalizacao -- pontuacao
          e acentos sao removidos durante a tokenizacao.
        </p>
      </Section>

      {/* ================================================================== */}
      {/* RECIPES */}
      {/* ================================================================== */}
      <Section
        title="Receitas Prontas -- Exemplos Praticos"
        color="border-emerald-200 bg-emerald-50/30"
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          Clique <strong>Run</strong> para executar cada receita contra o dataset real:
        </p>

        <div className="space-y-3">
          <RunnableExample
            title="Risco de Cancelamento (alta precisao)"
            titleColor="text-red-500"
            code={`RULE risco_cancelamento
WHEN
    speaker == "customer"
    AND semantic.intent("cancelamento") > 0.60
    AND lexical.contains_any(["cancelar", "encerrar", "desistir"])
    AND lexical.excludes_any(["teste", "debug"])
THEN
    tag("risco_cancelamento") score(0.95) priority("high")`}
          />

          <RunnableExample
            title="Frustacao do Cliente (lexical + estrutural)"
            titleColor="text-orange-500"
            code={`RULE frustacao_cliente
WHEN
    speaker == "customer"
    AND lexical.contains_any(["absurdo", "ridiculo", "pessimo", "horrivel", "terrivel"])
    AND lexical.contains_any(["problema", "reclamacao", "erro"])
THEN
    tag("frustacao_cliente") priority("high")`}
          />

          <RunnableExample
            title="Problema Tecnico Especifico"
            titleColor="text-blue-500"
            code={`RULE problema_internet
WHEN
    lexical.word("internet")
    AND lexical.near("internet", "funciona", 5)
    AND channel == "voice"
THEN
    tag("problema_internet") priority("medium")`}
          />

          <RunnableExample
            title="Elogio do Cliente"
            titleColor="text-green-500"
            code={`RULE elogio_cliente
WHEN
    speaker == "customer"
    AND lexical.contains_any(["obrigado", "excelente", "otimo", "parabens"])
    AND lexical.not_contains("ironia")
THEN
    tag("elogio") priority("low")`}
          />

          <RunnableExample
            title="Busca Hibrida (semantica + lexical)"
            titleColor="text-purple-500"
            code={`RULE cobranca_indevida
WHEN
    semantic.similarity("recebi uma cobranca que nao reconheco") > 0.40
    AND lexical.stem("cobr")
    AND speaker == "customer"
THEN
    tag("cobranca_indevida") score(0.9) priority("high")`}
          />

          <RunnableExample
            title="Escalacao (problema seguido de cancelamento)"
            titleColor="text-amber-500"
            code={`RULE escalacao_cancelamento
WHEN
    speaker == "customer"
    AND occurs_after("text", "problema", "cancelar")
    AND lexical.stem("cancel")
THEN
    tag("escalacao") priority("high")`}
          />
        </div>
      </Section>

      {/* ================================================================== */}
      {/* VISUAL BUILDER */}
      {/* ================================================================== */}
      <Section
        title="Como Usar o Visual Builder"
        color="border-gray-200 bg-white"
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          O Visual Builder permite construir regras sem escrever DSL
          manualmente:
        </p>
        <ol className="text-xs text-gray-600 leading-relaxed space-y-2 list-decimal list-inside">
          <li>
            <strong>Selecione o tipo de condicao</strong> no dropdown --
            organizado em grupos (Lexical Basic/Lists/Advanced, Semantic,
            Structural, Contextual)
          </li>
          <li>
            <strong>Preencha o valor</strong> -- cada tipo tem um placeholder
            explicativo com exemplo
          </li>
          <li>
            <strong>Para tipos com threshold</strong> (semantic) -- ajuste o
            operador (&gt;, &gt;=) e o valor minimo
          </li>
          <li>
            <strong>Para &quot;Words near each other&quot;</strong> -- preencha a
            primeira palavra no campo principal, a segunda palavra e a
            distancia maxima nos campos extras
          </li>
          <li>
            <strong>Combine condicoes</strong> com AND/OR clicando no
            conector entre elas
          </li>
          <li>
            <strong>Visualize o DSL gerado</strong> em tempo real abaixo
            das condicoes
          </li>
        </ol>

        <div className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 mt-2">
          <p className="text-xs text-blue-800 leading-relaxed">
            <strong>Dica:</strong> Comece com um template na aba
            &quot;Examples&quot; (Search) ou &quot;Templates&quot;
            (Categories), depois personalize no Visual Builder ou DSL Editor.
          </p>
        </div>
      </Section>

      {/* ================================================================== */}
      {/* ANALYTICS */}
      {/* ================================================================== */}
      <Section
        title="Painel de Analytics"
        color="border-gray-200 bg-white"
      >
        <p className="text-xs text-gray-600 leading-relaxed">
          A aba <strong>Analytics</strong> mostra estatisticas do corpus:
          total de conversas, janelas de contexto, embeddings gerados,
          confianca media do ASR (speech-to-text), media de turnos por
          conversa, e distribuicao por dominio. Use para entender o
          dataset antes de criar regras.
        </p>
      </Section>

      {/* ================================================================== */}
      {/* QUALITY EVALUATION */}
      {/* ================================================================== */}
      <Section
        title="Avaliacao de Qualidade das Queries"
        color="border-gray-200 bg-white"
      >
        <p className="text-xs text-gray-600 leading-relaxed mb-2">
          Apos executar uma busca ou preview de categoria, o sistema exibe
          um painel de avaliacao com:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <div className="border border-gray-100 rounded-lg px-3 py-2">
            <h4 className="text-xs font-semibold text-gray-700">
              Pre-execucao
            </h4>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Familias de predicados usados, complexidade, avisos de threshold,
              pitfalls potenciais
            </p>
          </div>
          <div className="border border-gray-100 rounded-lg px-3 py-2">
            <h4 className="text-xs font-semibold text-gray-700">
              Pos-execucao
            </h4>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Distribuicao de scores (min/max/media/p90), cobertura do corpus,
              concentracao, quality score (0-100)
            </p>
          </div>
        </div>
      </Section>
    </div>
  );
}
