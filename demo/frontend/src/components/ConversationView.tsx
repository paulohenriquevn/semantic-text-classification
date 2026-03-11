import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Clock, Mic, MessageSquare } from "lucide-react";
import { useEffect, useRef } from "react";
import { getConversation } from "@/lib/api";
import { HighlightedText } from "@/components/HighlightedText";

interface ConversationViewProps {
  conversationId: string;
  highlightFragments?: string[];
  onBack: () => void;
}

export function ConversationView({
  conversationId,
  highlightFragments = [],
  onBack,
}: ConversationViewProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => getConversation(conversationId),
  });

  // Strip speaker tags like [UNKNOWN], [customer], [agent] from fragments
  // so they match against turn raw_text which doesn't have these tags
  const cleanFragments = highlightFragments.map((f) =>
    f.replace(/\[[\w]+\]\s*/g, "").trim(),
  ).filter((f) => f.length > 3);

  // Scroll to first highlighted turn after render
  const firstHighlightRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (firstHighlightRef.current) {
      firstHighlightRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [data]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-10">
        <p className="text-red-600">Failed to load conversation.</p>
        <button onClick={onBack} className="mt-2 text-blue-600 hover:underline">
          Back to results
        </button>
      </div>
    );
  }

  // Check which turns contain any highlight fragment
  let firstHighlightFound = false;

  return (
    <div className="space-y-4">
      <button
        onClick={onBack}
        className="flex items-center gap-1 text-sm text-blue-600 hover:underline"
      >
        <ArrowLeft className="h-4 w-4" /> Voltar aos resultados
      </button>

      {/* Metadata header */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">
          Conversation {data.conversation_id}
        </h2>
        <div className="flex flex-wrap gap-4 text-sm text-gray-600">
          <span className="flex items-center gap-1">
            <MessageSquare className="h-4 w-4" /> {data.turn_count} turns
          </span>
          <span className="flex items-center gap-1">
            <Clock className="h-4 w-4" /> {data.audio_duration_seconds}s
          </span>
          <span className="flex items-center gap-1">
            <Mic className="h-4 w-4" /> ASR: {(data.asr_confidence * 100).toFixed(1)}%
          </span>
          <span className="bg-gray-100 px-2 py-0.5 rounded">{data.domain}</span>
          <span className="bg-gray-100 px-2 py-0.5 rounded">{data.topic}</span>
        </div>
      </div>

      {/* Highlight legend */}
      {cleanFragments.length > 0 && (
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className="bg-yellow-200 text-yellow-900 px-1.5 py-0.5 rounded text-[10px] font-medium">
            Trecho encontrado
          </span>
          <span>Trechos que correspondem à sua busca estão destacados abaixo</span>
        </div>
      )}

      {/* Turns */}
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm divide-y">
        {data.turns.map((turn) => {
          const hasHighlight =
            cleanFragments.length > 0 &&
            cleanFragments.some((f) =>
              turn.raw_text.toLowerCase().includes(f.toLowerCase()),
            );

          // Assign ref to first highlighted turn for auto-scroll
          let ref: React.Ref<HTMLDivElement> | undefined;
          if (hasHighlight && !firstHighlightFound) {
            ref = firstHighlightRef;
            firstHighlightFound = true;
          }

          return (
            <div
              key={turn.turn_id}
              ref={ref}
              className={`p-3 transition-colors ${
                hasHighlight
                  ? "bg-yellow-50 border-l-4 border-yellow-400"
                  : ""
              }`}
            >
              <span
                className={`inline-block text-xs font-medium px-2 py-0.5 rounded mb-1 ${
                  turn.speaker === "customer"
                    ? "bg-blue-100 text-blue-700"
                    : turn.speaker === "agent"
                      ? "bg-green-100 text-green-700"
                      : "bg-gray-100 text-gray-600"
                }`}
              >
                {turn.speaker}
              </span>
              <p className="text-sm text-gray-700 whitespace-pre-line select-text cursor-text">
                <HighlightedText text={turn.raw_text} fragments={cleanFragments} />
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

