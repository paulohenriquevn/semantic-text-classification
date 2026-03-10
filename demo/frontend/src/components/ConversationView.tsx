import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Clock, Mic, MessageSquare } from "lucide-react";
import { getConversation } from "@/lib/api";

interface ConversationViewProps {
  conversationId: string;
  onBack: () => void;
}

export function ConversationView({ conversationId, onBack }: ConversationViewProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => getConversation(conversationId),
  });

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

  return (
    <div className="space-y-4">
      <button
        onClick={onBack}
        className="flex items-center gap-1 text-sm text-blue-600 hover:underline"
      >
        <ArrowLeft className="h-4 w-4" /> Back to results
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

      {/* Turns */}
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm divide-y">
        {data.turns.map((turn) => (
          <div key={turn.turn_id} className="p-3">
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
            <p className="text-sm text-gray-700 whitespace-pre-line">{turn.raw_text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
