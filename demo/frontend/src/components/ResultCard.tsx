import { ChevronRight } from "lucide-react";
import { HighlightedText } from "@/components/HighlightedText";
import type { SearchHit } from "@/types/api";

interface ResultCardProps {
  hit: SearchHit;
  onClick: (conversationId: string) => void;
}

export function ResultCard({ hit, onClick }: ResultCardProps) {
  return (
    <div
      className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm
                 hover:shadow-md hover:border-blue-300 transition-all cursor-pointer"
      onClick={() => onClick(hit.conversation_id)}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              #{hit.rank}
            </span>
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
              {hit.domain}
            </span>
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
              {hit.topic}
            </span>
          </div>
          <p className="text-sm text-gray-700 line-clamp-3 whitespace-pre-line">
            <HighlightedText text={hit.text} fragments={hit.matched_text ? [hit.matched_text] : []} />
          </p>
        </div>

        <div className="flex flex-col items-end gap-1 shrink-0">
          <ScoreBadge label="Score" value={hit.score} color="blue" />
          {hit.lexical_score != null && (
            <ScoreBadge label="BM25" value={hit.lexical_score} color="green" />
          )}
          {hit.semantic_score != null && (
            <ScoreBadge label="Semantic" value={hit.semantic_score} color="purple" />
          )}
          <ChevronRight className="h-4 w-4 text-gray-400 mt-1" />
        </div>
      </div>
    </div>
  );
}

function ScoreBadge({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: "blue" | "green" | "purple";
}) {
  const colors = {
    blue: "bg-blue-50 text-blue-700",
    green: "bg-green-50 text-green-700",
    purple: "bg-purple-50 text-purple-700",
  };

  return (
    <span className={`text-xs px-2 py-0.5 rounded font-mono ${colors[color]}`}>
      {label}: {value.toFixed(3)}
    </span>
  );
}

