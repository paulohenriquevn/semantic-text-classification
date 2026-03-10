import { useQuery } from "@tanstack/react-query";
import { BarChart3, Database, Layers, Mic } from "lucide-react";
import { getAnalytics } from "@/lib/api";

export function AnalyticsPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ["analytics"],
    queryFn: getAnalytics,
    staleTime: Infinity,
  });

  if (isLoading || !data) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-gray-200 rounded w-1/3" />
          <div className="h-8 bg-gray-200 rounded w-1/2" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          icon={<Database className="h-5 w-5 text-blue-500" />}
          label="Conversations"
          value={data.total_conversations.toLocaleString()}
        />
        <StatCard
          icon={<Layers className="h-5 w-5 text-purple-500" />}
          label="Context Windows"
          value={data.total_windows.toLocaleString()}
        />
        <StatCard
          icon={<Mic className="h-5 w-5 text-green-500" />}
          label="ASR Confidence"
          value={`${(data.avg_asr_confidence * 100).toFixed(1)}%`}
        />
        <StatCard
          icon={<BarChart3 className="h-5 w-5 text-orange-500" />}
          label="Avg Turns"
          value={data.avg_turns_per_conversation.toFixed(1)}
        />
      </div>

      {/* Domain distribution */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Domain Distribution</h3>
        <div className="space-y-2">
          {data.domains
            .sort((a, b) => b.count - a.count)
            .map((d) => {
              const pct = (d.count / data.total_conversations) * 100;
              return (
                <div key={d.domain} className="flex items-center gap-3">
                  <span className="text-xs text-gray-600 w-28 truncate">{d.domain}</span>
                  <div className="flex-1 bg-gray-100 rounded-full h-2.5">
                    <div
                      className="bg-blue-500 h-2.5 rounded-full transition-all"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 w-16 text-right">
                    {d.count} ({pct.toFixed(0)}%)
                  </span>
                </div>
              );
            })}
        </div>
      </div>

      {/* Credibility panel */}
      <div className="bg-gray-900 text-white rounded-lg p-4 shadow-sm">
        <h3 className="text-sm font-semibold mb-2">Engine Details</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <span className="text-gray-400">Embedding model</span>
          <span className="font-mono">{data.embedding_model || "null (deterministic)"}</span>
          <span className="text-gray-400">Dimensions</span>
          <span className="font-mono">{data.index_dimensions}</span>
          <span className="text-gray-400">Total embeddings</span>
          <span className="font-mono">{data.total_embeddings.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <span className="text-xs text-gray-500">{label}</span>
      </div>
      <p className="text-xl font-bold text-gray-900">{value}</p>
    </div>
  );
}
