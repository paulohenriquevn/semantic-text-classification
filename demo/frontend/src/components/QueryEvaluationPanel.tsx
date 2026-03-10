import { AlertTriangle, BarChart3, ChevronDown, ChevronRight, Shield, Zap } from "lucide-react";
import { useState } from "react";
import type { QueryEvaluation } from "@/types/api";

// ---------------------------------------------------------------------------
// Quality score badge — circular with color coding
// ---------------------------------------------------------------------------

const QUALITY_TIERS = [
  { min: 80, label: "Excellent", color: "text-green-700 bg-green-100 border-green-300" },
  { min: 60, label: "Good", color: "text-blue-700 bg-blue-100 border-blue-300" },
  { min: 40, label: "Fair", color: "text-amber-700 bg-amber-100 border-amber-300" },
  { min: 0, label: "Poor", color: "text-red-700 bg-red-100 border-red-300" },
];

function getQualityTier(score: number) {
  return QUALITY_TIERS.find((t) => score >= t.min) ?? QUALITY_TIERS[QUALITY_TIERS.length - 1];
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface QueryEvaluationPanelProps {
  evaluation: QueryEvaluation;
}

export function QueryEvaluationPanel({ evaluation }: QueryEvaluationPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const { pre_execution: pre, post_execution: post } = evaluation;
  const tier = getQualityTier(post.quality_score);

  const hasWarnings =
    pre.threshold_warnings.length > 0 ||
    pre.pitfalls.length > 0 ||
    post.signal_warnings.length > 0;

  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm overflow-hidden">
      {/* Collapsed header — always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <BarChart3 className="h-4 w-4 text-gray-400" />
          <span className="text-sm font-medium text-gray-700">Query Evaluation</span>

          {/* Quality badge */}
          <span
            className={`inline-flex items-center gap-1 text-xs font-bold px-2 py-0.5 rounded-full border ${tier.color}`}
          >
            {post.quality_score}/100
            <span className="font-medium">{tier.label}</span>
          </span>

          {/* Warning indicator */}
          {hasWarnings && (
            <span className="inline-flex items-center gap-0.5 text-[10px] font-medium text-amber-600 bg-amber-50 px-1.5 py-0.5 rounded">
              <AlertTriangle className="h-3 w-3" />
              {pre.threshold_warnings.length + pre.pitfalls.length + post.signal_warnings.length}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Quick stats */}
          <div className="hidden sm:flex items-center gap-2 text-xs text-gray-400">
            <span>{pre.predicate_count} predicate{pre.predicate_count !== 1 ? "s" : ""}</span>
            <span className="text-gray-300">|</span>
            <span>{post.window_coverage_pct.toFixed(1)}% coverage</span>
          </div>
          {expanded ? (
            <ChevronDown className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronRight className="h-4 w-4 text-gray-400" />
          )}
        </div>
      </button>

      {/* Expanded details */}
      {expanded && (
        <div className="border-t border-gray-100 px-4 py-3 space-y-4">
          {/* Pre-execution analysis */}
          <div>
            <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-600 mb-2">
              <Shield className="h-3.5 w-3.5" />
              Pre-Execution Analysis
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <StatCard label="Predicates" value={pre.predicate_count} />
              <StatCard label="Complexity" value={pre.complexity} />
              <StatCard
                label="Families"
                value={pre.predicate_families.join(", ") || "none"}
              />
              <StatCard
                label="Missing"
                value={pre.missing_families.join(", ") || "none"}
                muted={pre.missing_families.length === 0}
              />
            </div>
          </div>

          {/* Post-execution analysis */}
          <div>
            <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-600 mb-2">
              <Zap className="h-3.5 w-3.5" />
              Post-Execution Analysis
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <StatCard label="Window Coverage" value={`${post.window_coverage_pct.toFixed(1)}%`} />
              <StatCard label="Conv. Coverage" value={`${post.conversation_coverage_pct.toFixed(1)}%`} />
              <StatCard label="Concentration" value={post.concentration_ratio.toFixed(2)} />
              <StatCard label="Quality" value={`${post.quality_score}/100`} highlight={tier.color} />
            </div>

            {/* Score distribution */}
            {post.score_distribution && (
              <div className="mt-2 bg-gray-50 rounded-lg px-3 py-2">
                <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">
                  Score Distribution
                </span>
                <div className="flex items-center gap-4 mt-1 text-xs text-gray-600">
                  <span>
                    min <span className="font-mono font-medium">{post.score_distribution.min.toFixed(3)}</span>
                  </span>
                  <span>
                    median <span className="font-mono font-medium">{post.score_distribution.median.toFixed(3)}</span>
                  </span>
                  <span>
                    mean <span className="font-mono font-medium">{post.score_distribution.mean.toFixed(3)}</span>
                  </span>
                  <span>
                    p90 <span className="font-mono font-medium">{post.score_distribution.p90.toFixed(3)}</span>
                  </span>
                  <span>
                    max <span className="font-mono font-medium">{post.score_distribution.max.toFixed(3)}</span>
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Warnings and pitfalls */}
          {hasWarnings && (
            <div>
              <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-600 mb-2">
                <AlertTriangle className="h-3.5 w-3.5" />
                Warnings & Pitfalls
              </div>
              <ul className="space-y-1">
                {pre.threshold_warnings.map((w, i) => (
                  <WarningItem key={`tw-${i}`} text={w} type="threshold" />
                ))}
                {pre.pitfalls.map((p, i) => (
                  <WarningItem key={`pf-${i}`} text={p} type="pitfall" />
                ))}
                {post.signal_warnings.map((w, i) => (
                  <WarningItem key={`sw-${i}`} text={w} type="signal" />
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatCard({
  label,
  value,
  muted,
  highlight,
}: {
  label: string;
  value: string | number;
  muted?: boolean;
  highlight?: string;
}) {
  return (
    <div className={`rounded-md px-2.5 py-1.5 ${highlight ? highlight : "bg-gray-50"}`}>
      <div className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">{label}</div>
      <div className={`text-sm font-semibold ${muted ? "text-gray-300" : "text-gray-700"}`}>
        {value}
      </div>
    </div>
  );
}

function WarningItem({ text, type }: { text: string; type: "threshold" | "pitfall" | "signal" }) {
  const colors = {
    threshold: "bg-orange-50 border-orange-200 text-orange-700",
    pitfall: "bg-yellow-50 border-yellow-200 text-yellow-700",
    signal: "bg-amber-50 border-amber-200 text-amber-700",
  };

  return (
    <li
      className={`text-xs px-2.5 py-1.5 rounded border ${colors[type]}`}
    >
      {text}
    </li>
  );
}
